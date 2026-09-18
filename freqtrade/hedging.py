"""Shared OKX hedge lifecycle. All callers serialize writes using the bot's exit lock.

Only this controller submits orders for linked trades. One durable pending intent
per group makes a lost response recoverable without issuing a duplicate order.
"""

import logging
from math import isclose, isfinite
from typing import TYPE_CHECKING
from uuid import uuid4

from freqtrade.constants import CUSTOM_TAG_MAX_LENGTH, NON_OPEN_EXCHANGE_STATES
from freqtrade.enums import ExitType, TradingMode
from freqtrade.exceptions import DependencyException, InvalidOrderException, OperationalException
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.hedge_group import HedgeGroup
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
from freqtrade.util import dt_now


if TYPE_CHECKING:
    from freqtrade.freqtradebot import FreqtradeBot

logger = logging.getLogger(__name__)


class HedgeManager:
    def __init__(self, bot: "FreqtradeBot") -> None:
        self.bot = bot
        self.enabled = bool(bot.config.get("hedge", {}).get("enabled", False))
        self._processing = False

    def validate(self) -> None:
        if not self.enabled:
            if HedgeGroup.active():
                raise OperationalException(
                    "Active hedge groups exist. Keep hedge enabled until closed."
                )
            return
        if self.bot.exchange.id != "okx":
            raise OperationalException("Hedging currently requires OKX.")
        if self.bot.strategy.hedge_requires_local_stoploss and (
            self.bot.strategy.order_types.get("stoploss_on_exchange")
            or any(t.has_open_sl_orders for t in Trade.get_open_trades())
        ):
            raise OperationalException(
                "This strategy's hedge trigger requires local stops: disable exchange stops "
                "(stoploss_on_exchange) and reconcile any existing exchange stop orders."
            )
        self.bot.strategy.validate_hedge()
        self.bot.exchange.validate_hedge_mode()
        for pair in self.bot.active_pair_whitelist:
            self.bot.exchange.validate_hedge_market(pair)

    def manages(self, trade: Trade) -> bool:
        return (
            self.enabled
            and (group := HedgeGroup.for_trade(trade.id)) is not None
            and (group.state != "closed")
        )

    def manages_pair(self, pair: str) -> bool:
        return self.enabled and any(
            self.manages(t) for t in Trade.get_trades_proxy(pair=pair, is_open=True)
        )

    def can_adopt_order(self, trade: Trade, order: dict) -> bool:
        return not self.enabled or (
            order.get("info", {}).get("posSide") == trade.trade_direction
            and order.get("side") == trade.exit_side
        )

    def observe_fill(self, trade: Trade, order: Order) -> None:
        if order.status in NON_OPEN_EXCHANGE_STATES and order.safe_filled > 0:
            self.evaluate(trade, order.safe_price, order=order)

    def evaluate(self, trade: Trade, current_rate: float, *, order: Order | None = None) -> bool:
        """Persist a strategy decision. The caller processes the group under the exit lock."""
        if (
            not self.enabled
            or not trade.is_open
            or trade.amount <= 0
            or not any(
                o.ft_order_side == trade.entry_side and o.safe_filled > 0 for o in trade.orders
            )
            or not isfinite(current_rate)
            or current_rate <= 0
            or HedgeGroup.for_trade(trade.id) is not None
        ):
            return False
        reason = strategy_safe_wrapper(self.bot.strategy.custom_hedge, supress_error=True)(
            pair=trade.pair,
            trade=trade,
            current_time=dt_now(),
            current_rate=current_rate,
            current_profit=trade.calc_profit_ratio(current_rate),
            order=order,
        )
        if reason is None:
            return False
        if not isinstance(reason, str) or not reason.strip():
            logger.warning("custom_hedge must return a nonempty reason string or None.")
            return False
        self.request_open(
            trade,
            trigger_order_id=order.order_id if order else None,
            trigger_source="strategy",
            trigger_reason=reason[:CUSTOM_TAG_MAX_LENGTH],
        )
        return True

    def request_open(
        self,
        trade: Trade,
        *,
        trigger_order_id: str | None = None,
        trigger_source: str = "manual",
        trigger_reason: str | None = None,
    ) -> HedgeGroup:
        """Record any trigger under the bot's exit lock; processing follows separately."""
        if not self.enabled:
            raise OperationalException("Enable hedge before requesting a hedge.")
        if existing := HedgeGroup.for_trade(trade.id):
            return existing
        if trade.exchange != "okx" or trade.trading_mode != TradingMode.FUTURES:
            raise OperationalException("Only OKX futures trades can be hedged.")
        has_fills = any(
            order.ft_order_side == trade.entry_side and order.safe_filled > 0
            for order in trade.orders
        )
        if not trade.is_open or not has_fills:
            raise DependencyException("An open trade with a filled position is required.")
        group = HedgeGroup(
            parent_id=trade.id,
            trigger_order_id=trigger_order_id,
            trigger_source=trigger_source,
            trigger_reason=trigger_reason,
            order_type=self.bot.config["hedge"].get("order_type", "market"),
            accounted_orders=[o.order_id for o in trade.orders if not o.ft_is_open],
        )
        Trade.session.add(group)
        Trade.commit()
        self.bot.notify_status(
            f"Hedge requested for trade {trade.id} ({trade.pair}). "
            "Automatic exits and DCA are suspended; both legs require manual closure."
        )
        return group

    def recover(self) -> None:
        """Catch a crash between persisting a parent fill and recording hedge intent."""
        if not self.enabled:
            return
        callback = self.bot.strategy.custom_hedge
        trades = (
            Trade.get_open_trades()
            if getattr(callback, "__func__", None) is not IStrategy.custom_hedge
            else []
        )
        for trade in trades:
            if HedgeGroup.for_trade(trade.id) is not None:
                continue
            for order in trade.orders:
                if order.status not in NON_OPEN_EXCHANGE_STATES or order.safe_filled <= 0:
                    continue
                # A crash may have persisted the fill before its strategy callback.
                # Callbacks must tolerate replay with the same order ID.
                strategy_safe_wrapper(self.bot.strategy.order_filled, supress_error=True)(
                    pair=trade.pair,
                    trade=trade,
                    order=order,
                    current_time=order.order_filled_utc or dt_now(),
                )
                self.observe_fill(trade, order)
                if HedgeGroup.for_trade(trade.id) is not None:
                    break
        self.process()

    def request_close(self, trade: Trade) -> bool:
        group = HedgeGroup.for_trade(trade.id) if self.enabled else None
        if group is None or group.state == "closed":
            return False
        group.close_requested = True
        # Resolve any in-flight opening before closing its actual filled amount.
        if not group.pending_client_id:
            group.state = "closing"
            group.last_error = None
        Trade.commit()
        self.process()
        return True

    def process(self) -> None:
        if not self.enabled or self._processing:
            return
        self._processing = True
        try:
            for group in HedgeGroup.active():
                try:
                    self._process_group(group)
                except (DependencyException, OperationalException) as exc:
                    self._error(group, str(exc))
                Trade.commit()
        finally:
            self._processing = False

    def _error(self, group: HedgeGroup, message: str) -> None:
        if message != group.last_error:
            logger.error("Hedge trade %s: %s", group.parent_id, message)
            group.last_error = message[:2048]
            Trade.commit()
            self.bot.notify_status(f"Hedge trade {group.parent_id}: {message}")

    def _cancel_orders(self, trade: Trade) -> bool:
        exchange = self.bot.exchange
        for order in list(trade.open_orders) + list(trade.open_sl_orders):
            is_stop = order.ft_order_side == "stoploss"
            try:
                if is_stop:
                    exchange.cancel_stoploss_order(order.order_id, trade.pair)
                elif order.order_type == "chase":
                    exchange.cancel_chase_order(order.order_id, trade.pair)
                else:
                    exchange.cancel_order(order.order_id, trade.pair)
            except InvalidOrderException:
                # Already filled/canceled is possible. Only a fresh fetch establishes state.
                pass
            result = exchange.fetch_order_or_stoploss_order(
                order.order_id, trade.pair, is_stop, order_type=order.order_type
            )
            if result.get("status") in NON_OPEN_EXCHANGE_STATES and (
                result.get("filled") is None or (result["filled"] > 0 and not result.get("average"))
            ):
                return False
            self.bot.update_trade_state(trade, order.order_id, result, stoploss_order=is_stop)
        return not trade.has_open_orders and not trade.has_open_sl_orders

    def _process_group(self, group: HedgeGroup) -> None:
        parent = Trade.session.get(Trade, group.parent_id)
        hedge = Trade.session.get(Trade, group.hedge_id) if group.hedge_id else None
        if parent is None:
            raise OperationalException("Linked parent trade is missing; manual repair required.")
        if (
            group.pending_client_id
            and not group.pending_exit
            and not group.close_requested
            and group.state != "blocked"
        ):
            self._check_pending_entry_positions(group, parent, hedge)
        if group.pending_client_id and not self._reconcile(group):
            return
        if group.close_requested:
            group.state = "closing"
        if group.state == "blocked":
            return
        if group.state == "preparing" and not self._cancel_orders(parent):
            return
        self._sync_positions(parent, hedge)
        if group.state == "held":
            # Holding is a terminal opening state. Manual changes NEVER reopen the hedge.
            if not parent.is_open and hedge and not hedge.is_open:
                group.state = "closed"
            return
        if group.state == "preparing":
            hedge = self._prepare(group, parent)
        if group.state == "opening":
            self._open(group, parent, hedge)
        elif group.state == "closing":
            self._close(group, parent, hedge)

    def _check_pending_entry_positions(
        self, group: HedgeGroup, parent: Trade, hedge: Trade | None
    ) -> None:
        """An indefinitely waiting entry must not undo a manual position reduction."""
        owned = parent.amount if parent.is_open else 0.0
        if not self.bot.config["dry_run"]:
            self.bot.wallets.update()
            owned = self.bot.wallets.get_owned(
                parent.pair, parent.safe_base_currency, parent.trade_direction
            )
        if not isclose(owned, group.target_amount, rel_tol=1e-9, abs_tol=1e-12):
            group.state = "blocked"
            self._error(group, "Parent changed during hedge opening. Close the group manually.")
            return
        if hedge is None or self.bot.config["dry_run"]:
            return
        filled = sum(o.safe_filled for o in hedge.orders if o.ft_order_side == hedge.entry_side)
        owned = self.bot.wallets.get_owned(
            hedge.pair, hedge.safe_base_currency, hedge.trade_direction
        )
        if owned < filled and not isclose(owned, filled, rel_tol=1e-9, abs_tol=1e-12):
            # Positions can lag a fresh fill. Require an identifiable exit before
            # treating the discrepancy as a manual reduction of the opening hedge.
            exits = self.bot.exchange.fetch_orders(hedge.pair, hedge.open_date_utc)
            if any(
                self.can_adopt_order(hedge, order) and (order.get("filled") or 0.0) > 0
                for order in exits
            ):
                group.state = "blocked"
                self._error(group, "Hedge reduced during opening. Close the group manually.")

    def _prepare(self, group: HedgeGroup, parent: Trade) -> Trade | None:
        if not parent.is_open or parent.amount <= 0:
            group.state = "closed"
            return None
        self.bot.exchange.validate_hedge_market(parent.pair)
        self.bot.wallets.update()
        if not self.bot.config["dry_run"]:
            opposite = "long" if parent.is_short else "short"
            if self.bot.wallets.get_owned(parent.pair, parent.safe_base_currency, opposite):
                group.state = "blocked"
                raise DependencyException("An unmanaged opposite position already exists.")
        group.target_amount = parent.amount
        hedge = self._new_hedge(parent, group.order_type)
        Trade.session.add(hedge)
        Trade.session.flush()
        group.hedge_id = hedge.id
        group.state = "opening"
        Trade.commit()
        return hedge

    def _open(self, group: HedgeGroup, parent: Trade, hedge: Trade | None) -> None:
        if hedge is None:
            raise OperationalException("Linked hedge trade is missing.")
        if hedge.nr_of_successful_exits:
            group.state = "blocked"
            raise DependencyException("Hedge reduced during opening. Close the group manually.")
        if not parent.is_open or not isclose(
            parent.amount, group.target_amount, rel_tol=1e-9, abs_tol=1e-12
        ):
            group.state = "blocked"
            raise DependencyException(
                "Parent changed during hedge opening. Close the group manually."
            )
        remaining = group.target_amount - hedge.amount
        if remaining < -1e-12:
            group.state = "blocked"
            raise DependencyException("Hedge exceeds target quantity; manual closure required.")
        if not isclose(remaining, 0.0, abs_tol=1e-12):
            self._submit(group, hedge, remaining, exiting=False)
        if (
            not group.pending_client_id
            and group.state != "blocked"
            and isclose(hedge.amount, group.target_amount, rel_tol=1e-9, abs_tol=1e-12)
        ):
            group.state = "held"
            group.last_error = None
            self.bot.notify_status(
                f"Hedge held: trades {parent.id}/{hedge.id}, "
                f"{group.target_amount} {parent.safe_base_currency} on each side. "
                "Force-exit either trade to close both."
            )

    def _close(self, group: HedgeGroup, *legs: Trade | None) -> None:
        for leg in legs:
            if leg is None or not leg.is_open:
                continue
            if not self._cancel_orders(leg):
                return
            if leg.amount > 0:
                self._submit(group, leg, leg.amount, exiting=True)
                return
            leg.is_open = False
            leg.close_date = dt_now()
        group.state = "closed"
        group.last_error = None

    def _sync_positions(self, *legs: Trade | None) -> None:
        """Adopt manual exchange reductions before sizing any subsequent operation."""
        if self.bot.config["dry_run"]:
            return
        self.bot.wallets.update()
        for leg in legs:
            if leg is None or not leg.is_open or leg.amount <= 0:
                continue
            owned = self.bot.wallets.get_owned(
                leg.pair, leg.safe_base_currency, leg.trade_direction
            )
            if isclose(owned, leg.amount, rel_tol=1e-9, abs_tol=1e-12):
                continue
            if owned < leg.amount:
                self.bot.handle_onexchange_order(leg)
            if leg.is_open and not isclose(owned, leg.amount, rel_tol=1e-9, abs_tol=1e-12):
                raise DependencyException(
                    f"Trade {leg.id} amount differs from its OKX {leg.trade_direction} position. "
                    "Waiting for identifiable fills; manual reconciliation may be required."
                )

    def _new_hedge(self, parent: Trade, order_type: str) -> Trade:
        rate = self.bot.exchange.get_rate(
            parent.pair, side="entry", is_short=not parent.is_short, refresh=True
        )
        fee = self.bot.exchange.get_fee(symbol=parent.pair, taker_or_maker="taker")
        entry_fee = (
            self.bot.exchange.get_fee(symbol=parent.pair, taker_or_maker="maker")
            if order_type == "chase"
            else fee
        )
        hedge = Trade(
            pair=parent.pair,
            base_currency=parent.base_currency,
            stake_currency=parent.stake_currency,
            exchange=parent.exchange,
            strategy=parent.strategy,
            timeframe=parent.timeframe,
            enter_tag=f"hedge:{parent.id}",
            amount=0.0,
            stake_amount=0.0,
            amount_requested=parent.amount,
            open_rate=rate,
            open_rate_requested=rate,
            open_date=dt_now(),
            is_open=True,
            is_short=not parent.is_short,
            leverage=parent.leverage,
            trading_mode=parent.trading_mode,
            fee_open=entry_fee,
            fee_close=fee,
            amount_precision=parent.amount_precision,
            price_precision=parent.price_precision,
            precision_mode=parent.precision_mode,
            precision_mode_price=parent.precision_mode_price,
            contract_size=parent.contract_size,
        )
        # Retain valid display fields; managed trades never execute these automatic stops.
        hedge.adjust_stop_loss(rate, self.bot.strategy.stoploss, initial=True)
        return hedge

    def _submit(self, group: HedgeGroup, trade: Trade, amount: float, *, exiting: bool) -> None:
        exchange = self.bot.exchange
        rounded = exchange.amount_to_contract_precision(trade.pair, amount)
        if (
            not isfinite(amount)
            or amount <= 0
            or not isclose(rounded, amount, rel_tol=1e-9, abs_tol=1e-12)
        ):
            group.state = "blocked"
            raise DependencyException("Exact hedge quantity is not representable in exchange lots.")
        rate = exchange.get_rate(
            trade.pair, side="exit" if exiting else "entry", is_short=trade.is_short, refresh=True
        )
        self.bot.wallets.update()
        if not exiting and self.bot.wallets.get_free(trade.stake_currency) < (
            amount * rate / trade.leverage + amount * rate * trade.fee_open
        ):
            group.state = "blocked"
            raise DependencyException("Insufficient free collateral for the entire hedge quantity.")
        if exiting:
            trade.set_funding_fees(
                exchange.get_funding_fees(
                    pair=trade.pair,
                    amount=trade.amount,
                    is_short=trade.is_short,
                    open_date=trade.date_last_filled_utc,
                )
            )
            trade.exit_reason = ExitType.FORCE_EXIT.value
            trade.close_rate_requested = rate
        group.pending_trade_id = trade.id
        group.pending_client_id = "fth" + uuid4().hex[:29]
        group.pending_amount = amount
        group.pending_rate = rate
        group.pending_exit = exiting
        Trade.commit()  # MUST precede exchange submission, including in dry-run.
        try:
            order = exchange.create_order(
                pair=trade.pair,
                ordertype="market" if exiting else group.order_type,
                side=trade.exit_side if exiting else trade.entry_side,
                amount=amount,
                rate=rate,
                leverage=trade.leverage,
                reduceOnly=exiting,
                client_order_id=group.pending_client_id,
            )
        except InvalidOrderException:
            # Definitive rejection: no order exists. Do not repeatedly retry it.
            group.clear_pending()
            group.state = "blocked"
            raise
        self._accept_order(group, order)

    def _reconcile(self, group: HedgeGroup) -> bool:
        if (
            group.pending_client_id is None
            or group.pending_amount is None
            or group.pending_rate is None
        ):
            raise OperationalException("Pending hedge intent is incomplete.")
        trade = Trade.session.get(Trade, group.pending_trade_id)
        if trade is None:
            raise OperationalException("Pending hedge trade is missing.")
        order_type = "market" if group.pending_exit else group.order_type
        if group.pending_order_id:
            if (group.close_requested or group.state == "blocked") and not group.pending_exit:
                try:
                    if order_type == "chase":
                        self.bot.exchange.cancel_chase_order(group.pending_order_id, trade.pair)
                    else:
                        self.bot.exchange.cancel_order(group.pending_order_id, trade.pair)
                except InvalidOrderException:
                    pass
            order = self.bot.exchange.fetch_order_or_stoploss_order(
                group.pending_order_id, trade.pair, order_type=order_type
            )
        else:
            order = self.bot.exchange.fetch_order_by_client_id(
                group.pending_client_id, trade.pair, order_type=order_type
            )
        if order is None:
            if self.bot.config["dry_run"]:
                # There was no external side effect. If a crash lost the in-memory simulated
                # order before it reached our database, replay this SAME simulated intent.
                order = self.bot.exchange.create_order(
                    pair=trade.pair,
                    ordertype=order_type,
                    side=trade.exit_side if group.pending_exit else trade.entry_side,
                    amount=group.pending_amount,
                    rate=group.pending_rate,
                    leverage=trade.leverage,
                    reduceOnly=group.pending_exit,
                    client_order_id=group.pending_client_id,
                )
                return self._accept_order(group, order)
            raise DependencyException(
                f"Awaiting definitive OKX result for client ID {group.pending_client_id}; "
                "no replacement order will be sent."
            )
        return self._accept_order(group, order)

    def _accept_order(self, group: HedgeGroup, order: dict) -> bool:
        trade = Trade.session.get(Trade, group.pending_trade_id)
        if trade is None:
            raise OperationalException("Pending hedge trade is missing.")
        expected_side = trade.exit_side if group.pending_exit else trade.entry_side
        if order.get("side") != expected_side or order.get("symbol") != trade.pair:
            raise OperationalException("Recovered hedge order has an unexpected symbol or side.")
        if order.get("clientOrderId") and order["clientOrderId"] != group.pending_client_id:
            raise OperationalException("Recovered hedge order has an unexpected client ID.")
        if (position_side := order.get("info", {}).get("posSide")) and (
            position_side != trade.trade_direction
        ):
            raise OperationalException("Recovered hedge order has an unexpected position side.")
        if (
            group.pending_amount is None
            or order.get("amount") is None
            or not isclose(order["amount"], group.pending_amount, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise OperationalException("Recovered hedge order has an unexpected amount.")
        group.pending_order_id = order["id"]
        # Do not let the ordinary fill handler book a terminal order at a placeholder
        # price. Keep the intent and fetch it again until authoritative fills arrive.
        if order.get("status") in NON_OPEN_EXCHANGE_STATES and (
            order.get("filled") is None or (order["filled"] > 0 and not order.get("average"))
        ):
            Trade.commit()
            return False
        existing = Order.order_by_id(order["id"], trade.pair)
        if existing and existing.ft_trade_id != trade.id:
            raise OperationalException("Recovered order already belongs to another trade.")
        if not existing:
            order_obj = Order.parse_from_ccxt_object(
                order, trade.pair, expected_side, group.pending_amount, group.pending_rate
            )
            order_obj.ft_order_tag = "hedge_close" if group.pending_exit else "hedge_open"
            trade.orders.append(order_obj)
        Trade.commit()
        self.bot.update_trade_state(trade, order["id"], order)
        if order.get("status") not in NON_OPEN_EXCHANGE_STATES:
            return False
        zero_fill = self.bot.exchange.check_order_canceled_empty(order)
        group.clear_pending()
        if group.state != "blocked":
            group.last_error = None
        if zero_fill:
            group.state = "blocked"
            self._error(group, "Hedge order terminated without a fill. Manual action required.")
        Trade.commit()
        self.bot.wallets.update()
        return True
