from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from freqtrade.enums import TradingMode
from freqtrade.exceptions import InvalidOrderException, TemporaryError
from freqtrade.hedging import HedgeManager
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.hedge_group import HedgeGroup
from tests.conftest import get_patched_freqtradebot
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy


PAIR = "ETH/USDT:USDT"


def ccxt_order(order_id, side, amount, *, status="closed", filled=None, price=80.0):
    filled = amount if filled is None else filled
    return {
        "id": order_id,
        "symbol": PAIR,
        "side": side,
        "type": "market",
        "status": status,
        "amount": amount,
        "filled": filled,
        "remaining": amount - filled,
        "price": price,
        "average": price,
        "cost": filled * price,
        "timestamp": 1788700000000,
        "fee": None,
        "info": {},
    }


@pytest.fixture
def hedge_bot(mocker, default_conf_usdt):
    bot = get_patched_freqtradebot(mocker, default_conf_usdt)
    bot.config.update({"hedge": {"enabled": True}, "trading_mode": TradingMode.FUTURES})
    bot.trading_mode = TradingMode.FUTURES
    bot.strategy = QFLDCAStrategy(bot.config)
    bot.strategy.max_safe_orders.value = 2
    bot.hedges = HedgeManager(bot)
    mocker.patch.object(bot.exchange, "get_rate", return_value=80.0)
    mocker.patch.object(bot.exchange, "get_fee", return_value=0.001)
    mocker.patch.object(bot.exchange, "get_funding_fees", return_value=0.0)
    mocker.patch.object(bot.exchange, "amount_to_contract_precision", side_effect=lambda p, a: a)
    mocker.patch.object(bot.exchange, "validate_hedge_market", create=True)
    mocker.patch.object(bot.exchange, "fetch_order_by_client_id", create=True, return_value=None)
    mocker.patch.object(bot, "_notify_enter")
    mocker.patch.object(bot, "_notify_exit")
    mocker.patch.object(bot, "handle_protections")
    mocker.patch.object(bot, "notify_status")
    return bot


def parent_trade(bot, *, is_short=False):
    trade = Trade(
        pair=PAIR,
        base_currency="ETH",
        stake_currency="USDT",
        exchange="okx",
        amount=0,
        stake_amount=0,
        open_rate=100,
        open_date=datetime.now(UTC),
        fee_open=0.001,
        fee_close=0.001,
        is_open=True,
        is_short=is_short,
        leverage=5.0,
        trading_mode=TradingMode.FUTURES,
        amount_precision=0.01,
        precision_mode=4,
        price_precision=0.01,
        precision_mode_price=4,
        contract_size=0.1,
        timeframe=5,
    )
    prefix = bot.strategy.SHORT_TAG_PREFIX if is_short else bot.strategy.LONG_TAG_PREFIX
    for index, amount in enumerate([1.0, 2.0, 4.0]):
        order = Order.parse_from_ccxt_object(
            ccxt_order(f"parent{index}", trade.entry_side, amount), PAIR, trade.entry_side
        )
        order.ft_order_tag = "base" if index == 0 else bot.strategy._safety_order_tag(prefix, index)
        trade.orders.append(order)
    trade.recalc_trade_from_orders()
    trade.adjust_stop_loss(trade.open_rate, -0.1, initial=True)
    Trade.session.add(trade)
    Trade.commit()
    return trade


def fill_exchange(bot, *, fill_fraction=1.0, status="closed"):
    def submit(**kwargs):
        # The recovery key and target trade exist durably before crossing the API boundary.
        groups = HedgeGroup.active()
        assert len(groups) == 1
        assert groups[0].pending_client_id == kwargs["client_order_id"]
        assert groups[0].pending_trade_id is not None
        return ccxt_order(
            kwargs["client_order_id"],
            kwargs["side"],
            kwargs["amount"],
            status=status,
            filled=kwargs["amount"] * fill_fraction,
        ) | {"type": kwargs["ordertype"]}

    bot.exchange.create_order = MagicMock(side_effect=submit)
    return bot.exchange.create_order


@pytest.mark.parametrize("is_short", [False, True])
def test_filled_safety_order_opens_entire_opposite_position_and_holds(hedge_bot, is_short):
    bot = hedge_bot
    trade = parent_trade(bot, is_short=is_short)
    submit = fill_exchange(bot)
    bot.update_trade_state(trade, trade.orders[-1].order_id, trade.orders[-1].to_ccxt_object())
    group = HedgeGroup.for_trade(trade.id)
    assert group is not None and group.state == "held"
    hedge = Trade.session.get(Trade, group.hedge_id)
    assert hedge.is_short is not is_short
    assert hedge.amount == trade.amount == 7.0
    assert hedge.leverage == trade.leverage
    assert submit.call_args.kwargs["amount"] == 7.0
    assert submit.call_args.kwargs["side"] == trade.exit_side
    assert submit.call_args.kwargs["ordertype"] == "market"
    assert submit.call_args.kwargs["reduceOnly"] is False
    assert bot.hedges.manages(trade) and bot.hedges.manages(hedge)
    bot.check_and_call_adjust_trade_position(trade)
    bot.check_and_call_adjust_trade_position(hedge)
    assert bot.exit_positions([trade, hedge]) == 0
    # Repeated callbacks and a fresh manager must not submit another hedge.
    bot.hedges = HedgeManager(bot)
    bot.update_trade_state(trade, trade.orders[-1].order_id, trade.orders[-1].to_ccxt_object())
    bot.hedges.process()
    assert submit.call_count == 1


@pytest.mark.parametrize(
    "index,status,filled", [(0, "closed", 1), (1, "closed", 2), (2, "open", 2), (2, "canceled", 2)]
)
def test_only_selected_fully_completed_safety_order_triggers(hedge_bot, index, status, filled):
    trade = parent_trade(hedge_bot)
    order = trade.orders[index]
    order.status, order.filled = status, filled
    hedge_bot.hedges.observe_fill(trade, order)
    assert HedgeGroup.for_trade(trade.id) is None


def test_uncertain_submission_is_looked_up_after_restart_without_resubmitting(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    bot.wallets.update()
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    bot.exchange.create_order = MagicMock(side_effect=TemporaryError("response lost"))
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    client_id = group.pending_client_id
    assert client_id and group.state == "opening"
    assert "response lost" in group.last_error
    bot.hedges = HedgeManager(bot)
    bot.hedges.process()  # absent order is not proof that submission failed
    assert bot.exchange.create_order.call_count == 1
    bot.exchange.fetch_order_by_client_id.return_value = ccxt_order(client_id, "sell", 7.0)
    bot.wallets.get_owned = MagicMock(return_value=7.0)
    bot.hedges.process()
    assert group.state == "held"
    assert Trade.session.get(Trade, group.hedge_id).amount == 7.0
    assert bot.exchange.create_order.call_count == 1


def test_confirmed_partial_fill_only_submits_remaining_quantity(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    submit = fill_exchange(bot, fill_fraction=0.5, status="canceled")
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "opening"
    submit = fill_exchange(bot)
    bot.hedges.process()
    assert submit.call_args.kwargs["amount"] == 3.5
    assert group.state == "held"


def test_manual_group_close_does_not_reopen_hedge(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    submit = fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    hedge = Trade.session.get(Trade, group.hedge_id)
    bot.hedges.request_close(hedge)
    for _ in range(3):
        bot.hedges.process()
    assert not trade.is_open and not hedge.is_open
    assert group.state == "closed"
    assert submit.call_count == 3
    assert all(c.kwargs["reduceOnly"] for c in submit.call_args_list[1:])
    bot.hedges.process()
    assert submit.call_count == 3


def test_parent_exit_racing_cancellation_reduces_hedge_target(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    pending = ccxt_order("parent_tp", trade.exit_side, 2.0, status="open", filled=0.0)
    trade.orders.append(Order.parse_from_ccxt_object(pending, PAIR, trade.exit_side))
    Trade.commit()
    bot.exchange.cancel_order = MagicMock(return_value={"id": "parent_tp"})
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(
        return_value=ccxt_order("parent_tp", trade.exit_side, 2.0)
    )
    submit = fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[2])
    bot.hedges.process()
    assert trade.amount == 5.0
    assert submit.call_args.kwargs["amount"] == 5.0
    assert HedgeGroup.for_trade(trade.id).state == "held"


def test_cancel_ack_without_terminal_snapshot_never_opens_hedge(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    pending = ccxt_order("parent_tp", trade.exit_side, 2.0, status="open", filled=0.0)
    trade.orders.append(Order.parse_from_ccxt_object(pending, PAIR, trade.exit_side))
    Trade.commit()
    bot.exchange.cancel_order = MagicMock(return_value={"id": "parent_tp"})
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(
        side_effect=InvalidOrderException("unknown")
    )
    submit = fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[2])
    bot.hedges.process()
    assert HedgeGroup.for_trade(trade.id).state == "preparing"
    submit.assert_not_called()


def test_terminal_snapshot_without_fill_details_is_not_accounted_as_complete(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    result = ccxt_order("missing_details", "sell", 7.0)
    result["average"] = None
    bot.exchange.create_order = MagicMock(return_value=result)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    assert group.pending_client_id
    assert group.state == "opening"
    assert Trade.session.get(Trade, group.hedge_id).amount == 0


def test_manual_close_cancels_pending_opening_before_closing_filled_legs(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    fill_exchange(bot, status="open", fill_fraction=0.5)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    pending_id = group.pending_order_id
    bot.exchange.cancel_order = MagicMock()
    bot.exchange.fetch_order = MagicMock(
        return_value=ccxt_order(pending_id, "sell", 7.0, status="canceled", filled=3.5)
    )
    submit = fill_exchange(bot)
    bot.hedges.request_close(trade)
    bot.hedges.process()
    bot.hedges.process()
    assert bot.exchange.cancel_order.call_args.args == (pending_id, PAIR)
    assert not trade.is_open
    assert not Trade.session.get(Trade, group.hedge_id).is_open
    assert [c.kwargs["amount"] for c in submit.call_args_list] == [7.0, 3.5]
    assert group.state == "closed"


def test_insufficient_collateral_blocks_without_clipping_quantity(hedge_bot, mocker):
    trade = parent_trade(hedge_bot)
    submit = fill_exchange(hedge_bot)
    mocker.patch.object(hedge_bot.wallets, "get_free", return_value=1.0)
    hedge_bot.hedges.observe_fill(trade, trade.orders[-1])
    hedge_bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "blocked"
    assert "collateral" in group.last_error
    assert group.target_amount == trade.amount
    submit.assert_not_called()


def test_restart_after_partial_close_accounting_does_not_apply_fill_twice(hedge_bot, monkeypatch):
    bot = hedge_bot
    trade = parent_trade(bot)
    fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    # Crash exactly after terminal accounting commits and before intent is cleared.
    fill_exchange(bot, status="canceled", fill_fraction=0.5)
    clear_pending = HedgeGroup.clear_pending

    def crash(_group):
        raise RuntimeError("simulated crash after accounting")

    monkeypatch.setattr(HedgeGroup, "clear_pending", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        bot.hedges.request_close(trade)
    assert trade.is_open and trade.amount == 3.5
    assert group.pending_order_id in group.accounted_orders
    order_id = group.pending_order_id
    Trade.session.expire_all()
    monkeypatch.setattr(HedgeGroup, "clear_pending", clear_pending)
    bot.exchange.fetch_order = MagicMock(
        return_value=ccxt_order(order_id, "sell", 7.0, status="canceled", filled=3.5)
    )
    submit = fill_exchange(bot)
    bot.hedges = HedgeManager(bot)
    bot.hedges.process()
    assert submit.call_args.kwargs["amount"] == 3.5
    assert submit.call_args.kwargs["side"] == "sell"
    bot.hedges.process()
    bot.hedges.process()
    assert group.state == "closed"


def test_parent_cancel_without_fill_details_keeps_preparing(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    order = ccxt_order("tp_race", "sell", 2.0, status="open", filled=0.0)
    trade.orders.append(Order.parse_from_ccxt_object(order, PAIR, "sell"))
    Trade.commit()
    bot.exchange.cancel_order = MagicMock()
    result = {**order, "status": "closed", "filled": None}
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(return_value=result)
    submit = fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[2])
    bot.hedges.process()
    assert trade.has_open_orders
    assert HedgeGroup.for_trade(trade.id).state == "preparing"
    submit.assert_not_called()


def test_manual_okx_parent_close_during_opening_does_not_block_hedge_close(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    fill_exchange(bot, status="open", fill_fraction=0.5)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    pending_id = group.pending_order_id
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    bot.wallets.get_owned = MagicMock(side_effect=lambda p, b, side: 0.0 if side == "long" else 3.5)
    bot.exchange.cancel_order = MagicMock()
    bot.exchange.fetch_order = MagicMock(
        return_value=ccxt_order(pending_id, "sell", 7.0, status="canceled", filled=3.5)
    )
    manual_exit = ccxt_order("manual_parent_close", "sell", 7.0)
    manual_exit["info"] = {"posSide": "long"}
    bot.exchange.fetch_orders = MagicMock(return_value=[manual_exit])
    submit = fill_exchange(bot)
    bot.hedges.request_close(trade)
    bot.hedges.process()
    assert not trade.is_open
    assert group.state == "closed"
    assert submit.call_count == 1
    assert submit.call_args.kwargs["side"] == "buy"
    assert submit.call_args.kwargs["amount"] == 3.5


def test_definitive_exchange_rejection_still_allows_manual_close(hedge_bot):
    from freqtrade.exceptions import InsufficientFundsError

    bot = hedge_bot
    trade = parent_trade(bot)
    bot.exchange.create_order = MagicMock(side_effect=InsufficientFundsError("margin rejected"))
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "blocked" and group.pending_client_id is None
    submit = fill_exchange(bot)
    bot.hedges.request_close(trade)
    bot.hedges.process()
    assert group.state == "closed"
    assert submit.call_count == 1
    assert submit.call_args.kwargs["reduceOnly"] is True


def test_force_exit_rpc_closes_group_and_rejects_partial_requests(hedge_bot):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC, RPCException

    bot = hedge_bot
    bot.state = State.RUNNING
    trade = parent_trade(bot)
    fill_exchange(bot)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    rpc = RPC(bot)
    status = rpc._rpc_trade_status()
    assert len(status) == 2
    assert all(t["hedge"]["state"] == "held" for t in status)
    assert {t["hedge"]["hedge_trade_id"] for t in status} == {group.hedge_id}
    for trade_id in (str(trade.id), "all"):
        with pytest.raises(RPCException, match="full market group close"):
            rpc._rpc_force_exit(trade_id, amount=1.0)
    result = rpc._rpc_force_exit(str(group.hedge_id))
    assert "both hedge legs" in result["result"]
    bot.hedges.process()
    bot.hedges.process()
    assert group.state == "closed"


@pytest.mark.parametrize("order_type", ["market", "limit", "chase"])
def test_dry_run_restarts_an_unrecorded_simulated_order_with_same_id(hedge_bot, order_type):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = order_type
    trade = parent_trade(bot)
    bot.exchange.create_order = MagicMock(side_effect=TemporaryError("simulated interruption"))
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    client_id = group.pending_client_id
    bot.config["hedge"]["order_type"] = "market"
    submit = fill_exchange(bot)
    bot.hedges = HedgeManager(bot)
    bot.hedges.process()
    assert submit.call_args.kwargs["client_order_id"] == client_id
    assert submit.call_args.kwargs["ordertype"] == order_type
    assert group.state == "held"


@pytest.mark.parametrize("order_type", ["limit", "chase"])
@pytest.mark.parametrize("is_short", [False, True])
def test_hedge_waits_for_configured_order_across_restart(hedge_bot, order_type, is_short):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = order_type
    trade = parent_trade(bot, is_short=is_short)
    submit = fill_exchange(bot, status="open", fill_fraction=0.25)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    assert submit.call_args.kwargs["ordertype"] == order_type
    assert submit.call_args.kwargs["rate"] == 80.0
    assert group.to_json()["order_type"] == order_type
    order_id = group.pending_order_id
    pending = ccxt_order(order_id, trade.exit_side, 7.0, status="open", filled=1.75)
    pending["type"] = order_type
    fetch = MagicMock(return_value=pending)
    if order_type == "chase":
        bot.exchange.fetch_chase_order = fetch
    else:
        bot.exchange.fetch_order = fetch
    # A new setting applies to future groups, never an existing opening hedge.
    bot.config["hedge"]["order_type"] = "market"
    bot.config["unfilledtimeout"] = {"entry": 0, "exit": 0, "unit": "seconds"}
    bot.strategy.check_entry_timeout = MagicMock(return_value=True)
    Trade.session.expire_all()
    bot.hedges = HedgeManager(bot)
    for _ in range(3):
        bot.manage_open_orders()
        bot.hedges.process()
    assert submit.call_count == 1
    assert group.state == "opening"
    assert group.pending_order_id == order_id
    assert bot.exit_positions([trade]) == 0
    bot.check_and_call_adjust_trade_position(trade)
    assert submit.call_count == 1
    fetch.return_value = ccxt_order(order_id, trade.exit_side, 7.0) | {"type": order_type}
    bot.hedges.process()
    assert group.state == "held"
    assert Trade.session.get(Trade, group.hedge_id).amount == trade.amount == 7.0
    assert submit.call_count == 1


@pytest.mark.parametrize("order_type", ["limit", "chase"])
def test_hedge_lost_response_recovers_persisted_order_type(hedge_bot, order_type):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = order_type
    trade = parent_trade(bot)
    bot.wallets.update()
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    bot.exchange.create_order = MagicMock(side_effect=TemporaryError("response lost"))
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    client_id = group.pending_client_id
    bot.config["hedge"]["order_type"] = "market"
    Trade.session.expire_all()
    bot.hedges = HedgeManager(bot)
    bot.hedges.process()
    bot.exchange.fetch_order_by_client_id.assert_called_with(client_id, PAIR, order_type=order_type)
    assert bot.exchange.create_order.call_count == 1
    bot.exchange.fetch_order_by_client_id.return_value = ccxt_order("recovered", "sell", 7.0) | {
        "type": order_type,
        "clientOrderId": client_id,
    }
    bot.wallets.get_owned = MagicMock(return_value=7.0)
    bot.hedges.process()
    assert group.state == "held"
    assert bot.exchange.create_order.call_count == 1


def test_manual_close_waits_for_chase_cancel_final_fills(hedge_bot):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = "chase"
    trade = parent_trade(bot)
    fill_exchange(bot, status="open", fill_fraction=0.25)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    order_id = group.pending_order_id
    bot.exchange.cancel_chase_order = MagicMock(return_value={"id": order_id})
    bot.exchange.fetch_chase_order = MagicMock(
        return_value=ccxt_order(order_id, "sell", 7.0, status="open", filled=1.75)
        | {"type": "chase"}
    )
    submit = fill_exchange(bot)
    bot.hedges.request_close(trade)
    submit.assert_not_called()
    bot.exchange.cancel_chase_order.assert_called_once_with(order_id, PAIR)
    # More contracts filled while the cancellation was in flight.
    bot.exchange.fetch_chase_order.return_value = ccxt_order(
        order_id, "sell", 7.0, status="canceled", filled=3.5
    ) | {"type": "chase"}
    for _ in range(3):
        bot.hedges.process()
    assert group.state == "closed"
    assert [c.kwargs["amount"] for c in submit.call_args_list] == [7.0, 3.5]
    assert all(c.kwargs["ordertype"] == "market" for c in submit.call_args_list)
    assert all(c.kwargs["reduceOnly"] for c in submit.call_args_list)


@pytest.mark.parametrize("order_type", ["limit", "chase"])
def test_partial_terminal_hedge_keeps_order_type_for_remainder(hedge_bot, order_type):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = order_type
    trade = parent_trade(bot)
    fill_exchange(bot, status="canceled", fill_fraction=0.5)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    bot.config["hedge"]["order_type"] = "market"
    submit = fill_exchange(bot)
    bot.hedges.process()
    assert submit.call_args.kwargs["amount"] == 3.5
    assert submit.call_args.kwargs["ordertype"] == order_type
    assert HedgeGroup.for_trade(trade.id).state == "held"


@pytest.mark.parametrize("order_type", ["limit", "chase"])
@pytest.mark.parametrize("changed_leg", ["parent", "hedge"])
def test_manual_position_reduction_cancels_waiting_hedge(hedge_bot, order_type, changed_leg):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = order_type
    trade = parent_trade(bot)
    submit = fill_exchange(bot, status="open", fill_fraction=0.5)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    order_id = group.pending_order_id
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    bot.wallets.get_owned = MagicMock(
        side_effect=lambda p, b, side: (
            (3.5 if changed_leg == "parent" else 7.0)
            if side == "long"
            else (1.75 if changed_leg == "hedge" else 3.5)
        )
    )
    manual_exit = ccxt_order("manualExit", "buy", 1.75)
    manual_exit["info"] = {"posSide": "short"}
    bot.exchange.fetch_orders = MagicMock(return_value=[manual_exit])
    cancel = MagicMock(return_value={"id": order_id})
    fetch = MagicMock(
        return_value=ccxt_order(order_id, "sell", 7.0, status="open", filled=3.5)
        | {"type": order_type}
    )
    if order_type == "chase":
        bot.exchange.cancel_chase_order = cancel
        bot.exchange.fetch_chase_order = fetch
    else:
        bot.exchange.cancel_order = cancel
        bot.exchange.fetch_order = fetch
    bot.hedges.process()
    cancel.assert_called_once_with(order_id, PAIR)
    assert group.state == "blocked"
    assert group.pending_order_id == order_id
    fetch.return_value = ccxt_order(order_id, "sell", 7.0, status="canceled", filled=3.5) | {
        "type": order_type
    }
    bot.hedges.process()
    bot.hedges.process()
    assert group.state == "blocked" and group.pending_order_id is None
    assert group.last_error
    assert submit.call_count == 1


@pytest.mark.parametrize("state", ["opening", "held"])
@pytest.mark.parametrize("changed_leg", ["parent", "hedge"])
def test_small_unexplained_wallet_difference_never_resizes_or_refills_hedge(
    hedge_bot,
    state,
    changed_leg,
):
    bot = hedge_bot
    parent = parent_trade(bot)
    submit = fill_exchange(bot)
    bot.hedges.observe_fill(parent, parent.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(parent.id)
    hedge = Trade.session.get(Trade, group.hedge_id)
    group.state = state
    Trade.commit()
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    changed_side = parent.trade_direction if changed_leg == "parent" else hedge.trade_direction
    bot.wallets.get_owned = MagicMock(
        side_effect=lambda p, b, side: 6.93 if side == changed_side else 7
    )
    bot.exchange.fetch_orders = MagicMock(return_value=[])
    bot.hedges.process()
    assert parent.amount == hedge.amount == 7.0
    assert submit.call_count == 1
    assert "Waiting for identifiable fills" in group.last_error


def test_manual_hedge_reduction_at_terminal_fill_never_reopens(hedge_bot):
    bot = hedge_bot
    bot.config["hedge"]["order_type"] = "chase"
    trade = parent_trade(bot)
    submit = fill_exchange(bot, status="open", fill_fraction=0.0)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    bot.hedges.process()
    group = HedgeGroup.for_trade(trade.id)
    bot.config["dry_run"] = False
    bot.wallets.update = MagicMock()
    bot.wallets.get_owned = MagicMock(side_effect=lambda p, b, side: 7.0 if side == "long" else 3.5)
    bot.exchange.fetch_chase_order = MagicMock(
        return_value=ccxt_order(group.pending_order_id, "sell", 7.0) | {"type": "chase"}
    )
    manual_exit = ccxt_order("manualHedgeExit", "buy", 3.5)
    manual_exit["info"] = {"posSide": "short"}
    bot.exchange.fetch_orders = MagicMock(return_value=[manual_exit])
    bot.hedges.process()
    assert group.state == "blocked"
    assert Trade.session.get(Trade, group.hedge_id).amount == 3.5
    assert submit.call_count == 1


@pytest.mark.parametrize("order_type", ["market", "limit", "chase"])
@pytest.mark.parametrize("is_short", [False, True])
def test_manual_hedge_rpc_uses_same_rules_before_selected_safety_order(
    hedge_bot, order_type, is_short
):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC

    bot = hedge_bot
    bot.state = State.PAUSED
    bot.config["hedge"].update({"after_safety_order": 9, "order_type": order_type})
    bot.config["force_entry_enable"] = False
    trade = parent_trade(bot, is_short=is_short)
    submit = fill_exchange(bot)
    rpc = RPC(bot)
    result = rpc._rpc_hedge(str(trade.id))
    group = HedgeGroup.for_trade(trade.id)
    hedge = Trade.session.get(Trade, group.hedge_id)
    assert result["hedge"]["state"] == "held"
    assert result["hedge"]["trigger_source"] == "manual"
    assert group.trigger_order_id is None
    assert hedge.amount == trade.amount == 7.0
    assert hedge.is_short is not is_short
    assert submit.call_args.kwargs["ordertype"] == order_type
    assert submit.call_args.kwargs["reduceOnly"] is False
    assert bot.exit_positions([trade, hedge]) == 0
    bot.check_and_call_adjust_trade_position(trade)
    bot.check_and_call_adjust_trade_position(hedge)
    # Repeated commands on either leg reuse the same group.
    assert rpc._rpc_hedge(str(trade.id))["hedge"] == result["hedge"]
    assert rpc._rpc_hedge(str(hedge.id))["hedge"] == result["hedge"]
    assert submit.call_count == 1
    rpc._rpc_force_exit(str(hedge.id))
    bot.hedges.process()
    bot.hedges.process()
    assert group.state == "closed"


@pytest.mark.parametrize("reason", ["disabled", "stopped", "closed", "empty", "invalid", "spot"])
def test_manual_hedge_rpc_rejects_ineligible_request(hedge_bot, reason):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC, RPCException

    bot = hedge_bot
    bot.state = State.RUNNING
    trade = parent_trade(bot)
    trade_id = str(trade.id)
    if reason == "disabled":
        bot.hedges.enabled = False
    elif reason == "stopped":
        bot.state = State.STOPPED
    elif reason == "closed":
        trade.is_open = False
    elif reason == "empty":
        trade.amount = 0.0
        for order in trade.orders:
            order.filled = 0.0
    elif reason == "invalid":
        trade_id = "all"
    elif reason == "spot":
        trade.trading_mode = TradingMode.SPOT
    Trade.commit()
    submit = fill_exchange(bot)
    with pytest.raises(RPCException):
        RPC(bot)._rpc_hedge(trade_id)
    assert HedgeGroup.for_trade(trade.id) is None
    submit.assert_not_called()


def test_manual_hedge_cancels_working_safety_order_before_sizing(hedge_bot):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC

    bot = hedge_bot
    bot.state = State.RUNNING
    trade = parent_trade(bot)
    pending = ccxt_order("working_so", "buy", 2.0, status="open", filled=0.5)
    trade.orders.append(Order.parse_from_ccxt_object(pending, PAIR, "buy"))
    Trade.commit()
    bot.exchange.cancel_order = MagicMock()
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(
        return_value=ccxt_order("working_so", "buy", 2.0, status="canceled", filled=1.0)
    )
    submit = fill_exchange(bot)
    RPC(bot)._rpc_hedge(str(trade.id))
    assert trade.amount == 8.0
    assert submit.call_args.kwargs["amount"] == 8.0
    bot.exchange.cancel_order.assert_called_once_with("working_so", PAIR)


def test_manual_hedge_request_survives_restart_while_canceling_parent_order(hedge_bot):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC

    bot = hedge_bot
    bot.state = State.RUNNING
    bot.config["hedge"]["order_type"] = "chase"
    trade = parent_trade(bot)
    pending = ccxt_order("working_tp", "sell", 2.0, status="open", filled=0.0)
    trade.orders.append(Order.parse_from_ccxt_object(pending, PAIR, "sell"))
    Trade.commit()
    bot.exchange.cancel_order = MagicMock()
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(return_value=pending)
    submit = fill_exchange(bot)
    result = RPC(bot)._rpc_hedge(str(trade.id))
    assert result["hedge"]["state"] == "preparing"
    submit.assert_not_called()
    bot.config["hedge"]["order_type"] = "market"
    Trade.session.expire_all()
    bot.hedges = HedgeManager(bot)
    bot.exchange.fetch_order_or_stoploss_order.return_value = ccxt_order(
        "working_tp", "sell", 2.0, status="canceled", filled=0.0
    )
    bot.hedges.recover()
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "held" and group.trigger_order_id is None
    assert submit.call_count == 1
    assert submit.call_args.kwargs["ordertype"] == "chase"


def test_manual_hedge_of_partial_base_fill_reconciles_quantity(hedge_bot):
    from freqtrade.enums import State
    from freqtrade.rpc import RPC

    bot = hedge_bot
    bot.state = State.RUNNING
    trade = parent_trade(bot)
    trade.orders = [trade.orders[0]]
    base = trade.orders[0]
    base.status, base.ft_is_open, base.filled = "open", True, 0.5
    trade.amount = trade.stake_amount = 0.0
    Trade.commit()
    bot.exchange.cancel_order = MagicMock()
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(
        return_value=ccxt_order(base.order_id, "buy", 1.0, status="canceled", filled=0.75)
    )
    submit = fill_exchange(bot)
    RPC(bot)._rpc_hedge(str(trade.id))
    assert trade.amount == 0.75
    assert submit.call_args.kwargs["amount"] == 0.75
    assert HedgeGroup.for_trade(trade.id).state == "held"
