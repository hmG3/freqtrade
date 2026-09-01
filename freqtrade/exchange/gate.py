"""Gate.io exchange subclass"""

import logging
from datetime import datetime

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import MarginMode, PriceType, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    InvalidOrderException,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas


logger = logging.getLogger(__name__)


class Gate(Exchange):
    """Gate.io exchange class.
    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    unified_account = False
    position_mode: str | None = None

    _ft_has: FtHas = {
        "order_time_in_force": ["GTC", "IOC"],
        "stoploss_on_exchange": True,
        "stoploss_order_types": {"limit": "limit"},
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_query_requires_stop_flag": True,
        "stoploss_algo_order_info_id": "fired_order_id",
        "l2_limit_upper": 1000,
        "marketOrderRequiresPrice": True,
        "trades_has_history": False,  # Endpoint would support this - but ccxt doesn't.
    }

    _ft_has_futures: FtHas = {
        "chase_order": True,
        "cross_margin_stake_currencies": ["USDT"],
        "cross_liquidation_price_fallback": True,
        "needs_trading_fees": True,
        "marketOrderRequiresPrice": False,
        "funding_fee_candle_limit": 90,
        "stop_price_type_field": "price_type",
        "l2_limit_upper": 300,
        "stoploss_blocks_assets": False,
        "stoploss_algo_order_info_id": "trade_id",
        "stop_price_type_value_mapping": {
            PriceType.LAST: 0,
            PriceType.MARK: 1,
            PriceType.INDEX: 2,
        },
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        # (TradingMode.MARGIN, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if not self._config["dry_run"]:
                self._api.load_unified_status()
                is_unified = self._api.options.get("unifiedAccount")

                # Returns a tuple of bools, first for margin, second for Account
                if is_unified:
                    self.unified_account = True
                    logger.info("Gate: Unified account.")
                else:
                    self.unified_account = False
                    logger.info("Gate: Classic account.")

                if (
                    self.trading_mode == TradingMode.FUTURES
                    and self.margin_mode == MarginMode.CROSS
                ):
                    settle = self._config["stake_currency"].lower()
                    account = self._api.fetch2(
                        "{settle}/accounts",
                        ["private", "futures"],
                        "GET",
                        {"settle": settle},
                    )
                    self._log_exchange_response("fetch_cross_margin_account", account)
                    try:
                        account_margin_mode_raw = account.get("margin_mode")
                    except AttributeError as exc:
                        raise OperationalException(
                            "Gate cross margin requires a futures account in classic or "
                            "single-currency margin mode."
                        ) from exc
                    valid_margin_mode = (
                        type(account_margin_mode_raw) is int and account_margin_mode_raw in (0, 3)
                    ) or (
                        type(account_margin_mode_raw) is str
                        and account_margin_mode_raw in ("0", "3")
                    )
                    position_mode = account.get("position_mode")
                    if not valid_margin_mode or position_mode not in (
                        "single",
                        "dual",
                        "dual_plus",
                    ):
                        raise OperationalException(
                            "Gate cross margin requires a classic or single-currency futures "
                            "account with a supported position mode."
                        )
                    self.position_mode = position_mode
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_params(
        self,
        side: BuySell,
        ordertype: str,
        leverage: float,
        reduceOnly: bool,
        time_in_force: str = "GTC",
    ) -> dict:
        params = super()._get_params(
            side=side,
            ordertype=ordertype,
            leverage=leverage,
            reduceOnly=reduceOnly,
            time_in_force=time_in_force,
        )
        if ordertype == "market" and self.trading_mode == TradingMode.FUTURES:
            params["type"] = "market"
            params.update({"timeInForce": "IOC"})
        if self.margin_mode == MarginMode.CROSS and self.position_mode == "dual_plus":
            params["pos_margin_mode"] = "cross"
        return params

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(side, ordertype, stop_price)
        if self.margin_mode == MarginMode.CROSS and self.position_mode == "dual_plus":
            params["pos_margin_mode"] = "cross"
        return params

    def _lev_prep(
        self,
        pair: str,
        leverage: float,
        side: BuySell,
        accept_fail: bool = False,
    ) -> None:
        if self.trading_mode != TradingMode.SPOT:
            self._set_leverage(
                leverage,
                pair,
                accept_fail,
                params={"marginMode": self.margin_mode.value},
            )

    @staticmethod
    def _chase_amount_to_string(amount: float) -> str:
        return str(int(amount)) if float(amount).is_integer() else str(amount)

    def _chase_market_params(self, pair: str) -> tuple[str, str]:
        market = self.markets[pair]
        return market["id"], market["settle"].lower()

    def _is_valid_liquidation_price(self, liquidation_price: float | None) -> bool:
        if liquidation_price is None or not super()._is_valid_liquidation_price(liquidation_price):
            return False
        return float(liquidation_price) < 99_999_999.0

    def create_chase_order(
        self,
        pair: str,
        side: BuySell,
        amount: float,
        rate: float | None,
        params: dict,
    ) -> CcxtOrder:
        contract, settle = self._chase_market_params(pair)
        signed_amount = amount if side == "buy" else -amount
        request = {
            "settle": settle,
            "contract": contract,
            "amount": self._chase_amount_to_string(signed_amount),
            "price_limit": "0",
            "reduce_only": bool(params.get("reduceOnly", False)),
            "price_type": 1,
        }
        if self.margin_mode == MarginMode.CROSS and self.position_mode == "dual_plus":
            request["pos_margin_mode"] = "cross"
        response = self._api.fetch2(
            "{settle}/autoorder/v1/chase/create", ["private", "futures"], "POST", request
        )
        self._log_exchange_response("create_chase_order", response)
        item = response.get("order", response)
        if not item.get("id"):
            raise ccxt.InvalidOrder(item.get("error_label") or str(response))
        return {
            "id": str(item["id"]),
            "symbol": pair,
            "type": "chase",
            "side": side,
            "price": rate,
            "average": None,
            "amount": amount,
            "filled": 0.0,
            "remaining": amount,
            "cost": 0.0,
            "status": "open",
            "fee": {},
            "info": item,
        }

    def _normalize_chase_order(self, pair: str, item: dict) -> CcxtOrder:
        signed_amount = float(item.get("amount") or 0.0)
        amount = abs(signed_amount)
        filled = abs(float(item.get("fill_amount") or 0.0))
        remaining = max(amount - filled, 0.0)
        average_raw = item.get("average_fill_price")
        average = None if average_raw is None or average_raw == "" else float(average_raw)
        price_raw = item.get("suborder_price")
        price = average if price_raw is None or price_raw == "" else float(price_raw)
        raw_status = str(item.get("status") or "").lower()
        status_code = item.get("status_code")
        failure = bool(item.get("error_label")) or status_code not in (None, "", "0", 0)
        if raw_status == "open":
            status = "open"
        elif raw_status == "finished" and amount > 0 and filled >= amount:
            status = "closed"
        elif raw_status in ("failed", "rejected") or (
            raw_status == "finished" and not filled and failure
        ):
            status = "rejected"
        elif raw_status in ("finished", "canceled", "cancelled", "stopped"):
            status = "canceled"
        else:
            status = "open"

        order: CcxtOrder = {
            "id": str(item["id"]),
            "symbol": pair,
            "type": "chase",
            "side": "buy" if signed_amount > 0 else "sell",
            "price": price,
            "average": average,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "cost": filled * average if average is not None else 0.0,
            "status": status,
            "fee": {},
            "info": item,
        }
        if item.get("suborder_id"):
            order["id_chase"] = str(item["suborder_id"])
        if item.get("create_time"):
            order["timestamp"] = int(float(item["create_time"]) * 1000)
        return self._order_contracts_to_amount(order)

    @retrier
    def fetch_chase_order(self, order_id: str, pair: str) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)
        contract, settle = self._chase_market_params(pair)
        try:
            response = self._api.fetch2(
                "{settle}/autoorder/v1/chase/detail",
                ["private", "futures"],
                "GET",
                {"settle": settle, "id": order_id},
            )
            self._log_exchange_response("fetch_chase_order", response)
            item = response.get("order", response)
            if not item.get("id"):
                raise ccxt.OrderNotFound(
                    item.get("error_label") or f"Chase order {order_id} was not returned."
                )
            if item.get("contract") not in (None, contract):
                raise ccxt.InvalidOrder(f"Chase order {order_id} belongs to another contract.")
            return self._normalize_chase_order(pair, item)
        except ccxt.OrderNotFound as e:
            raise RetryableOrderError(
                f"Chase order not found (pair: {pair} id: {order_id}). Message: {e}"
            ) from e
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(
                f"Tried to get an invalid chase order (pair: {pair} id: {order_id}). Message: {e}"
            ) from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not get chase order due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    @retrier
    def cancel_chase_order(self, order_id: str, pair: str) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.cancel_order(order_id, pair)
        _, settle = self._chase_market_params(pair)
        try:
            response = self._api.fetch2(
                "{settle}/autoorder/v1/chase/stop",
                ["private", "futures"],
                "POST",
                {"settle": settle, "id": order_id},
            )
            self._log_exchange_response("cancel_chase_order", response)
            item = response.get("order", response)
            if item.get("id"):
                return self._normalize_chase_order(pair, item)
            return self.fetch_chase_order(order_id, pair)
        except ccxt.InvalidOrder as e:
            raise InvalidOrderException(f"Could not cancel chase order. Message: {e}") from e
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not cancel chase order due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def get_trades_for_order(
        self, order_id: str, pair: str, since: datetime, params: dict | None = None
    ) -> list:
        trades = super().get_trades_for_order(order_id, pair, since, params)

        if self.trading_mode == TradingMode.FUTURES:
            # Futures usually don't contain fees in the response.
            # As such, futures orders on gate will not contain a fee, which causes
            # a repeated "update fee" cycle and wrong calculations.
            # Therefore we patch the response with fees if it's not available.
            # An alternative also containing fees would be
            # privateFuturesGetSettleAccountBook({"settle": "usdt"})
            pair_fees = self._trading_fees.get(pair, {})
            if pair_fees:
                for idx, trade in enumerate(trades):
                    fee = trade.get("fee", {})
                    if fee and fee.get("cost") is None:
                        takerOrMaker = trade.get("takerOrMaker", "taker")
                        if pair_fees.get(takerOrMaker) is not None:
                            trades[idx]["fee"] = {
                                "currency": self.get_pair_quote_currency(pair),
                                "cost": trade["cost"] * pair_fees[takerOrMaker],
                                "rate": pair_fees[takerOrMaker],
                            }
        return trades


class GateEU(Gate):
    """Gate.io EU exchange class.
    Minimal adjustment to disable futures trading for the EU version of gate.io.
    """

    _ft_has_futures: FtHas = {}
    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]
