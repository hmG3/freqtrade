import logging
from datetime import timedelta

import ccxt

from freqtrade.constants import BuySell
from freqtrade.enums import CandleType, MarginMode, PriceType, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    InvalidOrderException,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange import Exchange
from freqtrade.exchange.common import API_RETRY_COUNT, retrier
from freqtrade.exchange.exchange_types import CcxtOrder, FtHas
from freqtrade.util import dt_now, dt_ts


logger = logging.getLogger(__name__)


class Okx(Exchange):
    """Okx exchange class.

    Contains adjustments needed for Freqtrade to work with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 100,  # Warning, special case with data prior to X months
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "stoploss_on_exchange": True,
        "stoploss_query_requires_stop_flag": True,
        "trades_has_history": False,  # Endpoint doesn't have a "since" parameter
        "ws_enabled": True,
    }
    _ft_has_futures: FtHas = {
        "chase_order": True,
        "cross_margin_stake_currencies": ["USDT"],
        "cross_liquidation_price_fallback": True,
        "tickers_have_quoteVolume": False,
        "stop_price_type_field": "slTriggerPxType",
        "stop_price_type_value_mapping": {
            PriceType.LAST: "last",
            PriceType.MARK: "mark",
            PriceType.INDEX: "index",
        },
        "stoploss_blocks_assets": False,
        "ws_enabled": True,
        # ccxt maps "total" to the currency's "eq" (equity), which includes unrealized PnL
        "balance_includes_unrealized_pnl": True,
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        # (TradingMode.MARGIN, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.CROSS),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
    ]

    net_only = True

    _ccxt_params: dict = {"options": {"brokerId": "ffb5405ad327SUDE"}}

    def ohlcv_candle_limit(
        self, timeframe: str, candle_type: CandleType, since_ms: int | None = None
    ) -> int:
        """
        Exchange ohlcv candle limit
        OKX has the following behaviour:
        * spot and futures:
            * 300 candles for regular candles
        * mark and premium-index:
            * 300 candles for up-to-date data
            * 100 candles for historic data
        * additional data:
            * 100 candles for additional candles
        :param timeframe: Timeframe to check
        :param candle_type: Candle type to use (spot, futures, funding_rate, ...)
        :param since_ms: Starting timestamp
        :return: Candle limit as integer
        """
        if candle_type in (CandleType.FUTURES, CandleType.SPOT):
            return 300

        return super().ohlcv_candle_limit(timeframe, candle_type, since_ms)

    @retrier
    def additional_exchange_init(self) -> None:
        """
        Additional exchange initialization logic.
        .api will be available at this point.
        Must be overridden in child methods if required.
        """
        try:
            if self.trading_mode == TradingMode.FUTURES and not self._config["dry_run"]:
                accounts = self._api.fetch_accounts()
                self._log_exchange_response("fetch_accounts", accounts)
                account_info = accounts[0].get("info", {}) if len(accounts) > 0 else {}
                if self.margin_mode == MarginMode.CROSS and (
                    account_info.get("acctLv") != "2"
                    or account_info.get("posMode") not in ("net_mode", "long_short_mode")
                ):
                    raise OperationalException(
                        "OKX cross margin requires OKX account level 2 (Futures mode) and "
                        "a supported position mode."
                    )
                if account_info:
                    self.net_only = account_info.get("posMode") == "net_mode"
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Error in additional_exchange_init due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

    def _get_posSide(self, side: BuySell, reduceOnly: bool):
        if self.net_only:
            return "net"
        if not reduceOnly:
            # Enter
            return "long" if side == "buy" else "short"
        else:
            # Exit
            return "long" if side == "sell" else "short"

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
        if self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["tdMode"] = self.margin_mode.value
            params["posSide"] = self._get_posSide(side, reduceOnly)
        return params

    @staticmethod
    def _chase_amount_to_string(amount: float) -> str:
        return str(int(amount)) if float(amount).is_integer() else str(amount)

    @staticmethod
    def _get_chase_response_item(response: dict) -> dict:
        if response.get("code") != "0":
            raise ccxt.ExchangeError(response.get("msg") or str(response))
        data = response.get("data") or []
        if not data:
            raise ccxt.OrderNotFound("Chase order response contained no order data.")
        item = data[0]
        if item.get("sCode") not in (None, "", "0"):
            raise ccxt.InvalidOrder(item.get("sMsg") or str(item))
        return item

    def create_chase_order(
        self,
        pair: str,
        side: BuySell,
        amount: float,
        rate: float | None,
        params: dict,
    ) -> CcxtOrder:
        request = {
            "instId": self.markets[pair]["id"],
            "tdMode": params["tdMode"],
            "side": side,
            "posSide": params["posSide"],
            "ordType": "chase",
            "sz": self._chase_amount_to_string(amount),
            "chaseType": "distance",
            "chaseVal": "0",
        }
        if params.get("reduceOnly"):
            request["reduceOnly"] = True

        response = self._api.private_post_trade_order_algo(request)
        self._log_exchange_response("create_chase_order", response)
        item = self._get_chase_response_item(response)
        order_id = str(item["algoId"])
        return {
            "id": order_id,
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

    @staticmethod
    def _get_chase_child_id(item: dict) -> str | None:
        child_ids = [item.get("ordId")]
        linked_order = item.get("linkedOrd")
        if isinstance(linked_order, dict):
            child_ids.append(linked_order.get("ordId"))
        if isinstance(item.get("ordIdList"), list):
            child_ids.extend(reversed(item["ordIdList"]))
        return next((str(order_id) for order_id in child_ids if order_id), None)

    @staticmethod
    def _get_chase_child_response_items(response: dict) -> list[dict]:
        if response.get("code") != "0":
            raise ccxt.ExchangeError(response.get("msg") or str(response))
        return response.get("data") or []

    def _fetch_chase_child_order(self, pair: str, item: dict) -> dict | None:
        order_id = str(item["algoId"])
        market = self.markets[pair]
        child_id = self._get_chase_child_id(item)
        if child_id:
            response = self._api.private_get_trade_order(
                {"instId": market["id"], "ordId": child_id}
            )
            self._log_exchange_response("chase_child_order", response)
            child_orders = self._get_chase_child_response_items(response)
            if child_orders:
                return child_orders[0]

        request = {
            "instType": "FUTURES" if market.get("future") else "SWAP",
            "instId": market["id"],
            "limit": 100,
        }
        state = str(item.get("state") or "").lower()
        endpoints = (
            (
                ("chase_child_order_history", self._api.private_get_trade_orders_history),
                ("chase_child_open_orders", self._api.private_get_trade_orders_pending),
            )
            if state not in ("live", "pause")
            else (
                ("chase_child_open_orders", self._api.private_get_trade_orders_pending),
                ("chase_child_order_history", self._api.private_get_trade_orders_history),
            )
        )
        for log_name, endpoint in endpoints:
            page_request = request
            previous_cursor = None
            while True:
                response = endpoint(page_request)
                self._log_exchange_response(log_name, response)
                child_orders = self._get_chase_child_response_items(response)
                matches = [
                    child for child in child_orders if str(child.get("algoId") or "") == order_id
                ]
                if matches:
                    return max(
                        matches,
                        key=lambda child: (
                            abs(float(child.get("accFillSz") or 0.0)),
                            int(child.get("uTime") or child.get("cTime") or 0),
                        ),
                    )
                if len(child_orders) < request["limit"]:
                    break
                oldest_order = child_orders[-1]
                oldest_order_id = oldest_order.get("ordId")
                parent_time = int(item.get("cTime") or 0)
                oldest_time = int(oldest_order.get("cTime") or 0)
                if (
                    not oldest_order_id
                    or str(oldest_order_id) == previous_cursor
                    or (parent_time and oldest_time < parent_time)
                ):
                    break
                previous_cursor = str(oldest_order_id)
                page_request = request | {"after": previous_cursor}
        return None

    def _normalize_chase_order(
        self, pair: str, item: dict, child_order: dict | None = None
    ) -> CcxtOrder:
        amount = abs(float(item.get("sz") or 0.0))
        filled = abs(
            float(
                child_order.get("accFillSz")
                if child_order and child_order.get("accFillSz") not in (None, "")
                else item.get("actualSz") or 0.0
            )
        )
        remaining = max(amount - filled, 0.0)
        average_value = (child_order.get("avgPx") if child_order else None) or item.get("actualPx")
        average = float(average_value) if average_value not in (None, "") else None
        price_value = (child_order.get("px") if child_order else None) or item.get("ordPx")
        price = float(price_value) if price_value not in (None, "") else average
        state = str(item.get("state") or "").lower()
        child_state = str(child_order.get("state") or "").lower() if child_order else ""
        if child_state == "filled":
            status = "closed"
        elif child_state in ("live", "partially_filled"):
            status = "open"
        elif child_state in ("canceled", "cancelled", "mmp_canceled"):
            status = "open" if state in ("live", "pause") else "canceled"
        elif state in ("live", "pause"):
            status = "open"
        elif state == "effective":
            # OKX can mark a chase parent effective while omitting all execution fields.
            # Keep polling until its spawned regular order is visible in order history.
            parent_fill_complete = abs(float(item.get("actualSz") or 0.0)) > 0 and item.get(
                "actualPx"
            ) not in (None, "")
            status = "closed" if parent_fill_complete else "open"
        elif state in ("canceled", "cancelled", "partially_effective", "partially_canceled"):
            status = "canceled"
        elif state in ("order_failed", "failed", "rejected"):
            status = "rejected"
        else:
            status = "open"

        order: CcxtOrder = {
            "id": str(item["algoId"]),
            "symbol": pair,
            "type": "chase",
            "side": item.get("side"),
            "price": price,
            "average": average,
            "amount": amount,
            "filled": filled,
            "remaining": remaining,
            "cost": filled * average if average is not None else 0.0,
            "status": status,
            "fee": {},
            "info": item | ({"childOrder": child_order} if child_order else {}),
        }
        child_id = str(child_order["ordId"]) if child_order and child_order.get("ordId") else None
        child_id = child_id or self._get_chase_child_id(item)
        if child_id:
            order["id_chase"] = child_id
        if item.get("cTime"):
            order["timestamp"] = int(item["cTime"])
        return self._order_contracts_to_amount(order)

    @retrier(retries=API_RETRY_COUNT)
    def fetch_chase_order(self, order_id: str, pair: str) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)
        try:
            response = self._api.private_get_trade_order_algo({"algoId": order_id})
            self._log_exchange_response("fetch_chase_order", response)
            item = self._get_chase_response_item(response)
            state = str(item.get("state") or "").lower()
            parent_execution_complete = abs(float(item.get("actualSz") or 0.0)) > 0 and item.get(
                "actualPx"
            ) not in (None, "")
            needs_child_order = state in (
                "live",
                "pause",
                "effective",
                "canceled",
                "cancelled",
                "partially_effective",
                "partially_canceled",
            ) and (not self._get_chase_child_id(item) or not parent_execution_complete)
            child_order = self._fetch_chase_child_order(pair, item) if needs_child_order else None
            return self._normalize_chase_order(pair, item, child_order)
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
        try:
            response = self._api.private_post_trade_cancel_algos(
                [{"algoId": order_id, "instId": self.markets[pair]["id"]}]
            )
            self._log_exchange_response("cancel_chase_order", response)
            self._get_chase_response_item(response)
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

    def __fetch_leverage_already_set(self, pair: str, leverage: float, side: BuySell) -> bool:
        try:
            res_lev = self._api.fetch_leverage(
                symbol=pair,
                params={
                    "mgnMode": self.margin_mode.value,
                    "posSide": self._get_posSide(side, False),
                },
            )
            self._log_exchange_response("get_leverage", res_lev)
            already_set = all(float(x["lever"]) == leverage for x in res_lev["data"])
            return already_set

        except ccxt.BaseError:
            # Assume all errors as "not set yet"
            return False

    @retrier
    def _lev_prep(self, pair: str, leverage: float, side: BuySell, accept_fail: bool = False):
        if self.trading_mode != TradingMode.SPOT and self.margin_mode is not None:
            try:
                res = self._api.set_leverage(
                    leverage=leverage,
                    symbol=pair,
                    params={
                        "mgnMode": self.margin_mode.value,
                        "posSide": self._get_posSide(side, False),
                    },
                )
                self._log_exchange_response("set_leverage", res)

            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                already_set = self.__fetch_leverage_already_set(pair, leverage, side)
                if not already_set:
                    raise TemporaryError(
                        f"Could not set leverage due to {e.__class__.__name__}. Message: {e}"
                    ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e

    def get_max_pair_stake_amount(self, pair: str, price: float, leverage: float = 1.0) -> float:
        if self.trading_mode == TradingMode.SPOT:
            return float("inf")  # Not actually inf, but this probably won't matter for SPOT

        if pair not in self._leverage_tiers:
            return float("inf")

        pair_tiers = self._leverage_tiers[pair]
        last_max_notional = pair_tiers[-1]["maxNotional"]
        if last_max_notional is None:
            return float("inf")
        return last_max_notional / leverage

    def _get_stop_params(self, side: BuySell, ordertype: str, stop_price: float) -> dict:
        params = super()._get_stop_params(side, ordertype, stop_price)
        if self.trading_mode == TradingMode.SPOT:
            params["tdMode"] = "cash"
        elif self.trading_mode == TradingMode.FUTURES and self.margin_mode:
            params["tdMode"] = self.margin_mode.value
            params["posSide"] = self._get_posSide(side, True)
        return params

    def _convert_stop_order(self, pair: str, order_id: str, order: CcxtOrder) -> CcxtOrder:
        if (
            order.get("status", "open") == "closed"
            and (real_order_id := order.get("info", {}).get("ordId")) is not None
        ):
            # Once a order triggered, we fetch the regular followup order.
            order_reg = self.fetch_order(real_order_id, pair)
            self._log_exchange_response("fetch_stoploss_order1", order_reg)
            order_reg["id_stop"] = order_reg["id"]
            order_reg["id"] = order_id
            order_reg["type"] = "stoploss"
            order_reg["status_stop"] = "triggered"
            return order_reg
        order = self._order_contracts_to_amount(order)
        order["type"] = "stoploss"
        return order

    @retrier(retries=API_RETRY_COUNT)
    def fetch_stoploss_order(
        self, order_id: str, pair: str, params: dict | None = None
    ) -> CcxtOrder:
        if self._config["dry_run"]:
            return self.fetch_dry_run_order(order_id)

        try:
            params1 = {"stop": True}
            order_reg = self._api.fetch_order(order_id, pair, params=params1)
            self._log_exchange_response("fetch_stoploss_order", order_reg)
            return self._convert_stop_order(pair, order_id, order_reg)
        except (ccxt.OrderNotFound, ccxt.InvalidOrder):
            pass
        except ccxt.DDoSProtection as e:
            raise DDosProtection(e) from e
        except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
            raise TemporaryError(
                f"Could not get order due to {e.__class__.__name__}. Message: {e}"
            ) from e
        except ccxt.BaseError as e:
            raise OperationalException(e) from e

        return self._fetch_stop_order_fallback(order_id, pair)

    def _fetch_stop_order_fallback(self, order_id: str, pair: str) -> CcxtOrder:
        params2 = {"stop": True, "ordType": "conditional"}
        for method in (
            self._api.fetch_open_orders,
            self._api.fetch_closed_orders,
            self._api.fetch_canceled_orders,
        ):
            try:
                orders = method(pair, params=params2)
                orders_f = [order for order in orders if order["id"] == order_id]
                if orders_f:
                    order = orders_f[0]
                    return self._convert_stop_order(pair, order_id, order)
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except ccxt.DDoSProtection as e:
                raise DDosProtection(e) from e
            except (ccxt.OperationFailed, ccxt.ExchangeError) as e:
                raise TemporaryError(
                    f"Could not get order due to {e.__class__.__name__}. Message: {e}"
                ) from e
            except ccxt.BaseError as e:
                raise OperationalException(e) from e
        raise RetryableOrderError(f"StoplossOrder not found (pair: {pair} id: {order_id}).")

    def _fetch_orders_emulate(self, pair: str, since_ms: int) -> list[CcxtOrder]:
        orders = []

        orders = self._api.fetch_closed_orders(pair, since=since_ms)
        if since_ms < dt_ts(dt_now() - timedelta(days=6, hours=23)):
            # Regular fetch_closed_orders only returns 7 days of data.
            # Force usage of "archive" endpoint, which returns 3 months of data.
            params = {"method": "privateGetTradeOrdersHistoryArchive"}
            orders_hist = self._api.fetch_closed_orders(pair, since=since_ms, params=params)
            orders.extend(orders_hist)

        orders_open = self._api.fetch_open_orders(pair, since=since_ms)
        orders.extend(orders_open)
        return orders


class Myokx(Okx):
    """MyOkx exchange class.
    Minimal adjustment to disable futures trading for the EU subsidiary of Okx
    """

    _ft_has_futures: FtHas = {}
    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]


class Okxus(Okx):
    """Okxus exchange class.
    Minimal adjustment to disable futures trading for the US subsidiary of Okx
    """

    _ft_has_futures: FtHas = {}
    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
    ]
