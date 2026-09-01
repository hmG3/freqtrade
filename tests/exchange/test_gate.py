from datetime import UTC, datetime
from unittest.mock import MagicMock

import ccxt
import pytest

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exceptions import (
    DDosProtection,
    InvalidOrderException,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange.common import API_RETRY_COUNT
from tests.conftest import EXMS, get_patched_exchange


@pytest.mark.parametrize("margin_mode", [0, "0", 3, "3"])
@pytest.mark.parametrize("position_mode", ["single", "dual", "dual_plus"])
def test_additional_exchange_init_gate_cross_account(
    default_conf, mocker, margin_mode, position_mode
):
    api_mock = MagicMock()
    api_mock.options = {"unifiedAccount": margin_mode in (3, "3")}
    api_mock.fetch2.return_value = {
        "margin_mode": margin_mode,
        "position_mode": position_mode,
    }
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate", api_mock=api_mock)
    exchange._config["dry_run"] = False
    exchange._config["stake_currency"] = "USDT"
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    exchange.additional_exchange_init()

    assert exchange.position_mode == position_mode
    api_mock.load_unified_status.assert_called_once_with()
    api_mock.fetch2.assert_called_once_with(
        "{settle}/accounts",
        ["private", "futures"],
        "GET",
        {"settle": "usdt"},
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"margin_mode": 1, "position_mode": "single"},
        {"margin_mode": 2, "position_mode": "single"},
        {"margin_mode": False, "position_mode": "single"},
        {"margin_mode": 3.5, "position_mode": "single"},
        {"margin_mode": "invalid", "position_mode": "single"},
        {"margin_mode": 0, "position_mode": "invalid"},
        {},
        None,
    ],
)
def test_additional_exchange_init_gate_cross_rejects_account_mode(default_conf, mocker, payload):
    api_mock = MagicMock()
    api_mock.options = {"unifiedAccount": False}
    api_mock.fetch2.return_value = payload
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate", api_mock=api_mock)
    exchange._config["dry_run"] = False
    exchange._config["stake_currency"] = "USDT"
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    with pytest.raises(OperationalException, match="cross margin requires"):
        exchange.additional_exchange_init()


@pytest.mark.parametrize(
    "exception,expected_exception",
    [
        (ccxt.DDoSProtection("ddos"), DDosProtection),
        (ccxt.OperationFailed("temporary"), TemporaryError),
        (ccxt.BaseError("fatal"), OperationalException),
    ],
)
def test_additional_exchange_init_gate_cross_translates_api_errors(
    default_conf, mocker, exception, expected_exception
):
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()
    api_mock.options = {"unifiedAccount": False}
    api_mock.fetch2.side_effect = exception
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate", api_mock=api_mock)
    exchange._config["dry_run"] = False
    exchange._config["stake_currency"] = "USDT"
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    with pytest.raises(expected_exception):
        exchange.additional_exchange_init()


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_gate(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")

    fetch_order_mock = MagicMock()
    exchange.fetch_order = fetch_order_mock

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert fetch_order_mock.call_count == 1
    assert fetch_order_mock.call_args_list[0][0][0] == "1234"
    assert fetch_order_mock.call_args_list[0][0][1] == "ETH/BTC"
    assert fetch_order_mock.call_args_list[0][0][2] == {"stop": True}

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"

    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")

    exchange.fetch_order = MagicMock(
        return_value={
            "status": "closed",
            "id": "1234",
            "stopPrice": 5.62,
            "info": {"trade_id": "222555"},
        }
    )

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert exchange.fetch_order.call_count == 2
    assert exchange.fetch_order.call_args_list[0][0][0] == "1234"
    assert exchange.fetch_order.call_args_list[1][1]["order_id"] == "222555"


def test_cancel_stoploss_order_gate(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    cancel_order_mock = mocker.patch.object(exchange, "cancel_order", autospec=True)

    exchange.cancel_stoploss_order("1234", "ETH/BTC")
    assert cancel_order_mock.call_count == 1
    assert cancel_order_mock.call_args_list[0][0][0] == "1234"
    assert cancel_order_mock.call_args_list[0][0][1] == "ETH/BTC"
    assert cancel_order_mock.call_args_list[0][0][2] == {"stop": True}


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_gate(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    order = {
        "price": 1500,
        "stopPrice": 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side)
    assert not exchange.stoploss_adjust(sl2, order, side)


@pytest.mark.parametrize(
    "takerormaker,rate,cost",
    [
        ("taker", 0.0005, 0.0001554325),
        ("maker", 0.0, 0.0),
    ],
)
def test_fetch_my_trades_gate(mocker, default_conf, takerormaker, rate, cost):
    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    tick = {
        "ETH/USDT:USDT": {
            "info": {
                "user_id": "",
                "taker_fee": "0.0018",
                "maker_fee": "0.0018",
                "gt_discount": False,
                "gt_taker_fee": "0",
                "gt_maker_fee": "0",
                "loan_fee": "0.18",
                "point_type": "1",
                "futures_taker_fee": "0.0005",
                "futures_maker_fee": "0",
            },
            "symbol": "ETH/USDT:USDT",
            "maker": 0.0,
            "taker": 0.0005,
        }
    }
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED

    api_mock = MagicMock()
    api_mock.fetch_my_trades = MagicMock(
        return_value=[
            {
                "fee": {"cost": None},
                "price": 3108.65,
                "cost": 0.310865,
                "order": "22255",
                "takerOrMaker": takerormaker,
                "amount": 1,  # 1 contract
            }
        ]
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock=api_mock, exchange="gate")
    exchange._trading_fees = tick
    trades = exchange.get_trades_for_order("22255", "ETH/USDT:USDT", datetime.now(UTC))
    trade = trades[0]
    assert trade["fee"]
    assert trade["fee"]["rate"] == rate
    assert trade["fee"]["currency"] == "USDT"
    assert trade["fee"]["cost"] == cost


@pytest.mark.parametrize("margin_mode", [MarginMode.ISOLATED, MarginMode.CROSS])
def test_gate_leverage_preparation_passes_margin_mode(default_conf, mocker, margin_mode):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = margin_mode
    set_leverage = mocker.patch.object(exchange, "_set_leverage")
    set_margin_mode = mocker.patch.object(exchange, "set_margin_mode")

    exchange._lev_prep("ETH/USDT:USDT", 3.0, "buy")

    set_leverage.assert_called_once_with(
        3.0,
        "ETH/USDT:USDT",
        False,
        params={"marginMode": margin_mode.value},
    )
    set_margin_mode.assert_not_called()


@pytest.mark.parametrize(
    "position_mode,expected",
    [("single", {}), ("dual", {}), ("dual_plus", {"pos_margin_mode": "cross"})],
)
def test_gate_order_params_cross_split_position_mode(default_conf, mocker, position_mode, expected):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS
    exchange.position_mode = position_mode

    params = exchange._get_params("buy", "limit", 3.0, False)

    assert params == expected


@pytest.mark.parametrize(
    "position_mode,expected_position_margin_mode",
    [("single", None), ("dual", None), ("dual_plus", "cross")],
)
def test_gate_stop_params_cross_split_position_mode(
    default_conf, mocker, position_mode, expected_position_margin_mode
):
    exchange = get_patched_exchange(mocker, default_conf, exchange="gate")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS
    exchange.position_mode = position_mode

    params = exchange._get_stop_params("sell", "limit", 100.0)

    assert params["stopPrice"] == 100.0
    if expected_position_margin_mode:
        assert params["pos_margin_mode"] == expected_position_margin_mode
    else:
        assert "pos_margin_mode" not in params


@pytest.mark.parametrize(
    "position_mode,expected_position_margin_mode",
    [("single", None), ("dual", None), ("dual_plus", "cross")],
)
def test_create_chase_order_gate_cross_position_mode(
    default_conf, mocker, markets, position_mode, expected_position_margin_mode
):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {"id": "12345"}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )
    exchange._config["dry_run"] = False
    exchange.position_mode = position_mode
    mocker.patch.object(exchange, "_lev_prep")

    exchange.create_order(
        pair="ETH/USDT:USDT",
        ordertype="chase",
        side="buy",
        amount=20.0,
        rate=2500.0,
        leverage=3.0,
    )

    request = api_mock.fetch2.call_args.args[3]
    if expected_position_margin_mode:
        assert request["pos_margin_mode"] == expected_position_margin_mode
    else:
        assert "pos_margin_mode" not in request


@pytest.mark.parametrize(
    ("side", "reduce_only", "signed_amount"),
    [
        ("buy", False, "2"),
        ("sell", False, "-2"),
        ("buy", True, "2"),
        ("sell", True, "-2"),
    ],
)
def test_create_chase_order_gate(default_conf, mocker, markets, side, reduce_only, signed_amount):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {"id": "12345"}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )
    mocker.patch.object(exchange, "_lev_prep")

    order = exchange.create_order(
        pair="ETH/USDT:USDT",
        ordertype="chase",
        side=side,
        amount=20.0,
        rate=2500.0,
        leverage=3.0,
        reduceOnly=reduce_only,
    )

    api_mock.fetch2.assert_called_once_with(
        "{settle}/autoorder/v1/chase/create",
        ["private", "futures"],
        "POST",
        {
            "settle": "usdt",
            "contract": "ETH_USDT",
            "amount": signed_amount,
            "price_limit": "0",
            "reduce_only": reduce_only,
            "price_type": 1,
        },
    )
    assert order["id"] == "12345"
    assert order["type"] == "chase"
    assert order["side"] == side
    assert order["amount"] == 20.0
    assert order["remaining"] == 20.0
    assert order["status"] == "open"


def test_create_chase_order_gate_accepts_success_envelope(default_conf, mocker, markets):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {
        "code": 0,
        "message": "ok",
        "data": {"id": "8964187"},
        "timestamp": 1787579705985,
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )
    exchange._config["dry_run"] = False
    mocker.patch.object(exchange, "_lev_prep")

    order = exchange.create_order(
        pair="ETH/USDT:USDT",
        ordertype="chase",
        side="buy",
        amount=20.0,
        rate=2500.0,
        leverage=3.0,
    )

    assert order["id"] == "8964187"
    assert order["status"] == "open"
    assert order["amount"] == 20.0


def test_fetch_chase_order_gate_normalizes_parent_and_child(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {
        "code": 0,
        "message": "ok",
        "data": {
            "order": {
                "id": "12345",
                "contract": "ETH_USDT",
                "settle": "usdt",
                "amount": "-2",
                "status": "finished",
                "reason": "filled",
                "fill_amount": "-2",
                "average_fill_price": "2501.5",
                "suborder_id": "67890",
                "suborder_price": "2501.5",
                "suborder_ongoing": False,
                "suborder_finish_as": "succeeded",
                "create_time": 1710000000,
                "finish_time": 1710000001,
                "status_code": "",
                "error_label": "",
                "reduce_only": True,
            }
        },
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    api_mock.fetch2.assert_called_once_with(
        "{settle}/autoorder/v1/chase/detail",
        ["private", "futures"],
        "GET",
        {"settle": "usdt", "id": "12345"},
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["type"] == "chase"
    assert order["side"] == "sell"
    assert order["status"] == "closed"
    assert order["amount"] == 20.0
    assert order["filled"] == 20.0
    assert order["remaining"] == 0.0
    assert order["average"] == 2501.5


def test_fetch_chase_order_gate_retries_error_envelope(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {
        "code": -1,
        "message": "Chase order was not found.",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )

    with pytest.raises(RetryableOrderError, match="Chase order was not found"):
        exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert api_mock.fetch2.call_count == API_RETRY_COUNT + 1


def test_cancel_chase_order_gate_preserves_partial_fill(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {
        "code": 0,
        "message": "ok",
        "data": {
            "order": {
                "id": "12345",
                "contract": "ETH_USDT",
                "amount": "-2",
                "status": "finished",
                "reason": "stopped",
                "fill_amount": "-1",
                "average_fill_price": "2501.5",
                "suborder_id": "67890",
                "suborder_price": "2501.5",
            }
        },
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )

    order = exchange.cancel_chase_order("12345", "ETH/USDT:USDT")

    api_mock.fetch2.assert_called_once_with(
        "{settle}/autoorder/v1/chase/stop",
        ["private", "futures"],
        "POST",
        {"settle": "usdt", "id": "12345"},
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["status"] == "canceled"
    assert order["filled"] == 10.0
    assert order["remaining"] == 10.0


def test_cancel_chase_order_gate_fetches_detail_after_success_ack(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.side_effect = [
        {
            "code": 0,
            "message": "ok",
            "data": {"id": "12345"},
        },
        {
            "code": 0,
            "message": "ok",
            "data": {
                "order": {
                    "id": "12345",
                    "contract": "ETH_USDT",
                    "amount": "2",
                    "status": "finished",
                    "reason": "stopped",
                    "fill_amount": "0.5",
                    "average_fill_price": "2501.5",
                    "suborder_id": "67890",
                }
            },
        },
    ]
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )

    order = exchange.cancel_chase_order("12345", "ETH/USDT:USDT")

    assert api_mock.fetch2.call_count == 2
    assert api_mock.fetch2.call_args_list[0].args == (
        "{settle}/autoorder/v1/chase/stop",
        ["private", "futures"],
        "POST",
        {"settle": "usdt", "id": "12345"},
    )
    assert api_mock.fetch2.call_args_list[1].args == (
        "{settle}/autoorder/v1/chase/detail",
        ["private", "futures"],
        "GET",
        {"settle": "usdt", "id": "12345"},
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["status"] == "canceled"
    assert order["filled"] == 5.0
    assert order["remaining"] == 15.0


@pytest.mark.parametrize(
    ("status_code", "error_label", "expected_status"),
    [
        ("INVALID_PARAM_VALUE", "INVALID_PARAM_VALUE", "rejected"),
        ("0", "", "canceled"),
    ],
)
def test_fetch_chase_order_gate_maps_terminal_status(
    default_conf, mocker, markets, status_code, error_label, expected_status
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.return_value = {
        "order": {
            "id": "12345",
            "contract": "ETH_USDT",
            "amount": "2",
            "status": "finished",
            "fill_amount": "0",
            "status_code": status_code,
            "error_label": error_label,
        }
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert order["status"] == expected_status
    assert order["filled"] == 0.0


@pytest.mark.parametrize(
    ("response", "error_match"),
    [
        ({"error_label": "INVALID_PARAM_VALUE"}, "INVALID_PARAM_VALUE"),
        (
            {
                "code": -1,
                "message": "Failed to submit: active chase limit order.",
                "timestamp": 1787579711976,
            },
            "active chase limit order",
        ),
    ],
)
def test_create_chase_order_gate_rejects_error_response(
    default_conf, mocker, markets, response, error_match
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    api_mock = MagicMock()
    api_mock.fetch2.return_value = response
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="gate",
        mock_markets={"ETH/USDT:USDT": markets["ETH/USDT:USDT"]},
    )
    mocker.patch.object(exchange, "_lev_prep")

    with pytest.raises(InvalidOrderException, match=error_match):
        exchange.create_order(
            pair="ETH/USDT:USDT",
            ordertype="chase",
            side="buy",
            amount=20.0,
            rate=2500.0,
            leverage=3.0,
        )
