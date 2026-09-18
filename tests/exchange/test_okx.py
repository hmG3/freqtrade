from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import ccxt
import pytest

from freqtrade.enums import CandleType, MarginMode, TradingMode
from freqtrade.exceptions import (
    InvalidOrderException,
    OperationalException,
    RetryableOrderError,
    TemporaryError,
)
from freqtrade.exchange.common import API_RETRY_COUNT
from freqtrade.exchange.exchange import timeframe_to_minutes
from tests.conftest import EXMS, get_patched_exchange, log_has
from tests.exchange.test_exchange import ccxt_exceptionhandlers


def test_okx_ohlcv_candle_limit(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    timeframes = ("1m", "5m", "1h")
    start_time = int(datetime(2021, 1, 1, tzinfo=UTC).timestamp() * 1000)

    for timeframe in timeframes:
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE) == 100

        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, start_time) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, start_time) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, start_time) == 100
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUNDING_RATE, start_time) == 100
        one_call = int(
            (
                datetime.now(UTC) - timedelta(minutes=290 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )

        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, one_call) == 100

        one_call = int(
            (
                datetime.now(UTC) - timedelta(minutes=320 * timeframe_to_minutes(timeframe))
            ).timestamp()
            * 1000
        )
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.SPOT, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.FUTURES, one_call) == 300
        assert exchange.ohlcv_candle_limit(timeframe, CandleType.MARK, one_call) == 100


def test_get_maintenance_ratio_and_amt_okx(
    default_conf,
    mocker,
):
    api_mock = MagicMock()
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["dry_run"] = False
    mocker.patch.multiple(
        "freqtrade.exchange.okx.Okx",
        exchange_has=MagicMock(return_value=True),
        load_leverage_tiers=MagicMock(
            return_value={
                "ETH/USDT:USDT": [
                    {
                        "tier": 1,
                        "minNotional": 0,
                        "maxNotional": 2000,
                        "maintenanceMarginRate": 0.01,
                        "maxLeverage": 75,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.013",
                            "instId": "",
                            "maxLever": "75",
                            "maxSz": "2000",
                            "minSz": "0",
                            "mmr": "0.01",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "1",
                            "uly": "ETH-USDT",
                        },
                    },
                    {
                        "tier": 2,
                        "minNotional": 2001,
                        "maxNotional": 4000,
                        "maintenanceMarginRate": 0.015,
                        "maxLeverage": 50,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.02",
                            "instId": "",
                            "maxLever": "50",
                            "maxSz": "4000",
                            "minSz": "2001",
                            "mmr": "0.015",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "2",
                            "uly": "ETH-USDT",
                        },
                    },
                    {
                        "tier": 3,
                        "minNotional": 4001,
                        "maxNotional": 8000,
                        "maintenanceMarginRate": 0.02,
                        "maxLeverage": 20,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.05",
                            "instId": "",
                            "maxLever": "20",
                            "maxSz": "8000",
                            "minSz": "4001",
                            "mmr": "0.02",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "3",
                            "uly": "ETH-USDT",
                        },
                    },
                ],
                "ADA/USDT:USDT": [
                    {
                        "tier": 1,
                        "minNotional": 0,
                        "maxNotional": 500,
                        "maintenanceMarginRate": 0.02,
                        "maxLeverage": 75,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.013",
                            "instId": "",
                            "maxLever": "75",
                            "maxSz": "500",
                            "minSz": "0",
                            "mmr": "0.01",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "1",
                            "uly": "ADA-USDT",
                        },
                    },
                    {
                        "tier": 2,
                        "minNotional": 501,
                        "maxNotional": 1000,
                        "maintenanceMarginRate": 0.025,
                        "maxLeverage": 50,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.02",
                            "instId": "",
                            "maxLever": "50",
                            "maxSz": "1000",
                            "minSz": "501",
                            "mmr": "0.015",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "2",
                            "uly": "ADA-USDT",
                        },
                    },
                    {
                        "tier": 3,
                        "minNotional": 1001,
                        "maxNotional": 2000,
                        "maintenanceMarginRate": 0.03,
                        "maxLeverage": 20,
                        "info": {
                            "baseMaxLoan": "",
                            "imr": "0.05",
                            "instId": "",
                            "maxLever": "20",
                            "maxSz": "2000",
                            "minSz": "1001",
                            "mmr": "0.02",
                            "optMgnFactor": "0",
                            "quoteMaxLoan": "",
                            "tier": "3",
                            "uly": "ADA-USDT",
                        },
                    },
                ],
            }
        ),
    )
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 2000) == (0.01, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 2001) == (0.015, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 4001) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt("ETH/USDT:USDT", 8000) == (0.02, None)

    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 1) == (0.02, None)
    assert exchange.get_maintenance_ratio_and_amt("ADA/USDT:USDT", 2000) == (0.03, None)


def test_get_max_pair_stake_amount_okx(default_conf, mocker, leverage_tiers):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    assert exchange.get_max_pair_stake_amount("BNB/BUSD", 1.0) == float("inf")

    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange._leverage_tiers = leverage_tiers

    assert exchange.get_max_pair_stake_amount("XRP/USDT:USDT", 1.0) == 30000000
    assert exchange.get_max_pair_stake_amount("BNB/USDT:USDT", 1.0) == 50000000
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0) == 1000000000
    assert exchange.get_max_pair_stake_amount("BTC/USDT:USDT", 1.0, 10.0) == 100000000

    assert exchange.get_max_pair_stake_amount("TTT/USDT:USDT", 1.0) == float("inf")  # Not in tiers


@pytest.mark.parametrize(
    "mode,side,reduceonly,result",
    [
        ("net", "buy", False, "net"),
        ("net", "sell", True, "net"),
        ("net", "sell", False, "net"),
        ("net", "buy", True, "net"),
        ("longshort", "buy", False, "long"),
        ("longshort", "sell", True, "long"),
        ("longshort", "sell", False, "short"),
        ("longshort", "buy", True, "short"),
    ],
)
def test__get_posSide(default_conf, mocker, mode, side, reduceonly, result):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.net_only = mode == "net"
    assert exchange._get_posSide(side, reduceonly) == result


def test_additional_exchange_init_okx(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "long_short_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    assert api_mock.fetch_accounts.call_count == 0
    exchange.trading_mode = TradingMode.FUTURES
    # Default to netOnly
    assert exchange.net_only
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert not exchange.net_only

    api_mock.fetch_accounts = MagicMock(
        return_value=[
            {
                "id": "2555",
                "type": "2",
                "currency": None,
                "info": {
                    "acctLv": "2",
                    "autoLoan": False,
                    "ctIsoMode": "automatic",
                    "greeksType": "PA",
                    "level": "Lv1",
                    "levelTmp": "",
                    "mgnIsoMode": "automatic",
                    "posMode": "net_mode",
                    "uid": "2555",
                },
            }
        ]
    )
    exchange.additional_exchange_init()
    assert api_mock.fetch_accounts.call_count == 1
    assert exchange.net_only
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    ccxt_exceptionhandlers(
        mocker, default_conf, api_mock, "okx", "additional_exchange_init", "fetch_accounts"
    )


@pytest.mark.parametrize(
    "accounts,net_only",
    [
        ([{"info": {"acctLv": "2", "posMode": "net_mode"}}], True),
        ([{"info": {"acctLv": "2", "posMode": "long_short_mode"}}], False),
    ],
)
def test_additional_exchange_init_okx_cross_account(default_conf, mocker, accounts, net_only):
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = accounts
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    exchange._config["dry_run"] = False
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    exchange.additional_exchange_init()

    assert exchange.net_only is net_only
    api_mock.fetch_accounts.assert_called_once_with()


@pytest.mark.parametrize(
    "accounts",
    [
        [{"info": {"acctLv": "3", "posMode": "net_mode"}}],
        [{"info": {"acctLv": "4", "posMode": "net_mode"}}],
        [{"info": {"posMode": "net_mode"}}],
        [{"info": {"acctLv": "2"}}],
        [{"info": {"acctLv": "2", "posMode": "invalid"}}],
        [],
    ],
)
def test_additional_exchange_init_okx_cross_rejects_account_level(default_conf, mocker, accounts):
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = accounts
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    exchange._config["dry_run"] = False
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    with pytest.raises(OperationalException, match="cross margin requires OKX account level 2"):
        exchange.additional_exchange_init()


def test_okx_hedge_mode_requires_long_short_account(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    default_conf.update(
        {
            "dry_run": False,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "hedge": {"enabled": True},
        }
    )
    with pytest.raises(OperationalException, match="long_short_mode"):
        get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)


def test_okx_hedge_mode_dry_run_uses_both_sides(default_conf, mocker):
    default_conf.update(
        {
            "dry_run": True,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "hedge": {"enabled": True},
        }
    )
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")

    exchange.validate_hedge_mode()

    assert exchange.net_only is False


def test_okx_validate_hedge_market(default_conf, mocker, markets):
    default_conf.update(
        {"trading_mode": "futures", "margin_mode": "isolated", "stake_currency": "USDT"}
    )
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    pair = "ETH/USDT:USDT"
    exchange.markets[pair] = {**markets[pair], "linear": True, "settle": "USDT"}

    exchange.validate_hedge_market(pair)
    exchange.markets[pair]["linear"] = False
    with pytest.raises(OperationalException, match="linear"):
        exchange.validate_hedge_market(pair)
    exchange.markets[pair]["linear"] = True
    exchange.markets[pair]["settle"] = "BTC"
    with pytest.raises(OperationalException, match="settlement"):
        exchange.validate_hedge_market(pair)


def test_okx_create_and_fetch_order_by_client_id_dry_run(
    default_conf, mocker, markets, init_persistence
):
    default_conf.update({"dry_run": True, "trading_mode": "futures", "margin_mode": "isolated"})
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    pair = "ETH/USDT:USDT"
    exchange.markets[pair] = markets[pair]
    mocker.patch.object(exchange, "exchange_has", return_value=False)

    order = exchange.create_order(
        pair=pair,
        ordertype="limit",
        side="buy",
        amount=1.0,
        rate=100.0,
        leverage=1.0,
        client_order_id="hedge-123",
    )

    assert order["id"] == "dry_run_hedge-123"
    assert exchange.fetch_order_by_client_id("hedge-123", pair) == order
    assert exchange.fetch_order_by_client_id("missing", pair) is None


def test_okx_hedge_market_rejects_dated_contracts(default_conf, mocker, markets):
    default_conf.update({"trading_mode": "futures", "stake_currency": "USDT"})
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    pair = "ETH/USDT:USDT"
    exchange.markets[pair] = {
        **markets[pair],
        "contract": True,
        "linear": True,
        "settle": "USDT",
        "type": "future",
        "swap": False,
    }
    with pytest.raises(OperationalException, match="linear swap"):
        exchange.validate_hedge_market(pair)


def test_okx_client_id_lookup_restores_persisted_dry_order_and_updates_fill(
    default_conf,
    mocker,
    init_persistence,
):
    from freqtrade.persistence import Order, Trade

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    pair = "ETH/USDT:USDT"
    order = {
        "id": "dry_run_recovery",
        "symbol": pair,
        "status": "open",
        "type": "limit",
        "side": "buy",
        "price": 80.0,
        "average": None,
        "amount": 1.0,
        "filled": 0.0,
        "remaining": 1.0,
        "cost": 0.0,
    }
    stored = Order.parse_from_ccxt_object(order, pair, "buy")
    trade = Trade(
        pair=pair,
        exchange="okx",
        open_rate=80,
        amount=0,
        stake_amount=80,
        fee_open=0.001,
        fee_close=0.001,
        orders=[stored],
    )
    Trade.session.add(trade)
    Trade.commit()
    fill = mocker.patch.object(
        exchange,
        "check_dry_limit_order_filled",
        side_effect=lambda snapshot: snapshot | {"status": "closed", "filled": 1.0, "remaining": 0},
    )
    result = exchange.fetch_order_by_client_id("recovery", pair)
    assert result is not None and result["status"] == "closed" and result["filled"] == 1.0
    assert result["id"] == order["id"]
    assert fill.call_count == 1


def test_okx_fetch_order_by_client_id_live(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.fetch_order.return_value = {
        "id": "exchange-id",
        "symbol": "ETH/USDT:USDT",
        "amount": 2.0,
        "filled": 1.0,
        "remaining": 1.0,
        "status": "open",
    }
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    mocker.patch.object(exchange, "_order_contracts_to_amount", side_effect=lambda order: order)
    mocker.patch.object(exchange, "exchange_has", return_value=True)

    result = exchange.fetch_order_by_client_id("hedge-123", "ETH/USDT:USDT")

    assert result == api_mock.fetch_order.return_value
    api_mock.fetch_order.assert_called_once_with(
        "hedge-123", "ETH/USDT:USDT", params={"clientOrderId": "hedge-123"}
    )

    api_mock.fetch_order.side_effect = ccxt.OrderNotFound("missing")
    assert exchange.fetch_order_by_client_id("missing", "ETH/USDT:USDT") is None


def test_create_order_passes_client_order_id(default_conf, mocker):
    api_mock = MagicMock()
    api_mock.create_order.return_value = {
        "id": "exchange-id",
        "symbol": "ETH/USDT:USDT",
        "amount": 1.0,
        "filled": 0.0,
        "remaining": 1.0,
        "status": "open",
    }
    default_conf["dry_run"] = False
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx", api_mock=api_mock)
    mocker.patch.object(exchange, "_order_contracts_to_amount", side_effect=lambda order: order)
    mocker.patch.object(exchange, "_lev_prep")

    exchange.create_order(
        pair="ETH/USDT:USDT",
        ordertype="limit",
        side="buy",
        amount=1.0,
        rate=100.0,
        leverage=1.0,
        client_order_id="hedge-123",
    )

    assert api_mock.create_order.call_args.args[-1]["clientOrderId"] == "hedge-123"


def test_okx_liquidation_price_selects_requested_side(default_conf, mocker):
    default_conf.update({"dry_run": False, "trading_mode": "futures", "margin_mode": "isolated"})
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange._config["dry_run"] = False
    mocker.patch.object(exchange, "exchange_has", return_value=True)
    mocker.patch.object(
        exchange,
        "fetch_positions",
        return_value=[
            {"side": "short", "liquidationPrice": 150.0},
            {"side": "long", "liquidationPrice": 50.0},
        ],
    )
    exchange.liquidation_buffer = 0.0

    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, False, 1, 100, 1, 100) == 50
    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, True, 1, 100, 1, 100) == 150


def test_okx_liquidation_price_side_alias_and_legacy_missing_side(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange._config["dry_run"] = False
    mocker.patch.object(exchange, "exchange_has", return_value=True)
    positions = [{"side": "buy", "liquidationPrice": 45.0}]
    mocker.patch.object(exchange, "fetch_positions", side_effect=lambda pair: positions)
    exchange.liquidation_buffer = 0.0

    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, False, 1, 100, 1, 100) == 45
    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, True, 1, 100, 1, 100) is None
    positions[:] = [{"side": None, "liquidationPrice": 55.0}]
    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, False, 1, 100, 1, 100) == 55


def test_okx_dry_run_hedge_liquidation_is_not_modeled(default_conf, mocker):
    default_conf["hedge"] = {"enabled": True}
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.CROSS

    assert exchange.get_liquidation_price("ETH/USDT:USDT", 100, False, 1, 100, 1, 100) is None


def test_load_leverage_tiers_okx(default_conf, mocker, markets, tmp_path, caplog, time_machine):
    default_conf["datadir"] = tmp_path
    # fd_mock = mocker.patch('freqtrade.exchange.exchange.file_dump_json')
    api_mock = MagicMock()
    type(api_mock).has = PropertyMock(
        return_value={
            "fetchLeverageTiers": False,
            "fetchMarketLeverageTiers": True,
        }
    )
    api_mock.fetch_market_leverage_tiers = AsyncMock(
        side_effect=[
            [
                {
                    "tier": 1,
                    "minNotional": 0,
                    "maxNotional": 500,
                    "maintenanceMarginRate": 0.02,
                    "maxLeverage": 75,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.013",
                        "instId": "",
                        "maxLever": "75",
                        "maxSz": "500",
                        "minSz": "0",
                        "mmr": "0.01",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "1",
                        "uly": "ADA-USDT",
                    },
                },
                {
                    "tier": 2,
                    "minNotional": 501,
                    "maxNotional": 1000,
                    "maintenanceMarginRate": 0.025,
                    "maxLeverage": 50,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.02",
                        "instId": "",
                        "maxLever": "50",
                        "maxSz": "1000",
                        "minSz": "501",
                        "mmr": "0.015",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "2",
                        "uly": "ADA-USDT",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 1001,
                    "maxNotional": 2000,
                    "maintenanceMarginRate": 0.03,
                    "maxLeverage": 20,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.05",
                        "instId": "",
                        "maxLever": "20",
                        "maxSz": "2000",
                        "minSz": "1001",
                        "mmr": "0.02",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "3",
                        "uly": "ADA-USDT",
                    },
                },
            ],
            TemporaryError("this Failed"),
            [
                {
                    "tier": 1,
                    "minNotional": 0,
                    "maxNotional": 2000,
                    "maintenanceMarginRate": 0.01,
                    "maxLeverage": 75,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.013",
                        "instId": "",
                        "maxLever": "75",
                        "maxSz": "2000",
                        "minSz": "0",
                        "mmr": "0.01",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "1",
                        "uly": "ETH-USDT",
                    },
                },
                {
                    "tier": 2,
                    "minNotional": 2001,
                    "maxNotional": 4000,
                    "maintenanceMarginRate": 0.015,
                    "maxLeverage": 50,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.02",
                        "instId": "",
                        "maxLever": "50",
                        "maxSz": "4000",
                        "minSz": "2001",
                        "mmr": "0.015",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "2",
                        "uly": "ETH-USDT",
                    },
                },
                {
                    "tier": 3,
                    "minNotional": 4001,
                    "maxNotional": 8000,
                    "maintenanceMarginRate": 0.02,
                    "maxLeverage": 20,
                    "info": {
                        "baseMaxLoan": "",
                        "imr": "0.05",
                        "instId": "",
                        "maxLever": "20",
                        "maxSz": "8000",
                        "minSz": "4001",
                        "mmr": "0.02",
                        "optMgnFactor": "0",
                        "quoteMaxLoan": "",
                        "tier": "3",
                        "uly": "ETH-USDT",
                    },
                },
            ],
        ]
    )
    default_conf["trading_mode"] = "futures"
    default_conf["margin_mode"] = "isolated"
    default_conf["stake_currency"] = "USDT"
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    exchange.trading_mode = TradingMode.FUTURES
    exchange.margin_mode = MarginMode.ISOLATED
    exchange.markets = markets
    # Initialization of load_leverage_tiers happens as part of exchange init.
    assert exchange._leverage_tiers == {
        "ADA/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 500,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 75,
                "maintAmt": None,
            },
            {
                "minNotional": 501,
                "maxNotional": 1000,
                "maintenanceMarginRate": 0.025,
                "maxLeverage": 50,
                "maintAmt": None,
            },
            {
                "minNotional": 1001,
                "maxNotional": 2000,
                "maintenanceMarginRate": 0.03,
                "maxLeverage": 20,
                "maintAmt": None,
            },
        ],
        "ETH/USDT:USDT": [
            {
                "minNotional": 0,
                "maxNotional": 2000,
                "maintenanceMarginRate": 0.01,
                "maxLeverage": 75,
                "maintAmt": None,
            },
            {
                "minNotional": 2001,
                "maxNotional": 4000,
                "maintenanceMarginRate": 0.015,
                "maxLeverage": 50,
                "maintAmt": None,
            },
            {
                "minNotional": 4001,
                "maxNotional": 8000,
                "maintenanceMarginRate": 0.02,
                "maxLeverage": 20,
                "maintAmt": None,
            },
        ],
    }
    filename = (
        default_conf["datadir"] / f"futures/leverage_tiers_{default_conf['stake_currency']}.json"
    )
    assert filename.is_file()

    logmsg = "Cached leverage tiers are outdated. Will update."
    assert not log_has(logmsg, caplog)

    api_mock.fetch_market_leverage_tiers.reset_mock()

    exchange.load_leverage_tiers()
    assert not log_has(logmsg, caplog)

    assert api_mock.fetch_market_leverage_tiers.call_count == 0
    # 2 day passes ...
    time_machine.move_to(datetime.now() + timedelta(weeks=5))
    exchange.load_leverage_tiers()

    assert log_has(logmsg, caplog)


@pytest.mark.parametrize("margin_mode", [MarginMode.ISOLATED, MarginMode.CROSS])
def test__set_leverage_okx(mocker, default_conf, margin_mode):
    api_mock = MagicMock()
    api_mock.set_leverage = MagicMock()
    type(api_mock).has = PropertyMock(return_value={"setLeverage": True})
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = margin_mode

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    exchange._config["dry_run"] = False
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    assert api_mock.set_leverage.call_count == 1
    # Leverage is rounded to 3.
    assert api_mock.set_leverage.call_args_list[0][1]["leverage"] == 3.2
    assert api_mock.set_leverage.call_args_list[0][1]["symbol"] == "BTC/USDT:USDT"
    assert api_mock.set_leverage.call_args_list[0][1]["params"] == {
        "mgnMode": margin_mode.value,
        "posSide": "net",
    }
    api_mock.set_leverage = MagicMock(side_effect=ccxt.NetworkError())
    exchange._lev_prep("BTC/USDT:USDT", 3.2, "buy")
    assert api_mock.fetch_leverage.call_count == 1

    api_mock.fetch_leverage = MagicMock(side_effect=ccxt.NetworkError())
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    ccxt_exceptionhandlers(
        mocker,
        default_conf,
        api_mock,
        "okx",
        "_lev_prep",
        "set_leverage",
        pair="XRP/USDT:USDT",
        leverage=5.0,
        side="buy",
    )


@pytest.mark.usefixtures("init_persistence")
def test_fetch_stoploss_order_okx(default_conf, mocker):
    default_conf["dry_run"] = False
    mocker.patch("freqtrade.exchange.common.time.sleep")
    api_mock = MagicMock()
    api_mock.fetch_order = MagicMock()

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_order.call_args_list[0][0][0] == "1234"
    assert api_mock.fetch_order.call_args_list[0][0][1] == "ETH/BTC"
    assert api_mock.fetch_order.call_args_list[0][1]["params"] == {"stop": True}

    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound)
    api_mock.fetch_open_orders = MagicMock(return_value=[])
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(creturn_value=[])

    with pytest.raises(RetryableOrderError):
        exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_open_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_closed_orders.call_count == API_RETRY_COUNT + 1
    assert api_mock.fetch_canceled_orders.call_count == API_RETRY_COUNT + 1

    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()

    api_mock.fetch_closed_orders = MagicMock(
        return_value=[{"id": "1234", "status": "closed", "info": {"ordId": "123455"}}]
    )
    mocker.patch(f"{EXMS}.fetch_order", MagicMock(return_value={"id": "123455"}))
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")
    assert api_mock.fetch_order.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert api_mock.fetch_canceled_orders.call_count == 0

    assert resp["id"] == "1234"
    assert resp["id_stop"] == "123455"
    assert resp["type"] == "stoploss"

    default_conf["dry_run"] = True
    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    dro_mock = mocker.patch(f"{EXMS}.fetch_dry_run_order", MagicMock(return_value={"id": "123455"}))

    api_mock.fetch_order.reset_mock()
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()
    api_mock.fetch_canceled_orders.reset_mock()
    resp = exchange.fetch_stoploss_order("1234", "ETH/BTC")

    assert api_mock.fetch_order.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 0
    assert api_mock.fetch_closed_orders.call_count == 0
    assert api_mock.fetch_canceled_orders.call_count == 0
    assert dro_mock.call_count == 1


def test_fetch_stoploss_order_okx_exceptions(default_conf_usdt, mocker):
    default_conf_usdt["dry_run"] = False
    api_mock = MagicMock()
    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_order",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )

    # Test 2nd part of the function
    api_mock.fetch_order = MagicMock(side_effect=ccxt.OrderNotFound())
    api_mock.fetch_closed_orders = MagicMock(return_value=[])
    api_mock.fetch_canceled_orders = MagicMock(return_value=[])

    ccxt_exceptionhandlers(
        mocker,
        default_conf_usdt,
        api_mock,
        "okx",
        "fetch_stoploss_order",
        "fetch_open_orders",
        retries=API_RETRY_COUNT + 1,
        order_id="12345",
        pair="ETH/USDT",
    )


@pytest.mark.parametrize(
    "sl1,sl2,sl3,side", [(1501, 1499, 1501, "sell"), (1499, 1501, 1499, "buy")]
)
def test_stoploss_adjust_okx(mocker, default_conf, sl1, sl2, sl3, side):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    order = {
        "type": "stoploss",
        "price": 1500,
        "stopLossPrice": 1500,
    }
    assert exchange.stoploss_adjust(sl1, order, side=side)
    assert not exchange.stoploss_adjust(sl2, order, side=side)


def test_stoploss_cancel_okx(mocker, default_conf):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    co_mock = mocker.patch.object(exchange, "cancel_order", autospec=True)

    exchange.cancel_stoploss_order("1234", "ETH/USDT")
    assert co_mock.call_count == 1
    args, _ = co_mock.call_args
    assert args[0] == "1234"
    assert args[1] == "ETH/USDT"
    assert args[2] == {"stop": True}


@pytest.mark.parametrize(
    "trading_mode,margin_mode,expected",
    [
        (TradingMode.SPOT, MarginMode.NONE, {"stopLossPrice": 1500, "tdMode": "cash"}),
        (
            TradingMode.FUTURES,
            MarginMode.ISOLATED,
            {"stopLossPrice": 1500, "tdMode": "isolated", "posSide": "net"},
        ),
    ],
)
def test__get_stop_params_okx(mocker, default_conf, trading_mode, margin_mode, expected):
    default_conf["trading_mode"] = trading_mode
    default_conf["margin_mode"] = margin_mode
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    params = exchange._get_stop_params("sell", "market", 1500)

    assert params == expected


def test_fetch_orders_okx(default_conf, mocker, limit_order):
    api_mock = MagicMock()
    api_mock.fetch_orders = MagicMock(
        return_value=[
            limit_order["buy"],
            limit_order["sell"],
        ]
    )
    api_mock.fetch_open_orders = MagicMock(return_value=[limit_order["buy"]])
    api_mock.fetch_closed_orders = MagicMock(return_value=[limit_order["buy"]])

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)
    start_time = datetime.now(UTC) - timedelta(days=20)

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")
    # Not available in dry-run
    assert exchange.fetch_orders("mocked", start_time) == []
    assert api_mock.fetch_orders.call_count == 0
    default_conf["dry_run"] = False

    exchange = get_patched_exchange(mocker, default_conf, api_mock, exchange="okx")

    def has_resp(_, endpoint):
        if endpoint == "fetchOrders":
            return False
        if endpoint == "fetchClosedOrders":
            return True
        if endpoint == "fetchOpenOrders":
            return True

    mocker.patch(f"{EXMS}.exchange_has", has_resp)

    history_params = {"method": "privateGetTradeOrdersHistoryArchive"}

    # happy path without fetchOrders
    exchange.fetch_orders("mocked", start_time)
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params

    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    # regular closed_orders endpoint only has history for 7 days.
    exchange.fetch_orders("mocked", datetime.now(UTC) - timedelta(days=6))
    assert api_mock.fetch_orders.call_count == 0
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 1
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]

    mocker.patch(f"{EXMS}.exchange_has", return_value=True)

    # Unhappy path - first fetch-orders call fails.
    api_mock.fetch_orders = MagicMock(side_effect=ccxt.NotSupported())
    api_mock.fetch_open_orders.reset_mock()
    api_mock.fetch_closed_orders.reset_mock()

    exchange.fetch_orders("mocked", start_time)

    assert api_mock.fetch_orders.call_count == 1
    assert api_mock.fetch_open_orders.call_count == 1
    assert api_mock.fetch_closed_orders.call_count == 2
    assert "params" not in api_mock.fetch_closed_orders.call_args_list[0][1]
    assert api_mock.fetch_closed_orders.call_args_list[1][1]["params"] == history_params


@pytest.mark.parametrize(
    ("net_only", "side", "reduce_only", "position_side"),
    [
        (True, "sell", True, "net"),
        (False, "buy", False, "long"),
        (False, "sell", False, "short"),
        (False, "sell", True, "long"),
        (False, "buy", True, "short"),
    ],
)
@pytest.mark.parametrize("margin_mode", [MarginMode.ISOLATED, MarginMode.CROSS])
@pytest.mark.parametrize("client_id", [None, "fthChaseHedge123"])
def test_create_chase_order_okx(
    default_conf,
    mocker,
    markets,
    net_only,
    side,
    reduce_only,
    position_side,
    margin_mode,
    client_id,
):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = margin_mode
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_post_trade_order_algo.return_value = {
        "code": "0",
        "data": [{"algoId": "12345", "sCode": "0", "sMsg": ""}],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    exchange._config["dry_run"] = False
    exchange.net_only = net_only
    mocker.patch.object(exchange, "_lev_prep")

    order = exchange.create_order(
        pair="ETH/USDT:USDT",
        ordertype="chase",
        side=side,
        amount=20.0,
        rate=2500.0,
        leverage=3.0,
        reduceOnly=reduce_only,
        client_order_id=client_id,
    )

    expected_request = {
        "instId": "ETH-USDT-SWAP",
        "tdMode": margin_mode.value,
        "side": side,
        "posSide": position_side,
        "ordType": "chase",
        "sz": "2",
        "chaseType": "distance",
        "chaseVal": "0",
    }
    if reduce_only:
        expected_request["reduceOnly"] = True
    if client_id:
        expected_request["algoClOrdId"] = client_id
    api_mock.private_post_trade_order_algo.assert_called_once_with(expected_request)
    assert order == {
        "id": "12345",
        "symbol": "ETH/USDT:USDT",
        "type": "chase",
        "side": side,
        "price": 2500.0,
        "average": None,
        "amount": 20.0,
        "filled": 0.0,
        "remaining": 20.0,
        "cost": 0.0,
        "status": "open",
        "fee": {},
        "info": {"algoId": "12345", "sCode": "0", "sMsg": ""},
    }


@pytest.mark.parametrize("by_client_id", [False, True])
def test_fetch_chase_order_okx_normalizes_parent_and_child(
    default_conf, mocker, markets, by_client_id
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "algoClOrdId": "fthChaseHedge123",
                "ordId": "67890",
                "instId": "ETH-USDT-SWAP",
                "state": "effective",
                "side": "buy",
                "sz": "2",
                "actualSz": "2",
                "actualPx": "2501.5",
                "cTime": "1710000000000",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_history.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_order.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "partially_filled",
                "accFillSz": "1",
                "avgPx": "2499",
                "uTime": "1710000001000",
            }
        ],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    if by_client_id:
        order = exchange.fetch_order_by_client_id(
            "fthChaseHedge123", "ETH/USDT:USDT", order_type="chase"
        )
    else:
        order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    api_mock.private_get_trade_order_algo.assert_called_once_with(
        {"algoClOrdId": "fthChaseHedge123"} if by_client_id else {"algoId": "12345"}
    )
    assert order["clientOrderId"] == "fthChaseHedge123"
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["type"] == "chase"
    assert order["status"] == "closed"
    assert order["amount"] == 20.0
    assert order["filled"] == 20.0
    assert order["remaining"] == 0.0
    assert order["average"] == 2501.5


def test_fetch_chase_order_okx_resolves_blank_effective_parent(default_conf, mocker, markets):
    default_conf["dry_run"] = True
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "",
                "ordIdList": [],
                "linkedOrd": {"ordId": ""},
                "instId": "ETH-USDT-SWAP",
                "state": "effective",
                "side": "buy",
                "sz": "2",
                "actualSz": "",
                "actualPx": "",
                "cTime": "1710000000000",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_history.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "instId": "ETH-USDT-SWAP",
                "state": "filled",
                "side": "buy",
                "sz": "2",
                "accFillSz": "2",
                "avgPx": "2501.5",
                "px": "2501.5",
                "source": "34",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {"code": "0", "data": [], "msg": ""}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    exchange._config["dry_run"] = False

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    api_mock.private_get_trade_orders_history.assert_called_once_with(
        {"instType": "SWAP", "instId": "ETH-USDT-SWAP", "limit": 100}
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["status"] == "closed"
    assert order["amount"] == 20.0
    assert order["filled"] == 20.0
    assert order["remaining"] == 0.0
    assert order["average"] == 2501.5


def test_fetch_chase_order_okx_waits_for_effective_child(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "ordIdList": [],
                "state": "effective",
                "side": "sell",
                "sz": "2",
                "actualSz": "0",
                "actualPx": "",
            }
        ],
        "msg": "",
    }
    empty_response = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_order.return_value = empty_response
    api_mock.private_get_trade_orders_history.return_value = empty_response
    api_mock.private_get_trade_orders_pending.return_value = empty_response
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    mocker.patch("freqtrade.exchange.common.time.sleep")

    child_request = {"instType": "SWAP", "instId": "ETH-USDT-SWAP", "limit": 100}
    with pytest.raises(RetryableOrderError, match="execution is not yet authoritative"):
        exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    api_mock.private_get_trade_order.assert_called_with(
        {"instId": "ETH-USDT-SWAP", "ordId": "67890"}
    )
    api_mock.private_get_trade_orders_history.assert_called_with(child_request)
    api_mock.private_get_trade_orders_pending.assert_called_with(child_request)


def test_fetch_chase_order_okx_aggregates_replaced_children(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "ordIdList": ["child-a", "child-b", "child-latest"],
                "linkedOrd": {"ordId": "child-latest"},
                "state": "canceled",
                "side": "sell",
                "sz": "3",
                "actualSz": "0",
                "actualPx": "",
                "cTime": "1000",
                "uTime": "2000",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "state": "live",
                "accFillSz": "0",
                "uTime": "2000",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_history.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "state": "canceled",
                "accFillSz": "0",
                "uTime": "3000",
            },
            {
                "algoId": "12345",
                "ordId": "child-a",
                "state": "filled",
                "accFillSz": "1",
                "avgPx": "2500",
                "uTime": "2500",
            },
            {
                "algoId": "12345",
                "ordId": "child-b",
                "state": "filled",
                "accFillSz": "2",
                "avgPx": "2503",
                "uTime": "2600",
            },
        ],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert order["id"] == "12345"
    assert order["id_chase"] == "child-b"
    assert set(order["id_chase_list"]) == {"child-a", "child-b", "child-latest"}
    assert order["status"] == "closed"
    assert order["amount"] == 30.0
    assert order["filled"] == 30.0
    assert order["remaining"] == 0.0
    assert order["average"] == pytest.approx(2502.0)
    assert order["cost"] == pytest.approx(75060.0)
    assert order["info"]["chaseDiagnostics"] == {
        "parentFill": 0.0,
        "childFill": 3.0,
        "selectedFill": 3.0,
        "requestedSize": 3.0,
    }
    assert len(order["info"]["childOrders"]) == 3
    api_mock.private_get_trade_order.assert_not_called()


def test_fetch_chase_order_okx_active_parent_ignores_canceled_replacement(
    default_conf, mocker, markets
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "state": "live",
                "side": "buy",
                "sz": "2",
                "actualSz": "0",
                "actualPx": "",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_history.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-filled",
                "state": "filled",
                "accFillSz": "1",
                "avgPx": "2501",
                "uTime": "2000",
            },
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "state": "canceled",
                "accFillSz": "0",
                "uTime": "3000",
            },
        ],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert order["status"] == "open"
    assert order["filled"] == 10.0
    assert order["remaining"] == 10.0
    assert order["id_chase"] == "child-filled"
    assert set(order["id_chase_list"]) == {"child-filled", "child-latest"}


def test_fetch_chase_order_okx_partially_effective_with_live_child_is_open(
    default_conf, mocker, markets
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "",
                "ordIdList": [],
                "linkedOrd": {"ordId": ""},
                "state": "partially_effective",
                "side": "sell",
                "sz": "2",
                "actualSz": "",
                "actualPx": "",
                "uTime": "1710000000000",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_history.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_pending.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "live",
                "side": "sell",
                "sz": "2",
                "accFillSz": "0",
                "avgPx": "",
                "px": "2501.5",
                "uTime": "1710000001000",
            }
        ],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    sleep_mock = mocker.patch("freqtrade.exchange.common.time.sleep")

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert order["status"] == "open"
    assert order["filled"] == 0.0
    assert order["remaining"] == 20.0
    assert order["id_chase"] == "67890"
    api_mock.private_get_trade_order_algo.assert_called_once_with({"algoId": "12345"})
    sleep_mock.assert_not_called()


def test_normalize_chase_order_okx_terminal_parent_with_live_child_is_open(
    default_conf, mocker, markets
):
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    parent = {
        "algoId": "12345",
        "state": "canceled",
        "side": "buy",
        "sz": "2",
        "actualSz": "0",
        "uTime": "1",
    }
    children = [
        {
            "algoId": "12345",
            "ordId": "67890",
            "state": "live",
            "accFillSz": "0",
            "px": "2501.5",
        }
    ]

    order = exchange._normalize_chase_order("ETH/USDT:USDT", parent, children)

    assert order["status"] == "open"
    assert order["filled"] == 0.0
    assert order["remaining"] == 20.0


def test_fetch_chase_order_okx_recent_zero_fill_cancel_retries(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "child-latest",
                "state": "canceled",
                "side": "sell",
                "sz": "2",
                "actualSz": "0",
                "actualPx": "",
                "uTime": str(int(datetime.now(UTC).timestamp() * 1000)),
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_order.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_pending.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_history.return_value = {"code": "0", "data": [], "msg": ""}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    mocker.patch("freqtrade.exchange.common.time.sleep")

    with pytest.raises(RetryableOrderError, match="still settling"):
        exchange.fetch_chase_order("12345", "ETH/USDT:USDT")


def test_chase_order_okx_terminal_fallback_window(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")
    item = {"algoId": "12345", "state": "canceled"}
    mocker.patch("freqtrade.exchange.okx.dt_ts", side_effect=[1000, 15999, 16000])

    with pytest.raises(RetryableOrderError, match="still settling"):
        exchange._get_chase_status(item, 2.0, 0.0)
    with pytest.raises(RetryableOrderError, match="still settling"):
        exchange._get_chase_status(item, 2.0, 0.0)
    assert exchange._get_chase_status(item, 2.0, 0.0) == "canceled"


def test_chase_order_okx_zero_fill_failure_is_rejected(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="okx")

    for state in ("order_failed", "partially_failed"):
        assert exchange._get_chase_status({"algoId": "12345", "state": state}, 2.0, 0.0) == (
            "rejected"
        )


def test_chase_order_okx_decimal_child_fills_close_parent(default_conf, mocker, markets):
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    parent = {
        "algoId": "12345",
        "state": "canceled",
        "side": "buy",
        "sz": "0.8",
        "actualSz": "0",
    }
    children = [
        {"ordId": "child-a", "state": "filled", "accFillSz": "0.1", "avgPx": "2500"},
        {"ordId": "child-b", "state": "filled", "accFillSz": "0.7", "avgPx": "2501"},
    ]

    order = exchange._normalize_chase_order("ETH/USDT:USDT", parent, children)

    assert order["status"] == "closed"
    assert order["filled"] == 8.0
    assert order["remaining"] == 0.0


@pytest.mark.parametrize(
    ("parent_state", "child_state", "future", "endpoint_name", "expected_status"),
    [
        ("live", "partially_filled", False, "private_get_trade_orders_pending", "open"),
        (
            "partially_effective",
            "partially_filled",
            False,
            "private_get_trade_orders_pending",
            "open",
        ),
        ("canceled", "canceled", True, "private_get_trade_orders_history", "canceled"),
    ],
)
def test_fetch_chase_order_okx_uses_partial_child(
    default_conf,
    mocker,
    markets,
    parent_state,
    child_state,
    future,
    endpoint_name,
    expected_status,
):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {
        "id": "ETH-USDT-SWAP",
        "future": future,
        "swap": not future,
    }
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "",
                "state": parent_state,
                "side": "buy",
                "sz": "2",
                "actualSz": "",
                "actualPx": "",
            }
        ],
        "msg": "",
    }
    getattr(api_mock, endpoint_name).return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": child_state,
                "side": "buy",
                "sz": "2",
                "accFillSz": "1",
                "avgPx": "2501.5",
                "px": "2501.5",
            }
        ],
        "msg": "",
    }
    other_endpoint_name = (
        "private_get_trade_orders_history"
        if endpoint_name == "private_get_trade_orders_pending"
        else "private_get_trade_orders_pending"
    )
    getattr(api_mock, other_endpoint_name).return_value = {"code": "0", "data": [], "msg": ""}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    getattr(api_mock, endpoint_name).assert_called_once_with(
        {
            "instType": "FUTURES" if future else "SWAP",
            "instId": "ETH-USDT-SWAP",
            "limit": 100,
        }
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["status"] == expected_status
    assert order["filled"] == 10.0
    assert order["remaining"] == 10.0
    assert order["average"] == 2501.5


def test_fetch_chase_order_okx_paginates_child_orders(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "",
                "state": "live",
                "side": "sell",
                "sz": "2",
                "actualSz": "",
                "actualPx": "",
                "cTime": "1000",
            }
        ],
        "msg": "",
    }
    first_page = [
        {"algoId": "other", "ordId": str(200 - index), "cTime": "2000"} for index in range(100)
    ]
    child_order = {
        "algoId": "12345",
        "ordId": "67890",
        "state": "live",
        "side": "sell",
        "sz": "2",
        "accFillSz": "0",
        "avgPx": "",
        "px": "2501.5",
        "cTime": "1500",
    }
    api_mock.private_get_trade_orders_pending.side_effect = [
        {"code": "0", "data": first_page, "msg": ""},
        {"code": "0", "data": [child_order], "msg": ""},
    ]
    api_mock.private_get_trade_orders_history.return_value = {"code": "0", "data": [], "msg": ""}
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    child_request = {"instType": "SWAP", "instId": "ETH-USDT-SWAP", "limit": 100}
    assert [args[0] for args, _ in api_mock.private_get_trade_orders_pending.call_args_list] == [
        child_request,
        child_request | {"after": "101"},
    ]
    assert order["id_chase"] == "67890"
    assert order["status"] == "open"


def test_fetch_chase_order_okx_translates_child_error(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "",
                "state": "live",
                "side": "buy",
                "sz": "2",
                "actualSz": "",
                "actualPx": "",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {
        "code": "50000",
        "data": [],
        "msg": "Order history unavailable",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    with pytest.raises(TemporaryError, match="Order history unavailable"):
        exchange.fetch_chase_order("12345", "ETH/USDT:USDT")


def test_cancel_chase_order_okx_keeps_active_child_open(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_post_trade_cancel_algos.return_value = {
        "code": "0",
        "data": [{"algoId": "12345", "sCode": "0", "sMsg": ""}],
        "msg": "",
    }
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "partially_effective",
                "side": "sell",
                "sz": "2",
                "actualSz": "1",
                "actualPx": "2501.5",
            }
        ],
        "msg": "",
    }
    api_mock.private_get_trade_orders_pending.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_history.return_value = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_order.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "partially_filled",
                "accFillSz": "1",
                "avgPx": "2501.5",
            }
        ],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.cancel_chase_order("12345", "ETH/USDT:USDT")

    api_mock.private_post_trade_cancel_algos.assert_called_once_with(
        [{"algoId": "12345", "instId": "ETH-USDT-SWAP"}]
    )
    assert order["id"] == "12345"
    assert order["id_chase"] == "67890"
    assert order["status"] == "open"
    assert order["filled"] == 10.0
    assert order["remaining"] == 10.0


def test_fetch_chase_order_okx_old_zero_fill_cancel_resolves(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "canceled",
                "side": "buy",
                "sz": "2",
                "actualSz": "0",
                "actualPx": "",
                "uTime": str(int((datetime.now(UTC) - timedelta(seconds=20)).timestamp() * 1000)),
            }
        ],
        "msg": "",
    }
    empty_response = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_pending.return_value = empty_response
    api_mock.private_get_trade_orders_history.return_value = empty_response
    api_mock.private_get_trade_order.return_value = empty_response
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )

    order = exchange.fetch_chase_order("12345", "ETH/USDT:USDT")

    assert order["status"] == "canceled"
    assert order["filled"] == 0.0
    assert order["remaining"] == 20.0


def test_cancel_chase_order_okx_delayed_fill_wins(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.CROSS
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.fetch_accounts.return_value = [{"info": {"acctLv": "2", "posMode": "net_mode"}}]
    api_mock.private_post_trade_cancel_algos.return_value = {
        "code": "0",
        "data": [{"algoId": "12345", "sCode": "0", "sMsg": ""}],
        "msg": "",
    }
    api_mock.private_get_trade_order_algo.return_value = {
        "code": "0",
        "data": [
            {
                "algoId": "12345",
                "ordId": "67890",
                "state": "canceled",
                "side": "sell",
                "sz": "2",
                "actualSz": "0",
                "actualPx": "",
                "uTime": str(int(datetime.now(UTC).timestamp() * 1000)),
            }
        ],
        "msg": "",
    }
    empty_response = {"code": "0", "data": [], "msg": ""}
    api_mock.private_get_trade_orders_history.side_effect = [
        empty_response,
        {
            "code": "0",
            "data": [
                {
                    "algoId": "12345",
                    "ordId": "67890",
                    "state": "filled",
                    "accFillSz": "2",
                    "avgPx": "2502",
                    "uTime": "3000",
                }
            ],
            "msg": "",
        },
    ]
    api_mock.private_get_trade_orders_pending.return_value = empty_response
    api_mock.private_get_trade_order.return_value = empty_response
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    mocker.patch("freqtrade.exchange.common.time.sleep")

    order = exchange.cancel_chase_order("12345", "ETH/USDT:USDT")

    assert order["status"] == "closed"
    assert order["filled"] == 20.0
    assert order["average"] == 2502.0
    api_mock.private_post_trade_cancel_algos.assert_called_once()
    assert api_mock.private_get_trade_order_algo.call_count == 2


def test_create_chase_order_okx_rejects_item_error(default_conf, mocker, markets):
    default_conf["dry_run"] = False
    default_conf["trading_mode"] = TradingMode.FUTURES
    default_conf["margin_mode"] = MarginMode.ISOLATED
    market = markets["ETH/USDT:USDT"] | {"id": "ETH-USDT-SWAP", "future": False}
    api_mock = MagicMock()
    api_mock.private_post_trade_order_algo.return_value = {
        "code": "0",
        "data": [{"algoId": "", "sCode": "51000", "sMsg": "Invalid chase order"}],
        "msg": "",
    }
    exchange = get_patched_exchange(
        mocker,
        default_conf,
        api_mock=api_mock,
        exchange="okx",
        mock_markets={"ETH/USDT:USDT": market},
    )
    mocker.patch.object(exchange, "_lev_prep")

    with pytest.raises(InvalidOrderException, match="Invalid chase order"):
        exchange.create_order(
            pair="ETH/USDT:USDT",
            ordertype="chase",
            side="buy",
            amount=20.0,
            rate=2500.0,
            leverage=3.0,
        )
