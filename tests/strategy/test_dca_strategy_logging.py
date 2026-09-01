"""The four DCA strategies share the same order logging contract."""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pandas import DataFrame

from freqtrade.enums import RunMode
from freqtrade.persistence import CustomDataWrapper, Order, Trade
from user_data.strategies.adx_vw_dca_strategy import ADXVWDCAStrategy
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy
from user_data.strategies.rsi_ml_dca_strategy import RSIMLDCAStrategy
from user_data.strategies.rtb_dca_strategy import RTBDCAStrategy


FILL_TIME = datetime(2026, 8, 31, 12, 32, tzinfo=UTC)
CANDLE_TIME = datetime(2026, 8, 31, 12, 31, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _in_memory_custom_data(monkeypatch):
    monkeypatch.setattr(CustomDataWrapper, "use_db", False)
    CustomDataWrapper.reset_custom_data()
    yield
    CustomDataWrapper.reset_custom_data()


@pytest.fixture(params=[ADXVWDCAStrategy, RSIMLDCAStrategy, RTBDCAStrategy, QFLDCAStrategy])
def strategy(request, monkeypatch):
    instance = request.param({"stake_currency": "USDT"})
    monkeypatch.setattr(instance.vol_scale, "value", 1.5)
    monkeypatch.setattr(instance.max_safe_orders, "value", 5)
    monkeypatch.setattr(instance.enable_longs, "value", True)
    monkeypatch.setattr(instance.enable_shorts, "value", True)
    if not isinstance(instance, QFLDCAStrategy):
        monkeypatch.setattr(instance.so_atr_mult, "value", 2.0)
    instance.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (
            DataFrame({"date": [CANDLE_TIME], "close": [48.82], "basis": [50.0], "atr": [1.0]}),
            None,
        ),
    )
    return instance


def _trade(entry_count=1, is_short=False, contract_size=1.0, stake_currency="USDT"):
    trade = Trade(
        id=71,
        pair=f"LTC/{stake_currency}:{stake_currency}",
        stake_currency=stake_currency,
        stake_amount=2318.95 * entry_count,
        amount=475.0 * entry_count,
        open_rate=48.82,
        open_date=CANDLE_TIME,
        is_open=True,
        is_short=is_short,
        fee_open=0.001,
        fee_close=0.001,
        exchange="okx",
        leverage=10.0,
        contract_size=contract_size,
    )
    for index in range(entry_count):
        # A partially executed, canceled order still counts as a filled entry.
        # Price and requested amount deliberately differ from the actual fill.
        order = Order.parse_from_ccxt_object(
            {
                "id": f"entry-{index}",
                "symbol": trade.pair,
                "status": "canceled",
                "side": trade.entry_side,
                "type": "limit",
                "price": 50.0,
                "average": 48.82,
                "amount": 500.0,
                "filled": 475.0,
                "remaining": 25.0,
                "cost": 23189.5,
            },
            trade.pair,
            trade.entry_side,
        )
        order.order_date = FILL_TIME
        order.order_filled_date = FILL_TIME
        trade.orders.append(order)
    return trade


@pytest.mark.parametrize("runmode", [RunMode.BACKTEST, RunMode.LIVE, RunMode.DRY_RUN])
@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize(
    ("entry_count", "contract_size", "expected_body"),
    [
        (1, 1.0, "BO filled | 475 contracts / 475 LTC / 23189.5 USDT"),
        (2, 0.1, "SO1 filled | 4750 contracts / 475 LTC / 23189.5 USDT"),
        (3, 10.0, "SO2 filled | 47.5 contracts / 475 LTC / 23189.5 USDT"),
    ],
)
def test_entry_fill_logs_numbered_order_and_executed_quantities(
    strategy, caplog, runmode, is_short, entry_count, contract_size, expected_body
):
    strategy.config["runmode"] = runmode
    trade = _trade(entry_count, is_short, contract_size)

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        strategy.order_filled(trade.pair, trade, trade.orders[-1], FILL_TIME)

    side = "short" if is_short else "long"
    prefix = f"[LTC/USDT:USDT {side} 2026-08-31 12:32] " if runmode == RunMode.BACKTEST else ""
    assert caplog.messages == [prefix + expected_body]


def test_entry_fill_uses_exchange_contract_size_when_not_saved(strategy, caplog):
    trade = _trade(contract_size=None)
    strategy.dp._exchange = SimpleNamespace(get_contract_size=lambda pair: 0.1)

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        strategy.order_filled(trade.pair, trade, trade.orders[-1], FILL_TIME)

    assert caplog.messages == ["BO filled | 4750 contracts / 475 LTC / 23189.5 USDT"]


def test_entry_fill_preserves_currency_specific_amount_precision(strategy, caplog):
    trade = _trade(contract_size=0.1)
    trade.pair = "ETH/USDT:USDT"
    trade.amount = 0.123456789
    trade.open_rate = 100.0
    trade.stake_amount = 1.23456789
    order = trade.orders[-1]
    order.ft_pair = trade.pair
    order.amount = 0.2
    order.filled = 0.123456789
    order.remaining = 0.076543211
    order.average = order.price = 100.0
    order.cost = 12.3456789

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        strategy.order_filled(trade.pair, trade, order, FILL_TIME)

    assert caplog.messages == ["BO filled | 1.235 contracts / 0.12346 ETH / 12.346 USDT"]


@pytest.mark.parametrize("contract_size", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_entry_fill_does_not_invent_unknown_contract_quantity(strategy, caplog, contract_size):
    trade = _trade(contract_size=contract_size)

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        strategy.order_filled(trade.pair, trade, trade.orders[-1], FILL_TIME)

    assert caplog.messages == ["BO filled | N/A contracts / 475 LTC / 23189.5 USDT"]


@pytest.mark.parametrize("runmode", [RunMode.BACKTEST, RunMode.LIVE, RunMode.DRY_RUN])
@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("entry_count", [1, 2, 3])
@pytest.mark.parametrize(("max_stake", "expected_stake"), [(5000.0, "3478.425"), (3000.0, "3000")])
def test_safety_order_request_logs_number_and_final_collateral(
    strategy, caplog, runmode, is_short, entry_count, max_stake, expected_stake
):
    strategy.config.update(runmode=runmode, stake_currency="USDC")
    trade = _trade(entry_count, is_short, stake_currency="USDC")
    candle = DataFrame(
        {
            "date": [CANDLE_TIME],
            "close": [60.0 if is_short else 30.0],
            "atr": [1.0],
            "signal": ["↓" if is_short else "↑"],
        }
    )
    strategy.dp.get_analyzed_dataframe = lambda pair, timeframe: (candle, None)

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        if isinstance(strategy, QFLDCAStrategy):
            result = strategy.adjust_trade_position(
                trade,
                FILL_TIME,
                float(candle["close"].iat[0]),
                -0.1,
                None,
                max_stake,
                current_entry_rate=float(candle["close"].iat[0]),
                current_exit_rate=float(candle["close"].iat[0]),
                current_entry_profit=-0.1,
                current_exit_profit=-0.1,
            )
        else:
            result = strategy._safety_order_adjustment(
                trade, candle.iloc[0], FILL_TIME, None, max_stake
            )

    assert result is not None
    assert result[0] == pytest.approx(float(expected_stake))
    side = "short" if is_short else "long"
    event_time = "12:32" if isinstance(strategy, QFLDCAStrategy) else "12:31"
    prefix = (
        f"[LTC/USDC:USDC {side} 2026-08-31 {event_time}] " if runmode == RunMode.BACKTEST else ""
    )
    assert len(caplog.messages) == 1
    # QFL additionally retains its order tag and exact limit diagnostics.
    fields = caplog.messages[0].split(" | ")
    assert fields[0] == f"{prefix}SO{entry_count} requested"
    assert f"{expected_stake} USDC" in fields
    assert "#71" not in caplog.messages[0]


def test_rejected_safety_order_does_not_log_a_request(strategy, caplog):
    trade = _trade()
    candle = DataFrame({"date": [CANDLE_TIME], "close": [30.0], "atr": [1.0], "signal": ["↑"]})
    strategy.dp.get_analyzed_dataframe = lambda pair, timeframe: (candle, None)

    with caplog.at_level(logging.INFO, logger=strategy.__module__):
        if isinstance(strategy, QFLDCAStrategy):
            result = strategy.adjust_trade_position(
                trade,
                FILL_TIME,
                30.0,
                -0.1,
                4000.0,
                5000.0,
                current_entry_rate=30.0,
                current_exit_rate=30.0,
                current_entry_profit=-0.1,
                current_exit_profit=-0.1,
            )
        else:
            result = strategy._safety_order_adjustment(
                trade, candle.iloc[0], FILL_TIME, 4000.0, 5000.0
            )

    assert result is None
    assert caplog.messages == []
