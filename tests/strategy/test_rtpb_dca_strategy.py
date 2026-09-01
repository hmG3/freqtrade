import logging
import math
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pandas import DataFrame, date_range

from freqtrade.exchange import ROUND_DOWN, ROUND_UP
from freqtrade.persistence import Order, Trade
from user_data.strategies.rtpb_dca_strategy import (
    RTPBDCAStrategy,
    calc_rtpb_traps_nb,
)


def _strategy() -> RTPBDCAStrategy:
    strategy = RTPBDCAStrategy({"stake_currency": "USDT"})
    strategy.vol_scale.value = 1.5
    strategy.max_safe_orders.value = 5
    strategy.so_atr_mult.value = 2.0
    strategy.tp_atr_mult.value = 1.0
    return strategy


def _ohlcv(close: np.ndarray) -> DataFrame:
    return DataFrame(
        {
            "date": date_range("2026-01-01", periods=len(close), freq="1min", tz="UTC"),
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


def _trap_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    trap_window: int,
    signal_gap: int,
    enable_longs: bool,
    enable_shorts: bool,
) -> tuple[np.ndarray, np.ndarray]:
    bull = np.zeros(len(close), dtype=np.int8)
    bear = np.zeros(len(close), dtype=np.int8)
    close_above_count = 0
    close_below_count = 0
    last_signal = -1_000_000

    for index in range(len(close)):
        raw_bear = False
        raw_bull = False
        if index > 0 and close[index] < upper[index]:
            raw_bear = (
                high[index] > upper[index]
                and close[index - 1] < upper[index - 1]
                and close_above_count <= trap_window
            ) or (close[index - 1] > upper[index - 1] and close_above_count <= trap_window)
        if index > 0 and close[index] > lower[index]:
            raw_bull = (
                low[index] < lower[index]
                and close[index - 1] > lower[index - 1]
                and close_below_count <= trap_window
            ) or (close[index - 1] < lower[index - 1] and close_below_count <= trap_window)

        can_fire = index - last_signal >= signal_gap
        if raw_bull and enable_longs and can_fire:
            bull[index] = 1
        if raw_bear and enable_shorts and can_fire:
            bear[index] = 1
        if bull[index] or bear[index]:
            last_signal = index

        close_above_count = close_above_count + 1 if close[index] > upper[index] else 0
        close_below_count = close_below_count + 1 if close[index] < lower[index] else 0

    return bull, bear


def _filled_order(trade: Trade, price: float, index: int) -> Order:
    order = Order.parse_from_ccxt_object(
        {
            "id": f"entry-{index}",
            "symbol": trade.pair,
            "status": "closed",
            "side": trade.entry_side,
            "type": "limit",
            "price": price,
            "average": price,
            "amount": 1.0,
            "filled": 1.0,
            "cost": price,
            "remaining": 0.0,
        },
        trade.pair,
        trade.entry_side,
    )
    side_prefix = "📉" if trade.is_short else "📈"
    order.ft_order_tag = (
        f"{side_prefix}-📍"
        if index == 0
        else f"{side_prefix}-🛡️{RTPBDCAStrategy.SAFETY_ORDER_NUMERALS[index - 1]}"
    )
    return order


def _trade(is_short: bool, entry_prices: tuple[float, ...] = (100.0,)) -> Trade:
    trade = Trade(
        id=81,
        pair="ETH/USDT:USDT",
        stake_amount=sum(entry_prices),
        amount=float(len(entry_prices)),
        fee_open=0.001,
        fee_close=0.001,
        is_open=True,
        open_date=datetime(2026, 1, 1, tzinfo=UTC),
        open_rate=sum(entry_prices) / len(entry_prices),
        exchange="binance",
        strategy="RTPBDCAStrategy",
        timeframe=1,
        leverage=1.0,
        is_short=is_short,
    )
    for index, price in enumerate(entry_prices):
        trade.orders.append(_filled_order(trade, price, index))
    return trade


def _add_open_exit(trade: Trade, price: float = 105.0) -> None:
    order = Order.parse_from_ccxt_object(
        {
            "id": "exit-open",
            "symbol": trade.pair,
            "status": "open",
            "side": trade.exit_side,
            "type": "limit",
            "price": price,
            "amount": trade.amount,
            "filled": 0.0,
            "cost": 0.0,
            "remaining": trade.amount,
        },
        trade.pair,
        trade.exit_side,
    )
    order.ft_order_tag = RTPBDCAStrategy.TAKE_PROFIT_EXIT_TAG
    trade.orders.append(order)


def _custom_data(monkeypatch, trade: Trade, initial: dict | None = None) -> dict:
    data = dict(initial or {})
    monkeypatch.setattr(trade, "get_custom_data", lambda key, default=None: data.get(key, default))
    monkeypatch.setattr(
        trade,
        "set_custom_data",
        lambda key, value: data.__setitem__(key, value),
    )
    return data


def _attach_exchange(strategy: RTPBDCAStrategy, dataframe: DataFrame) -> None:
    def round_price(pair: str, price: float, *, rounding_mode: int) -> float:
        ticks = price / 0.01
        if rounding_mode == ROUND_UP:
            return math.ceil(ticks - 1e-12) * 0.01
        if rounding_mode == ROUND_DOWN:
            return math.floor(ticks + 1e-12) * 0.01
        raise AssertionError("unexpected rounding mode")

    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(price_to_precision=round_price),
        get_analyzed_dataframe=lambda pair, timeframe: (dataframe, None),
    )


def test_trap_kernel_matches_state_machine_reference() -> None:
    x = np.arange(160, dtype=np.float64)
    upper = np.full(len(x), 105.0)
    lower = np.full(len(x), 95.0)
    close = 100.0 + np.sin(x / 4.0) * 7.0
    high = close + 1.2
    low = close - 1.2

    expected = _trap_reference(high, low, close, upper, lower, 4, 6, True, True)
    actual = calc_rtpb_traps_nb(high, low, close, upper, lower, 4, 6, True, True)

    assert actual[0].dtype == np.int8
    assert actual[1].dtype == np.int8
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert np.count_nonzero(actual[0]) + np.count_nonzero(actual[1]) > 0


def test_disabled_traps_do_not_consume_shared_signal_gap() -> None:
    upper = np.full(5, 105.0)
    lower = np.full(5, 95.0)
    close = np.array([100.0, 106.0, 104.0, 94.0, 96.0])
    high = np.array([101.0, 107.0, 106.0, 95.0, 97.0])
    low = np.array([99.0, 105.0, 103.0, 93.0, 94.0])

    bull, bear = calc_rtpb_traps_nb(
        high,
        low,
        close,
        upper,
        lower,
        trap_window=3,
        signal_gap=10,
        enable_longs=True,
        enable_shorts=False,
    )

    assert np.flatnonzero(bear).tolist() == []
    assert np.flatnonzero(bull).tolist() == [4]


def test_populate_indicators_keeps_only_operational_and_plot_columns() -> None:
    strategy = _strategy()
    strategy.envelope_length.value = 5
    strategy.atr_length.value = 3
    dataframe = _ohlcv(100.0 + np.sin(np.arange(30) / 2.0) * 8.0)
    original_columns = set(dataframe.columns)

    result = strategy.populate_indicators(dataframe, {"pair": "ETH/USDT:USDT"})

    assert set(result.columns) - original_columns == {
        "basis",
        "upper_band",
        "lower_band",
        "atr",
        "signal",
    }
    assert not any(column.startswith("_") for column in result.columns)


def test_entry_population_uses_only_the_current_trap_candle() -> None:
    strategy = _strategy()
    dataframe = _ohlcv(np.full(5, 100.0))
    dataframe["signal"] = ["", "↑", "", "↓", ""]

    result = strategy.populate_entry_trend(dataframe, {})

    assert result["enter_tag"].tolist() == [None, "📈-📍", None, "📉-📍", None]


def test_tags_follow_directional_composition_pattern() -> None:
    strategy = _strategy()

    assert strategy.LONG_ENTRY_TAG == "📈-📍"
    assert strategy.SHORT_ENTRY_TAG == "📉-📍"
    assert strategy._safety_order_tag(strategy.LONG_TAG_PREFIX, 1) == "📈-🛡️⓵"
    assert strategy._safety_order_tag(strategy.SHORT_TAG_PREFIX, 2) == "📉-🛡️⓶"
    assert strategy.TAKE_PROFIT_EXIT_TAG == "🎯"
    assert strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG == "🛟"


def test_order_fill_uses_submission_basis_and_fill_time_atr(caplog, monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade(False)
    data = _custom_data(monkeypatch, trade)
    dataframe = _ohlcv(np.array([100.0, 101.0, 102.0, 103.0]))
    dataframe["basis"] = [100.5, 101.5, 102.5, 103.5]
    dataframe["atr"] = [1.5, 2.5, 3.5, 4.5]
    _attach_exchange(strategy, dataframe)
    trade.orders[0].order_date = dataframe["date"].iat[2].to_pydatetime()

    with caplog.at_level(logging.INFO, logger="user_data.strategies.rtpb_dca_strategy"):
        strategy.order_filled(
            pair=trade.pair,
            trade=trade,
            order=trade.orders[0],
            current_time=dataframe["date"].iat[-1].to_pydatetime(),
        )

        safety_order = _filled_order(trade, 95.0, 1)
        safety_order.order_date = dataframe["date"].iat[2].to_pydatetime()
        trade.orders.append(safety_order)
        strategy.order_filled(
            pair=trade.pair,
            trade=trade,
            order=safety_order,
            current_time=dataframe["date"].iat[-1].to_pydatetime(),
        )

    assert data["rtpb_tp_basis"] == pytest.approx(101.5)
    assert data["rtpb_tp_atr"] == pytest.approx(4.5)
    assert data["rtpb_last_dca_signal_time"] == str(dataframe["date"].iat[1])
    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith("Initial order filled | #81 ETH/USDT:USDT long")
    assert messages[1].startswith("SO filled | #81 ETH/USDT:USDT long")


@pytest.mark.parametrize("is_short", [False, True])
def test_custom_exit_price_uses_exact_fee_adjustment_and_directional_tick_rounding(
    caplog,
    monkeypatch,
    is_short: bool,
) -> None:
    strategy = _strategy()
    trade = _trade(is_short)
    _custom_data(
        monkeypatch,
        trade,
        {"rtpb_tp_basis": 100.0, "rtpb_tp_atr": 2.0},
    )
    dataframe = _ohlcv(np.array([100.0]))
    dataframe["signal"] = ""
    _attach_exchange(strategy, dataframe)

    break_even = trade.calc_close_rate_for_roi(0.0)
    raw_tp_rate = (
        break_even - 2.0 / (1.0 + trade.fee_close)
        if is_short
        else break_even + 2.0 / (1.0 - trade.fee_close)
    )
    expected_tp_rate = (
        math.floor(raw_tp_rate * 100.0) / 100.0
        if is_short
        else math.ceil(raw_tp_rate * 100.0) / 100.0
    )

    with caplog.at_level(logging.INFO, logger="user_data.strategies.rtpb_dca_strategy"):
        result = strategy.custom_exit_price(
            pair=trade.pair,
            trade=trade,
            current_time=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            proposed_rate=100.0,
            current_profit=0.0,
            exit_tag=strategy.TAKE_PROFIT_EXIT_TAG,
        )

    assert result == pytest.approx(expected_tp_rate)
    assert next(record.getMessage() for record in caplog.records).startswith(
        "TP limit | #81 ETH/USDT:USDT"
    )


def test_profitable_opposite_trap_does_not_replace_take_profit(monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade(False)
    _custom_data(
        monkeypatch,
        trade,
        {"rtpb_tp_basis": 101.0, "rtpb_tp_atr": 1.0},
    )
    dataframe = _ohlcv(np.array([100.0]))
    dataframe["signal"] = "↓"
    _attach_exchange(strategy, dataframe)
    minimum_tp_rate = strategy._minimum_tp_rate(trade)

    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            current_rate=minimum_tp_rate,
            current_profit=trade.calc_profit_ratio(minimum_tp_rate),
        )
        == strategy.TAKE_PROFIT_EXIT_TAG
    )

    _add_open_exit(trade)
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2026, 1, 1, 1, 1, tzinfo=UTC),
            current_rate=minimum_tp_rate,
            current_profit=trade.calc_profit_ratio(minimum_tp_rate),
        )
        is None
    )


def test_resting_tp_does_not_block_qualifying_safety_order(caplog, monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade(False)
    data = _custom_data(
        monkeypatch,
        trade,
        {"rtpb_tp_basis": 101.0, "rtpb_tp_atr": 1.0},
    )
    _add_open_exit(trade)
    dataframe = _ohlcv(np.array([97.9]))
    dataframe["atr"] = 1.0
    dataframe["signal"] = "↑"
    _attach_exchange(strategy, dataframe)

    with caplog.at_level(logging.INFO, logger="user_data.strategies.rtpb_dca_strategy"):
        adjustment = strategy.adjust_trade_position(
            trade=trade,
            current_time=dataframe["date"].iat[-1].to_pydatetime(),
            current_rate=97.9,
            current_profit=-0.03,
            min_stake=None,
            max_stake=1000.0,
            current_entry_rate=97.9,
            current_exit_rate=97.9,
            current_entry_profit=-0.03,
            current_exit_profit=-0.03,
        )

    assert adjustment is not None
    assert adjustment[0] == pytest.approx(trade.orders[0].stake_amount_filled * 1.5)
    assert adjustment[1] == "📈-🛡️⓵"
    assert data["rtpb_last_dca_signal_time"] == str(dataframe["date"].iat[-1])
    assert next(record.getMessage() for record in caplog.records).startswith(
        "SO requested | #81 ETH/USDT:USDT long"
    )


def test_pending_non_tp_exit_blocks_safety_order(monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade(False)
    _custom_data(
        monkeypatch,
        trade,
        {"rtpb_tp_basis": 101.0, "rtpb_tp_atr": 1.0},
    )
    _add_open_exit(trade)
    trade.orders[-1].ft_order_tag = "manual-exit"
    dataframe = _ohlcv(np.array([97.9]))
    dataframe["atr"] = 1.0
    dataframe["signal"] = "↑"
    _attach_exchange(strategy, dataframe)

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=dataframe["date"].iat[-1].to_pydatetime(),
        current_rate=97.9,
        current_profit=-0.03,
        min_stake=None,
        max_stake=1000.0,
        current_entry_rate=97.9,
        current_exit_rate=97.9,
        current_entry_profit=-0.03,
        current_exit_profit=-0.03,
    )

    assert adjustment is None


def test_emergency_break_even_replaces_normal_target_after_final_safety_order(
    caplog,
    monkeypatch,
) -> None:
    strategy = _strategy()
    strategy.max_safe_orders.value = 1
    strategy.use_emergency_be.value = True
    trade = _trade(False, (100.0, 95.0))
    _custom_data(
        monkeypatch,
        trade,
        {"rtpb_tp_basis": 110.0, "rtpb_tp_atr": 3.0},
    )
    dataframe = _ohlcv(np.array([97.0]))
    dataframe["signal"] = ""
    _attach_exchange(strategy, dataframe)

    with caplog.at_level(logging.INFO, logger="user_data.strategies.rtpb_dca_strategy"):
        result = strategy.custom_exit_price(
            pair=trade.pair,
            trade=trade,
            current_time=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            proposed_rate=97.0,
            current_profit=0.0,
            exit_tag=strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG,
        )

    expected = math.ceil(trade.calc_close_rate_for_roi(0.0) * 100.0) / 100.0
    assert result == pytest.approx(expected)
    assert next(record.getMessage() for record in caplog.records).startswith(
        "Emergency BE limit | #81 ETH/USDT:USDT"
    )


def test_leverage_logs_all_loaded_tiers_as_a_table_once(caplog) -> None:
    strategy = _strategy()
    strategy.leverage_tier.value = 2
    pair = "LTC/USDT:USDT"
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(
            _leverage_tiers={
                pair: [
                    {
                        "minNotional": 500.1,
                        "maxNotional": 2000.0,
                        "maintenanceMarginRate": 0.01,
                        "maxLeverage": 40.0,
                    },
                    {
                        "minNotional": 0.0,
                        "maxNotional": 500.0,
                        "maintenanceMarginRate": 0.0065,
                        "maxLeverage": 50.0,
                    },
                ]
            }
        )
    )
    caplog.set_level(logging.INFO, logger="user_data.strategies.rtpb_dca_strategy")
    callback_args = {
        "pair": pair,
        "current_time": datetime(2026, 1, 1, tzinfo=UTC),
        "current_rate": 70.0,
        "proposed_leverage": 1.0,
        "max_leverage": 50.0,
        "entry_tag": "📈-📍",
        "side": "long",
    }

    assert strategy.leverage(**callback_args) == pytest.approx(40.0)
    assert strategy.leverage(**callback_args) == pytest.approx(40.0)

    table_messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Leverage tiers loaded |")
    ]
    assert len(table_messages) == 1
    table = table_messages[0]
    assert "Leverage tiers loaded | LTC/USDT:USDT" in table
    assert "Use | Tier | Notional range (USDT) | MMR   | Initial margin | Max leverage" in table
    assert "| 1    | 0 to 500" in table
    assert "0.65%" in table
    assert "2%" in table
    assert "50x" in table
    assert "*   | 2    | 500.1 to 2000" in table
    assert "1%" in table
    assert "2.5%" in table
    assert "40x" in table
