import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pandas import DataFrame, Series, date_range

from freqtrade.persistence import Order, Trade
from user_data.strategies import adx_vw_dca_strategy as adx_module
from user_data.strategies.adx_vw_dca_strategy import (
    ADXVWDCAStrategy,
    calc_adx_vw_signals_nb,
)


ENGINE_SETTINGS = (8, 0.7, 5, 4, 0.5, 0.0, 3.0, 6, 5)


def test_okx_config_supports_break_even_limit_exit() -> None:
    config_path = Path(__file__).parents[2] / "user_data" / "config_okx_baryga.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["order_types"]["exit"] == "limit"
    assert config["custom_price_max_distance_ratio"] == 1.0
    assert config["exit_pricing"] == {
        "price_side": "same",
        "use_order_book": True,
        "order_book_top": 1,
    }


def _strategy() -> ADXVWDCAStrategy:
    strategy = ADXVWDCAStrategy({"stake_currency": "USDT"})
    strategy.leverage_tier.value = 3
    strategy.max_safe_orders.value = 5
    strategy.vol_scale.value = 1.5
    strategy.so_atr_mult.value = 2.0
    strategy.tp_atr_mult.value = 1.0
    return strategy


def _ohlcv_dataframe(close: np.ndarray, frequency: str = "1min") -> DataFrame:
    return DataFrame(
        {
            "date": date_range("2026-01-01", periods=len(close), freq=frequency, tz="UTC"),
            "open": close - 0.05,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


def _trade(is_short: bool, entry_prices: tuple[float, ...] = (100.0,)) -> Trade:
    trade = Trade(
        id=71,
        pair="ETH/USDT:USDT",
        stake_amount=100.0,
        amount=1.0,
        fee_open=0.001,
        fee_close=0.001,
        is_open=True,
        open_date=datetime(2026, 1, 1, tzinfo=UTC),
        open_rate=100.0,
        exchange="binance",
        strategy="ADXVWDCAStrategy",
        timeframe=1,
        leverage=10.0,
        is_short=is_short,
    )
    for index, price in enumerate(entry_prices):
        order = Order.parse_from_ccxt_object(
            {
                "id": f"entry-{index}",
                "symbol": trade.pair,
                "status": "closed",
                "side": trade.entry_side,
                "type": "limit",
                "price": price,
                "average": price,
                "amount": 10.0,
                "filled": 10.0,
                "cost": price * 10.0,
                "remaining": 0.0,
            },
            trade.pair,
            trade.entry_side,
        )
        trade.orders.append(order)
    return trade


def _rma_reference(values: np.ndarray, length: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float64)
    seed_sum = 0.0
    seed_count = 0
    previous = np.nan

    for index, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if seed_count < length:
            seed_sum += value
            seed_count += 1
            if seed_count == length:
                previous = seed_sum / length
                output[index] = previous
        else:
            previous = (previous * (length - 1) + value) / length
            output[index] = previous

    return output


def _signals_reference(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    settings: tuple[int, float, int, int, float, float, float, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        bb_length,
        bb_multiplier,
        adx_length,
        adx_smoothing,
        adx_influence,
        zone_offset,
        zone_expansion,
        smooth_length,
        signal_cooldown,
    ) = settings
    n = len(close)
    true_range = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for index in range(n):
        if index == 0:
            true_range[index] = high[index] - low[index]
            continue
        high_change = high[index] - high[index - 1]
        low_change = low[index - 1] - low[index]
        plus_dm[index] = high_change if high_change > low_change and high_change > 0 else 0.0
        minus_dm[index] = low_change if low_change > high_change and low_change > 0 else 0.0
        true_range[index] = max(
            high[index] - low[index],
            abs(high[index] - close[index - 1]),
            abs(low[index] - close[index - 1]),
        )

    atr = _rma_reference(true_range, adx_length)
    plus_smoothed = _rma_reference(plus_dm, adx_length)
    minus_smoothed = _rma_reference(minus_dm, adx_length)
    dx = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(atr)
    plus_di = np.divide(100.0 * plus_smoothed, atr, out=np.zeros(n), where=valid & (atr > 0))
    minus_di = np.divide(100.0 * minus_smoothed, atr, out=np.zeros(n), where=valid & (atr > 0))
    di_sum = plus_di + minus_di
    dx[valid] = np.divide(
        100.0 * np.abs(plus_di[valid] - minus_di[valid]),
        di_sum[valid],
        out=np.zeros(np.count_nonzero(valid)),
        where=di_sum[valid] > 0,
    )
    adx = _rma_reference(dx, adx_smoothing)

    close_series = Series(close)
    basis = close_series.rolling(bb_length, min_periods=bb_length).mean().to_numpy()
    deviation = close_series.rolling(bb_length, min_periods=bb_length).std(ddof=0).to_numpy()
    adjusted = bb_multiplier * deviation * (1.0 + adx / 100.0 * adx_influence)
    upper = (
        Series(basis + adjusted).rolling(smooth_length, min_periods=smooth_length).mean().to_numpy()
    )
    lower = (
        Series(basis - adjusted).rolling(smooth_length, min_periods=smooth_length).mean().to_numpy()
    )
    band_range = upper - lower
    top_bottom = upper + band_range * zone_offset
    top_top = top_bottom + band_range * zone_expansion
    bottom_top = lower - band_range * zone_offset
    bottom_bottom = bottom_top - band_range * zone_expansion
    in_top = (close >= top_bottom) & (close <= top_top)
    in_bottom = (close <= bottom_top) & (close >= bottom_bottom)

    bull = np.zeros(n, dtype=np.int8)
    bear = np.zeros(n, dtype=np.int8)
    last_bull = -1000000
    last_bear = -1000000
    for index in range(n):
        if in_bottom[index] and (index == 0 or not in_bottom[index - 1]):
            if index - last_bull >= signal_cooldown:
                bull[index] = 1
                last_bull = index
        if in_top[index] and (index == 0 or not in_top[index - 1]):
            if index - last_bear >= signal_cooldown:
                bear[index] = 1
                last_bear = index

    return bull, bear, top_bottom, top_top, bottom_top, bottom_bottom


def test_adx_vw_kernel_matches_reference_signals_and_zone_boundaries() -> None:
    x = np.arange(240, dtype=np.float64)
    close = 100.0 + np.sin(x / 5.0) * 3.0 + np.sin(x / 17.0) * 7.0
    close[[70, 135, 195]] += np.array([12.0, -14.0, 16.0])
    high = close + 0.6 + np.cos(x / 9.0) * 0.1
    low = close - 0.6 - np.sin(x / 11.0) * 0.1

    expected = _signals_reference(high, low, close, ENGINE_SETTINGS)
    result = calc_adx_vw_signals_nb(high, low, close, *ENGINE_SETTINGS)
    bull, bear, upper_inner, upper_outer, lower_inner, lower_outer = result

    assert bull.dtype == np.int8
    assert bear.dtype == np.int8
    assert np.array_equal(bull, expected[0])
    assert np.array_equal(bear, expected[1])
    for actual, expected_boundary in zip(result[2:], expected[2:], strict=True):
        assert actual.dtype == np.float64
        assert np.allclose(actual, expected_boundary, equal_nan=True)
    assert np.all(upper_outer >= upper_inner, where=np.isfinite(upper_outer))
    assert np.all(lower_outer <= lower_inner, where=np.isfinite(lower_outer))
    assert np.count_nonzero(bull) + np.count_nonzero(bear) > 0


def test_adx_vw_kernel_handles_empty_input() -> None:
    empty = np.empty(0, dtype=np.float64)

    result = calc_adx_vw_signals_nb(empty, empty, empty, *ENGINE_SETTINGS)
    bull, bear, *boundaries = result

    assert bull.dtype == np.int8
    assert bear.dtype == np.int8
    assert len(result) == 6
    assert len(bull) == len(bear) == 0
    assert all(boundary.dtype == np.float64 and len(boundary) == 0 for boundary in boundaries)


def test_entry_engine_has_no_separate_timeframe_setting() -> None:
    strategy = _strategy()

    assert not hasattr(strategy, "entry_adx_timeframe")


def test_lower_management_timeframes_are_rejected() -> None:
    strategy = _strategy()
    strategy.timeframe = "5m"
    strategy.exit_adx_timeframe = "1m"

    with pytest.raises(ValueError, match="exit_adx_timeframe"):
        strategy._adx_timeframes()


def test_higher_timeframe_signal_is_confirmed_deduplicated_and_temp_columns_drop(
    monkeypatch,
) -> None:
    strategy = _strategy()
    dataframe = _ohlcv_dataframe(np.full(16, 100.0))
    informative = _ohlcv_dataframe(np.full(4, 100.0), "5min")
    strategy.dp = SimpleNamespace(
        get_pair_dataframe=lambda pair, timeframe: informative,
    )
    calculation_lengths: list[int] = []

    def fake_signals(high, low, close, *settings):
        calculation_lengths.append(len(close))
        bull = np.zeros(len(close), dtype=np.int8)
        bear = np.zeros(len(close), dtype=np.int8)
        bull[1] = 1
        boundaries = tuple(np.full(len(close), np.nan) for _ in range(4))
        return bull, bear, *boundaries

    monkeypatch.setattr(adx_module, "calc_adx_vw_signals_nb", fake_signals)

    result, _, exit_engine, sl_engine = strategy._ADXVWDCAStrategy__prepare_signal_engines(
        dataframe=dataframe,
        metadata={"pair": "ETH/USDT:USDT"},
        exit_timeframe="5m",
        sl_timeframe="5m",
        entry_settings=ENGINE_SETTINGS,
        exit_settings=ENGINE_SETTINGS,
    )

    assert np.flatnonzero(exit_engine[0]).tolist() == [9]
    assert np.flatnonzero(sl_engine[0]).tolist() == [9]
    assert calculation_lengths == [len(dataframe), len(informative)]
    assert not any(column.startswith("_") for column in result.columns)
    assert not any(column.startswith("date_") for column in result.columns)


def test_indicator_signal_columns_and_reverse_candidate_window() -> None:
    strategy = _strategy()
    dataframe = _ohlcv_dataframe(np.full(7, 100.0))
    entry_engine = (
        np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
        np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int8),
        np.arange(101.0, 108.0),
        np.arange(111.0, 118.0),
        np.arange(91.0, 98.0),
        np.arange(81.0, 88.0),
    )
    exit_engine = (
        np.array([0, 0, 0, 0, 1, 0, 0], dtype=np.int8),
        np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int8),
    )
    sl_engine = (
        np.array([0, 0, 1, 0, 0, 0, 0], dtype=np.int8),
        np.array([0, 0, 0, 0, 0, 1, 0], dtype=np.int8),
    )

    result = strategy._ADXVWDCAStrategy__populate_adx_vw_dca(
        dataframe,
        entry_engine,
        exit_engine,
        sl_engine,
        atr_length=2,
    )

    assert result["signal"].tolist() == ["", "↑", "", "↓", "", "", ""]
    assert result["tp_signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    assert result["sl_signal"].tolist() == [0, 0, 1, 0, 0, -1, 0]
    assert result["reverse_signal"].tolist() == [0, -1, -1, 0, 1, 1, 0]
    assert result["upper_zone_inner"].tolist() == list(np.arange(101.0, 108.0))
    assert result["upper_zone_outer"].tolist() == list(np.arange(111.0, 118.0))
    assert result["lower_zone_inner"].tolist() == list(np.arange(91.0, 98.0))
    assert result["lower_zone_outer"].tolist() == list(np.arange(81.0, 88.0))


def test_populate_indicators_keeps_only_callback_and_plot_columns() -> None:
    strategy = _strategy()
    x = np.arange(140, dtype=np.float64)
    dataframe = _ohlcv_dataframe(100.0 + np.sin(x / 6.0) * 5.0)
    original_columns = set(dataframe.columns)

    result = strategy.populate_indicators(dataframe, {"pair": "ETH/USDT:USDT"})

    assert set(result.columns) - original_columns == {
        "atr",
        "signal",
        "tp_signal",
        "sl_signal",
        "reverse_signal",
        "upper_zone_inner",
        "upper_zone_outer",
        "lower_zone_inner",
        "lower_zone_outer",
    }
    assert "signal_stop_signal" not in result.columns
    assert not any(column.startswith("_") for column in result.columns)


def test_plot_config_draws_entry_zone_inner_and_outer_boundaries() -> None:
    strategy = _strategy()
    main_plot = strategy.plot_config["main_plot"]

    assert set(main_plot) == {
        "upper_zone_inner",
        "upper_zone_outer",
        "lower_zone_inner",
        "lower_zone_outer",
    }
    assert main_plot["upper_zone_inner"]["fill_to"] == "upper_zone_outer"
    assert main_plot["lower_zone_inner"]["fill_to"] == "lower_zone_outer"


def test_long_and_short_tags_use_shared_prefix_suffix_combinations() -> None:
    strategy = _strategy()
    dataframe = DataFrame(
        {
            "signal": ["↑", "↓", "", ""],
            "reverse_signal": [1, -1, 1, -1],
            "volume": [100.0, 100.0, 100.0, 100.0],
        }
    )

    result = strategy.populate_entry_trend(dataframe, {"pair": "ETH/USDT:USDT"})

    assert strategy.LONG_TAG_PREFIX == "📈"
    assert strategy.SHORT_TAG_PREFIX == "📉"
    assert strategy.ENTRY_TAG_SUFFIX == "-📍"
    assert strategy.REVERSE_ENTRY_TAG_SUFFIX == "-🔄"
    assert strategy.SAFETY_ORDER_TAG_SUFFIX == "-🛡️"
    assert strategy.LONG_ENTRY_TAG == "📈-📍"
    assert strategy.SHORT_ENTRY_TAG == "📉-📍"
    assert strategy.LONG_REVERSE_TAG == "📈-🔄"
    assert strategy.SHORT_REVERSE_TAG == "📉-🔄"
    assert strategy.TAKE_PROFIT_EXIT_TAG == "🎯"
    assert strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG == "🛟"
    assert result["enter_tag"].tolist() == ["📈-📍", "📉-📍", "📈-🔄", "📉-🔄"]


def test_custom_stake_reserves_complete_geometric_dca_budget() -> None:
    strategy = _strategy()
    strategy.vol_scale.value = 1.5
    strategy.max_safe_orders.value = 5

    stake = strategy.custom_stake_amount(
        pair="ETH/USDT:USDT",
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        proposed_stake=207.8125,
        min_stake=None,
        max_stake=207.8125,
        leverage=20.0,
        entry_tag=strategy.LONG_ENTRY_TAG,
        side="long",
    )

    assert strategy._max_dca_multiplier() == pytest.approx(20.78125)
    assert stake == pytest.approx(10.0)


def test_safety_order_requires_fresh_signal_and_atr_distance(monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade(False, (100.0,))
    custom_data: dict[str, str] = {}
    monkeypatch.setattr(
        trade,
        "get_custom_data",
        lambda key, default=None: custom_data.get(key, default),
    )
    monkeypatch.setattr(
        trade,
        "set_custom_data",
        lambda key, value: custom_data.__setitem__(key, value),
    )
    candle = Series(
        {
            "date": datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            "close": 97.9,
            "atr": 1.0,
            "signal": "↑",
        }
    )

    adjustment = strategy._safety_order_adjustment(
        trade,
        candle,
        datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        min_stake=None,
        max_stake=1000.0,
    )

    assert adjustment is not None
    assert adjustment[0] == pytest.approx(
        trade.select_filled_orders(trade.entry_side)[-1].stake_amount_filled * 1.5
    )
    assert adjustment[1] == "📈-🛡️⓵"

    candle["date"] = datetime(2026, 1, 1, 1, 1, tzinfo=UTC)
    candle["close"] = 98.1
    assert (
        strategy._safety_order_adjustment(
            trade,
            candle,
            datetime(2026, 1, 1, 1, 1, tzinfo=UTC),
            min_stake=None,
            max_stake=1000.0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("is_short", "tp_signal", "exit_tag"),
    [
        (False, 1, "🎯"),
        (True, -1, "🎯"),
    ],
)
def test_take_profit_uses_fee_aware_atr_target_and_arms_reversal(
    is_short: bool,
    tp_signal: int,
    exit_tag: str,
) -> None:
    strategy = _strategy()
    trade = _trade(is_short)
    break_even = trade.calc_close_rate_for_roi(0.0)
    direction = -1.0 if is_short else 1.0
    tp_target_rate = break_even + direction * 2.0
    candle = Series(
        {
            "date": datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            "close": tp_target_rate,
            "atr": 2.0,
            "tp_signal": tp_signal,
        }
    )

    assert (
        strategy._take_profit_adjustment(
            trade,
            candle,
            datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            tp_target_rate - direction * 0.01,
        )
        is None
    )
    assert strategy._take_profit_adjustment(
        trade,
        candle,
        datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        tp_target_rate,
    ) == (-trade.stake_amount, exit_tag)


def test_signal_stop_loss_requires_completed_ladder_same_direction_and_distance(caplog) -> None:
    strategy = _strategy()
    strategy.max_safe_orders.value = 1
    strategy.use_signal_stop_loss.value = True
    trade = _trade(False, (100.0, 97.0))
    candle = Series(
        {
            "date": datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            "close": 94.9,
            "atr": 1.0,
            "sl_signal": 1,
        }
    )

    with caplog.at_level(logging.INFO, logger="user_data.strategies.adx_vw_dca_strategy"):
        assert strategy._signal_stop_loss_adjustment(trade, candle) == (
            -trade.stake_amount,
            strategy.SIGNAL_STOP_LOSS_EXIT_TAG,
        )

    assert [record.getMessage() for record in caplog.records] == [
        "Signal SL | #71 ETH/USDT:USDT long | 2026-01-01 01:00 | close 94.9 | "
        "required <= 95 | last entry 97 | ATR distance 1 x2 = 2"
    ]

    candle["close"] = 95.1
    assert strategy._signal_stop_loss_adjustment(trade, candle) is None


def test_emergency_break_even_requires_enabled_completed_ladder(monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.use_emergency_be, "value", True)
    complete_trade = _trade(False, (100.0, 97.0))
    incomplete_trade = _trade(False, (100.0,))

    assert strategy._emergency_break_even_adjustment(complete_trade) == (
        -complete_trade.stake_amount,
        "🛟",
    )
    assert strategy._emergency_break_even_adjustment(incomplete_trade) is None

    strategy.use_emergency_be.value = False
    assert strategy._emergency_break_even_adjustment(complete_trade) is None


def test_emergency_break_even_precedes_dataframe_exits_at_negative_profit(monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.use_emergency_be, "value", True)
    trade = _trade(False, (100.0, 97.0))
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda *args, **kwargs: pytest.fail(
            "emergency break-even must run before dataframe exits"
        )
    )

    assert strategy.adjust_trade_position(
        trade=trade,
        current_time=datetime(2026, 8, 17, tzinfo=UTC),
        current_rate=90.0,
        current_profit=-0.10,
        min_stake=None,
        max_stake=1000.0,
        current_entry_rate=90.0,
        current_exit_rate=90.0,
        current_entry_profit=-0.10,
        current_exit_profit=-0.10,
    ) == (-trade.stake_amount, "🛟")


@pytest.mark.parametrize("is_short", [False, True])
def test_custom_exit_price_uses_fee_aware_break_even_for_lifebuoy(is_short: bool) -> None:
    strategy = _strategy()
    trade = _trade(is_short)

    assert strategy.custom_exit_price(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 8, 17, tzinfo=UTC),
        proposed_rate=91.0,
        current_profit=-0.10,
        exit_tag="🛟",
    ) == pytest.approx(trade.calc_close_rate_for_roi(0.0))


def test_custom_exit_price_preserves_unrelated_adx_exit_price() -> None:
    strategy = _strategy()
    trade = _trade(False)

    assert (
        strategy.custom_exit_price(
            pair=trade.pair,
            trade=trade,
            current_time=datetime(2026, 8, 17, tzinfo=UTC),
            proposed_rate=105.0,
            current_profit=0.05,
            exit_tag=strategy.TAKE_PROFIT_EXIT_TAG,
        )
        == 105.0
    )


def test_adjust_trade_position_logs_emergency_break_even_limit(caplog, monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.use_emergency_be, "value", True)
    trade = _trade(False, (100.0, 97.0))
    candle = DataFrame(
        [
            {
                "date": datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
                "close": 100.5,
                "atr": 1.0,
                "signal": "",
                "tp_signal": 0,
                "sl_signal": 0,
            }
        ]
    )
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (candle, None),
    )

    with caplog.at_level(logging.INFO, logger="user_data.strategies.adx_vw_dca_strategy"):
        adjustment = strategy.adjust_trade_position(
            trade=trade,
            current_time=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            current_rate=100.5,
            current_profit=0.003,
            min_stake=None,
            max_stake=1000.0,
            current_entry_rate=100.5,
            current_exit_rate=100.5,
            current_entry_profit=0.003,
            current_exit_profit=-0.003,
        )

    assert adjustment == (-trade.stake_amount, "🛟")
    assert [record.getMessage() for record in caplog.records] == [
        (
            "Emergency BE limit | #71 ETH/USDT:USDT long | 2026-01-01 01:00 | "
            f"target {strategy._format_log_number(trade.calc_close_rate_for_roi(0.0))}"
        )
    ]


@pytest.mark.parametrize(
    ("previous_short", "entry_tag", "side"),
    [
        (False, ADXVWDCAStrategy.SHORT_REVERSE_TAG, "short"),
        (True, ADXVWDCAStrategy.LONG_REVERSE_TAG, "long"),
    ],
)
def test_recent_matching_reverse_exit_allows_opposite_entry(
    monkeypatch,
    previous_short: bool,
    entry_tag: str,
    side: str,
) -> None:
    strategy = _strategy()
    current_time = datetime(2026, 1, 1, 1, 1, tzinfo=UTC)
    previous_trade = SimpleNamespace(
        is_short=previous_short,
        exit_reason=strategy.TAKE_PROFIT_EXIT_TAG,
        close_date_utc=current_time - timedelta(minutes=1),
    )
    monkeypatch.setattr(
        Trade,
        "get_trades_proxy",
        staticmethod(lambda **kwargs: [previous_trade]),
    )

    assert strategy.confirm_trade_entry(
        pair="ETH/USDT:USDT",
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="gtc",
        current_time=current_time,
        entry_tag=entry_tag,
        side=side,
    )


@pytest.mark.parametrize(
    ("side", "entry_tag", "previous_short", "exit_reason", "age_minutes"),
    [
        ("short", ADXVWDCAStrategy.SHORT_REVERSE_TAG, False, "🛟", 1),
        ("short", ADXVWDCAStrategy.SHORT_REVERSE_TAG, True, "🎯", 1),
        ("short", ADXVWDCAStrategy.SHORT_REVERSE_TAG, False, "🎯", 2),
        ("long", ADXVWDCAStrategy.LONG_REVERSE_TAG, False, "🎯", 1),
    ],
)
def test_reverse_entry_rejects_wrong_or_stale_closed_trade(
    monkeypatch,
    side: str,
    entry_tag: str,
    previous_short: bool,
    exit_reason: str,
    age_minutes: int,
) -> None:
    strategy = _strategy()
    current_time = datetime(2026, 1, 1, 1, 2, tzinfo=UTC)
    previous_trade = SimpleNamespace(
        is_short=previous_short,
        exit_reason=exit_reason,
        close_date_utc=current_time - timedelta(minutes=age_minutes),
    )
    monkeypatch.setattr(
        Trade,
        "get_trades_proxy",
        staticmethod(lambda **kwargs: [previous_trade]),
    )

    assert not strategy.confirm_trade_entry(
        pair="ETH/USDT:USDT",
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="gtc",
        current_time=current_time,
        entry_tag=entry_tag,
        side=side,
    )


def test_ordinary_entry_does_not_require_closed_trade_history(monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(
        Trade,
        "get_trades_proxy",
        staticmethod(lambda **kwargs: []),
    )

    assert strategy.confirm_trade_entry(
        pair="ETH/USDT:USDT",
        order_type="market",
        amount=1.0,
        rate=100.0,
        time_in_force="gtc",
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        entry_tag=strategy.LONG_ENTRY_TAG,
        side="long",
    )


def test_leverage_uses_requested_exchange_tier() -> None:
    strategy = _strategy()
    pair = "LTC/USDT:USDT"
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(
            _leverage_tiers={
                pair: [
                    {"minNotional": 0.0, "maxLeverage": 50.0},
                    {"minNotional": 500.1, "maxLeverage": 40.0},
                    {"minNotional": 2000.1, "maxLeverage": 20.0},
                ]
            }
        )
    )

    assert (
        strategy.leverage(
            pair=pair,
            current_time=datetime(2026, 1, 1, tzinfo=UTC),
            current_rate=100.0,
            proposed_leverage=1.0,
            max_leverage=50.0,
            entry_tag=strategy.LONG_ENTRY_TAG,
            side="long",
        )
        == 20.0
    )


def test_take_profit_log_is_emitted_once_per_candle(caplog) -> None:
    strategy = _strategy()
    trade = _trade(False)
    break_even = trade.calc_close_rate_for_roi(0.0)
    candle = Series(
        {
            "date": datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
            "close": break_even,
            "atr": 1.0,
            "tp_signal": 1,
        }
    )

    with caplog.at_level(logging.INFO, logger="user_data.strategies.adx_vw_dca_strategy"):
        for _ in range(2):
            strategy._take_profit_adjustment(
                trade,
                candle,
                datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
                break_even,
            )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert messages[0].startswith("TP skipped | #71 ETH/USDT:USDT long | 2026-01-01 01:00 |")
    assert "ATR distance 1 x1 = 1" in messages[0]
