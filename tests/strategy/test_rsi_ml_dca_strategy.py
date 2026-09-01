import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pandas import DataFrame, Series, date_range

from freqtrade.enums import MarginMode
from freqtrade.persistence import CustomDataWrapper, Order, Trade
from user_data.strategies.rsi_ml_dca_strategy import (
    RSIMLDCAStrategy,
    calc_rsi_entry_state_nb,
    calc_rsi_ml_nb,
)


def _strategy() -> RSIMLDCAStrategy:
    strategy = RSIMLDCAStrategy({})
    strategy.tp_atr_mult.value = 3.0
    strategy.leverage_tier.value = 3
    return strategy


@pytest.fixture(autouse=True)
def _reset_trade_custom_data() -> Iterator[None]:
    previous_use_db = CustomDataWrapper.use_db
    CustomDataWrapper.use_db = False
    CustomDataWrapper.reset_custom_data()
    yield
    CustomDataWrapper.reset_custom_data()
    CustomDataWrapper.use_db = previous_use_db


def _trade(is_short: bool, open_rate: float = 100.0) -> Trade:
    return Trade(
        pair="ETH/USDT:USDT",
        stake_amount=100.0,
        amount=1.0,
        fee_open=0.001,
        fee_close=0.001,
        is_open=True,
        open_date=datetime(2026, 1, 1, tzinfo=UTC),
        open_rate=open_rate,
        exchange="binance",
        strategy="RSIMLDCAStrategy",
        timeframe=1,
        leverage=10.0,
        is_short=is_short,
    )


def _add_filled_entries(trade: Trade, entry_prices: tuple[float, ...]) -> None:
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


def _ohlcv_dataframe(
    close: np.ndarray,
    frequency: str = "1min",
) -> DataFrame:
    return DataFrame(
        {
            "date": date_range("2026-01-01", periods=len(close), freq=frequency, tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )


@pytest.mark.parametrize(
    ("is_short", "tp_signal"),
    [
        (False, 1),
        (True, -1),
    ],
)
def test_take_profit_requires_fresh_rsi_cross_and_atr_profit(
    is_short: bool,
    tp_signal: int,
) -> None:
    strategy = _strategy()
    trade = _trade(is_short)
    trade.id = 900 + int(is_short)
    current_time = datetime(2026, 7, 3, 6, 44, tzinfo=UTC)
    candle = Series(
        {
            "date": current_time,
            "close": 100.0,
            "atr": 2.0,
            "tp_signal": tp_signal,
        }
    )
    break_even_rate = trade.calc_close_rate_for_roi(0.0)
    tp_atr_distance = candle["atr"] * strategy.tp_atr_mult.value
    tp_target_rate = (
        break_even_rate - tp_atr_distance if is_short else break_even_rate + tp_atr_distance
    )
    rate_before_target = tp_target_rate + 0.0001 if is_short else tp_target_rate - 0.0001

    assert (
        strategy._take_profit_adjustment(
            trade,
            candle,
            current_time,
            rate_before_target,
        )
        is None
    )

    assert strategy._take_profit_adjustment(
        trade,
        candle,
        current_time,
        tp_target_rate,
    ) == (-trade.stake_amount, strategy.TAKE_PROFIT_EXIT_TAG)


def test_take_profit_does_not_latch_rsi_cross() -> None:
    strategy = _strategy()
    trade = _trade(False)
    trade.id = 902
    candle = Series(
        {
            "date": datetime(2026, 7, 3, 6, 44, tzinfo=UTC),
            "close": 110.0,
            "atr": 2.0,
            "tp_signal": 1,
        }
    )
    tp_target_rate = trade.calc_close_rate_for_roi(0.0) + (
        candle["atr"] * strategy.tp_atr_mult.value
    )

    assert strategy._take_profit_adjustment(trade, candle, candle["date"], 100.0) is None

    candle["date"] = datetime(2026, 7, 3, 6, 45, tzinfo=UTC)
    candle["tp_signal"] = 0
    assert strategy._take_profit_adjustment(trade, candle, candle["date"], tp_target_rate) is None


def test_signal_stop_loss_requires_completed_ladder_same_direction_and_atr_distance(
    caplog,
    monkeypatch,
) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.so_atr_mult, "value", 2.0)
    monkeypatch.setattr(strategy.use_signal_stop_loss, "value", True)
    trade = _trade(False)
    trade.id = 903
    _add_filled_entries(trade, (100.0, 97.0))
    candle = Series(
        {
            "date": datetime(2026, 7, 3, 6, 44, tzinfo=UTC),
            "close": 95.0,
            "atr": 1.0,
            "sl_signal": 1,
        }
    )

    with caplog.at_level(logging.INFO, logger="user_data.strategies.rsi_ml_dca_strategy"):
        adjustment = strategy._signal_stop_loss_adjustment(trade, candle)

    assert adjustment == (-trade.stake_amount, strategy.SIGNAL_STOP_LOSS_EXIT_TAG)
    assert [record.getMessage() for record in caplog.records] == [
        "Signal SL | #903 ETH/USDT:USDT long | 2026-07-03 06:44 | close 95 | "
        "required <= 95 | last entry 97 | ATR distance 1 x2 = 2"
    ]

    candle["close"] = 95.1
    assert strategy._signal_stop_loss_adjustment(trade, candle) is None

    candle["close"] = 95.0
    candle["sl_signal"] = -1
    assert strategy._signal_stop_loss_adjustment(trade, candle) is None

    incomplete_trade = _trade(False)
    incomplete_trade.id = 905
    _add_filled_entries(incomplete_trade, (100.0,))
    candle["sl_signal"] = 1
    assert strategy._signal_stop_loss_adjustment(incomplete_trade, candle) is None

    strategy.use_signal_stop_loss.value = False
    assert strategy._signal_stop_loss_adjustment(trade, candle) is None


def test_take_profit_skips_unavailable_atr() -> None:
    strategy = _strategy()
    trade = _trade(False)
    trade.id = 904
    candle = Series(
        {
            "date": datetime(2026, 7, 3, 6, 44, tzinfo=UTC),
            "close": 110.0,
            "atr": np.nan,
            "tp_signal": 1,
        }
    )

    assert (
        strategy._take_profit_adjustment(
            trade,
            candle,
            datetime.now(tz=UTC),
            110.0,
        )
        is None
    )


def test_emergency_break_even_requires_enabled_completed_ladder(monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.use_emergency_be, "value", True)
    complete_trade = _trade(False)
    incomplete_trade = _trade(False)
    _add_filled_entries(complete_trade, (100.0, 97.0))
    _add_filled_entries(incomplete_trade, (100.0,))

    assert strategy._emergency_break_even_adjustment(complete_trade) == (
        -complete_trade.stake_amount,
        strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG,
    )
    assert strategy._emergency_break_even_adjustment(incomplete_trade) is None

    strategy.use_emergency_be.value = False
    assert strategy._emergency_break_even_adjustment(complete_trade) is None


def test_emergency_break_even_precedes_dataframe_exits_and_logs_target(
    caplog,
    monkeypatch,
) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.max_safe_orders, "value", 1)
    monkeypatch.setattr(strategy.use_emergency_be, "value", True)
    trade = _trade(False)
    trade.id = 907
    _add_filled_entries(trade, (100.0, 97.0))
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda *args, **kwargs: pytest.fail(
            "emergency break-even must run before dataframe exits"
        )
    )

    current_time = datetime(2026, 8, 17, tzinfo=UTC)
    with caplog.at_level(logging.INFO, logger="user_data.strategies.rsi_ml_dca_strategy"):
        adjustment = strategy.adjust_trade_position(
            trade=trade,
            current_time=current_time,
            current_rate=90.0,
            current_profit=-0.10,
            min_stake=None,
            max_stake=1000.0,
            current_entry_rate=90.0,
            current_exit_rate=90.0,
            current_entry_profit=-0.10,
            current_exit_profit=-0.10,
        )

    assert adjustment == (-trade.stake_amount, strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG)
    assert [record.getMessage() for record in caplog.records] == [
        "Emergency BE limit | #907 ETH/USDT:USDT long | 2026-08-17 00:00 | "
        f"target {strategy._format_log_number(trade.calc_close_rate_for_roi(0.0))}"
    ]


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
        exit_tag=strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG,
    ) == pytest.approx(trade.calc_close_rate_for_roi(0.0))


def test_custom_exit_price_preserves_unrelated_rsi_exit_price() -> None:
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


def test_entry_state_requires_full_extreme_before_zero_count_pivot() -> None:
    overbuy_count = np.array([0, 11, 4, 0, 0, 5, 0], dtype=np.int16)
    oversell_count = np.array([11, 3, 0, 0, 7, 0, 0], dtype=np.int16)

    *_, bull_signal, bear_signal = calc_rsi_entry_state_nb(
        overbuy_count,
        oversell_count,
        11,
    )

    assert bull_signal.tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert bear_signal.tolist() == [0, 0, 0, 1, 0, 0, 0]


def test_higher_filter_rsi_timeframe_filters_base_timeframe_engine() -> None:
    strategy = _strategy()
    base_close = 100.0 + np.sin(np.arange(30, dtype=np.float64) / 2.0)
    informative_close = 200.0 + np.arange(8, dtype=np.float64)
    dataframe = _ohlcv_dataframe(base_close)
    informative = _ohlcv_dataframe(informative_close, "5min")
    strategy.dp = SimpleNamespace(
        get_pair_dataframe=lambda pair, timeframe: informative,
    )

    (
        result,
        entry_engine,
        entry_filter_avg_rsi,
        exit_engine,
        sl_engine,
    ) = strategy._RSIMLDCAStrategy__prepare_rsi_engines(
        dataframe=dataframe,
        metadata={"pair": "ETH/USDT:USDT"},
        entry_filter_timeframe="5m",
        exit_timeframe="1m",
        sl_timeframe="5m",
        entry_min_length=2,
        entry_max_length=3,
        exit_min_length=2,
        exit_max_length=3,
        overbought=70.0,
        oversold=30.0,
    )

    expected_entry_engine = calc_rsi_ml_nb(base_close, 2, 3, 70.0, 30.0)
    expected_filter_avg = calc_rsi_ml_nb(
        informative_close,
        2,
        3,
        70.0,
        30.0,
    )[0]

    assert np.allclose(entry_engine[0], expected_entry_engine[0], equal_nan=True)
    assert np.allclose(exit_engine[0], expected_entry_engine[0], equal_nan=True)
    assert np.isnan(sl_engine[0][:4]).all()
    assert np.allclose(sl_engine[0][4:9], expected_filter_avg[0])
    assert entry_filter_avg_rsi is not None
    assert np.isnan(entry_filter_avg_rsi[:4]).all()
    assert np.allclose(entry_filter_avg_rsi[4:9], expected_filter_avg[0])
    assert np.allclose(entry_filter_avg_rsi[9:14], expected_filter_avg[1])
    assert not any(column.startswith("_") for column in result.columns)
    assert not any(column.startswith("date_") for column in result.columns)


def test_base_filter_rsi_timeframe_disables_filter() -> None:
    strategy = _strategy()
    close = 100.0 + np.sin(np.arange(30, dtype=np.float64) / 2.0)
    dataframe = _ohlcv_dataframe(close)

    _, entry_engine, entry_filter_avg_rsi, _, sl_engine = (
        strategy._RSIMLDCAStrategy__prepare_rsi_engines(
            dataframe=dataframe,
            metadata={"pair": "ETH/USDT:USDT"},
            entry_filter_timeframe="1m",
            exit_timeframe="1m",
            sl_timeframe="1m",
            entry_min_length=2,
            entry_max_length=3,
            exit_min_length=2,
            exit_max_length=3,
            overbought=70.0,
            oversold=30.0,
        )
    )

    expected_entry_avg = calc_rsi_ml_nb(close, 2, 3, 70.0, 30.0)[0]
    assert np.allclose(entry_engine[0], expected_entry_avg, equal_nan=True)
    assert np.allclose(sl_engine[0], expected_entry_avg, equal_nan=True)
    assert entry_filter_avg_rsi is None


def test_filter_rsi_timeframe_registers_informative_pair() -> None:
    strategy = _strategy()
    strategy.filter_rsi_timeframe = "5m"
    strategy.signal_stop_loss_timeframe = "15m"
    strategy.dp = SimpleNamespace(current_whitelist=lambda: ["ETH/USDT:USDT"])

    assert strategy.informative_pairs() == [
        ("ETH/USDT:USDT", "5m"),
        ("ETH/USDT:USDT", "15m"),
    ]


@pytest.mark.parametrize(
    ("entry_filter_avg_rsi", "expected_signal"),
    [
        (np.array([50.0, 80.0, 50.0, 20.0]), ["", "↓", "", "↑"]),
        (np.array([50.0, 69.0, 50.0, 31.0]), ["", "", "", ""]),
        (None, ["", "↓", "", "↑"]),
    ],
)
def test_higher_timeframe_average_rsi_gates_entry_signals(
    entry_filter_avg_rsi: np.ndarray | None,
    expected_signal: list[str],
) -> None:
    strategy = _strategy()
    dataframe = _ohlcv_dataframe(np.full(4, 100.0, dtype=np.float64))
    average = np.full(4, 50.0, dtype=np.float64)
    entry_engine = (
        average,
        average.copy(),
        average.copy(),
        np.array([1, 0, 0, 0], dtype=np.int16),
        np.array([0, 0, 1, 0], dtype=np.int16),
    )
    exit_engine = (
        average.copy(),
        average.copy(),
        average.copy(),
        np.zeros(4, dtype=np.int16),
        np.zeros(4, dtype=np.int16),
    )

    result = strategy._RSIMLDCAStrategy__populate_rsi_ml_dca(
        dataframe=dataframe,
        entry_engine=entry_engine,
        entry_filter_avg_rsi=entry_filter_avg_rsi,
        exit_engine=exit_engine,
        sl_engine=entry_engine,
        entry_length_count=1,
        overbought=70.0,
        oversold=30.0,
        mrc_length=2,
        mrc_outer_multiplier=2.618,
        atr_length=2,
    )

    assert result["signal"].tolist() == expected_signal


def test_populate_indicators_uses_fresh_exit_rsi_crosses_for_take_profit() -> None:
    strategy = _strategy()
    dataframe = _ohlcv_dataframe(np.full(6, 100.0, dtype=np.float64))
    average = np.full(6, 50.0, dtype=np.float64)
    entry_engine = (
        average,
        average.copy(),
        average.copy(),
        np.zeros(6, dtype=np.int16),
        np.zeros(6, dtype=np.int16),
    )
    exit_engine = (
        np.array([50.0, 69.0, 71.0, 72.0, 31.0, 29.0]),
        np.full(6, 70.0),
        np.full(6, 30.0),
        np.zeros(6, dtype=np.int16),
        np.zeros(6, dtype=np.int16),
    )

    result = strategy._RSIMLDCAStrategy__populate_rsi_ml_dca(
        dataframe=dataframe,
        entry_engine=entry_engine,
        entry_filter_avg_rsi=None,
        exit_engine=exit_engine,
        sl_engine=entry_engine,
        entry_length_count=1,
        overbought=70.0,
        oversold=30.0,
        mrc_length=2,
        mrc_outer_multiplier=2.618,
        atr_length=2,
    )

    assert result["tp_signal"].tolist() == [0, 0, 1, 0, 0, -1]


def test_populate_indicators_keeps_only_callback_and_plot_columns() -> None:
    strategy = _strategy()
    close = 100.0 + np.sin(np.arange(260, dtype=np.float64) / 5.0) * 10.0
    dataframe = DataFrame(
        {
            "date": date_range("2026-01-01", periods=len(close), freq="1min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(len(close), 100.0),
        }
    )

    result = strategy.populate_indicators(dataframe, {"pair": "ETH/USDT:USDT"})

    assert {
        "atr",
        "avg_rsi",
        "buy_rsi_ma",
        "sell_rsi_ma",
        "mrc_mean",
        "mrc_upper",
        "mrc_lower",
        "overbuy_100",
        "overbuy_zone",
        "oversell_100",
        "oversell_zone",
        "signal",
        "tp_signal",
        "sl_signal",
    } <= set(result.columns)
    assert "signal_stop_signal" not in result.columns
    assert "long_tp_arm" not in result.columns
    assert "short_tp_arm" not in result.columns
    assert "so_atr" not in result.columns
    assert not any(column.startswith("_") for column in result.columns)


def test_populate_entry_trend_uses_extracted_order_tags(monkeypatch) -> None:
    strategy = _strategy()
    monkeypatch.setattr(strategy.mrc_enable, "value", False)
    monkeypatch.setattr(strategy.enable_longs, "value", True)
    monkeypatch.setattr(strategy.enable_shorts, "value", True)
    dataframe = DataFrame(
        {
            "signal": ["↑", "↓"],
            "volume": [100.0, 100.0],
        }
    )

    result = strategy.populate_entry_trend(dataframe, {"pair": "ETH/USDT:USDT"})

    assert strategy.LONG_TAG_PREFIX == "📈"
    assert strategy.SHORT_TAG_PREFIX == "📉"
    assert strategy.ENTRY_TAG_SUFFIX == "-📍"
    assert strategy.LONG_ENTRY_TAG == "📈-📍"
    assert strategy.SHORT_ENTRY_TAG == "📉-📍"
    assert strategy.LONG_ENTRY_TAG == strategy.LONG_TAG_PREFIX + strategy.ENTRY_TAG_SUFFIX
    assert strategy.SHORT_ENTRY_TAG == strategy.SHORT_TAG_PREFIX + strategy.ENTRY_TAG_SUFFIX
    assert result.loc[0, "enter_tag"] == strategy.LONG_ENTRY_TAG
    assert result.loc[1, "enter_tag"] == strategy.SHORT_ENTRY_TAG


@pytest.mark.parametrize(
    ("configured_stake", "proposed_stake", "expected_initial_stake"),
    [
        (630.0, 630.0, 10.0),
        ("unlimited", 1000.0, 1000.0 / 63.0),
    ],
)
def test_custom_stake_reserves_dca_budget_for_fixed_and_unlimited_stakes(
    configured_stake: float | str,
    proposed_stake: float,
    expected_initial_stake: float,
) -> None:
    strategy = RSIMLDCAStrategy(
        {
            "stake_amount": configured_stake,
            "stake_currency": "USDT",
        }
    )

    stake = strategy.custom_stake_amount(
        pair="ETH/USDT:USDT",
        current_time=datetime(2026, 7, 3, 6, 44, tzinfo=UTC),
        current_rate=100.0,
        proposed_stake=proposed_stake,
        min_stake=None,
        max_stake=proposed_stake,
        leverage=10.0,
        entry_tag="L-BO",
        side="long",
    )

    assert stake == pytest.approx(expected_initial_stake)


def test_custom_stake_logs_requested_stake_without_trade_id(caplog) -> None:
    strategy = RSIMLDCAStrategy(
        {
            "stake_amount": 1130.81668108,
            "stake_currency": "USDT",
        }
    )

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        stake = strategy.custom_stake_amount(
            pair="DOGE/USDT:USDT",
            current_time=datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
            current_rate=0.0997,
            proposed_stake=1130.81668108,
            min_stake=None,
            max_stake=1130.81668108,
            leverage=20.0,
            entry_tag="L-BO",
            side="long",
        )

    assert stake == pytest.approx(1130.81668108 / 63.0)
    assert [record.getMessage() for record in caplog.records] == [
        "Initial stake requested | DOGE/USDT:USDT long | 2026-07-24 21:22 | "
        "17.949 USDT | DCA budget 1130.817 USDT / multiplier 63"
    ]


def test_order_filled_logs_actual_initial_stake_but_not_safety_orders(caplog) -> None:
    strategy = RSIMLDCAStrategy({"stake_currency": "USDT"})
    requested_time = datetime(2026, 7, 24, 21, 22, tzinfo=UTC)
    trade = _trade(False)
    trade.id = 41
    trade.pair = "DOGE/USDT:USDT"
    trade.leverage = 20.0
    price = 17.947 * trade.leverage / 3600.0
    first_order = Order.parse_from_ccxt_object(
        {
            "id": "base-order",
            "symbol": trade.pair,
            "status": "closed",
            "side": trade.entry_side,
            "type": "limit",
            "price": price,
            "average": price,
            "amount": 3600.0,
            "filled": 3600.0,
            "cost": 358.94,
            "remaining": 0.0,
        },
        trade.pair,
        trade.entry_side,
    )
    trade.orders.append(first_order)

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        strategy.order_filled(
            pair=trade.pair,
            trade=trade,
            order=first_order,
            current_time=requested_time,
        )

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Initial order filled | #41 DOGE/USDT:USDT long | 2026-07-24 21:22 | 17.947 USDT",
    ]

    second_order = Order.parse_from_ccxt_object(
        {
            "id": "safety-order",
            "symbol": trade.pair,
            "status": "closed",
            "side": trade.entry_side,
            "type": "limit",
            "price": price,
            "average": price,
            "amount": 7200.0,
            "filled": 7200.0,
            "cost": 717.88,
            "remaining": 0.0,
        },
        trade.pair,
        trade.entry_side,
    )
    trade.orders.append(second_order)
    caplog.clear()

    strategy.order_filled(
        pair=trade.pair,
        trade=trade,
        order=second_order,
        current_time=datetime(2026, 7, 24, 21, 23, tzinfo=UTC),
    )

    assert caplog.text == ""


def test_initial_fill_consumes_entry_signal_before_safety_order() -> None:
    strategy = _strategy()
    strategy.config["stake_currency"] = "USDT"
    trade = _trade(False)
    trade.id = 42
    _add_filled_entries(trade, (100.0,))
    signal_time = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    order_time = datetime(2026, 1, 1, 1, 1, tzinfo=UTC)
    trade.orders[0].order_date = order_time
    candle = Series(
        {
            "date": signal_time,
            "close": 97.0,
            "atr": 1.0,
            "signal": "↑",
        }
    )
    previous_candle = candle.copy()
    previous_candle["date"] = signal_time.replace(minute=59, hour=0)
    previous_candle["signal"] = ""
    execution_candle = candle.copy()
    execution_candle["date"] = order_time
    execution_candle["signal"] = ""
    dataframe = DataFrame([previous_candle, candle, execution_candle])
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda *args: (dataframe, None))

    strategy.order_filled(
        pair=trade.pair,
        trade=trade,
        order=trade.orders[0],
        current_time=order_time,
    )

    assert trade.get_custom_data(strategy.LAST_DCA_SIGNAL_KEY) == str(signal_time)
    assert (
        strategy._safety_order_adjustment(
            trade=trade,
            last_candle=candle,
            current_time=order_time,
            min_stake=None,
            max_stake=1000.0,
        )
        is None
    )


def test_initial_fill_does_not_consume_a_post_order_signal() -> None:
    strategy = _strategy()
    strategy.config["stake_currency"] = "USDT"
    trade = _trade(False)
    trade.id = 43
    _add_filled_entries(trade, (100.0,))
    order_time = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    trade.orders[0].order_date = order_time
    future_candle = Series(
        {
            "date": datetime(2026, 1, 1, 1, 1, tzinfo=UTC),
            "close": 97.0,
            "atr": 1.0,
            "signal": "↑",
        }
    )
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda *args: (DataFrame([future_candle]), None)
    )

    strategy.order_filled(
        pair=trade.pair,
        trade=trade,
        order=trade.orders[0],
        current_time=order_time,
    )

    assert trade.get_custom_data(strategy.LAST_DCA_SIGNAL_KEY) is None


def test_take_profit_logs_each_candle_result_once(caplog) -> None:
    strategy = _strategy()
    trade = _trade(False)
    trade.id = 145
    trade.pair = "DOGE/USDT:USDT"
    trade.amount = 35677.0
    trade.leverage = 20.0
    trade.open_rate = 0.06987
    trade.recalc_open_trade_value()
    candle = Series(
        {
            "date": datetime(2026, 7, 24, 6, 19, tzinfo=UTC),
            "close": 0.06934,
            "atr": 0.00003625,
            "tp_signal": 1,
        }
    )
    tp_target_rate = trade.calc_close_rate_for_roi(0.0) + (
        candle["atr"] * strategy.tp_atr_mult.value
    )

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        for _ in range(2):
            strategy._take_profit_adjustment(
                trade,
                candle,
                datetime(2026, 7, 24, 6, 19, tzinfo=UTC),
                tp_target_rate - 0.00001,
            )
        for _ in range(2):
            strategy._take_profit_adjustment(
                trade,
                candle,
                datetime(2026, 7, 24, 6, 19, tzinfo=UTC),
                tp_target_rate,
            )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("TP skipped | #145 DOGE/USDT:USDT long | 2026-07-24 06:19 |")
    assert messages[1].startswith("TP reached | #145 DOGE/USDT:USDT long | 2026-07-24 06:19 |")
    assert all("Trade(" not in message for message in messages)
    assert all("+00:00" not in message for message in messages)
    assert all("ATR distance 0.00003625 x3 = 0.00010875" in message for message in messages)


def test_safety_order_tags_use_circled_numerals() -> None:
    strategy = _strategy()

    assert strategy.SAFETY_ORDER_TAG_SUFFIX == "-🛡️"
    assert [
        strategy._safety_order_tag(strategy.LONG_TAG_PREFIX, safety_index)
        for safety_index in range(1, 10)
    ] == [
        "📈-🛡️⓵",
        "📈-🛡️⓶",
        "📈-🛡️⓷",
        "📈-🛡️⓸",
        "📈-🛡️⓹",
        "📈-🛡️⓺",
        "📈-🛡️⓻",
        "📈-🛡️⓼",
        "📈-🛡️⓽",
    ]


@pytest.mark.parametrize(
    ("is_short", "signal", "candle_close", "comparison"),
    [
        (False, "↑", 0.06934, "<="),
        (True, "↓", 0.06890, ">="),
    ],
)
def test_safety_order_atr_skip_is_logged_once(
    caplog,
    is_short: bool,
    signal: str,
    candle_close: float,
    comparison: str,
) -> None:
    strategy = _strategy()
    strategy.so_atr_mult.value = 3.0
    trade = _trade(is_short)
    trade.id = 501 if is_short else 500
    trade.pair = "DOGE/USDT:USDT"
    trade.leverage = 20.0
    entry_price = 0.06902
    entry_order = Order.parse_from_ccxt_object(
        {
            "id": f"entry-{trade.id}",
            "symbol": trade.pair,
            "status": "closed",
            "side": trade.entry_side,
            "type": "limit",
            "price": entry_price,
            "average": entry_price,
            "amount": 1000.0,
            "filled": 1000.0,
            "cost": entry_price * 1000.0,
            "remaining": 0.0,
        },
        trade.pair,
        trade.entry_side,
    )
    trade.orders.append(entry_order)
    reloaded_trade = _trade(is_short)
    reloaded_trade.id = trade.id
    reloaded_trade.pair = trade.pair
    reloaded_trade.leverage = trade.leverage
    reloaded_entry_order = Order.parse_from_ccxt_object(
        {
            "id": f"reloaded-entry-{trade.id}",
            "symbol": reloaded_trade.pair,
            "status": "closed",
            "side": reloaded_trade.entry_side,
            "type": "limit",
            "price": entry_price,
            "average": entry_price,
            "amount": 1000.0,
            "filled": 1000.0,
            "cost": entry_price * 1000.0,
            "remaining": 0.0,
        },
        reloaded_trade.pair,
        reloaded_trade.entry_side,
    )
    reloaded_trade.orders.append(reloaded_entry_order)
    candle = Series(
        {
            "date": datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
            "close": candle_close,
            "atr": 0.00003625,
            "signal": signal,
        }
    )
    so_atr_distance = candle["atr"] * strategy.so_atr_mult.value
    so_trigger_rate = entry_price + so_atr_distance if is_short else entry_price - so_atr_distance

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        for current_trade in (trade, reloaded_trade):
            assert (
                strategy._safety_order_adjustment(
                    trade=current_trade,
                    last_candle=candle,
                    current_time=datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
                    min_stake=None,
                    max_stake=1000.0,
                )
                is None
            )

    side = "short" if is_short else "long"
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        f"SO skipped | #{trade.id} DOGE/USDT:USDT {side} | 2026-07-24 21:22 | "
        f"close {strategy._format_log_number(candle_close)} | "
        f"required {comparison} {strategy._format_log_number(so_trigger_rate)} | "
        f"last entry {strategy._format_log_number(entry_price)} | "
        "ATR distance 0.00003625 x3 = 0.00010875"
    ]


def _ltc_leverage_tiers() -> list[dict]:
    return [
        {"minNotional": 0.0, "maxNotional": 500.0, "maxLeverage": 50.0},
        {"minNotional": 500.1, "maxNotional": 2000.0, "maxLeverage": 40.0},
        {"minNotional": 2000.1, "maxNotional": 6000.0, "maxLeverage": 20.0},
        {"minNotional": 6000.1, "maxNotional": 12000.0, "maxLeverage": 18.18},
        {"minNotional": 12000.1, "maxNotional": 18000.0, "maxLeverage": 16.66},
        {"minNotional": 18000.1, "maxNotional": 24000.0, "maxLeverage": 15.38},
        {"minNotional": 24000.1, "maxNotional": 30000.0, "maxLeverage": 14.28},
    ]


def _set_leverage_tiers(strategy: RSIMLDCAStrategy, pair: str, tiers: list[dict]) -> None:
    strategy.dp = SimpleNamespace(_exchange=SimpleNamespace(_leverage_tiers={pair: tiers}))


def test_leverage_uses_configured_exchange_tier(caplog) -> None:
    strategy = _strategy()
    pair = "LTC/USDT:USDT"
    _set_leverage_tiers(strategy, pair, _ltc_leverage_tiers())

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        leverage = strategy.leverage(
            pair=pair,
            current_time=datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
            current_rate=120.0,
            proposed_leverage=1.0,
            max_leverage=50.0,
            entry_tag="📈",
            side="long",
        )

    assert strategy.leverage_tier.value == 3
    assert leverage == 20.0
    assert [record.getMessage() for record in caplog.records] == [
        "Leverage selected | LTC/USDT:USDT long | 2026-07-24 21:22 | tier 3/7 | 20x"
    ]


def test_leverage_uses_last_available_tier() -> None:
    strategy = _strategy()
    strategy.leverage_tier.value = 50
    pair = "LTC/USDT:USDT"
    _set_leverage_tiers(strategy, pair, _ltc_leverage_tiers())

    leverage = strategy.leverage(
        pair=pair,
        current_time=datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
        current_rate=120.0,
        proposed_leverage=1.0,
        max_leverage=50.0,
        entry_tag="📈",
        side="long",
    )

    assert leverage == 14.28


def test_leverage_respects_callback_exchange_cap() -> None:
    strategy = _strategy()
    pair = "LTC/USDT:USDT"
    _set_leverage_tiers(strategy, pair, _ltc_leverage_tiers())

    leverage = strategy.leverage(
        pair=pair,
        current_time=datetime(2026, 7, 24, 21, 22, tzinfo=UTC),
        current_rate=120.0,
        proposed_leverage=1.0,
        max_leverage=18.18,
        entry_tag="📈",
        side="long",
    )

    assert leverage == 18.18


@pytest.mark.parametrize(
    ("margin_mode", "is_short", "open_rate", "expected_stop"),
    [
        (MarginMode.CROSS, False, 80.0, 0.008),
        (MarginMode.CROSS, True, 120.0, 239.988),
        (MarginMode.ISOLATED, False, 80.0, 72.0008),
        (MarginMode.ISOLATED, True, 120.0, 131.9988),
    ],
)
def test_custom_stoploss_reanchors_from_weighted_open_rate_after_fill(
    margin_mode: MarginMode,
    is_short: bool,
    open_rate: float,
    expected_stop: float,
) -> None:
    strategy = RSIMLDCAStrategy({"margin_mode": margin_mode})
    trade = _trade(is_short, open_rate)
    current_rate = 100.0

    stoploss = strategy.custom_stoploss(
        trade.pair,
        trade,
        datetime(2026, 1, 2, tzinfo=UTC),
        current_rate,
        trade.calc_profit_ratio(current_rate),
        after_fill=True,
    )

    assert strategy.use_custom_stoploss is True
    assert stoploss is not None
    trade.adjust_stop_loss(current_rate, stoploss, allow_refresh=True)
    assert trade.stop_loss == pytest.approx(expected_stop)


@pytest.mark.parametrize("margin_mode", [MarginMode.CROSS, MarginMode.ISOLATED])
def test_custom_stoploss_does_not_move_between_fills(margin_mode: MarginMode) -> None:
    strategy = RSIMLDCAStrategy({"margin_mode": margin_mode})
    trade = _trade(False)

    assert (
        strategy.custom_stoploss(
            trade.pair,
            trade,
            datetime(2026, 1, 2, tzinfo=UTC),
            105.0,
            trade.calc_profit_ratio(105.0),
            after_fill=False,
        )
        is None
    )
