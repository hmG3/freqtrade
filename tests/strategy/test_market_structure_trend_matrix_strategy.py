from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ccxt import TICK_SIZE
from pandas import DataFrame, date_range
from pandas.testing import assert_frame_equal

from freqtrade.exchange import amount_to_contract_precision
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.custom_data import CustomDataWrapper
from freqtrade.resolvers import StrategyResolver
from freqtrade.util import FtPrecise
from user_data.strategies.market_structure_trend_matrix_strategy import (
    MarketStructureTrendMatrixStrategy,
)


@pytest.fixture(autouse=True)
def custom_data(monkeypatch):
    monkeypatch.setattr(CustomDataWrapper, "use_db", False)
    monkeypatch.setattr(CustomDataWrapper, "custom_data", [])


def strategy(**parameters):
    result = MarketStructureTrendMatrixStrategy({"stake_currency": "USDT"})
    settings = {
        "ms_length": 10,
        "atr_length": 14,
        "atr_multiplier": 4.0,
        "target_step_multiplier": 2.0,
        "target_count": 3,
    }
    settings.update(parameters)
    for name, value in settings.items():
        getattr(result, name).value = value
    return result


def candles(high, low=None, close=None):
    high = np.asarray(high, dtype=float)
    low = high - 2 if low is None else np.asarray(low, dtype=float)
    close = (high + low) / 2 if close is None else np.asarray(close, dtype=float)
    return DataFrame(
        {
            "date": date_range("2026-01-01", periods=len(high), freq="15min", tz="UTC"),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(high), 100.0),
        }
    )


def test_pivots_are_only_published_on_confirmation_candle():
    s = strategy(ms_length=2, atr_length=2)
    result = s.populate_indicators(candles([3, 4, 7, 5, 4, 3, 4, 5]), {})
    assert result["pivot_high"].iloc[:4].isna().all()
    assert result["pivot_high"].iloc[4:].tolist() == [7.0] * 4
    assert result["pivot_low"].iloc[:7].isna().all()
    assert result["pivot_low"].iloc[7] == 1.0


def test_atr_uses_pine_first_bar_true_range_and_wilder_seed():
    s = strategy(ms_length=1, atr_length=3)
    result = s.populate_indicators(candles([11, 13, 16, 14], [9, 11, 12, 10]), {})
    # TR = 2, 3, 4, 4; seed = 3; next RMA = (3 * 2 + 4) / 3.
    np.testing.assert_allclose(result["atr"], [np.nan, np.nan, 3.0, 10 / 3])


def test_structure_flips_and_stop_ratchet_match_hand_calculation():
    s = strategy(ms_length=1, atr_length=1, atr_multiplier=2.0, target_step_multiplier=2.0)
    result = s.populate_indicators(
        candles(
            [10, 12, 11, 13, 14, 13, 12, 9],
            [8, 10, 9, 11, 12, 11, 10, 7],
            [9, 11, 10, 12.5, 13, 12, 11, 8],
        ),
        {},
    )
    assert result["choch"].tolist() == [0, 0, 0, 1, 0, 0, 0, -1]
    np.testing.assert_allclose(result["atr_stop"].iloc[3:7], [6.5, 9, 9, 9])
    assert result["atr_stop"].iloc[7] == 16.0
    assert result["structure_entry"].iloc[3] == 12.0
    assert result["initial_target"].iloc[3] == 18.0
    assert result["structure_entry"].iloc[7] == 9.0
    assert result["initial_target"].iloc[7] == 1.0


def test_future_candles_do_not_change_existing_indicators_or_signals():
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 2, 250))
    data = candles(close + 1, close - 1, close)
    s = strategy(ms_length=3, atr_length=5)

    def analyze(frame):
        return s.populate_exit_trend(
            s.populate_entry_trend(s.populate_indicators(frame, {}), {}), {}
        )

    full = analyze(data.copy())
    assert (full["choch"] == 1).any() and (full["choch"] == -1).any()
    for length in (1, 5, 20, 60, 125, 249):
        assert_frame_equal(analyze(data.iloc[:length].copy()), full.iloc[:length])


def test_indicator_target_uses_current_atr_and_advances_only_once_per_bar():
    s = strategy(ms_length=1, atr_length=1, target_step_multiplier=0.2)
    result = s.populate_indicators(
        candles(
            [10, 12, 11, 13, 14],
            [8, 10, 9, 11, 12],
            [9, 11, 10, 12.5, 13],
        ),
        {},
    )
    assert result["initial_target"].iloc[3] == pytest.approx(12.6)
    assert result["current_target"].iloc[3] == pytest.approx(13.2)
    # High=14 passes two potential levels; Pine advances only one, using ATR=2.
    assert result["current_target"].iloc[4] == pytest.approx(13.6)
    assert result["target_hit"].iloc[3:].tolist() == [1, 1]


def test_entries_trade_both_sides_only_on_valid_structure_changes():
    s = strategy()
    data = DataFrame(
        {
            "choch": [1, 0, -1, 1, -1],
            "volume": [1, 1, 1, 0, 1],
            "atr": [2, 2, 2, 2, np.nan],
            "atr_stop": [90, 90, 110, 90, np.nan],
            "initial_target": [110, 110, 90, 110, np.nan],
        }
    )
    result = s.populate_exit_trend(s.populate_entry_trend(data, {}), {})
    assert result["enter_long"].tolist() == [1, 0, 0, 0, 0]
    assert result["enter_short"].tolist() == [0, 0, 1, 0, 0]
    assert result["exit_long"].tolist() == [0, 0, 1, 0, 1]
    assert result["exit_short"].tolist() == [1, 0, 0, 1, 0]


def filled_order(trade, side, tag, amount, number=0, status="closed", when=None):
    order = Order.parse_from_ccxt_object(
        {
            "id": f"{tag}-{number}",
            "symbol": trade.pair,
            "status": status,
            "side": side,
            "type": "limit",
            "price": 100.0,
            "average": 100.0,
            "amount": amount,
            "filled": amount,
            "cost": amount * 100,
            "remaining": 0.0,
        },
        trade.pair,
        side,
    )
    order.ft_order_tag = tag
    order.order_filled_date = when or datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    trade.orders.append(order)
    return order


def trade_context(is_short=False, count=3, leverage=1.0, initial_amount=1.0, open_rate=100.0):
    s = strategy(target_count=count)
    data = DataFrame(
        {
            "date": date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
            "atr": [2.0, 3.0],
            "direction": [-1 if is_short else 1] * 2,
            "atr_stop": [110.0, 108.0] if is_short else [90.0, 92.0],
            "initial_target": [96.0, 96.0] if is_short else [104.0, 104.0],
        }
    )
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data, None))
    t = Trade(
        id=17,
        pair="ETH/USDT:USDT",
        stake_amount=initial_amount * open_rate / leverage,
        amount=initial_amount,
        open_rate=open_rate,
        fee_open=0.0,
        fee_close=0.0,
        exchange="binance",
        open_date=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        is_open=True,
        is_short=is_short,
        leverage=leverage,
        enter_tag=("mstm_bearish_choch" if is_short else "mstm_bullish_choch")
        + ("_tp0" if count == 0 else ""),
    )
    entry = filled_order(t, t.entry_side, "choch", initial_amount, when=t.open_date_utc)
    s.order_filled(t.pair, t, entry, t.open_date_utc)
    return s, t, data


def adjust(s, t, rate, when=None, min_stake=1.0):
    return s.adjust_trade_position(
        t,
        when or datetime(2026, 1, 1, 0, 30, tzinfo=UTC),
        rate,
        t.calc_profit_ratio(rate),
        min_stake,
        10000,
        rate,
        rate,
        t.calc_profit_ratio(rate),
        t.calc_profit_ratio(rate),
    )


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("count", [0, 3])
def test_hedge_on_stop_touch_before_any_target_fill(is_short, count):
    s, t, data = trade_context(is_short, count=count)
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    stop = 108.0 if is_short else 92.0
    direction = -1 if is_short else 1
    # A chart wick hit is not a target fill in this trade.
    data["target_hit"] = 1
    assert s.custom_hedge(t.pair, t, now, stop + direction * 0.01, 0) is None
    assert s.custom_hedge(t.pair, t, now, stop, 0) == "mstm_stop_before_target"
    assert s.custom_hedge(t.pair, t, now, stop - direction, 0) == "mstm_stop_before_target"
    # Saved historical fills must not be interpreted as fresh stop touches at startup.
    assert s.custom_hedge(t.pair, t, now, stop, 0, order=t.orders[0]) is None


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("status,amount", [("closed", 0.25), ("canceled", 0.1), ("open", 0.1)])
def test_any_target_fill_disables_stop_hedge(is_short, status, amount):
    s, t, _ = trade_context(is_short)
    filled_order(t, t.exit_side, "mstm_tp_1", amount, status=status)
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    assert s.custom_hedge(t.pair, t, now, 109 if is_short else 91, 0) is None


def test_zero_fill_target_does_not_disable_stop_hedge():
    s, t, _ = trade_context()
    filled_order(t, t.exit_side, "mstm_tp_1", 0, status="canceled")
    assert s.custom_hedge(t.pair, t, datetime(2026, 1, 1, 0, 30, tzinfo=UTC), 92, 0) == (
        "mstm_stop_before_target"
    )


def test_missing_stop_or_invalid_price_does_not_request_hedge():
    s, t, data = trade_context()
    data["atr_stop"] = np.nan
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    for rate in (92, 0, np.nan):
        assert s.custom_hedge(t.pair, t, now, rate, 0) is None


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize(
    ("entry_offset", "target_offset"),
    [(3.99, 4.0), (4.0, 8.0), (7.5, 8.0), (8.0, 12.0), (13.0, 16.0)],
)
def test_first_target_skips_levels_at_or_behind_entry(is_short, entry_offset, target_offset):
    direction = -1 if is_short else 1
    s, t, _ = trade_context(is_short, open_rate=100.0 + direction * entry_offset)
    target = 100.0 + direction * target_offset
    assert adjust(s, t, target - direction * 0.01) is None
    stake, tag = adjust(s, t, target)
    assert stake == pytest.approx(-t.stake_amount / 4)
    assert tag == "mstm_tp_1"


@pytest.mark.parametrize("is_short", [False, True])
def test_delayed_entry_skips_using_fill_price_and_signal_atr(is_short):
    s, t, data = trade_context(is_short)
    CustomDataWrapper.reset_custom_data()
    direction = -1 if is_short else 1
    # The signal closed before TP1, but the delayed entry fills beyond TP2.
    data["close"] = 100.0 + direction * 3.0
    t.open_rate = 100.0 + direction * 9.0
    t.stake_amount = t.open_rate
    t.recalc_open_trade_value()
    entry = t.orders[0]
    entry.average = t.open_rate
    entry.order_filled_date = t.open_date_utc + timedelta(minutes=15)
    s.order_filled(t.pair, t, entry, entry.order_filled_utc)
    # Signal ATR=2 gives a step of 4; the later ATR=3 must not shift this grid.
    target = 100.0 + direction * 12.0
    assert adjust(s, t, target - direction * 0.01) is None
    assert adjust(s, t, target) == (-t.stake_amount / 4, "mstm_tp_1")
    replacement = strategy(target_count=1, target_step_multiplier=9)
    replacement.dp = s.dp
    assert adjust(replacement, t, target) == (-t.stake_amount / 4, "mstm_tp_1")


@pytest.mark.parametrize(
    ("is_short", "open_rate", "target"),
    [
        (False, 10.102, 10.103),
        (True, 10.098, 10.097),
        (False, 10.1019, 10.102),
        (True, 10.0981, 10.098),
    ],
)
def test_first_target_respects_fractional_price_boundaries(is_short, open_rate, target):
    s, t, data = trade_context(is_short, open_rate=open_rate)
    CustomDataWrapper.reset_custom_data()
    data["initial_target"] = 10.1
    data["atr"] = 0.0005  # Target spacing is 0.001.
    data["atr_stop"] = 11.0 if is_short else 9.0
    s.order_filled(t.pair, t, t.orders[0], t.open_date_utc)
    direction = -1 if is_short else 1
    assert adjust(s, t, target - direction * 0.00001) is None
    assert adjust(s, t, target) == (-t.stake_amount / 4, "mstm_tp_1")


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("count", [0, 1, 3, 5])
@pytest.mark.parametrize("entry_offset", [0.0, 9.0])
def test_equal_targets_leave_one_equal_runner_and_advance_using_fill_atr(
    is_short, count, entry_offset
):
    direction = -1 if is_short else 1
    open_rate = 100.0 + direction * entry_offset
    s, t, data = trade_context(is_short, count, open_rate=open_rate)
    target = 100.0 + direction * (12.0 if entry_offset else 4.0)
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    for index in range(count):
        assert adjust(s, t, target - direction * 0.01, now) is None
        amount, tag = adjust(s, t, target, now)
        assert amount == pytest.approx(-open_rate / (count + 1))
        assert tag == f"mstm_tp_{index + 1}"
        # Repeated callbacks before a fill must offer the same target, not advance.
        assert adjust(s, t, target, now) == (amount, tag)
        order = filled_order(t, t.exit_side, tag, 1 / (count + 1), index, when=now)
        t.amount -= 1 / (count + 1)
        t.stake_amount += amount
        t.recalc_open_trade_value()
        s.order_filled(t.pair, t, order, now)
        s.order_filled(t.pair, t, order, now)  # duplicate fill notification
        target += direction * 6  # confirmed ATR 3 * step 2
        assert adjust(s, t, target, now) is None  # only one target per candle
        now += timedelta(minutes=15)
        data.loc[len(data)] = [
            now - timedelta(minutes=15),
            3.0,
            direction,
            108.0 if is_short else 92.0,
            96.0 if is_short else 104.0,
        ]
        # New strategy instance still uses the persisted trade plan.
        replacement = strategy(target_count=2, target_step_multiplier=9.0)
        replacement.dp = s.dp
        s = replacement
    assert t.amount == pytest.approx(1 / (count + 1))
    assert adjust(s, t, target, now) is None


def test_partially_filled_canceled_target_retries_only_unfilled_quantity():
    s, t, _ = trade_context()
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.1, status="canceled")
    t.amount = 0.9
    t.stake_amount = 90.0
    t.recalc_open_trade_value()
    s.order_filled(t.pair, t, order, datetime(2026, 1, 1, 0, 30, tzinfo=UTC))
    amount, tag = adjust(s, t, 104)
    assert amount == pytest.approx(-15.0)
    assert tag == "mstm_tp_1"


def test_exchange_rounding_does_not_leave_target_stuck_on_dust():
    s, t, _ = trade_context()
    t.amount_precision = 0.1
    t.precision_mode = TICK_SIZE
    t.contract_size = 1.0
    assert adjust(s, t, 104) == (-20.0, "mstm_tp_1")
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.2)
    t.amount = 0.8
    t.stake_amount = 80.0
    t.recalc_open_trade_value()
    s.order_filled(t.pair, t, order, datetime(2026, 1, 1, 0, 30, tzinfo=UTC))
    # The remaining 0.05 of the theoretical 0.25 slice cannot be traded.
    assert adjust(s, t, 104, datetime(2026, 1, 1, 0, 45, tzinfo=UTC)) is None
    assert adjust(s, t, 110, datetime(2026, 1, 1, 0, 45, tzinfo=UTC)) == (-20.0, "mstm_tp_2")


def test_canceled_target_retries_exact_exchange_ticks_without_float_truncation():
    s, t, _ = trade_context()
    t.amount_precision = 0.01
    t.precision_mode = TICK_SIZE
    t.contract_size = 1.0
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.2, status="canceled")
    t.amount = 0.8
    t.stake_amount = 80.0
    t.recalc_open_trade_value()
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    s.order_filled(t.pair, t, order, now)
    assert adjust(s, t, 104) == (-5.0, "mstm_tp_1")
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.04, number=1, status="canceled")
    t.amount = 0.76
    t.stake_amount = 76.0
    t.recalc_open_trade_value()
    s.order_filled(t.pair, t, order, now)
    assert adjust(s, t, 104) == (-1.0, "mstm_tp_1")


@pytest.mark.parametrize(("count", "expected_amount"), [(3, 0.002), (4, 0.001)])
def test_returned_stake_survives_freqtrade_conversion_to_contract_amount(count, expected_amount):
    s, t, _ = trade_context(count=count, leverage=3, initial_amount=0.008, open_rate=61432.79)
    t.amount_precision = 0.001
    t.precision_mode = TICK_SIZE
    t.contract_size = 1.0
    stake, _ = adjust(s, t, 65000)
    # This is the reverse conversion performed by Freqtrade before order creation.
    amount = abs(float(FtPrecise(stake) * FtPrecise(t.amount) / FtPrecise(t.stake_amount)))
    actual = amount_to_contract_precision(
        amount, t.amount_precision, t.precision_mode, t.contract_size
    )
    assert actual == expected_amount


def test_targets_wait_for_open_orders_and_exchange_minimum():
    s, t, _ = trade_context()
    assert adjust(s, t, 104, min_stake=30) is None
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.25)
    order.ft_is_open = True
    order.status = "open"
    assert adjust(s, t, 104) is None


def test_targets_use_exit_quote_and_cannot_take_profit_at_a_loss():
    s, t, _ = trade_context()
    assert (
        s.adjust_trade_position(
            t, datetime(2026, 1, 1, 0, 30, tzinfo=UTC), 105, 0.05, 1, 10000, 105, 103, 0.05, 0.03
        )
        is None
    )
    t.open_rate = 110
    t.recalc_open_trade_value()
    assert adjust(s, t, 104) is None


@pytest.mark.parametrize("is_short", [False, True])
def test_stop_uses_closed_candles_leverage_and_never_widens_after_fill(is_short):
    s, t, data = trade_context(is_short, leverage=3)
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    rate = 95.0 if is_short else 105.0
    stop = 108.0 if is_short else 92.0
    actual = s.custom_stoploss(t.pair, t, now, rate, 0.1, after_fill=False)
    assert actual == pytest.approx(abs(1 - stop / rate) * 3)
    s.ft_stoploss_adjust(rate, t, now, 0.1, 0)
    data.loc[1, "atr_stop"] = 115 if is_short else 85
    assert s.custom_stoploss(t.pair, t, now, rate, 0.1, after_fill=True) == actual
    # A not-yet-closed candle must not tighten the stop early.
    data.loc[2] = [
        now,
        3.0,
        -1 if is_short else 1,
        96 if is_short else 104,
        96 if is_short else 104,
    ]
    assert s.custom_stoploss(t.pair, t, now, rate, 0.1, after_fill=False) == actual


@pytest.mark.parametrize("is_short", [False, True])
@pytest.mark.parametrize("count", [0, 3])
def test_crossed_stop_and_opposite_structure_close_runner(is_short, count):
    s, t, data = trade_context(is_short, count=count)
    now = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    rate = 109.0 if is_short else 91.0
    assert s.custom_exit(t.pair, t, now, rate, -0.1) == "mstm_atr_stop"
    assert s.custom_stoploss(t.pair, t, now, rate, -0.1, after_fill=False) is None
    data.loc[1, "direction"] *= -1
    assert s.custom_exit(t.pair, t, now, 100, 0) == "mstm_opposite_structure"


@pytest.mark.parametrize("is_short", [False, True])
def test_zero_targets_do_not_require_a_valid_profit_target(is_short):
    s, t, data = trade_context(is_short, count=0)
    # With targets disabled, target spacing must not prevent entries or stop setup.
    CustomDataWrapper.reset_custom_data()
    data["initial_target"] = np.nan
    data["choch"] = -1 if is_short else 1
    data["volume"] = 100.0
    signals = s.populate_entry_trend(data.copy(), {})
    entry_column = "enter_short" if is_short else "enter_long"
    assert signals[entry_column].tolist() == [1, 1]
    t.enter_tag = signals["enter_tag"].iloc[0]
    s.order_filled(t.pair, t, t.orders[0], t.open_date_utc)
    now = t.open_date_utc + timedelta(minutes=15)
    rate = 95.0 if is_short else 105.0
    assert adjust(s, t, rate, now) is None
    assert s.custom_exit(t.pair, t, now, rate, 0.05) is None
    assert s.custom_stoploss(t.pair, t, now, rate, 0.05, after_fill=False) == pytest.approx(
        abs(1 - (108.0 if is_short else 92.0) / rate)
    )
    assert t.get_all_custom_data() == []
    # The original zero-target mode survives a parameter change and a restart.
    replacement = strategy(target_count=5)
    replacement.dp = s.dp
    assert adjust(replacement, t, rate, now) is None
    assert t.get_all_custom_data() == []


def test_stop_updates_use_trade_stop_without_rewriting_target_data():
    s, t, _ = trade_context()
    before = dict(t.get_custom_data(s._STATE_KEY))
    s.ft_stoploss_adjust(100, t, t.open_date_utc + timedelta(minutes=15), 0, 0)
    assert t.stop_loss == pytest.approx(92)
    assert t.get_custom_data(s._STATE_KEY) == before


@pytest.mark.parametrize("is_short", [False, True])
def test_legacy_plan_migration_retains_crossed_tighter_stop(is_short):
    s, t, _ = trade_context(is_short)
    legacy = {
        "target_count": 3,
        "step_multiplier": 2.0,
        "completed": 0,
        "next_target": 96.0 if is_short else 104.0,
        "atr": 2.0,
        "initial_amount": 1.0,
        "stop": 95.0 if is_short else 105.0,
        "last_target_candle": None,
    }
    t.set_custom_data(s._STATE_KEY, legacy)
    t.adjust_stop_loss(100, s.stoploss, initial=True)
    now = t.open_date_utc + timedelta(minutes=15)
    rate = 96.0 if is_short else 104.0
    assert adjust(s, t, rate, now) is None
    assert s.custom_exit(t.pair, t, now, rate, 0.04) == "mstm_atr_stop"
    assert t.stop_loss == pytest.approx(95.0 if is_short else 105.0)
    assert "stop" not in t.get_custom_data(s._STATE_KEY)


def test_target_fill_without_candle_data_uses_last_saved_atr():
    s, t, data = trade_context()
    now = t.open_date_utc + timedelta(minutes=15)
    order = filled_order(t, t.exit_side, "mstm_tp_1", 0.25, when=now)
    t.amount, t.stake_amount = 0.75, 75
    t.recalc_open_trade_value()
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data.iloc[:0], None))
    s.order_filled(t.pair, t, order, now)
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data, None))
    assert adjust(s, t, 108, now) is None
    assert adjust(s, t, 108, now + timedelta(minutes=15)) == (-25.0, "mstm_tp_2")


def test_target_cooldown_uses_last_constituent_fill_after_restart():
    s, t, _ = trade_context()
    now = t.open_date_utc + timedelta(minutes=15)
    first = filled_order(
        t, t.exit_side, "mstm_tp_1", 0.1, status="canceled", when=now - timedelta(minutes=15)
    )
    s.order_filled(t.pair, t, first, now - timedelta(minutes=15))
    last = filled_order(t, t.exit_side, "mstm_tp_1", 0.15, number=1, when=now)
    t.amount, t.stake_amount = 0.75, 75
    t.recalc_open_trade_value()
    s.order_filled(t.pair, t, last, now)
    replacement = strategy(target_count=1, target_step_multiplier=9)
    replacement.dp = s.dp
    assert adjust(replacement, t, 110, now) is None
    assert adjust(replacement, t, 110, now + timedelta(minutes=15)) == (-25.0, "mstm_tp_2")


def test_legacy_zero_plan_remains_disabled_after_settings_change():
    s, t, _ = trade_context()
    legacy = {
        "target_count": 0,
        "step_multiplier": 2.0,
        "completed": 0,
        "next_target": None,
        "atr": 2.0,
        "initial_amount": 1.0,
        "stop": 90.0,
        "last_target_candle": None,
    }
    t.set_custom_data(s._STATE_KEY, legacy)
    assert adjust(s, t, 110) is None
    now = t.open_date_utc + timedelta(minutes=15)
    assert s.custom_exit(t.pair, t, now, 91, -0.09) == "mstm_atr_stop"


def test_legacy_stop_migration_does_not_loosen_a_tighter_native_stop():
    s, t, _ = trade_context()
    legacy = dict(
        t.get_custom_data(s._STATE_KEY), initial_amount=1.0, stop=95.0, last_target_candle=None
    )
    t.set_custom_data(s._STATE_KEY, legacy)
    t.adjust_stop_loss(100, -0.02)
    assert adjust(s, t, 104) == (-25.0, "mstm_tp_1")
    assert t.stop_loss == pytest.approx(98.0)


@pytest.mark.parametrize("is_short", [False, True])
def test_migrated_stop_survives_native_partial_fill_recalculation(is_short):
    s, t, _ = trade_context(is_short)
    stop = 110.0 if is_short else 90.0
    legacy = dict(
        t.get_custom_data(s._STATE_KEY), initial_amount=1.0, stop=stop, last_target_candle=None
    )
    t.set_custom_data(s._STATE_KEY, legacy)
    t.adjust_stop_loss(100, s.stoploss, initial=True)
    s.order_filled(t.pair, t, t.orders[0], t.open_date_utc)
    assert t.stop_loss == pytest.approx(stop)
    filled_order(t, t.exit_side, "mstm_tp_1", 0.25)
    # Live Freqtrade recalculates the position BEFORE notifying order_filled.
    t.recalc_trade_from_orders()
    assert t.stop_loss == pytest.approx(stop)


def test_missing_target_plan_exits_when_signal_candle_cannot_be_recovered():
    s, t, data = trade_context()
    CustomDataWrapper.reset_custom_data()
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data.iloc[1:], None))
    now = t.open_date_utc + timedelta(minutes=15)
    s.order_filled(t.pair, t, t.orders[0], now)
    assert s.custom_exit(t.pair, t, now, 105, 0.05) == "mstm_missing_entry_data"


def test_zero_mode_does_not_need_an_entry_candle_for_target_initialization():
    s, t, data = trade_context(count=0)
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data.iloc[1:], None))
    now = t.open_date_utc + timedelta(minutes=15)
    assert s.custom_exit(t.pair, t, now, 105, 0.05) is None
    assert t.get_all_custom_data() == []


@pytest.mark.parametrize("count", [0, 3])
def test_installed_stop_remains_effective_after_entry_history_is_gone(count):
    s, t, data = trade_context(count=count)
    now = t.open_date_utc + timedelta(minutes=15)
    s.ft_stoploss_adjust(100, t, now, 0, 0)
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data.iloc[:0], None))
    assert s.custom_exit(t.pair, t, now, 100, 0) is None
    assert s.custom_exit(t.pair, t, now, 91, -0.09) == "mstm_atr_stop"


def test_emergency_fallback_alone_does_not_count_as_initialized_atr_stop():
    s, t, data = trade_context(count=0)
    t.adjust_stop_loss(100, s.stoploss, initial=True)
    s.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (data.iloc[:0], None))
    now = t.open_date_utc + timedelta(minutes=15)
    assert s.custom_exit(t.pair, t, now, 100, 0) == "mstm_missing_entry_data"


def test_empty_data_is_safe_and_preserves_columns():
    s = strategy()
    result = s.populate_indicators(candles([]), {})
    assert result.empty
    assert {"atr", "atr_stop", "current_target"} <= set(result.columns)


@pytest.mark.parametrize("is_short", [False, True])
def test_freqtrade_installs_absolute_atr_stop_on_entry_and_partial_fill(is_short):
    s, t, _ = trade_context(is_short)
    s._ft_stop_uses_after_fill = True
    s.ft_stoploss_adjust(100, t, t.open_date_utc, 0, 0, after_fill=True)
    assert t.stop_loss == pytest.approx(110 if is_short else 90)
    s.ft_stoploss_adjust(100, t, t.open_date_utc + timedelta(minutes=15), 0, 0, after_fill=True)
    assert t.stop_loss == pytest.approx(108 if is_short else 92)


def test_freqtrade_resolver_loads_futures_strategy(default_conf):
    for override in ("minimal_roi", "timeframe", "stoploss"):
        default_conf.pop(override, None)
    default_conf.update(
        {
            "strategy": "MarketStructureTrendMatrixStrategy",
            "strategy_path": str(Path(__file__).parents[2] / "user_data/strategies"),
            "trading_mode": "futures",
            "margin_mode": "isolated",
        }
    )
    loaded = StrategyResolver.load_strategy(default_conf)
    frame = loaded.advise_indicators(candles([10.0] * 600), {"pair": "ETH/USDT:USDT"})
    result = loaded.ft_advise_signals(frame, {"pair": "ETH/USDT:USDT"})
    assert loaded.can_short
    assert len(result) == 600
    assert (result["enter_long"] == 0).all()
    assert (result["enter_short"] == 0).all()
