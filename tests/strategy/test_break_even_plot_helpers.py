from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from ccxt import DECIMAL_PLACES
from pandas import DataFrame

from freqtrade.ft_types.plot_annotation_type import AnnotationTypeTA
from user_data.strategies.adx_vw_dca_strategy import ADXVWDCAStrategy
from user_data.strategies.break_even_plot_helpers import FeeAwareBreakEvenPlotMixin
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy
from user_data.strategies.rsi_ml_dca_strategy import RSIMLDCAStrategy
from user_data.strategies.rtb_dca_strategy import RTBDCAStrategy


PAIR = "ETH/USDT:USDT"
START = datetime(2026, 1, 1, tzinfo=UTC)
RUNTIME_CONFIGS = [
    "config.json",
    "config_bybit_demo.json",
    "config_gate_baryga.json",
    "config_okx_baryga.json",
    "config_spot.json",
]


class Subject(FeeAwareBreakEvenPlotMixin):
    pass


@pytest.mark.parametrize("config_name", RUNTIME_CONFIGS)
def test_runtime_configs_do_not_enable_fiat_conversion(config_name: str) -> None:
    config_path = Path(__file__).parents[2] / "user_data" / config_name
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert "fiat_display_currency" not in config


def _trade(*, is_short: bool = False, rate: float = 100.0) -> SimpleNamespace:
    custom_data: dict[str, object] = {}
    close_rate_calls: list[float] = []
    trade = SimpleNamespace(
        pair=PAIR,
        contract_size=1.0,
        safe_base_currency="ETH",
        safe_quote_currency="USDT",
        is_open=True,
        is_short=is_short,
        entry_side="sell" if is_short else "buy",
        id=1,
        nr_of_successful_entries=2,
        open_date_utc=START,
        orders=[],
        get_custom_data=lambda key, default=None: custom_data.get(key, default),
        set_custom_data=lambda key, value: custom_data.__setitem__(key, value),
        _custom_data=custom_data,
        _close_rate_calls=close_rate_calls,
    )
    trade.calc_close_rate_for_roi = lambda roi: close_rate_calls.append(roi) or rate
    return trade


def _entry_order(
    trade: SimpleNamespace,
    order_id: object,
    fill_time: datetime,
) -> SimpleNamespace:
    order = SimpleNamespace(
        order_id=order_id,
        ft_order_side=trade.entry_side,
        order_filled_utc=fill_time,
        safe_filled=1.0,
        safe_price=1.0,
        stake_amount_filled=1.0,
    )
    trade.orders.append(order)
    return order


def _plot(strategy: Subject, start: datetime, end: datetime) -> list[dict]:
    annotations = strategy.plot_annotations(PAIR, start, end, DataFrame())
    for annotation in annotations:
        AnnotationTypeTA.validate_python(annotation, extra="forbid")
    return annotations


@pytest.mark.parametrize(
    ("is_short", "rate"),
    [(False, 100.2001), (True, 99.7999)],
)
def test_recorder_persists_exact_fee_aware_entry_snapshot(is_short: bool, rate: float) -> None:
    strategy = Subject()
    trade = _trade(is_short=is_short, rate=rate)
    fill_time = START + timedelta(minutes=1)
    order = _entry_order(trade, "entry-1", fill_time)

    strategy._record_break_even_fill(trade, order, START)

    assert trade._custom_data == {
        "dca_break_even_steps_v1": [
            {
                "order_id": "entry-1",
                "filled_at": "2026-01-01T00:01:00+00:00",
                "rate": rate,
            }
        ]
    }
    assert trade._close_rate_calls == [0.0]


def test_recorder_deduplicates_by_order_id() -> None:
    strategy = Subject()
    trade = _trade(rate=101.0)
    order = _entry_order(trade, "entry-1", START + timedelta(minutes=1))

    strategy._record_break_even_fill(trade, order, START)
    trade.calc_close_rate_for_roi = lambda roi: 102.0
    strategy._record_break_even_fill(trade, order, START + timedelta(minutes=2))

    assert trade._custom_data["dca_break_even_steps_v1"] == [
        {
            "order_id": "entry-1",
            "filled_at": "2026-01-01T00:01:00+00:00",
            "rate": 101.0,
        }
    ]
    assert trade._close_rate_calls == [0.0]


@pytest.mark.parametrize(
    "order",
    [
        SimpleNamespace(order_id="exit", ft_order_side="sell", order_filled_utc=START),
        SimpleNamespace(order_id=None, ft_order_side="buy", order_filled_utc=START),
        SimpleNamespace(order_id="", ft_order_side="buy", order_filled_utc=START),
    ],
)
def test_recorder_ignores_invalid_or_non_entry_fills(order: SimpleNamespace) -> None:
    strategy = Subject()
    trade = _trade()

    strategy._record_break_even_fill(trade, order, START)

    assert trade._custom_data == {}


def test_recorder_uses_utc_normalized_callback_time_when_fill_time_is_missing() -> None:
    strategy = Subject()
    trade = _trade()
    order = _entry_order(trade, "entry-1", START)
    order.order_filled_utc = None
    callback_time = datetime(2026, 1, 1, 2, tzinfo=timezone(timedelta(hours=2)))

    strategy._record_break_even_fill(trade, order, callback_time)

    assert trade._custom_data["dca_break_even_steps_v1"][0]["filled_at"] == (
        "2026-01-01T00:00:00+00:00"
    )


@pytest.mark.parametrize("filled", [None, "invalid", float("nan"), 0.0, -1.0])
def test_recorder_ignores_missing_or_non_positive_entry_quantity(filled: object) -> None:
    strategy = Subject()
    trade = _trade()
    order = _entry_order(trade, "entry-1", START)
    order.safe_filled = filled

    strategy._record_break_even_fill(trade, order, START)

    assert trade._custom_data == {}


def test_recorder_normalizes_numeric_order_id_to_a_non_empty_string() -> None:
    strategy = Subject()
    trade = _trade()
    order = _entry_order(trade, 7, START)

    strategy._record_break_even_fill(trade, order, START)

    assert trade._custom_data["dca_break_even_steps_v1"][0]["order_id"] == "7"


@pytest.mark.parametrize("rate", [0.0, -0.01])
def test_recorder_rejects_non_positive_break_even_rates(rate: float) -> None:
    strategy = Subject()
    trade = _trade(rate=rate)
    order = _entry_order(trade, "entry-1", START)

    strategy._record_break_even_fill(trade, order, START)

    assert trade._custom_data == {}


def test_plot_renders_base_and_safety_order_staircase(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=98.0)
    trade._custom_data["dca_break_even_steps_v1"] = [
        {"order_id": "bo", "filled_at": (START + timedelta(minutes=1)).isoformat(), "rate": 101.0},
        {"order_id": "so1", "filled_at": (START + timedelta(minutes=3)).isoformat(), "rate": 99.0},
    ]
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=5))

    assert annotations == [
        {
            "type": "line",
            "start": START + timedelta(minutes=1),
            "end": START + timedelta(minutes=3),
            "y_start": 101.0,
            "y_end": 101.0,
            "color": "#FFD700",
            "width": 2,
            "z_level": 10,
            "line_style": "solid",
        },
        {
            "type": "line",
            "start": START + timedelta(minutes=3),
            "end": START + timedelta(minutes=5),
            "y_start": 98.0,
            "y_end": 98.0,
            "color": "#FFD700",
            "width": 2,
            "z_level": 10,
            "line_style": "solid",
            "label": "BE: 98",
        },
    ]


def test_plot_recalculates_the_live_final_tail(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=103.0)
    trade._custom_data["dca_break_even_steps_v1"] = [
        {"order_id": "bo", "filled_at": (START + timedelta(minutes=1)).isoformat(), "rate": 101.0},
        {"order_id": "so1", "filled_at": (START + timedelta(minutes=3)).isoformat(), "rate": 99.0},
    ]
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START + timedelta(minutes=2), START + timedelta(minutes=4))

    assert [(line["start"], line["end"], line["y_start"]) for line in annotations] == [
        (START + timedelta(minutes=2), START + timedelta(minutes=3), 101.0),
        (START + timedelta(minutes=3), START + timedelta(minutes=4), 103.0),
    ]
    assert "label" not in annotations[0]
    assert annotations[1]["label"] == "BE: 103"


@pytest.mark.parametrize(
    ("is_short", "expected_label"),
    [
        (False, "BE: 98.124"),
        (True, "BE: 98.123"),
    ],
)
def test_plot_formats_current_label_to_the_trades_profitable_price_tick(
    monkeypatch, is_short: bool, expected_label: str
) -> None:
    strategy = Subject()
    trade = _trade(is_short=is_short, rate=98.12356)
    trade.price_precision = 3
    trade.precision_mode_price = DECIMAL_PLACES
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=1))

    assert annotations[0]["y_start"] == 98.12356
    assert annotations[0]["label"] == expected_label


def test_plot_clips_segments_to_the_requested_window(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=98.0)
    trade._custom_data["dca_break_even_steps_v1"] = [
        {"order_id": "bo", "filled_at": (START + timedelta(minutes=1)).isoformat(), "rate": 101.0},
        {"order_id": "so1", "filled_at": (START + timedelta(minutes=3)).isoformat(), "rate": 99.0},
    ]
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START + timedelta(minutes=2), START + timedelta(minutes=3))

    assert annotations == [
        {
            "type": "line",
            "start": START + timedelta(minutes=2),
            "end": START + timedelta(minutes=3),
            "y_start": 101.0,
            "y_end": 101.0,
            "color": "#FFD700",
            "width": 2,
            "z_level": 10,
            "line_style": "solid",
        }
    ]


def test_plot_ignores_malformed_snapshots(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=98.0)
    trade._custom_data["dca_break_even_steps_v1"] = [
        {"order_id": "bad-time", "filled_at": "not-a-date", "rate": 100.0},
        {"order_id": "bad-rate", "filled_at": START.isoformat(), "rate": "nan"},
        {"order_id": [], "filled_at": START.isoformat(), "rate": 100.0},
        {"order_id": 1, "filled_at": START.isoformat(), "rate": 100.0},
        {"order_id": "zero", "filled_at": START.isoformat(), "rate": 0.0},
        {"order_id": "negative", "filled_at": START.isoformat(), "rate": -1.0},
        {"order_id": "bo", "filled_at": (START + timedelta(minutes=1)).isoformat(), "rate": 101.0},
    ]
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=2))

    assert len(annotations) == 1
    assert annotations[0]["start"] == START + timedelta(minutes=1)
    assert annotations[0]["y_start"] == 98.0


def test_plot_falls_back_to_current_rate_from_latest_entry_fill(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=97.5)
    _entry_order(trade, "bo", START + timedelta(minutes=1))
    _entry_order(trade, "so1", START + timedelta(minutes=4))
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=5))

    assert annotations[0]["start"] == START + timedelta(minutes=4)
    assert annotations[0]["end"] == START + timedelta(minutes=5)
    assert annotations[0]["y_start"] == 97.5
    assert annotations[0]["label"] == "BE: 97.5"


def test_plot_falls_back_to_trade_open_time_without_filled_entries(monkeypatch) -> None:
    strategy = Subject()
    trade = _trade(rate=97.5)
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=5))

    assert annotations[0]["start"] == START
    assert annotations[0]["y_start"] == 97.5


def test_plot_selects_the_newest_open_trade_for_the_pair(monkeypatch) -> None:
    strategy = Subject()
    older_trade = _trade(rate=101.0)
    newest_trade = _trade(rate=97.5)
    newest_trade.open_date_utc = START + timedelta(minutes=1)
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        lambda: [older_trade, newest_trade],
    )

    annotations = _plot(strategy, START, START + timedelta(minutes=5))

    assert annotations[0]["start"] == newest_trade.open_date_utc
    assert annotations[0]["y_start"] == 97.5


def test_plot_returns_no_annotations_without_an_open_trade(monkeypatch) -> None:
    monkeypatch.setattr(
        "user_data.strategies.break_even_plot_helpers.Trade.get_open_trades",
        list,
    )

    assert _plot(Subject(), START, START + timedelta(minutes=1)) == []


@pytest.mark.parametrize(
    "strategy_class",
    [ADXVWDCAStrategy, RSIMLDCAStrategy, RTBDCAStrategy, QFLDCAStrategy],
)
def test_strategy_hooks_record_entries_before_dataframe_early_returns(
    monkeypatch, strategy_class
) -> None:
    strategy = strategy_class({"stake_currency": "USDT"})
    trade = _trade()
    order = _entry_order(trade, "entry-1", START + timedelta(minutes=1))
    recorded: list[tuple[object, object]] = []
    monkeypatch.setattr(
        strategy,
        "_record_break_even_fill",
        lambda callback_trade, callback_order, callback_time: recorded.append(
            (callback_trade, callback_order)
        ),
    )
    strategy.dp = None

    strategy.order_filled(PAIR, trade, order, START)

    assert recorded == [(trade, order)]


@pytest.mark.parametrize(
    "strategy_class",
    [ADXVWDCAStrategy, RSIMLDCAStrategy, RTBDCAStrategy, QFLDCAStrategy],
)
def test_strategy_hooks_do_not_record_exit_fills(monkeypatch, strategy_class) -> None:
    strategy = strategy_class({"stake_currency": "USDT"})
    trade = _trade()
    exit_order = SimpleNamespace(
        order_id="exit-1",
        ft_order_side="sell",
        order_filled_utc=START + timedelta(minutes=1),
    )
    monkeypatch.setattr(
        strategy,
        "_record_break_even_fill",
        lambda *args: pytest.fail("exit fills must not be recorded"),
    )

    strategy.order_filled(PAIR, trade, exit_order, START)
