import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from user_data.strategies.adx_vw_dca_strategy import ADXVWDCAStrategy
from user_data.strategies.dca_strategy_helpers import select_dca_leverage
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy
from user_data.strategies.rsi_ml_dca_strategy import RSIMLDCAStrategy
from user_data.strategies.rtb_dca_strategy import RTBDCAStrategy


LTC_TIERS = [
    {"minNotional": 0.0, "maxNotional": 500.0, "maxLeverage": 50.0},
    {"minNotional": 500.1, "maxNotional": 2000.0, "maxLeverage": 40.0},
    {"minNotional": 2000.1, "maxNotional": 6000.0, "maxLeverage": 20.0},
    {"minNotional": 6000.1, "maxNotional": 12000.0, "maxLeverage": 18.18},
    {"minNotional": 12000.1, "maxNotional": 18000.0, "maxLeverage": 16.66},
    {"minNotional": 18000.1, "maxNotional": 24000.0, "maxLeverage": 15.38},
    {"minNotional": 24000.1, "maxNotional": 30000.0, "maxLeverage": 14.28},
]


def test_select_dca_leverage_accounts_for_equal_hedge_and_buffer() -> None:
    selection = select_dca_leverage(
        tiers=LTC_TIERS,
        dca_budget=500.0,
        max_leverage=50.0,
        buffer_ratio=0.05,
    )

    assert selection.leverage == 16.0
    assert selection.tier_index == 5
    assert selection.gross_collateral == 1000.0
    assert selection.gross_notional == 16000.0
    assert selection.usable_max_notional == 17100.0
    assert selection.fits_tier


def test_select_dca_leverage_handles_fewer_exchange_tiers() -> None:
    selection = select_dca_leverage(
        tiers=[
            {"minNotional": 0.0, "maxNotional": 10000.0, "maxLeverage": 20.0},
            {"minNotional": 10000.0, "maxNotional": 100000.0, "maxLeverage": 10.0},
        ],
        dca_budget=400.0,
        max_leverage=20.0,
        buffer_ratio=0.05,
    )

    assert selection.leverage == 11.0
    assert selection.tier_index == 1
    assert selection.gross_notional == 8800.0


def test_select_dca_leverage_uses_buffered_boundary() -> None:
    selection = select_dca_leverage(
        tiers=[{"minNotional": 0.0, "maxNotional": 1000.0, "maxLeverage": 50.0}],
        dca_budget=10.0,
        max_leverage=50.0,
        buffer_ratio=0.05,
    )

    assert selection.leverage == 47.0
    assert selection.gross_notional == 940.0
    assert selection.usable_max_notional == 950.0


def test_select_dca_leverage_converts_contract_tiers_to_quote_notional() -> None:
    selection = select_dca_leverage(
        tiers=[{"minNotional": 0.0, "maxNotional": 15000.0, "maxLeverage": 50.0}],
        dca_budget=2000.0,
        max_leverage=50.0,
        buffer_ratio=0.05,
        tier_notional_multiplier=8.68,
    )

    assert selection.leverage == 30.0
    assert selection.gross_notional == 120000.0
    assert selection.usable_max_tier_units == 14250.0
    assert selection.usable_max_notional == 123690.0


def test_select_dca_leverage_accounts_for_minimum_base_order() -> None:
    selection = select_dca_leverage(
        tiers=[
            {"minNotional": 0.0, "maxNotional": 1000.0, "maxLeverage": 50.0},
            {"minNotional": 1000.0, "maxNotional": 10000.0, "maxLeverage": 10.0},
        ],
        dca_budget=10.0,
        max_leverage=50.0,
        buffer_ratio=0.05,
        minimum_dca_notional=600.0,
    )

    assert selection.leverage == 10.0
    assert selection.tier_index == 2
    assert selection.gross_collateral == 120.0
    assert selection.gross_notional == 1200.0


def test_select_dca_leverage_respects_exchange_cap_and_unbounded_tier() -> None:
    selection = select_dca_leverage(
        tiers=[{"minNotional": 0.0, "maxNotional": None, "maxLeverage": 75.0}],
        dca_budget=1000.0,
        max_leverage=20.8,
        buffer_ratio=0.05,
    )

    assert selection.leverage == 20.0
    assert selection.usable_max_notional is None


def test_select_dca_leverage_falls_back_to_one_without_tiers() -> None:
    selection = select_dca_leverage(
        tiers=[],
        dca_budget=500.0,
        max_leverage=50.0,
        buffer_ratio=0.05,
    )

    assert selection.leverage == 1.0
    assert selection.tier_index is None
    assert not selection.fits_tier


@pytest.mark.parametrize(
    "strategy_class",
    [RSIMLDCAStrategy, ADXVWDCAStrategy, RTBDCAStrategy, QFLDCAStrategy],
)
def test_dca_strategies_use_exposure_aware_leverage(strategy_class) -> None:
    pair = "LTC/USDT:USDT"
    strategy = strategy_class({"stake_currency": "USDT"})
    strategy.leverage_buffer_pct.value = 5.0
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(_leverage_tiers={pair: LTC_TIERS})
    )

    leverage = strategy.leverage(
        pair=pair,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=70.0,
        proposed_leverage=1.0,
        max_leverage=50.0,
        entry_tag=None,
        side="long",
        proposed_stake=500.0,
    )

    assert leverage == 16.0


@pytest.mark.parametrize(
    "strategy_class",
    [RSIMLDCAStrategy, ADXVWDCAStrategy, RTBDCAStrategy, QFLDCAStrategy],
)
def test_dca_strategies_include_exchange_minimum_in_leverage_budget(strategy_class) -> None:
    pair = "LTC/USDT:USDT"
    strategy = strategy_class({"stake_currency": "USDT"})
    strategy.leverage_buffer_pct.value = 5.0
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(
            _leverage_tiers={pair: LTC_TIERS},
            get_min_pair_stake_amount=lambda *args, **kwargs: 100.0,
        )
    )

    leverage = strategy.leverage(
        pair=pair,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=70.0,
        proposed_leverage=1.0,
        max_leverage=50.0,
        entry_tag=None,
        side="long",
        proposed_stake=10.0,
    )

    assert leverage == 20.0


def test_okx_contract_tier_capacity_is_converted_and_logged(caplog) -> None:
    pair = "ZEC/USDT:USDT"
    strategy = RSIMLDCAStrategy({"stake_currency": "USDT"})
    strategy.leverage_buffer_pct.value = 5.0
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(
            id="okx",
            _leverage_tiers={
                pair: [
                    {
                        "minNotional": 0.0,
                        "maxNotional": 15000.0,
                        "maxLeverage": 50.0,
                    }
                ]
            },
            get_contract_size=lambda _: 0.01,
        )
    )

    with caplog.at_level(
        logging.INFO,
        logger="user_data.strategies.rsi_ml_dca_strategy",
    ):
        leverage = strategy.leverage(
            pair=pair,
            current_time=datetime(2026, 8, 30, 15, 12, tzinfo=UTC),
            current_rate=868.0,
            proposed_leverage=1.0,
            max_leverage=50.0,
            entry_tag=None,
            side="short",
            proposed_stake=109.14471,
        )

    assert leverage == 50.0
    assert [record.getMessage() for record in caplog.records] == [
        (
            "Leverage selected | ZEC/USDT:USDT short | 2026-08-30 15:12 | 50x | "
            "DCA 109.145 USDT + hedge 109.145 USDT | "
            "gross notional 10914.471 USDT | tier 1/1, usable "
            "14250 contracts / 142.5 ZEC / 123690 USDT (5% buffer) | "
            "utilization 8.82%"
        )
    ]
