import logging
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pandas import DataFrame

from freqtrade.persistence import Order, Trade
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy


def _strategy() -> QFLDCAStrategy:
    strategy = QFLDCAStrategy({"stake_currency": "USDT"})
    strategy.vol_scale.value = 2.0
    strategy.so_price_deviation_pct.value = 0.5
    strategy.so_price_step_scale.value = 2.0
    strategy.max_safe_orders.value = 3
    strategy.tp_pct.value = 1.0
    strategy.tp_from_base_order.value = False
    strategy.base_order_rate_buffer_pct.value = 0.05
    return strategy


def _filled_entry(trade: Trade, price: float, stake: float, index: int) -> Order:
    amount = stake / price
    order = Order.parse_from_ccxt_object(
        {
            "id": f"entry-{index}",
            "symbol": trade.pair,
            "status": "closed",
            "side": trade.entry_side,
            "type": "limit",
            "price": price,
            "average": price,
            "amount": amount,
            "filled": amount,
            "cost": stake,
            "remaining": 0.0,
        },
        trade.pair,
        trade.entry_side,
    )
    order.ft_order_tag = (
        QFLDCAStrategy.LONG_ENTRY_TAG
        if index == 0
        else QFLDCAStrategy._safety_order_tag(QFLDCAStrategy.LONG_TAG_PREFIX, index)
    )
    return order


def _trade(entry_prices: tuple[float, ...] = (100.0,), is_short: bool = False) -> Trade:
    stakes = tuple(10.0 * 2**index for index in range(len(entry_prices)))
    amount = sum(stake / price for stake, price in zip(stakes, entry_prices, strict=True))
    trade = Trade(
        id=91,
        pair="ETH/USDT:USDT",
        stake_amount=sum(stakes),
        amount=amount,
        fee_open=0.001,
        fee_close=0.001,
        is_open=True,
        open_date=datetime(2026, 1, 1, tzinfo=UTC),
        open_rate=sum(stakes) / amount,
        exchange="binance",
        strategy="QFLDCAStrategy",
        timeframe=1,
        leverage=1.0,
        is_short=is_short,
    )
    for index, (price, stake) in enumerate(zip(entry_prices, stakes, strict=True)):
        trade.orders.append(_filled_entry(trade, price, stake, index))
    return trade


def _custom_data(monkeypatch, trade: Trade) -> dict[str, object]:
    data: dict[str, object] = {}
    monkeypatch.setattr(trade, "get_custom_data", lambda key, default=None: data.get(key, default))
    monkeypatch.setattr(
        trade,
        "set_custom_data",
        lambda key, value: data.__setitem__(key, value),
    )
    return data


def _add_open_order(trade: Trade, side: str, tag: str, price: float) -> None:
    order = Order.parse_from_ccxt_object(
        {
            "id": f"open-{tag}",
            "symbol": trade.pair,
            "status": "open",
            "side": side,
            "type": "limit",
            "price": price,
            "amount": trade.amount,
            "filled": 0.0,
            "cost": 0.0,
            "remaining": trade.amount,
        },
        trade.pair,
        side,
    )
    order.ft_order_tag = tag
    trade.orders.append(order)


def _last_candle() -> DataFrame:
    return DataFrame(
        {
            "date": [datetime(2026, 1, 1, 1, 0, tzinfo=UTC)],
            "open": [100.0],
            "high": [100.2],
            "low": [99.8],
            "close": [100.0],
            "volume": [100.0],
        }
    )


def test_custom_stake_reserves_complete_qfl_ladder_budget() -> None:
    strategy = _strategy()

    stake = strategy.custom_stake_amount(
        pair="ETH/USDT:USDT",
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        proposed_stake=150.0,
        min_stake=None,
        max_stake=150.0,
        leverage=10.0,
        entry_tag="📈-📍",
        side="long",
    )

    assert strategy._max_dca_multiplier() == pytest.approx(15.0)
    assert stake == pytest.approx(10.0)


def test_order_filled_uses_shared_initial_and_so_log_labels(monkeypatch, caplog) -> None:
    strategy = _strategy()
    initial_trade = _trade()
    safety_trade = _trade((100.0, 95.0))
    _custom_data(monkeypatch, initial_trade)
    _custom_data(monkeypatch, safety_trade)
    current_time = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    caplog.set_level(logging.INFO, logger="user_data.strategies.qfl_dca_strategy")

    strategy.order_filled(
        initial_trade.pair,
        initial_trade,
        initial_trade.orders[0],
        current_time,
    )
    strategy.order_filled(
        safety_trade.pair,
        safety_trade,
        safety_trade.orders[-1],
        current_time,
    )

    messages = [record.getMessage() for record in caplog.records]
    assert messages[0].startswith("Initial order filled | #91 ETH/USDT:USDT long")
    assert "📈-📍" not in messages[0]
    assert messages[1].startswith("SO filled | #91 ETH/USDT:USDT long")


def test_next_safety_order_is_not_requested_before_its_limit_is_reached(monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade()
    data = _custom_data(monkeypatch, trade)
    dataframe = _last_candle()
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (dataframe, None),
    )

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=dataframe["date"].iat[-1],
        current_rate=100.0,
        current_profit=0.0,
        min_stake=None,
        max_stake=1000.0,
        current_entry_rate=100.0,
        current_exit_rate=100.0,
        current_entry_profit=0.0,
        current_exit_profit=0.0,
    )

    assert adjustment is None
    assert "qfl_next_safety_order_price" not in data
    assert "qfl_next_safety_order_tag" not in data


def test_next_safety_order_uses_exact_limit_after_close_reaches_level(
    monkeypatch,
    caplog,
) -> None:
    strategy = _strategy()
    trade = _trade()
    data = _custom_data(monkeypatch, trade)
    dataframe = _last_candle()
    dataframe.loc[:, "close"] = 99.4
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (dataframe, None),
    )
    caplog.set_level(logging.INFO, logger="user_data.strategies.qfl_dca_strategy")

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=dataframe["date"].iat[-1],
        current_rate=99.4,
        current_profit=-0.006,
        min_stake=None,
        max_stake=1000.0,
        current_entry_rate=99.4,
        current_exit_rate=99.4,
        current_entry_profit=-0.006,
        current_exit_profit=-0.006,
    )

    assert adjustment is not None
    assert adjustment[0] == pytest.approx(20.0)
    assert adjustment[1] == "📈-🛡️⓵"
    assert data["qfl_next_safety_order_price"] == pytest.approx(99.5)
    assert next(record.getMessage() for record in caplog.records).startswith(
        "SO requested | #91 ETH/USDT:USDT long"
    )


def test_resting_take_profit_does_not_block_next_safety_order(monkeypatch) -> None:
    strategy = _strategy()
    trade = _trade()
    _custom_data(monkeypatch, trade)
    _add_open_order(trade, trade.exit_side, "🎯", 101.0)
    dataframe = _last_candle()
    dataframe.loc[:, "close"] = 99.4
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (dataframe, None),
    )

    adjustment = strategy.adjust_trade_position(
        trade=trade,
        current_time=dataframe["date"].iat[-1],
        current_rate=100.0,
        current_profit=0.0,
        min_stake=None,
        max_stake=1000.0,
        current_entry_rate=100.0,
        current_exit_rate=100.0,
        current_entry_profit=0.0,
        current_exit_profit=0.0,
    )

    assert adjustment is not None
    assert adjustment[1] == "📈-🛡️⓵"


def test_take_profit_limit_is_requested_without_waiting_for_price_touch() -> None:
    strategy = _strategy()
    trade = _trade()
    dataframe = _last_candle()
    strategy.dp = SimpleNamespace(
        get_analyzed_dataframe=lambda pair, timeframe: (dataframe, None),
    )

    assert (
        strategy.custom_exit(
            pair=trade.pair,
            trade=trade,
            current_time=dataframe["date"].iat[-1],
            current_rate=100.0,
            current_profit=0.0,
        )
        == strategy.TAKE_PROFIT_EXIT_TAG
    )


def test_open_safety_order_is_not_replaced_by_take_profit() -> None:
    strategy = _strategy()
    trade = _trade()
    _add_open_order(trade, trade.entry_side, "📈-🛡️⓵", 99.5)

    assert (
        strategy.custom_exit(
            pair=trade.pair,
            trade=trade,
            current_time=datetime(2026, 1, 1, tzinfo=UTC),
            current_rate=100.0,
            current_profit=0.0,
        )
        is None
    )


def test_take_profit_price_is_fee_aware() -> None:
    strategy = _strategy()
    trade = _trade()

    tp_rate = strategy.custom_exit_price(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=100.0,
        current_profit=0.0,
        exit_tag=strategy.TAKE_PROFIT_EXIT_TAG,
    )

    assert tp_rate == pytest.approx(trade.calc_close_rate_for_roi(0.01))


def test_take_profit_from_base_order_uses_actual_filled_base_stake() -> None:
    strategy = _strategy()
    strategy.tp_from_base_order.value = True
    trade = _trade((100.0, 95.0))
    expected_profit = trade.orders[0].stake_amount_filled * 0.01
    expected_roi = expected_profit / trade.stake_amount

    tp_rate = strategy.custom_exit_price(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=trade.open_rate,
        current_profit=0.0,
        exit_tag=strategy.TAKE_PROFIT_EXIT_TAG,
    )

    assert tp_rate == pytest.approx(trade.calc_close_rate_for_roi(expected_roi))


def test_custom_entry_price_separates_marketable_base_and_exact_safety_limit(
    monkeypatch,
) -> None:
    strategy = _strategy()
    trade = _trade()
    data = _custom_data(monkeypatch, trade)
    data["qfl_next_safety_order_price"] = 99.5
    data["qfl_next_safety_order_tag"] = "📈-🛡️⓵"

    base_rate = strategy.custom_entry_price(
        pair=trade.pair,
        trade=None,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=100.0,
        entry_tag="📈-📍",
        side="long",
    )
    so_rate = strategy.custom_entry_price(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=100.0,
        entry_tag="📈-🛡️⓵",
        side="long",
    )

    assert base_rate == pytest.approx(100.05)
    assert so_rate == pytest.approx(99.5)


def test_adjust_exit_price_tracks_average_after_safety_order_fill() -> None:
    strategy = _strategy()
    trade = _trade((100.0, 95.0))

    tp_rate = strategy.adjust_exit_price(
        trade=trade,
        order=None,
        pair=trade.pair,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=trade.open_rate,
        current_order_rate=101.0,
        entry_tag=trade.enter_tag,
        side=trade.exit_side,
    )

    assert tp_rate == pytest.approx(trade.calc_close_rate_for_roi(0.01))


def test_emergency_break_even_replaces_tp_after_final_safety_order() -> None:
    strategy = _strategy()
    strategy.max_safe_orders.value = 1
    strategy.use_emergency_be.value = True
    trade = _trade((100.0, 95.0))

    assert (
        strategy.custom_exit(
            pair=trade.pair,
            trade=trade,
            current_time=datetime(2026, 1, 1, tzinfo=UTC),
            current_rate=trade.open_rate,
            current_profit=0.0,
        )
        == strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG
    )
    assert strategy.custom_exit_price(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        proposed_rate=trade.open_rate,
        current_profit=0.0,
        exit_tag=strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG,
    ) == pytest.approx(trade.calc_close_rate_for_roi(0.0))


def test_strategy_uses_only_static_stoploss() -> None:
    strategy = _strategy()
    trade = _trade()

    result = strategy.custom_stoploss(
        pair=trade.pair,
        trade=trade,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=90.0,
        current_profit=-0.10,
        after_fill=False,
    )

    assert strategy.use_custom_stoploss is False
    assert result == pytest.approx(-0.99)


def test_entry_and_safety_order_tags_follow_shared_pattern() -> None:
    strategy = _strategy()
    dataframe = DataFrame(
        {
            "signal": ["↑", "↓"],
            "volume": [100.0, 100.0],
        }
    )

    result = strategy.populate_entry_trend(dataframe, {})

    assert result["enter_tag"].tolist() == ["📈-📍", "📉-📍"]
    assert strategy._safety_order_tag(strategy.LONG_TAG_PREFIX, 1) == "📈-🛡️⓵"
    assert strategy._safety_order_tag(strategy.SHORT_TAG_PREFIX, 2) == "📉-🛡️⓶"
    assert strategy.TAKE_PROFIT_EXIT_TAG == "🎯"
    assert strategy.EMERGENCY_BREAK_EVEN_EXIT_TAG == "🛟"


def test_leverage_respects_exchange_cap_and_buffered_tier() -> None:
    strategy = _strategy()
    strategy.leverage_buffer_pct.value = 5.0
    pair = "ETH/USDT:USDT"
    strategy.dp = SimpleNamespace(
        _exchange=SimpleNamespace(
            _leverage_tiers={
                pair: [
                    {"minNotional": 0.0, "maxNotional": 5000.0, "maxLeverage": 50.0},
                    {"minNotional": 5000.0, "maxNotional": 10000.0, "maxLeverage": 25.0},
                ]
            }
        )
    )

    leverage = strategy.leverage(
        pair=pair,
        current_time=datetime(2026, 1, 1, tzinfo=UTC),
        current_rate=100.0,
        proposed_leverage=1.0,
        max_leverage=20.0,
        entry_tag="📈-📍",
        side="long",
        proposed_stake=100.0,
    )

    assert leverage == pytest.approx(20.0)
