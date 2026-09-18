from unittest.mock import MagicMock

import pytest

from freqtrade.configuration.config_validation import _validate_hedge
from freqtrade.enums import RunMode, State
from freqtrade.exceptions import (
    ConfigurationError,
    DependencyException,
    OperationalException,
    PricingError,
)
from freqtrade.persistence import Order, Trade
from freqtrade.persistence.hedge_group import HedgeGroup
from freqtrade.rpc import RPC
from freqtrade.strategy import IStrategy
from tests import test_dca_hedge as hedge_fixtures
from tests.test_dca_hedge import ccxt_order, fill_exchange, parent_trade
from user_data.strategies.market_structure_trend_matrix_strategy import (
    MarketStructureTrendMatrixStrategy,
)


hedge_bot = hedge_fixtures.hedge_bot


def test_periodic_strategy_trigger_preempts_exit_and_holds_full_position(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    bot.strategy.custom_hedge = MagicMock(return_value="stop_before_target")
    bot.strategy.get_exit_signal = MagicMock(return_value=(False, True, "exit_signal"))
    bot._check_and_execute_exit = MagicMock(return_value=True)
    submit = fill_exchange(bot)
    assert bot.handle_trade(trade) is False
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "held"
    assert group.target_amount == trade.amount == 7
    assert group.trigger_source == "strategy"
    assert group.trigger_reason == "stop_before_target"
    assert group.trigger_order_id is None
    bot._check_and_execute_exit.assert_not_called()
    args = bot.strategy.custom_hedge.call_args.kwargs
    assert args["order"] is None and args["current_rate"] == 80
    assert args["current_profit"] == trade.calc_profit_ratio(80)
    bot.handle_trade(trade)
    bot.hedges.recover()
    assert submit.call_count == 1
    assert bot.strategy.custom_hedge.call_count == 1
    Trade.session.expire_all()
    assert HedgeGroup.for_trade(trade.id).trigger_reason == "stop_before_target"


@pytest.mark.parametrize("result", [None, False, True, "", 1, {"hedge": True}])
def test_invalid_or_empty_request_preserves_normal_exit(hedge_bot, result):
    bot = hedge_bot
    trade = parent_trade(bot)
    bot.strategy.custom_hedge = MagicMock(return_value=result)
    bot._check_and_execute_exit = MagicMock(return_value=True)
    assert bot.handle_trade(trade)
    assert HedgeGroup.for_trade(trade.id) is None


def test_callback_exception_preserves_normal_exit(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    bot.strategy.custom_hedge = MagicMock(side_effect=ValueError("invalid indicator"))
    bot._check_and_execute_exit = MagicMock(return_value=True)
    assert bot.handle_trade(trade)
    assert HedgeGroup.for_trade(trade.id) is None


def test_disabled_feature_never_evaluates_callback(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    bot.hedges.enabled = False
    bot.strategy.custom_hedge = MagicMock(return_value="hedge")
    bot._check_and_execute_exit = MagicMock(return_value=True)
    assert bot.handle_trade(trade)
    bot.strategy.custom_hedge.assert_not_called()


def test_fill_callback_observes_order_filled_state_first(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    order = trade.orders[0]
    events = []
    bot.strategy.order_filled = lambda **kwargs: events.append("filled")

    def callback(**kwargs):
        assert events == ["filled"]
        assert kwargs["order"].order_id == order.order_id
        return "entry_completed"

    bot.strategy.custom_hedge = callback
    fill_exchange(bot)
    bot._update_trade_after_fill(trade, order, False)
    group = HedgeGroup.for_trade(trade.id)
    assert group.trigger_order_id == order.order_id
    assert group.trigger_source == "strategy"
    assert group.trigger_reason == "entry_completed"


def test_recovery_restores_fill_callback_state_before_hedge_decision(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)

    def on_fill(**kwargs):
        kwargs["trade"].set_custom_data("last_fill", kwargs["order"].order_id)

    def callback(**kwargs):
        if kwargs["trade"].get_custom_data("last_fill") == kwargs["order"].order_id:
            return "recovered_fill"
        return None

    bot.strategy.order_filled = on_fill
    bot.strategy.custom_hedge = callback
    fill_exchange(bot)
    bot.hedges.recover()
    assert HedgeGroup.for_trade(trade.id).trigger_reason == "recovered_fill"


@pytest.mark.parametrize("timed_out", [False, True])
def test_hedge_precedes_exit_order_replacement_and_timeout(hedge_bot, timed_out):
    bot = hedge_bot
    trade = parent_trade(bot)
    pending = ccxt_order("pending_exit", trade.exit_side, 7, status="open", filled=0)
    order = Order.parse_from_ccxt_object(pending, trade.pair, trade.exit_side)
    trade.orders.append(order)
    Trade.commit()
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(return_value=pending)
    bot.strategy.custom_hedge = MagicMock(return_value="stop_before_target")
    bot.strategy.ft_check_timed_out = MagicMock(return_value=timed_out)
    bot.replace_order = MagicMock()
    bot.handle_cancel_order = MagicMock()
    # Keep preparation pending, so the test covers the freeze before cancellation settles.
    bot.exchange.cancel_order = MagicMock()
    bot.hedges._cancel_orders = MagicMock(return_value=False)
    bot.manage_open_orders()
    assert HedgeGroup.for_trade(trade.id).state == "preparing"
    bot.replace_order.assert_not_called()
    bot.handle_cancel_order.assert_not_called()


def test_default_strategy_callback_does_not_request_hedging(hedge_bot):
    assert IStrategy.custom_hedge(hedge_bot.strategy, "ETH/USDT:USDT", None, None, 80, 0) is None


def test_missing_hedge_quote_defers_open_order_actions(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    pending = ccxt_order("pending_exit", trade.exit_side, 7, status="open", filled=0)
    trade.orders.append(Order.parse_from_ccxt_object(pending, trade.pair, trade.exit_side))
    Trade.commit()
    bot.exchange.fetch_order_or_stoploss_order = MagicMock(return_value=pending)
    bot.exchange.get_rate.side_effect = PricingError("Empty order book")
    bot.replace_order = MagicMock()
    bot.handle_cancel_order = MagicMock()
    bot.manage_open_orders()
    bot.replace_order.assert_not_called()
    bot.handle_cancel_order.assert_not_called()
    assert HedgeGroup.for_trade(trade.id) is None


def test_unfilled_entry_is_not_an_automatic_or_manual_hedge_parent(hedge_bot):
    bot = hedge_bot
    trade = parent_trade(bot)
    for order in trade.orders:
        order.filled = 0
        order.status = "open"
        order.ft_is_open = True
    bot.strategy.custom_hedge = MagicMock(return_value="hedge")
    assert bot.hedges.evaluate(trade, 80) is False
    bot.strategy.custom_hedge.assert_not_called()
    with pytest.raises(DependencyException, match="filled position"):
        bot.hedges.request_open(trade)


@pytest.mark.parametrize("is_short", [False, True])
def test_real_trend_matrix_hedge_wins_over_stop_and_opposite_signal(hedge_bot, is_short):
    from pandas import DataFrame

    bot = hedge_bot
    trade = parent_trade(bot, is_short=is_short)
    trade.enter_tag = "mstm_bearish_choch" if is_short else "mstm_bullish_choch"
    trade.adjust_stop_loss(80, -0.01, allow_refresh=True)
    bot.strategy = MarketStructureTrendMatrixStrategy(bot.config)
    bot.strategy.dp = bot.dataprovider
    stop = trade.stop_loss
    bot.exchange.get_rate.return_value = stop
    bot.strategy.get_exit_signal = MagicMock(return_value=(False, True, "mstm_opposite_structure"))
    bot.dataprovider.get_analyzed_dataframe = MagicMock(return_value=(DataFrame(), None))
    # Recovery can use a persisted initialized stop even without indicator history.
    trade.is_stop_loss_trailing = True
    submit = fill_exchange(bot)
    assert bot.exit_positions([trade]) == 0
    group = HedgeGroup.for_trade(trade.id)
    assert group.state == "held"
    assert group.trigger_reason == "mstm_stop_before_target"
    assert submit.call_args.kwargs["amount"] == 7
    assert submit.call_args.kwargs["side"] == trade.exit_side


def test_generic_config_manual_hedge_without_strategy_trigger(hedge_bot):
    from freqtrade.hedging import HedgeManager

    bot = hedge_bot
    trade = parent_trade(bot)
    bot.config["hedge"]["order_type"] = "chase"
    bot.strategy.custom_hedge = IStrategy.custom_hedge.__get__(bot.strategy)
    bot.hedges = HedgeManager(bot)
    bot.hedges.observe_fill(trade, trade.orders[-1])
    assert HedgeGroup.for_trade(trade.id) is None
    submit = fill_exchange(bot)
    bot.state = State.RUNNING
    result = RPC(bot)._rpc_hedge(str(trade.id))
    assert set(result) == {"result", "hedge"}
    assert result["hedge"]["trigger_source"] == "manual"
    assert result["hedge"]["order_type"] == "chase"
    assert submit.call_args.kwargs["amount"] == 7


@pytest.mark.parametrize("stop_setting", ["configured", "existing"])
def test_price_trigger_strategy_rejects_exchange_stops(hedge_bot, mocker, stop_setting):
    bot = hedge_bot
    bot.strategy.hedge_requires_local_stoploss = True
    bot.strategy.order_types = dict(bot.strategy.order_types)
    bot.strategy.order_types["stoploss_on_exchange"] = stop_setting == "configured"
    mocker.patch.object(type(bot.exchange), "id", new_callable=lambda: property(lambda _: "okx"))
    mocker.patch.object(bot.exchange, "validate_hedge_mode", create=True)
    if stop_setting == "existing":
        trade = parent_trade(bot)
        trade.orders[-1].ft_order_side = "stoploss"
        trade.orders[-1].ft_is_open = True
        Trade.commit()
    with pytest.raises(OperationalException, match="exchange stop"):
        bot.hedges.validate()


def test_shared_config_rejects_unsupported_exchange():
    with pytest.raises(ConfigurationError, match="OKX futures"):
        _validate_hedge(
            {
                "hedge": {"enabled": True},
                "exchange": {"name": "binance"},
                "trading_mode": "futures",
                "runmode": RunMode.LIVE,
            }
        )
