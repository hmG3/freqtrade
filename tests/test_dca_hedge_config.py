from pathlib import Path

import pytest

from freqtrade.configuration.config_validation import _validate_hedge
from freqtrade.enums import RunMode
from freqtrade.exceptions import ConfigurationError, OperationalException
from freqtrade.freqtradebot import FreqtradeBot
from freqtrade.hedging import HedgeManager
from freqtrade.persistence import Trade
from freqtrade.persistence.hedge_group import HedgeGroup
from tests.conftest import patch_exchange, patch_freqtradebot
from tests.test_dca_hedge import ccxt_order
from user_data.strategies.adx_vw_dca_strategy import ADXVWDCAStrategy
from user_data.strategies.market_structure_trend_matrix_strategy import (
    MarketStructureTrendMatrixStrategy,
)
from user_data.strategies.qfl_dca_strategy import QFLDCAStrategy
from user_data.strategies.rsi_ml_dca_strategy import RSIMLDCAStrategy
from user_data.strategies.rtb_dca_strategy import RTBDCAStrategy


STRATEGIES = [ADXVWDCAStrategy, QFLDCAStrategy, RSIMLDCAStrategy, RTBDCAStrategy]


@pytest.mark.parametrize("order_type", ["market", "limit", "chase"])
def test_hedge_order_type_schema_accepts_supported_types(order_type):
    from jsonschema import validate

    from freqtrade.config_schema.config_schema import CONF_SCHEMA

    validate({"enabled": True, "order_type": order_type}, CONF_SCHEMA["properties"]["hedge"])


@pytest.mark.parametrize("order_type", ["stop", "MARKET", "", None])
def test_hedge_order_type_schema_rejects_unsupported_types(order_type):
    from jsonschema import ValidationError, validate

    from freqtrade.config_schema.config_schema import CONF_SCHEMA

    with pytest.raises(ValidationError):
        validate({"enabled": True, "order_type": order_type}, CONF_SCHEMA["properties"]["hedge"])


@pytest.mark.parametrize("strategy_class", [*STRATEGIES, MarketStructureTrendMatrixStrategy])
@pytest.mark.parametrize("order_type", ["market", "limit", "chase"])
def test_enabled_okx_bot_initializes_with_each_dca_strategy(
    default_conf_usdt, mocker, strategy_class, order_type
):
    conf = default_conf_usdt
    conf.update(
        {
            "strategy": strategy_class.__name__,
            "strategy_path": str(Path("user_data/strategies").resolve()),
            "trading_mode": "futures",
            "margin_mode": "cross",
            "hedge": {"enabled": True, "order_type": order_type},
            "runmode": RunMode.DRY_RUN,
            "order_types": {
                "entry": "limit",
                "exit": "limit",
                "stoploss": "market",
                "stoploss_on_exchange": False,
            },
        }
    )
    conf["exchange"]["name"] = "okx"
    conf["exchange"]["pair_whitelist"] = ["ETH/USDT:USDT"]
    patch_freqtradebot(mocker, conf)
    patch_exchange(mocker, exchange="okx")
    bot = FreqtradeBot(conf)
    assert bot.hedges.enabled
    assert bot.exchange.net_only is False
    assert bot.strategy.get_strategy_name() == strategy_class.__name__


@pytest.mark.parametrize("strategy_class", STRATEGIES)
@pytest.mark.parametrize("is_short", [False, True])
def test_safety_order_metadata_identifies_the_tag_not_attempt_count(strategy_class, is_short):
    from freqtrade.persistence import Order

    strategy = strategy_class({})
    trade = Trade(pair="ETH/USDT:USDT", is_short=is_short, amount=1, open_rate=80, fee_open=0.001)
    prefix = strategy.SHORT_TAG_PREFIX if is_short else strategy.LONG_TAG_PREFIX
    order = Order.parse_from_ccxt_object(
        ccxt_order("so", trade.entry_side, 1.0), trade.pair, trade.entry_side
    )
    order.ft_order_tag = strategy._safety_order_tag(prefix, 2)
    assert strategy._hedge_safety_order_index(trade, order) == 2
    order.ft_order_tag = "base"
    assert strategy._hedge_safety_order_index(trade, order) is None


@pytest.mark.parametrize("runmode", [RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.WEBSERVER])
def test_unsupported_execution_modes_fail_explicitly(runmode):
    with pytest.raises(ConfigurationError, match="live/dry-run only"):
        _validate_hedge(
            {
                "exchange": {"name": "okx"},
                "trading_mode": "futures",
                "runmode": runmode,
                "hedge": {"enabled": True},
            }
        )


@pytest.mark.parametrize("exchange,mode", [("binance", "futures"), ("okx", "spot")])
def test_wrong_exchange_or_trading_mode_rejected(exchange, mode):
    with pytest.raises(ConfigurationError, match="OKX futures"):
        _validate_hedge(
            {
                "exchange": {"name": exchange},
                "trading_mode": mode,
                "runmode": RunMode.DRY_RUN,
                "hedge": {"enabled": True},
            }
        )


def test_active_groups_cannot_be_disabled(init_persistence):
    from types import SimpleNamespace

    Trade.session.add(HedgeGroup(parent_id=1, trigger_order_id="trigger"))
    Trade.commit()
    manager = HedgeManager(SimpleNamespace(config={}))
    with pytest.raises(OperationalException, match="Keep hedge enabled"):
        manager.validate()
