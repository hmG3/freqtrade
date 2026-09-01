"""Mexc exchange subclass"""

import logging

from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Mexc(Exchange):
    _ft_has: FtHas = {
        "stoploss_on_exchange": True,
        "stop_price_param": "stopPrice",
        "stop_price_prop": "stopPrice",
        "stoploss_order_types": {"limit": "limit", "market": "market"},
        "ohlcv_candle_limit": 500,
        "order_time_in_force": ["GTC", "IOC", "FOK"],
        "l2_limit_range": [20, 100, 500],
    }
