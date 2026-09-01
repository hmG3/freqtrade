import logging
from typing import Any

from freqtrade.enums import MarginMode, RunMode
from freqtrade.exceptions import DependencyException
from freqtrade.exchange import Exchange
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.wallets import Wallets


logger = logging.getLogger(__name__)


def update_liquidation_prices(
    trade: LocalTrade | None = None,
    *,
    exchange: Exchange,
    wallets: Wallets,
    stake_currency: str,
    dry_run: bool = False,
):
    """
    Update trade liquidation price in isolated margin mode.
    Updates liquidation price for all trades in cross margin mode.
    """
    try:
        if exchange.margin_mode == MarginMode.CROSS:
            total_wallet_stake = 0.0
            if dry_run:
                # Parameters only needed for cross margin
                wallets.update()
                total_wallet_stake = wallets.get_collateral()
                logger.info(
                    "Updating liquidation price for all open trades. "
                    f"Collateral {total_wallet_stake} {stake_currency}."
                )

            open_trades: list[Trade] = Trade.get_open_trades()
            supports_cross_fallback = exchange.get_option("cross_liquidation_price_fallback", False)
            mark_prices = None
            if (
                dry_run
                and supports_cross_fallback
                and exchange._config.get("runmode") in (RunMode.LIVE, RunMode.DRY_RUN)
            ):
                pairs = list(dict.fromkeys(t.pair for t in open_trades if t.has_open_position))
                if pairs:
                    mark_prices = exchange.fetch_mark_prices(pairs)
            for t in open_trades:
                if t.has_open_position:
                    # TODO: This should be done in a batch update
                    liquidation_kwargs: dict[str, Any] = {
                        "pair": t.pair,
                        "open_rate": t.open_rate,
                        "is_short": t.is_short,
                        "amount": t.amount,
                        "stake_amount": t.stake_amount,
                        "leverage": t.leverage,
                        "wallet_balance": total_wallet_stake,
                        "open_trades": open_trades,
                    }
                    if supports_cross_fallback:
                        liquidation_kwargs["mark_prices"] = mark_prices
                    t.set_liquidation_price(exchange.get_liquidation_price(**liquidation_kwargs))
        elif trade:
            trade.set_liquidation_price(
                exchange.get_liquidation_price(
                    pair=trade.pair,
                    open_rate=trade.open_rate,
                    is_short=trade.is_short,
                    amount=trade.amount,
                    stake_amount=trade.stake_amount,
                    leverage=trade.leverage,
                    wallet_balance=trade.stake_amount,
                )
            )
        else:
            raise DependencyException(
                "Trade object is required for updating liquidation price in isolated margin mode."
            )
    except DependencyException:
        logger.warning("Unable to calculate liquidation price")
