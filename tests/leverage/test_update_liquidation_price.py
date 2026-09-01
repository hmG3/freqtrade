from unittest.mock import MagicMock, call

import pytest

from freqtrade.enums import RunMode
from freqtrade.enums.marginmode import MarginMode
from freqtrade.leverage.liquidation_price import update_liquidation_prices


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("margin_mode", [MarginMode.CROSS, MarginMode.ISOLATED])
def test_update_liquidation_prices(mocker, margin_mode, dry_run):
    # Heavily mocked test - Only testing the logic of the function
    # update liquidation price for trade in isolated mode
    # update liquidation price for all trades in cross mode
    exchange = MagicMock()
    exchange.margin_mode = margin_mode
    exchange.get_option.return_value = False
    wallets = MagicMock()
    trade_mock = MagicMock()

    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock])

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )

    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.get_collateral.call_count == (
        0 if margin_mode == MarginMode.ISOLATED or not dry_run else 1
    )

    # Test with multiple trades
    trade_mock.reset_mock()
    trade_mock_2 = MagicMock()

    mocker.patch(
        "freqtrade.persistence.Trade.get_open_trades", return_value=[trade_mock, trade_mock_2]
    )

    update_liquidation_prices(
        trade=trade_mock,
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=dry_run,
    )
    # Trade2 is only updated in cross mode
    assert trade_mock_2.set_liquidation_price.call_count == (
        1 if margin_mode == MarginMode.CROSS else 0
    )
    assert trade_mock.set_liquidation_price.call_count == 1

    assert wallets.call_count == 0 if not dry_run else 1


def test_update_cross_liquidation_prices_uses_one_mark_snapshot(mocker):
    exchange = MagicMock()
    exchange.margin_mode = MarginMode.CROSS
    exchange.get_option.return_value = True
    exchange._config = {"runmode": RunMode.DRY_RUN}
    mark_prices = {
        "ETH/USDT:USDT": {"markPrice": 100.0},
        "ADA/USDT:USDT": {"markPrice": 20.0},
    }
    exchange.fetch_mark_prices.return_value = mark_prices
    wallets = MagicMock()
    wallets.get_collateral.return_value = 250.0
    wallet_calls = MagicMock()
    wallet_calls.attach_mock(wallets.update, "update")
    wallet_calls.attach_mock(wallets.get_collateral, "get_collateral")
    trades = [
        MagicMock(pair="ETH/USDT:USDT", has_open_position=True),
        MagicMock(pair="ADA/USDT:USDT", has_open_position=True),
    ]
    mocker.patch("freqtrade.persistence.Trade.get_open_trades", return_value=trades)

    update_liquidation_prices(
        exchange=exchange,
        wallets=wallets,
        stake_currency="USDT",
        dry_run=True,
    )

    exchange.fetch_mark_prices.assert_called_once_with(["ETH/USDT:USDT", "ADA/USDT:USDT"])
    assert wallet_calls.mock_calls[:2] == [call.update(), call.get_collateral()]
    assert exchange.get_liquidation_price.call_count == 2
    for call_args in exchange.get_liquidation_price.call_args_list:
        assert call_args.kwargs["wallet_balance"] == 250.0
        assert call_args.kwargs["mark_prices"] is mark_prices
