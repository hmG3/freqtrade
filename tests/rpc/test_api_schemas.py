import pytest
from pydantic import ValidationError

from freqtrade.rpc.api_server.api_schemas import ForceEnterPayload, ForceExitPayload, OrderTypes


def test_order_types_allow_chase_only_for_entry_and_exit():
    order_types = OrderTypes(
        entry="chase",
        exit="chase",
        emergency_exit="market",
        force_exit="limit",
        force_entry="limit",
        stoploss="market",
        stoploss_on_exchange=False,
    )

    assert order_types.entry == "chase"
    assert order_types.exit == "chase"

    with pytest.raises(ValidationError):
        OrderTypes(
            entry="limit",
            exit="limit",
            force_entry="chase",
            stoploss="market",
            stoploss_on_exchange=False,
        )


@pytest.mark.parametrize(
    "payload",
    [
        lambda: ForceEnterPayload(pair="ETH/USDT:USDT", ordertype="chase"),
        lambda: ForceExitPayload(tradeid=1, ordertype="chase"),
    ],
)
def test_force_order_payloads_reject_chase(payload):
    with pytest.raises(ValidationError):
        payload()
