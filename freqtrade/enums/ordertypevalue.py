from enum import StrEnum


class OrderTypeValues(StrEnum):
    limit = "limit"
    market = "market"


class EntryExitOrderTypeValues(StrEnum):
    limit = "limit"
    market = "market"
    chase = "chase"
