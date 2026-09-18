"""Durable intent for one position and its protective opposite position."""

import json
from typing import ClassVar

from sqlalchemy import ForeignKey, String, Text, or_, select
from sqlalchemy.orm import Mapped, mapped_column

from freqtrade.persistence.base import ModelBase, SessionType


class HedgeGroup(ModelBase):
    __tablename__ = "hedge_groups"
    session: ClassVar[SessionType]

    parent_id: Mapped[int] = mapped_column(ForeignKey("trades.id"), primary_key=True)
    hedge_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"), unique=True)
    # Price-based and manual requests have no triggering exchange order.
    trigger_order_id: Mapped[str | None] = mapped_column(String(255))
    trigger_source: Mapped[str] = mapped_column(String(32), default="manual")
    trigger_reason: Mapped[str | None] = mapped_column(String(255))
    # Snapshot at the trigger. Configuration changes only affect future groups.
    order_type: Mapped[str] = mapped_column(String(16), default="market", server_default="market")
    target_amount: Mapped[float] = mapped_column(default=0.0)
    state: Mapped[str] = mapped_column(String(16), default="preparing")
    close_requested: Mapped[bool] = mapped_column(default=False)
    pending_trade_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"))
    pending_client_id: Mapped[str | None] = mapped_column(String(32))
    pending_order_id: Mapped[str | None] = mapped_column(String(255))
    pending_amount: Mapped[float | None]
    pending_rate: Mapped[float | None]
    pending_exit: Mapped[bool] = mapped_column(default=False)
    last_error: Mapped[str | None] = mapped_column(String(2048))
    # Persisted in the SAME transaction as trade accounting. Replaying a partial
    # exit against an already reduced trade amount would otherwise close it twice.
    # Use text-backed JSON, as native custom data does, for all supported SQL dialects.
    _accounted_orders: Mapped[str] = mapped_column("accounted_orders", Text, default="[]")

    @property
    def accounted_orders(self) -> list[str]:
        return json.loads(self._accounted_orders or "[]")

    @accounted_orders.setter
    def accounted_orders(self, value: list[str]) -> None:
        self._accounted_orders = json.dumps(value)

    @classmethod
    def for_trade(cls, trade_id: int) -> "HedgeGroup | None":
        return cls.session.scalar(
            select(cls).where(or_(cls.parent_id == trade_id, cls.hedge_id == trade_id))
        )

    @classmethod
    def active(cls) -> list["HedgeGroup"]:
        return list(cls.session.scalars(select(cls).where(cls.state != "closed")))

    def clear_pending(self) -> None:
        self.pending_trade_id = self.pending_client_id = self.pending_order_id = None
        self.pending_amount = self.pending_rate = None
        self.pending_exit = False

    def to_json(self) -> dict:
        return {
            "parent_trade_id": self.parent_id,
            "hedge_trade_id": self.hedge_id,
            "state": self.state,
            "trigger_source": self.trigger_source,
            "trigger_reason": self.trigger_reason,
            "order_type": self.order_type,
            "target_amount": self.target_amount,
            "close_requested": self.close_requested,
            "pending_client_id": self.pending_client_id,
            "pending_order_id": self.pending_order_id,
            "last_error": self.last_error,
        }
