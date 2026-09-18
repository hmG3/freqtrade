import pytest
from sqlalchemy import create_engine, event, inspect

from freqtrade.persistence import Trade, init_db
from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.hedge_group import HedgeGroup


@pytest.mark.parametrize("existing_database", [False, True])
def test_first_install_creates_complete_hedge_schema_once(mocker, existing_database):
    engine = create_engine("sqlite://")
    if existing_database:
        ModelBase.metadata.create_all(
            engine,
            tables=[t for t in ModelBase.metadata.sorted_tables if t.name != "hedge_groups"],
        )
    statements = []
    event.listen(
        engine,
        "before_cursor_execute",
        lambda conn, cursor, statement, parameters, context, executemany: statements.append(
            " ".join(statement.lower().split())
        ),
    )
    mocker.patch("freqtrade.persistence.models.create_engine", return_value=engine)
    init_db("sqlite://")
    init_db("sqlite://")
    columns = {c["name"]: c for c in inspect(engine).get_columns("hedge_groups")}
    assert set(columns) == set(HedgeGroup.__table__.columns.keys())
    assert columns["trigger_order_id"]["nullable"]
    assert not columns["trigger_source"]["nullable"]
    assert "order_type" in columns and "trigger_reason" in columns
    assert sum(s.startswith("create table hedge_groups") for s in statements) == 1
    assert not any(
        "hedge_groups" in s and s.startswith(("alter table", "update", "drop table"))
        for s in statements
    )
    assert set(inspect(engine).get_table_names()) == set(ModelBase.metadata.tables)
    assert {fk["referred_table"] for fk in inspect(engine).get_foreign_keys("hedge_groups")} == {
        "trades"
    }


def test_complete_schema_preserves_pending_intent_and_json_accounting_on_restart(
    init_persistence,
    mocker,
):
    engine = Trade.session.get_bind()
    group = HedgeGroup(
        parent_id=1,
        trigger_source="strategy",
        trigger_reason="mstm_stop_before_target",
        order_type="chase",
        state="opening",
        pending_client_id="pending-client",
        pending_order_id="algo-123",
        pending_amount=7.0,
        pending_rate=80.0,
        accounted_orders=["entry-1"],
    )
    other = HedgeGroup(parent_id=2)
    Trade.session.add_all([group, other])
    Trade.commit()
    group.accounted_orders = [*group.accounted_orders, "entry-2"]
    Trade.commit()
    Trade.session.remove()
    mocker.patch("freqtrade.persistence.models.create_engine", return_value=engine)
    init_db("sqlite://")
    group = Trade.session.get(HedgeGroup, 1)
    assert group.trigger_source == "strategy"
    assert group.trigger_reason == "mstm_stop_before_target"
    assert group.order_type == "chase" and group.state == "opening"
    assert group.pending_client_id == "pending-client"
    assert group.pending_order_id == "algo-123"
    assert group.pending_amount == 7.0 and group.pending_rate == 80.0
    assert group.accounted_orders == ["entry-1", "entry-2"]
    assert Trade.session.get(HedgeGroup, 2).accounted_orders == []
    assert Trade.session.get(HedgeGroup, 2).trigger_source == "manual"
