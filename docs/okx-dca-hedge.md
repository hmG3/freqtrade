# OKX strategy hedging and manual hold

This document covers hedging configuration, strategy triggers, execution,
persistence, recovery, and implementation details.

This fork can open an opposite OKX futures position when a strategy requests it,
or through `/hedge`. DCA strategies request it after a selected Safety Order
completes. It uses the **entire remaining position quantity**, including the Base
Order and all filled Safety Orders. For example, a long position containing
0.01 + 0.02 + 0.04 BTC opens a 0.07 BTC short. A short parent opens a long hedge.
The hedge uses the parent's leverage; its margin cost depends on the current price.

## Enable

Add this section to the existing trading configuration:

```json
"hedge": {
  "enabled": true,
  "after_safety_order": "last",
  "order_type": "chase"
}
```

`"last"` uses the strategy's loaded `max_safe_orders` value, including its parameter
file. An integer such as `3` selects SO3 instead. The Base Order is not SO1.
The configured number must not exceed `max_safe_orders`. `after_safety_order`
defaults to `"last"` when omitted and is used only by DCA strategies; other
strategies can omit it. `hedge` is the configuration section for both automatic
and manual hedge requests.

`order_type` selects how the **hedge entry** executes:

| Value | Behavior |
| --- | --- |
| `market` | Default when omitted. Submit a market order for the remaining hedge quantity. |
| `limit` | Submit a GTC limit order at the configured `entry_pricing` rate for the hedge direction; leave its price unchanged while it is open. |
| `chase` | Submit an OKX native post-only chase order following the best bid/ask (`chaseType=distance`, `chaseVal=0`). |

Limit and chase entries wait until filled or canceled. They bypass the ordinary
`unfilledtimeout` and strategy order-adjustment callbacks. There is **no automatic
timeout or market fallback**. The unfilled portion remains unhedged while waiting.
A limit order may execute as a taker if its price crosses the spread; chase uses
post-only execution. See [OKX chase orders](https://www.okx.com/en-eu/help/how-to-use-chase-order).

The bot saves this choice when the group is triggered. Changing configuration
and restarting affects only future groups, including when a pending order's
submission response was lost. Manual group force-exit closes both legs with market orders.

DCA strategies in this workspace are `ADXVWDCAStrategy`, `QFLDCAStrategy`,
`RSIMLDCAStrategy`, and `RTBDCAStrategy`. Their shared `DcaHedgeMixin` implements
`custom_hedge()` using their directional Safety Order tags. The execution controller
does not interpret strategy tags.
Base entries, exit fills, and partially filled canceled Safety Orders do not
trigger a hedge. A completed SO whose size was previously reduced through order
replacement remains identified by its SO tag.

Requirements:

- `exchange.name` is `okx`, `trading_mode` is `futures`, and `margin_mode` is `cross`
  or `isolated`.
- Instruments are linear perpetual swaps whose settlement currency matches
  `stake_currency`, for example `BTC/USDT:USDT`.
- A live OKX account already uses **long/short mode** (`long_short_mode`). Cross
  margin retains this fork's requirement for OKX account level 2 (Futures mode).
- Use one bot/database as the owner of both legs. Another bot or a manually opened
  opposite position on the same instrument cannot be combined with this group.
- Keep free collateral for the entire hedge plus execution fees. The strategies'
  leverage-tier calculation includes a hedge allowance, but **does not reserve
  wallet funds**. Other trades, price movement and fees can consume that capacity.

The bot validates these settings and never changes the live account's position
mode. OKX requires positions and pending orders to be cleared before switching
position mode. See [OKX account and trading guidance](https://www.okx.com/docs-v5/trick_en/).

Start with the normal dry-run configuration and a separate dry-run database.
No live configuration is enabled by this implementation. Existing configurations
without `hedge.enabled` keep their normal trading behavior.

## Define a strategy trigger

Implement this optional `IStrategy` callback:

```python
def custom_hedge(
    self, pair: str, trade: Trade, current_time: datetime,
    current_rate: float, current_profit: float,
    order: Order | None = None, **kwargs,
) -> str | None:
    if order is None and self.should_hedge(trade, current_rate):
        return "my_hedge_reason"
    return None
```

`should_hedge` represents a strategy-specific eligibility rule. Return a nonempty reason,
up to 255 characters; return `None` for no action. Do not submit orders here.
The default implementation does nothing, allowing manual-only hedging.
Automatic and manual requests require an open trade with a known positive entry
fill; merely submitting an unfilled entry does not make it eligible.
Use `validate_hedge(self, **kwargs)` for strategy-specific startup checks; it runs
after parameter loading and only when the feature is enabled.

With `order=None`, the callback runs each bot iteration before local automatic
exits, using the fresh executable exit quote and corresponding profit. It also runs
after `order_filled` for terminal orders with positive fills. These calls include
the order and use its fill price. Startup replays `order_filled` before
`custom_hedge` for saved terminal fills on unlinked positions, to recover a crash
between fill persistence and strategy processing. Make both callbacks idempotent
by order ID. Recovery uses the saved fill time for `order_filled`; normal calls use
the processing time. Manual-only strategies using the default hedge callback skip replay.
Price-based strategies should ignore these fill calls. Exceptions or invalid return
values log an error/warning and leave normal exits active. Existing linked groups
are never reevaluated. A committed request freezes the decision even if orders fill
during cancellation; the reconciled remaining quantity determines the hedge size.

`MarketStructureTrendMatrixStrategy` returns `mstm_stop_before_target` for an ATR
stop touch before any target fill. Any positive target fill counts, including a
partial fill on a subsequently canceled order; plotted wick hits do not.
Zero-target mode is eligible too. A simultaneous local exit signal yields to the
hedge request. It uses the working monotonic ATR stop, including its persisted
value after restart. See the [strategy documentation](market-structure-trend-matrix-strategy.md).

Strategies whose hedge trigger replaces a price stop must declare
`hedge_requires_local_stoploss = True`. Startup then rejects
`stoploss_on_exchange: true` and outstanding tracked exchange stop orders, which
could execute before the bot notices the trigger. Reconcile and cancel existing
exchange stops before switching such a strategy to hedging. Local stops remain
active until a hedge request is committed. Polling can miss a price touch that
happens and reverses between bot iterations. Open orders are reconciled after
analysis and before the hedge decision; the hedge gets priority over automatic
order replacements and timeout exits too.
If the fresh quote needed for an open-order hedge decision is unavailable, the
bot defers that trade's order replacements and timeout handling for that pass.

## What happens at the trigger

1. The bot records a durable hedge group when the strategy or manual command requests it.
2. Further DCA and all automatic exits are suspended for that parent.
3. Existing TP, stop and other working parent orders are canceled. The bot fetches
   their final fills before calculating the remaining parent quantity. A racing
   exit therefore reduces the hedge target.
4. The bot records an opposite trade and order intent, then submits the configured
   order for that exact quantity. This bypasses strategy initial-stake sizing,
   entry signals, pair locks and the normal maximum-open-trade entry limit.
5. An open, partially filled order stays pending without another submission.
   Confirmed terminal partial hedge fills are topped up by their remaining quantity
   using the group's saved order type (a new limit order uses a fresh entry price).
   Once the target is reached, the group stays held.

A hedge order that terminates without any fill blocks the group and requires
manual action. It is not automatically replaced. If cancellation of the parent's
orders reveals that the parent has already fully exited, the group closes without
opening a hedge.

Submission happens in the fill-processing pass, without waiting for a new candle.
Detection still depends on Freqtrade's polling loop, network latency, and OKX's
final fill reporting. Native chase orders retain their existing settlement wait.
This is not a sub-millisecond exchange-side contingent order.

Each leg has its own normal Trade/Order records and P&L. Both count toward the
ordinary open-trade count, so opening a protective hedge can put the bot above
`max_open_trades` and prevent new unrelated entries. Hedge orders never silently
shrink to fit available stake. An unrepresentable lot quantity or insufficient
collateral blocks opening and sends a status notification.

## Open a hedge manually with Telegram

Send `/hedge <trade_id>`, for example `/hedge 42`, to hedge an existing position
before the configured Safety Order triggers. Find the trade ID with `/status table`.
The command requires `hedge.enabled: true` and a running or paused bot. It uses
the existing Telegram chat, topic, and authorized-user restrictions.

Manual requests enter the same controller as automatic triggers: cancel working
parent orders, reconcile any fills, hedge the entire remaining quantity using
`hedge.order_type`, suspend DCA and automatic exits, and hold both legs until
manual closure. It also supports a partially filled entry after that fill is known
to the bot. It does not require `force_entry_enable`.

The response reports the group state; `preparing` or `opening` means the request
is still in progress. Repeating `/hedge` on either linked trade returns the existing
group and never starts a second hedge. `/hedge all`, amounts, and per-command order
type overrides are not supported. `/forceexit <trade_id>` still closes both legs.

## Persistence and initialization

Both legs use Freqtrade's existing `trades` and `orders` tables. The feature adds
one `hedge_groups` table and no columns to those existing tables. Each group stores
the linked trade IDs, trigger details, saved order type, target quantity, lifecycle
state, pending order intent, and identifiers of fills already accounted for.

Normal bot startup creates the complete hedge schema in a single idempotent
migration after native Freqtrade trade migrations; no separate migration command
is needed. The same step supports a new database or an existing Freqtrade database
without hedging. Later startups preserve saved groups and pending recovery IDs.
`trigger_source` is `manual` or `strategy`; `trigger_reason` stores the strategy's
reason. `trigger_order_id` is nullable for price-based and manual requests.
This first-release schema has no upgrades from intermediate hedge implementations.

`migrate_hedge_groups()` creates the complete table from the SQLAlchemy model using
`Table.create(checkfirst=True)`. Initial metadata creation excludes this table
until native trade migrations have run, so its foreign keys cannot follow an old
`trades` table when that table is renamed to a migration backup.

| Stored fields | Purpose |
| --- | --- |
| `parent_id`, `hedge_id` | Parent primary key and optional unique opposite-leg ID, referencing ordinary trades. |
| `trigger_source`, `trigger_reason`, `trigger_order_id` | Why the group was requested; price/manual triggers need no exchange order ID. |
| `order_type`, `target_amount` | Saved hedge entry type and remaining parent quantity after preparation. |
| `state`, `close_requested`, `last_error` | Lifecycle progress, durable manual-close request, and latest execution error. |
| `pending_trade_id`, `pending_client_id`, `pending_order_id` | The leg and exchange identifiers of the one pending submission per group. |
| `pending_amount`, `pending_rate`, `pending_exit` | Exact submitted quantity, requested price, and whether the intent closes a leg. |
| `accounted_orders` | Order IDs recorded atomically with terminal fill accounting to prevent replay. |

`accounted_orders` follows native custom-data storage: a JSON list encoded in a
`Text` column. The SQLAlchemy JSON column type fails the repository's Oracle DDL
compilation check, so text preserves compatibility with its tested SQL dialects.

## Hold and manual closure

Once triggered, both linked legs are excluded from further DCA, negative position
adjustments, ROI exits, exit signals, custom exits, stop-loss, trailing stops and
automatic exchange stop placement. Existing automatic orders are canceled during
preparation. This suspension also applies while a hedge is pending or blocked.
Exchange liquidation and funding charges still apply.

Use the bot's **force-exit on either trade ID** to request a market close of both
legs. The bot closes each leg's actual remaining amount and persists the closing
state across restarts. These are separate exchange orders, so execution is not
atomic. Partial amounts, explicit prices, and limit/chase group closes are rejected.
Force-exit `all` also closes linked groups. Use status to check actual completion.
If the hedge entry is still open, the bot first cancels it through the appropriate
OKX endpoint and waits for its final fills. A cancellation acknowledgment alone
does not authorize closing the legs or submitting another opening order.

Positions can also be closed directly on OKX. While running, the bot attempts
to identify those exit orders by instrument and `posSide` and update its records.
A manual reduction or closure never causes a held hedge to reopen. Closing only
one side leaves the other side open until it is closed manually. If positions and identifiable
fills disagree, the bot reports the discrepancy and waits rather than inventing
an exit or trading an uncertain amount.
During opening, a changed parent quantity or an identifiable manual reduction of
the hedge cancels its pending entry and blocks further opening. Final entry fills
are still reconciled; the bot does not refill the manually reduced quantity.
Use force-exit to close the resulting group, including any partially filled hedge.

Do not disable the feature while an active group exists: startup rejects that
configuration to prevent ordinary strategy exits from taking over held legs.
The bot also rejects force-entry/DCA and history deletion for linked groups.
Keep the database when restarting; the group and order intent are stored there.
If enabling the feature on an existing database, an open trade with an already
completed selected SO is eligible during startup recovery.

## Recovery and status

The trade-status RPC/API includes a `hedge` object with linked trade IDs, trigger source/reason,
state, saved entry `order_type`, target quantity, closing request, pending
client/order IDs and last error.
Status notifications announce requested/held groups and execution errors.

| State | Meaning |
| --- | --- |
| `preparing` | Waiting for parent orders to be canceled and final fills reconciled. |
| `opening` | Opening or recovering the opposite position. |
| `held` | Opening has finished; both legs require manual management. |
| `closing` | A manual request is closing the remaining legs. |
| `blocked` | A definitive rejection or inconsistent quantity needs manual action. |
| `closed` | The group has finished. |

The bot saves a unique OKX client order ID before submission (`clOrdId` for market
and limit, `algoClOrdId` for chase). If a response is lost, it looks up that ID
through the corresponding regular/algo endpoint and adopts the resulting order.
Chase fills are reconciled across its replacement child orders. A temporarily missing order
does **not** authorize a new live submission: OKX client IDs are only required to
be unique among pending orders, so reusing one after a fill could create another
order. Terminal fills are marked as accounted in the same transaction as trade
accounting, preventing duplicate partial-exit accounting after a crash.
See [OKX order guidance](https://www.okx.com/docs-v5/trick_en/) and
[CCXT client order IDs](https://github.com/ccxt/ccxt/wiki/Manual#user-defined-clientorderid).
Chase recovery uses [OKX algo order details](https://www.okx.com/docs-v5/en/#order-book-trading-algo-trading-get-algo-order-details).

After a definitive rejected opening, use force-exit to close the parent and any
filled hedge quantity. The bot does not repeatedly retry rejected orders.
When a live submission remains ambiguous, even a manual bot close must first
resolve it. Check the reported client ID on OKX and resolve account/order history
before changing the database or starting another bot on that position.

## Shared controller and native code reuse

Strategy callbacks and `/hedge` use the same `HedgeManager` lifecycle. Strategies
decide when to request a hedge; the controller persists the request, freezes
automatic management, coordinates the two legs, and recovers pending orders.
Callers serialize these operations with the bot's existing `_exit_lock`, including
Telegram requests, so the trading loop and remote commands cannot submit competing
group actions. This process-local lock is why one bot must own the database/legs.

| Concern | Native code reused |
| --- | --- |
| Regular client-ID lookup | `Exchange.fetch_order`, including retries, normalization, logging, and error handling. |
| Dry-run recovery | `fetch_dry_run_order`, including database recovery and simulated fill checks. |
| Instrument validation | `market_is_future`, plus a matching settlement-currency check. |
| Submission and exact sizing | `Exchange.create_order` and `amount_to_contract_precision`. |
| Order parsing and fill/fee accounting | `Order.parse_from_ccxt_object` and `FreqtradeBot.update_trade_state`. |
| Manual exchange exits | `handle_onexchange_order`, with exact-quantity protection for linked trades. |
| Empty terminal orders | `check_order_canceled_empty`. |
| Callback errors and strategy state | `strategy_safe_wrapper` and persistent Trade custom data. |
| Reason length and close label | `CUSTOM_TAG_MAX_LENGTH` and `ExitType.FORCE_EXIT`. |

OKX chase lookup retains its exchange-specific algo handling. Futures wallets are
keyed by `(pair, side)`; exit ownership checks and live liquidation lookup also
distinguish long from short, so the two positions cannot overwrite one another.

The controller submits through the native exchange layer rather than the ordinary
`execute_entry`/`execute_trade_exit` workflow: it must persist the client ID before
submission and retain the exact remaining quantity without strategy stake sizing
or exit vetoes.
Cancellation also requires authoritative fetched results; helpers that synthesize
fallback cancellation results cannot establish the final fill quantity.
For linked trades, wallet differences must be resolved by identifiable fills;
the ordinary small-balance adjustment is disabled to prevent accidental top-ups.

### Source and regression-test map

Paths below are relative to the repository root.

| Area | Main files |
| --- | --- |
| Shared lifecycle and bot integration | `freqtrade/hedging.py`, `freqtrade/freqtradebot.py` |
| Strategy contract and triggers | `freqtrade/strategy/interface.py`, `user_data/strategies/dca_strategy_helpers.py`, `user_data/strategies/market_structure_trend_matrix_strategy.py` |
| Persistence and initialization | `freqtrade/persistence/hedge_group.py`, `freqtrade/persistence/models.py`, `freqtrade/persistence/migrations.py`, `freqtrade/persistence/db_migration.py` |
| Exchange execution and side-aware balances | `freqtrade/exchange/okx.py`, `freqtrade/exchange/exchange.py`, `freqtrade/wallets.py` |
| Configuration and remote control | `freqtrade/config_schema/config_schema.py`, `freqtrade/configuration/config_validation.py`, `freqtrade/rpc/rpc.py`, `freqtrade/rpc/telegram.py`, `freqtrade/rpc/api_server/api_schemas.py` |
| Lifecycle, recovery, and shared callbacks | `tests/test_dca_hedge.py`, `tests/test_strategy_hedge.py`, `tests/test_dca_hedge_config.py` |
| Schema and database conversion | `tests/persistence/test_hedge_schema.py`, `tests/persistence/test_migrations.py`, `tests/persistence/test_db_migration.py` |
| Strategy, exchange, wallets, and Telegram | `tests/strategy/test_market_structure_trend_matrix_strategy.py`, `tests/exchange/test_okx.py`, `tests/test_wallets.py`, `tests/rpc/test_rpc_telegram.py` |

The DCA-named lifecycle tests also exercise the shared controller. Regression
coverage includes both legs in `opening` and `held` states when small unexplained
wallet differences occur, as well as callback failure, missing quotes, unfilled
parent entries, cancellation races, partial fills, and restart recovery.

## Initial implementation limits

Live and dry-run execution are supported. Backtesting and hyperopt reject an
enabled hedge configuration because their one-position accounting does not model
these groups. Dry-run hedge liquidation prices are deliberately unmodeled; exchange
margin and liquidation behavior must be checked in OKX demo trading before live use.
Dry-run chase uses this fork's existing limit-fill simulation; it does not reproduce
OKX's continuous repricing, queue priority, or partial-fill timing.
Existing per-leg profit reports remain separate; combined group reporting is not
added in this version. An equal-quantity hedge offsets subsequent directional
exposure; it does not erase the loss accumulated before the hedge opened.

The exchange-specific methods are isolated in the OKX adapter. Supporting another
exchange requires its hedge-mode validation, side-specific order semantics,
contract conversion and reliable client-ID recovery to be implemented and tested.

## Validation

The 2026-09-16 regression run passed **1,324 tests**, covering persistence, bot
behavior, RPC/Telegram/HTTP API, OKX, wallets, configuration, shared strategy
callbacks, DCA triggers, and the market-structure strategy. Coverage includes
complete schema creation on fresh and ordinary existing databases, restart
recovery, and preventing quantity changes from unexplained wallet discrepancies.
One unrelated FreqAI API test was excluded because the optional `datasieve`
dependency was unavailable. Ruff lint and formatting checks passed.

Exchange transport was mocked; no authenticated demo or live orders were placed.
The test environment used CCXT 4.5.71, while the repository pins 4.5.76.
