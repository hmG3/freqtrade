# MEXC Futures, Cross Margin, and Chase Orders Design

**Status:** Proposed

## Summary

Freqtrade will extend the existing `Mexc` exchange adapter with support for
USDT-settled linear perpetual futures in isolated and cross-margin modes. The
same adapter will support MEXC chase orders for entry and exit after a guarded
live probe confirms the exchange behavior that is missing from the public REST
schema.

MEXC chase creation is a two-step operation. Freqtrade must first create a
normal unfilled futures order and then submit that order's ID to MEXC's chase
conversion endpoint. The converted order will remain part of Freqtrade's
existing order lifecycle: persistence, polling, restart recovery, timeout and
shutdown cancellation, partial-fill reconciliation, and fee lookup.

The implementation will not create a second MEXC adapter. Spot behavior will
remain in `freqtrade/exchange/mexc.py`, and futures-specific behavior will be
added through `_ft_has_futures`, supported-mode declarations, and targeted
method overrides.

Relevant external contracts are documented by:

- [MEXC Futures API](https://www.mexc.com/api-docs/futures/introduction)
- [MEXC chase order endpoint](https://www.mexc.com/api-docs/futures/account-and-trading-endpoints/chase-order)
- [MEXC chase order FAQ](https://www.mexc.com/support/article/mexc-futures-chase-limit-order-faqs-318923114392927232)
- [CCXT 4.5.75 MEXC adapter](https://github.com/ccxt/ccxt/blob/v4.5.75/python/ccxt/mexc.py)

## Goals

- Add isolated and cross futures support to the existing `Mexc` adapter.
- Support MEXC chase orders for futures entry and exit in both margin modes.
- Fail startup unless a live futures account is already in hedge position mode.
- Reject contracts that are disabled for API trading or do not support the
  configured margin mode.
- Normalize MEXC futures balances, positions, orders, trades, fees, funding
  payments, mark prices, and risk limits into Freqtrade's expected contracts.
- Compensate safely for failures in the two-step chase activation workflow.
- Preserve existing dry-run, backtesting, hyperopt, timeout, recovery, and
  shutdown behavior.
- Verify undocumented chase semantics with a minimum-size live diagnostic before
  advertising chase support.

## Non-Goals

- A new or replacement MEXC exchange class.
- MEXC coin-settled, inverse, delivery, or non-USDT futures.
- MEXC spot margin trading.
- One-way position mode.
- Automatically changing the account's position mode.
- Portfolio margin or multi-currency shared collateral.
- Configurable chase distance, maximum chase distance, trigger price, or cap.
- Native stoploss-on-exchange support for MEXC futures in this change.
- WebSocket order management.
- A CCXT dependency update or database migration.
- Exact exchange-side chase simulation in dry-run, backtesting, or hyperopt.

## Public Configuration

### Trading modes

The primary `mexc` adapter will advertise these combinations:

- Spot with no margin mode.
- Linear futures with isolated margin.
- Linear futures with cross margin.

MEXC futures will require `stake_currency: "USDT"`. Freqtrade will continue to
use standard perpetual symbols such as `BTC/USDT:USDT`.

### Account position mode

MEXC chase orders require hedge position mode. During live futures startup, the
adapter will query the account position mode and require MEXC mode `1`.

The adapter will not switch the account automatically. A missing, malformed, or
one-way-mode response will raise an actionable `OperationalException` before
the bot can place orders. Dry-run and offline data commands will not require a
private account-mode request.

### Chase order type

`order_types.entry` and `order_types.exit` may use `"chase"` on MEXC only after
the live-probe acceptance gate in this specification has passed. Existing
validation will continue to reject chase for force orders, emergency exits,
stoploss orders, and every other order-type slot.

A chase entry or exit will require `GTC`. No `order_chase` configuration section
will be added. MEXC's REST conversion endpoint exposes only an existing order ID
and does not expose the product UI's trigger-price or maximum-distance options.

### Stoploss on exchange

The current MEXC adapter advertises stoploss-on-exchange for spot. MEXC futures
use separate plan-order endpoints whose query, trigger, and cancellation
lifecycle is not covered by the current adapter.

`_ft_has_futures` will therefore set `stoploss_on_exchange` to `False`. A futures
configuration that requests stoploss on exchange will fail validation instead
of sending a trigger order through an unverified lifecycle.

## Supported Market Boundary

For every configured futures pair, the adapter will inspect the raw MEXC market
metadata returned by the contract-detail endpoint.

A market is eligible only when:

- It is a linear perpetual swap settled in USDT.
- `state == 0`.
- `apiAllowed == true`.
- `positionOpenType == 1` for isolated-only markets, `2` for cross-only markets,
  or `3` for markets supporting both modes.
- The configured margin mode is included by `positionOpenType`.

CCXT's unified `active` field is not sufficient because the pinned MEXC parser
does not incorporate `apiAllowed`. Eligibility checks will use the raw market
`info` object and produce a pair-specific configuration error.

## Futures Capability Declaration

The MEXC futures capability set will declare:

- USDT as the supported cross-margin stake currency.
- Exchange-specific balance handling because futures `equity` includes
  unrealized PnL.
- Adapter-owned mark-price fetching.
- Adapter-owned leverage and risk-limit handling rather than CCXT's unified
  leverage tiers.
- Account-specific trading-fee loading.
- Futures stoploss-on-exchange disabled.
- Chase support only after successful live validation.

The exact `_ft_has_futures` values will be covered by adapter tests so a later
CCXT update cannot silently change the supported contract.

## Regular Futures Orders

### Request construction

All MEXC futures order requests will identify:

- `openType: 1` for isolated margin or `openType: 2` for cross margin.
- `positionMode: 1` for hedge mode.
- The requested leverage.
- The exchange's four-way side code.

The side mapping is:

| Freqtrade action | MEXC side |
| --- | ---: |
| Buy to open long | 1 |
| Buy to close short | 2 |
| Sell to open short | 3 |
| Sell to close long | 4 |

Entries use the open side. Reduce-only exits use the corresponding close side.
The adapter will not rely on CCXT's unified futures-side parser because it does
not normalize all four values reliably.

MEXC's documented order-type mapping is:

| Freqtrade type | MEXC type |
| --- | ---: |
| Limit | 1 |
| Post-only | 2 |
| IOC | 3 |
| FOK | 4 |
| Market | 5 |

The adapter will force native market type `5`; pinned CCXT maps the unified
`market` value differently. Limit-order time-in-force behavior will retain the
existing Freqtrade validation.

### Leverage preparation

The order request itself will carry MEXC's required leverage and margin mode.
The adapter will not call the base `_lev_prep()` sequence because CCXT's generic
`set_margin_mode()` contract does not match MEXC's position-specific leverage
endpoint.

MEXC `_lev_prep()` will therefore be a no-op for futures. Initial orders will
establish leverage through the create-order payload. Existing-position DCA will
retain the trade's established leverage and send that same value. The adapter
will not call MEXC's change-leverage endpoint, which has different requirements
for new and existing positions and does not support every cross-margin state.

### Order normalization

The adapter will normalize raw regular-order responses rather than trust the
incomplete unified futures parser. The normalized order will provide:

- Stable string ID.
- Symbol, type, side, and timestamp.
- Base-currency amount, filled amount, and remaining amount after contract-size
  conversion.
- Price, average execution price, and cost.
- Open, closed, canceled, or rejected status.
- Reduce-only and opening/closing intent in `info`.

MEXC order states will be mapped explicitly. Unknown non-terminal states will
remain open only when the raw response still describes an active order;
unrecognized terminal or failure states will not be treated as active.

Fetch, cancel, order-history, trade-history, startup-recovery, and shutdown paths
will all use the same normalizer.

## Chase Order Lifecycle

### Live validation gate

The public REST schema documents `POST /api/v1/private/order/chase_limit_order`
as an operation on an existing `orderId`. The newer product documentation
describes a persistent best-bid/best-ask chase order, but the REST schema does
not document:

- Whether post-only or only ordinary limit orders can be converted.
- Whether the original order ID remains authoritative.
- Whether the order continuously reprices or moves only once.
- How the product's maximum chase distance is selected for REST orders.
- Whether behavior is identical in isolated and cross mode.

The adapter will not advertise `chase_order` until the guarded live diagnostic
has confirmed these properties. If the endpoint performs only a one-time price
change, futures support will proceed but MEXC chase support will remain disabled.

### Creation

After validation establishes the accepted initial order type, creation will:

1. Create a minimum-normalized GTC futures order using the normal order path.
2. Capture the authoritative initial order ID.
3. Submit that ID to the signed chase conversion endpoint.
4. Fetch normal order detail and return a normalized chase-shaped order.

The pinned CCXT version already exposes the signed raw endpoint as
`contractPrivatePostOrderChaseLimitOrder`; no `fetch2` fallback or dependency
update is required.

### Stable identity

If the live probe confirms in-place conversion, `Order.order_id` will remain the
original MEXC order ID. Polling, cancellation, fills, trades, and fee lookup will
all use that ID.

The adapter will not synthesize `id_chase`. It will return `id_chase` only if the
live response contains a distinct executable-order ID. This differs from OKX
and Gate, where the native APIs explicitly expose separate parent and child
identities.

### Conversion race and compensation

Creation is not atomic. The initial order can fill, cancel, or become invalid
between the create and conversion requests.

If conversion fails, the adapter will fetch the initial order immediately:

- A fully or partially filled terminal order will be returned with its
  authoritative execution data so Freqtrade records the exposure.
- A partially filled active order will be canceled, fetched again, and returned
  with the authoritative partial fill.
- An unfilled active order will be canceled before the conversion error is
  raised.
- If final state cannot be established after cancellation, creation will raise
  a temporary error that preserves the order ID in the message and logs the raw
  exchange response for reconciliation.

This compensation prevents a successfully created exchange order from being
discarded by Freqtrade merely because chase activation failed.

### Fetch and cancel

MEXC exposes no separate chase-detail or chase-cancel REST endpoint. After
successful in-place conversion:

- `fetch_chase_order()` will call the normal order-detail endpoint and force the
  normalized type to `chase`.
- `cancel_chase_order()` will use normal order cancellation and then fetch the
  authoritative detail.
- Deal details and fee lookup will use the stable normal order ID.

Existing unfilled-timeout, startup recovery, shutdown, and RPC cancellation
paths will route through these hooks using the persisted order type.

### Simulation

Dry-run, backtesting, and hyperopt will treat `chase` as a limit order at
Freqtrade's calculated reference rate. The existing limit fill model will apply
to long and short entries and exits.

Simulation will not model the initial conversion request, continuous
best-price movement, exchange-side maximum distance, or conversion races.
Freqtrade's bot-side limit replacement logic will remain disabled for chase
orders because live repricing belongs to MEXC.

## Balance and Position Normalization

### Balances

For futures assets, the adapter will construct unified balances from the raw
account-assets endpoint:

- `free = availableBalance`.
- `total = equity`.
- `used = total - free`.

The futures capability will declare that total includes unrealized PnL. Wallet
synchronization will therefore remove normalized position unrealized PnL before
recombining cross collateral.

Negative or internally inconsistent raw components will be retained only when
they represent the exchange response; small floating-point residuals may be
clamped to zero. Missing required numeric fields will raise a temporary exchange
error rather than silently produce a zero wallet.

### Positions

The adapter will normalize MEXC's raw position fields, including:

- `positionId` as the position ID.
- `positionType` as long or short.
- `holdVol` as contract quantity.
- `openType` as isolated or cross.
- `openAvgPrice`, leverage, initial margin, maintenance margin, unrealized PnL,
  and liquidation price.
- Contract size and symbol from the loaded market.

Zero-size rows may remain in the raw response but will normalize consistently so
Freqtrade's wallet layer can discard non-open positions.

In live mode, a valid exchange-reported liquidation price remains authoritative.
If MEXC omits a cross liquidation price, the adapter will return `None` and emit
a rate-limited warning. It will not substitute a locally estimated live price.

## Risk Limits and Liquidation

### Risk-limit source

CCXT's unified MEXC leverage-tier parser treats contract-volume thresholds as
quote-notional tiers and ignores current custom risk-level structures. The
adapter will disable those unified capabilities and parse raw MEXC contract and
account risk-limit data.

The internal MEXC risk-tier representation will retain:

- Minimum and maximum contract volume.
- Maintenance margin rate.
- Initial margin rate.
- Maximum leverage.
- Risk-level identity when supplied.

Account-specific maximum leverage from the fee/risk endpoints will cap the
public contract maximum.

### Maximum leverage

Freqtrade's internal `get_max_leverage()` contract will gain an optional
reference-rate argument. Existing adapters may ignore it. Live entry creation
will pass the calculated entry reference rate, while backtesting will pass the
current candle's proposed entry rate.

MEXC will use stake amount, reference rate, contract size, and each candidate
leverage to calculate contract volume and select the highest permitted risk
tier. The result will also be capped by the account and market maximums. Utility
callers that do not have a rate will fetch one in live operation; a zero stake
may use the first tier's capped maximum.

The calculation will not relabel contract volume as quote notional. If a safe
tier cannot be selected, leverage will fail closed instead of falling back to
the market's headline maximum.

### Simulated liquidation

MEXC-specific dry-run liquidation code will implement the documented isolated
and cross formulas with contract quantity, contract size, maintenance margin
rate, entry price, and the configured liquidation fee where required.

For cross mode, the calculation will include all bot-owned open trades sharing
the USDT wallet:

- Wallet collateral.
- Other positions' unrealized PnL.
- Other positions' maintenance requirements.
- The target position's current risk tier.

Dry-run will use one current fair-price snapshot for other positions.
Backtesting and hyperopt will use their entry prices, producing deterministic
zero unrealized PnL for the other positions at recalculation time. The existing
liquidation buffer will be applied after the raw liquidation price is obtained.

## Mark Prices, Funding, and Fees

### Mark prices

CCXT's parsed MEXC futures ticker does not reliably populate unified mark and
index prices. `fetch_mark_prices()` will call the raw futures ticker endpoint and
map `fairPrice` to `markPrice`, filtering to requested symbols when supplied.

### Funding payments

Live funding accounting will use the private
`GET /position/funding_records` endpoint. The adapter will:

- Supply the pair and position direction when available.
- Translate Freqtrade's opening timestamp to MEXC `start_time`.
- Paginate with `page_num` and `page_size` until the requested interval is
  complete.
- Normalize payment currency and timestamp.
- Sum only records belonging to the requested contract and position direction.

Historical simulation will continue to use mark-price and funding-rate candles
through Freqtrade's existing calculation path.

### Trading fees

The adapter will fetch actual account rates from
`GET /account/tiered_fee_rate/v2`. It will cache `realMakerFee` and
`realTakerFee` by configured futures pair and fall back to the corresponding
documented original rates only when actual rates are absent.

Raw futures trades will be normalized with their order ID, direction, amount,
price, cost, maker/taker role, and fee. When MEXC omits a fill fee, the cached
account rate will provide the same fallback pattern used by other futures
adapters.

## Error Handling

- Private startup validation will map authentication and permission failures to
  actionable startup errors.
- API-disabled and margin-mode-incompatible markets will fail configuration
  validation before order placement.
- MEXC response envelopes will require `success == true` and `code == 0`.
- Chase error codes for unsupported mode, filled/canceled source order,
  insufficient assets, liquidation risk, distance limits, account/pair chase
  limits, invalid chase state, and timeout will map to the existing Freqtrade
  exception hierarchy.
- Order-not-found results will remain retryable where eventual consistency is
  plausible.
- Unknown API failures will retain the raw code and message.
- Logs and probe output will never include API keys, secrets, signatures, or
  complete request headers.

## Guarded Live Diagnostic

The standalone `scripts/mexc_chase_probe.py` diagnostic will use an existing
Freqtrade configuration and the pinned CCXT dependency. It will require:

- An explicit pair.
- `isolated` or `cross` margin mode.
- An explicit `--confirm-live-order` acknowledgement.
- A futures-enabled MEXC API key with order permission.
- An account already in hedge position mode.

The diagnostic will not change account position mode. It will:

1. Load and validate the selected market metadata.
2. Query and verify hedge position mode.
3. Determine the minimum valid contract quantity.
4. Place the safest non-marketable initial order supported by the conversion
   endpoint.
5. Convert the order to chase.
6. Poll normal order detail and top-of-book prices long enough to observe more
   than one reprice when the market moves.
7. Record whether the ID changes, whether partial fills remain attached, and
   whether normal fetch and cancel operations remain authoritative.
8. Cancel any remaining quantity in a `finally` block.
9. Print redacted raw response shapes suitable for test fixtures.

The probe will run once in isolated mode and once in cross mode. A minimum-size
fill remains possible because MEXC does not provide a futures sandbox; this risk
will be stated before confirmation.

Chase support passes the gate only if both modes demonstrate:

- Persistent repricing rather than a one-time price move.
- A deterministic authoritative order ID.
- Normal detail and cancellation compatibility.
- Correct partial/full-fill reporting.
- No requirement for an undisclosed distance parameter.

## Testing Strategy

### Adapter tests

A new `tests/exchange/test_mexc.py` will cover:

- Spot capability preservation.
- Isolated and cross supported-mode declarations.
- Hedge-mode startup success and failure.
- Market `state`, `apiAllowed`, and `positionOpenType` validation.
- Exact regular-order payloads for long/short entry and exit.
- Native market-order type and contract conversion.
- Regular order, cancellation, trade, balance, and position normalization.
- Actual fee retrieval and fallback behavior.
- Funding-history filtering and pagination.
- Fair-price normalization.
- Risk-tier selection and maximum leverage.
- Isolated and cross dry-run liquidation formulas.

### Chase tests

Fixtures captured by the redacted live probe will cover:

- Initial order creation and conversion payloads.
- Stable or distinct ID behavior established by the probe.
- Open, full fill, partial fill, cancellation, rejection, and timeout.
- Conversion success with an empty acknowledgement payload.
- Fill during the create/convert race.
- Partial fill followed by conversion failure and compensating cancellation.
- Unfilled conversion failure and compensating cancellation.
- Failure to establish final state without losing the exchange order ID.
- Restart recovery, timeout cancellation, shutdown cancellation, and fee lookup.
- Absence of Freqtrade limit-order replacement for chase orders.

### Integration and regression tests

Existing configuration, RPC, bot lifecycle, wallet, liquidation, and
backtesting tests will be extended for MEXC where behavior is exchange-specific.
The implementation will run focused MEXC tests first, then the affected broader
suites, formatting, lint, and type checks.

## Documentation Changes

User documentation will describe:

- MEXC isolated and cross futures support for USDT linear perpetuals.
- The hedge-position-mode requirement and fail-closed startup behavior.
- Contract-level `apiAllowed` and margin-mode restrictions.
- Chase availability for entry and exit only, with GTC.
- The two-step conversion race and exchange-owned repricing.
- The absence of configurable distance and cap fields in the REST API.
- Limit-order simulation behavior.
- Disabled futures stoploss-on-exchange support.
- Live cross liquidation behavior when MEXC omits its estimate.
- The requirement to dedicate the leveraged account or subaccount to one bot.

## Acceptance Criteria

### Futures foundation

- Existing MEXC spot behavior remains unchanged.
- Isolated and cross futures configurations initialize only for USDT linear
  perpetuals and hedge-mode accounts.
- One-way mode, unsupported settlement, API-disabled contracts, and incompatible
  `positionOpenType` values fail before trading.
- Long and short regular entry and reduce-only exit payloads use correct MEXC
  sides, margin mode, position mode, leverage, amount, and order type.
- Normalized balances and positions keep Freqtrade wallet ownership accurate.
- Fees and funding payments reconcile against MEXC account records.
- Leverage and simulated liquidation use contract-volume risk tiers without
  treating them as quote-notional tiers.
- Live and backtesting leverage selection supply a reference rate so MEXC can
  enforce volume-based risk tiers before order placement.

### Chase behavior

- The live probe passes independently in isolated and cross mode.
- Chase creation never loses an initial order that fills during conversion.
- Failed conversion cancels any remaining active initial order.
- Polling and cancellation use the authoritative ID established by the probe.
- Partial fills survive terminal cancellation and rejection paths.
- Startup recovery, timeout, shutdown, RPC cancellation, and fee lookup operate
  through the persisted chase order type.
- No bot-side repricing or replacement callback runs for a live chase order.
- Dry-run, backtesting, and hyperopt use existing limit-order behavior.

### Safety

- The adapter never changes account position mode automatically.
- Futures stoploss-on-exchange is rejected until separately implemented.
- Unknown market, account, risk, or order states fail closed.
- Probe and runtime logs contain no credentials or signatures.
- If persistent REST chase semantics cannot be demonstrated, MEXC futures ship
  without the `chase_order` capability.

## Delivery Constraints

- Work will continue on the current `develop` branch.
- The existing `freqtrade/exchange/mexc.py` adapter will be extended in place.
- Existing unrelated modified and untracked files will be preserved.
- The implementation will not update CCXT or migrate the database.
- Account position mode will be validated but never changed.
- The live diagnostic requires explicit user confirmation for every run.
- Manual positions and additional bots on the same futures account remain
  outside Freqtrade's supported cross-collateral model.
