# Native Chase Orders and Cross-Margin Futures Design

**Status:** Proposed

## Summary

Freqtrade will support native best-price chase orders on OKX and Gate for
USDT-settled linear perpetual futures in both isolated and cross-margin modes.

A chase order will continuously follow the exchange's best bid or ask without a
configurable distance or maximum-price cap. Repricing will be managed entirely
by the exchange. Freqtrade will retain responsibility for order persistence,
polling, timeout cancellation, restart recovery, shutdown handling, trade and
fee reconciliation, and final order-state normalization.

Cross mode will apply to regular and chase orders. It will use a
single-currency shared-collateral model comparable to Freqtrade's existing
cross-futures support. Multi-currency and portfolio-margin risk models will
remain unsupported.

Relevant exchange contracts are documented by the
[OKX API](https://www.okx.com/docs-v5/) and
[Gate Futures API](https://www.gate.com/docs/developers/apiv4/en/futures/).

## Goals

- Add `chase` as a native entry and exit order type for OKX and Gate futures.
- Support both `isolated` and `cross` margin for regular and chase orders on the
  primary OKX and Gate adapters.
- Preserve the exchange parent order and executable child order identities.
- Reuse the existing Freqtrade order lifecycle without bot-side repricing.
- Provide deterministic dry-run, backtesting, and hyperopt behavior.
- Calculate shared-wallet liquidation prices for supported cross simulations
  and as a live fallback when the exchange omits a liquidation price.
- Reject account and market configurations whose risk model Freqtrade cannot
  represent safely.

## Non-Goals

- Configurable chase distance, price gap, or maximum chase price.
- Chase support for force-entry, force-exit, emergency-exit, or stoploss slots.
- Spot, margin, inverse futures, delivery futures, or non-USDT settlement.
- Gate multi-currency or portfolio-margin accounts.
- OKX multi-currency or portfolio-margin accounts.
- Chase or futures support for GateEU, MyOKX, or OKX US.
- A CCXT dependency update or database migration.
- Reproduction of exchange-side chase repricing in simulated modes.

## Public Configuration

### Order types

`order_types.entry` and `order_types.exit` will accept `"chase"` in addition to
their existing values. Other order-type slots will remain restricted to their
current `limit` and `market` values.

Entry and exit order types will use an entry/exit-specific enum. RPC force-order
payloads will continue to use an enum that excludes chase, preventing force
commands from creating a native chase order.

No `order_chase` configuration section will be introduced. Native chase orders
will always use the current best bid or ask with no distance and no maximum
price cap.

### Time in force

A chase entry or exit will require `GTC`. Validation will reject a chase slot
whose corresponding `order_time_in_force` value is not `GTC`.

This restriction exists because the exchange maintains its own persistent
post-only child order. IOC, FOK, and other immediate execution policies are not
compatible with that lifecycle.

### Trading and margin modes

The primary `okx` and `gate` adapters will advertise these combinations:

- Spot with no margin mode.
- Linear futures with isolated margin.
- Linear futures with cross margin.

Cross mode on these adapters will require `stake_currency: "USDT"`. Existing
Freqtrade market filtering will continue to admit only linear perpetual swaps
and reject inverse contracts.

The `gateeu`, `myokx`, and `okxus` adapters will remain spot-only and will not
inherit futures, cross-margin, or chase capabilities.

## Shared Chase Order Lifecycle

The exchange interface will expose these hooks:

```python
def create_chase_order(
    self,
    pair: str,
    side: BuySell,
    amount: float,
    rate: float | None,
    params: dict,
) -> CcxtOrder: ...

def fetch_chase_order(self, order_id: str, pair: str) -> CcxtOrder: ...

def cancel_chase_order(self, order_id: str, pair: str) -> CcxtOrder: ...
```

The existing order creation, polling, cancellation, startup recovery, timeout,
and shutdown paths will select the native chase hooks from the persisted order
type. Chase orders will not use Freqtrade's limit-order replacement or price
adjustment callbacks because the exchange performs repricing.

Existing unfilled-timeout logic will remain active. A timeout will cancel the
exchange parent chase order and reconcile any partial fill returned by the
detail endpoint.

### Parent and child identity

The exchange parent chase ID will be persisted as `Order.order_id`. The
currently executable child-order ID will be returned as `id_chase` in the
normalized CCXT-shaped order.

Trade and fee lookup will use `id_chase` when it is present. Persistence and
subsequent parent-order polling will continue to use `Order.order_id`. This
separation avoids a database migration and prevents a child replacement from
losing the stable parent identity.

### Normalized order contract

Creation will return a normalized open order using Freqtrade's calculated
reference rate. The exchange detail endpoint will subsequently become
authoritative for:

- Filled and remaining amount.
- Average execution price and cost.
- Executable child ID.
- Open, closed, canceled, or rejected status.
- Partial fills associated with terminal outcomes.

## OKX Design

### Chase creation

OKX chase creation will call the algo-order endpoint with:

```json
{
  "instId": "<exchange contract id>",
  "tdMode": "isolated | cross",
  "side": "buy | sell",
  "posSide": "net | long | short",
  "ordType": "chase",
  "sz": "<contract quantity>",
  "chaseType": "distance",
  "chaseVal": "0"
}
```

`reduceOnly: true` will be added when required for an exit. Maximum-chase fields
will be omitted.

The returned `algoId` will be the stable parent order ID. The executable
`ordId`, when available, will be normalized as `id_chase`.

### Fetch and cancellation

Order details will be fetched by `algoId`. Cancellation will use
`cancel-algos` with the parent ID and instrument ID, followed by detail
reconciliation.

Live states will normalize to `open`. Effective terminal states will normalize
to `closed`, cancellation states to `canceled`, and explicit failures to
`rejected`. Partial fills will be retained regardless of the terminal status.

### Cross account validation

Live startup already obtains the OKX account configuration through
`fetch_accounts()`. When Freqtrade is configured for cross futures, the adapter
will require `acctLv == "2"`, representing OKX Futures mode.

An empty response, a missing account level, multi-currency mode, or portfolio
margin mode will fail startup with an actionable `OperationalException`.
Isolated behavior will remain compatible with the currently supported OKX
account configurations.

The existing `posMode` inspection will continue to determine whether OKX uses
net or long/short position mode.

### Cross order and leverage parameters

Regular and chase orders will use `tdMode: "cross"`. Leverage setup and
verification will use `mgnMode: "cross"` together with the position side
required by the configured OKX position mode.

## Gate Design

### Chase creation

Gate chase endpoints will be called through CCXT's generic signed `fetch2`
path because the pinned Gate adapter does not declare dedicated unified chase
methods.

Creation will send:

```json
{
  "settle": "usdt",
  "contract": "<exchange contract id>",
  "amount": "<signed contract quantity>",
  "price_limit": "0",
  "reduce_only": false,
  "price_type": 1
}
```

Buy quantities will be positive and sell quantities negative.
`reduce_only` will reflect the Freqtrade exit request.
`pos_margin_mode: "cross"` will be included in cross mode. Isolated requests
will preserve the existing payload and rely on the configured position mode.

Distance, gap, cap, and optional hedge-position fields will be omitted.

### Fetch and cancellation

Details will be fetched through `chase/detail`, and cancellation will use
`chase/stop`. The parent `id` will remain the persisted order ID, while
`suborder_id` will be normalized as `id_chase`.

State normalization will follow these rules:

- `open` becomes `open`.
- A fully filled `finished` order becomes `closed`.
- A partial or empty terminal cancellation becomes `canceled`.
- An explicit zero-fill failure becomes `rejected`.
- Partial fills remain attached to canceled or rejected terminal results.

### Cross account validation

For live cross futures, startup will query Gate's signed futures account
endpoint for the configured settlement currency. It will accept:

- `margin_mode == 0`: classic futures account.
- `margin_mode == 3`: unified single-currency account.

It will reject:

- `margin_mode == 1`: multi-currency margin.
- `margin_mode == 2`: portfolio margin.
- A missing, malformed, or otherwise unverifiable mode response.

The existing Gate unified-account detection will remain in place for balance
endpoint selection.

### Cross leverage parameters

Gate leverage preparation will pass `marginMode: "cross"` explicitly to
CCXT's `set_leverage()` call. With the pinned CCXT adapter, this selects cross
mode by sending `leverage: "0"` and the requested leverage as
`cross_leverage_limit`.

Regular orders will use the position mode established by leverage preparation.
Native chase creation will additionally identify cross mode through
`pos_margin_mode`.

## Cross-Margin Liquidation Model

### Source of liquidation prices

Live trading will prefer the liquidation price reported by the exchange's
position endpoint. This price reflects exchange state and remains authoritative
when present.

OKX can omit an estimated liquidation price when several cross positions share
account risk. Gate may also return an empty estimate. The OKX and Gate adapters
will therefore advertise an internal capability allowing the shared local
cross formula to act as a live fallback.

Dry-run, backtesting, and hyperopt will always use the local formula.

### Wallet collateral

`update_liquidation_prices()` will obtain `wallets.get_collateral()` for cross
mode in both live and simulated operation and pass it to each position
calculation.

The existing wallet normalization will remain unchanged:

- OKX balance totals include unrealized PnL, so wallet synchronization removes
  position unrealized PnL before collateral is recombined.
- Gate balance totals represent wallet balance without unrealized PnL, so no
  stripping flag will be added.

### Mark prices

A shared retried `fetch_funding_rates(symbols)` exchange wrapper will expose
CCXT's batch funding-rate response and its mark prices. Binance's existing
override will remain unchanged.

Live fallback and dry-run will use current mark prices for other positions.
Backtesting and hyperopt will use each other trade's entry price, giving zero
unrealized PnL at recalculation time. This matches the existing deterministic
Binance cross simulation and avoids introducing synchronized multi-pair mark
state into the backtester.

### Formula

For each other open position, Freqtrade will calculate:

```text
signed_unrealized_pnl =
    direction * amount * (mark_price - entry_price)

maintenance_margin =
    amount * mark_price * (maintenance_rate + taker_fee)
    - maintenance_amount
```

`direction` will be `+1` for a long and `-1` for a short. A missing maintenance
amount will be treated as zero. Maintenance tiers will be selected using
position notional rather than collateral.

For the target position, let `side` be `+1` for a long and `-1` for a short.
The conditional liquidation price will be:

```text
liquidation_price =
    (
        wallet_balance
        + other_unrealized_pnl
        - other_maintenance_margin
        + target_maintenance_amount
        - side * amount * entry_price
    )
    /
    (
        amount * (target_maintenance_rate + target_taker_fee - side)
    )
```

The calculation will assume that all non-target mark prices remain fixed. The
existing liquidation buffer will be applied after either an exchange or local
price is obtained, and the final buffered value will remain bounded at zero.

Binance and Hyperliquid will keep their exchange-specific liquidation formulas.

## Simulation Behavior

Dry-run, backtesting, and hyperopt will treat a chase order as a limit order at
Freqtrade's calculated reference rate. Existing limit-order fill rules will
apply to long and short entries and exits.

The simulation will not reproduce exchange-side best-price selection,
post-only child replacement, or continuous repricing. Strategy custom-price and
order-adjustment callbacks will not run for chase orders in live mode because
the exchange owns repricing.

## Error Handling

- Configuration errors will reject unsupported order slots, time-in-force
  values, exchanges, regional adapters, trading modes, settlement currencies,
  and contract types before trading starts.
- Live account-mode validation will fail closed when the adapter cannot prove
  that the configured cross account uses a supported single-currency model.
- Exchange authentication, rate-limit, temporary, invalid-order, and
  not-found failures will map through the existing Freqtrade exception
  hierarchy.
- Chase creation responses without a parent ID will be rejected.
- Detail responses for another contract will be rejected rather than attached
  to the requested trade.
- Terminal chase responses will retain authoritative partial fills even when
  cancellation or rejection is reported.
- Missing live liquidation prices will use the local fallback only on adapters
  that explicitly advertise the capability.

## Documentation Changes

The user documentation will describe:

- `chase` availability for OKX and Gate entry and exit orders.
- The GTC requirement and fixed best-price/no-cap behavior.
- Isolated and cross support for primary OKX and Gate adapters.
- Supported OKX and Gate account modes and the USDT linear-perpetual boundary.
- Parent and child order behavior where relevant to reconciliation.
- Limit-order simulation in dry-run, backtesting, and hyperopt.
- Cross-simulation assumptions for other positions' mark prices.
- The existing requirement that one Freqtrade bot exclusively owns a leveraged
  account or subaccount.

## Acceptance Criteria

### Configuration and interfaces

- Chase is accepted for OKX and Gate entry and exit orders in isolated and
  cross futures with GTC.
- Chase is rejected in unsupported slots, spot mode, non-GTC configurations,
  and unsupported adapters.
- Regular limit and market orders work in OKX and Gate cross futures.
- RPC force-order schemas cannot request chase orders.
- Cross mode rejects non-USDT settlement and non-linear contracts.

### OKX behavior

- Create, fetch, and cancel payloads exactly match the OKX algo-order contract.
- `tdMode` and `mgnMode` reflect isolated or cross configuration.
- Net and long/short position sides are preserved for entries and exits.
- `algoId` remains the parent ID and `ordId` becomes `id_chase`.
- Account level `2` is accepted for cross; missing, multi-currency, and
  portfolio levels are rejected.

### Gate behavior

- Create, detail, and stop requests exactly match Gate's signed chase contract.
- Contract quantity direction and reduce-only behavior are correct.
- Cross creation includes `pos_margin_mode: "cross"`.
- Cross leverage setup supplies `marginMode: "cross"` to CCXT.
- Account modes `0` and `3` are accepted; modes `1`, `2`, and unverifiable
  responses are rejected.
- Parent `id` and child `suborder_id` remain distinct.

### Order lifecycle

- Open, full-fill, partial-fill, cancellation, rejection, and API-error paths
  normalize correctly.
- Restart recovery resumes polling with the persisted parent ID.
- Timeout and shutdown cancellation target the parent chase order.
- Trade and fee lookup use the executable child ID when present.
- No bot-side limit replacement or adjustment callback runs for chase orders.

### Liquidation and simulation

- Long and short cross calculations work with one or several positions.
- Mixed directions, maintenance tiers, maintenance amounts, taker fees,
  current mark PnL, and liquidation buffers are covered.
- Live exchange liquidation prices take precedence over the fallback.
- An empty live price invokes the fallback only for OKX and Gate.
- Dry-run uses current mark prices for other positions.
- Backtesting and hyperopt use deterministic entry-price assumptions.
- Chase orders use existing limit-order fill behavior in all simulated modes.

## Delivery Constraints

- Work will remain on the current `develop` branch.
- Existing unrelated modified and untracked files will be preserved.
- The implementation will not update CCXT, migrate the database, or introduce
  new chase configuration.
- Account mode will be validated at startup. Changing account, position, or
  margin mode while the bot is running will remain unsupported.
- The leveraged account or subaccount will be dedicated to one Freqtrade bot;
  manual positions and additional bots are outside the supported risk model.
