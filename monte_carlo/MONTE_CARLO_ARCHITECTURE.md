# Monte Carlo Architecture for Residential Development Cashflow

This document describes the current Monte Carlo stack in the project and is meant
for manual logic review.

The implementation is inference-only:
- no training inside Monte Carlo
- all learned objects are loaded from artifacts
- deterministic baseline paths remain supported
- stochastic drivers are layered on top as runtime modulations

## 1. Big Picture

The engine simulates one residential development project quarter by quarter.
At each quarter it computes:

1. market price path
2. hedonic project price per sqm
3. sales count
4. revenue and escrow movement
5. costs
6. debt draw / repayment / interest
7. ratios such as DSCR and ISCR

The cashflow engine is deterministic if you do not attach a driver scenario and
do not inject random local shocks. When a driver scenario is attached, it
modulates the path, but still does not train anything.

## 2. Main Modules

### 2.1 `mc_cashflow_engine.py`

Contains the core project finance engine:
- `ProjectConfig`
- `CostSchedule`
- `GlobalScenario`
- `LocalScenarioPath`
- `Scenario`
- `QuarterContext`
- `QuarterState`
- `PathSummary`
- `PriceModel`
- `SalesModel`
- `StubPriceModel`
- `StubSalesModel`
- `FittedNBFeatureSalesModel`
- `OnePathCashflowEngine`
- `ScenarioFactory`
- `MonteCarloRunner`

This is the main cashflow loop.

### 2.2 `hedonic_runtime.py`

Contains the inference-only hedonic stack:
- `ProjectSimulationInput`
- `HedonicArtifacts`
- `GeoFeatureBuilder`
- `QualityFeatureBuilder`
- `SpatialContextBuilder`
- `HedonicFeatureBuilder`
- `MarketPathModel`
- `HedonicPriceModel`

This layer loads trained artifacts and produces price-per-sqm for each quarter.

### 2.3 `mc_stochastic_drivers.py`

Contains the runtime stochastic driver layer:
- `MonteCarloDriverConfig`
- `PathDriverState`
- `QuarterDriverState`
- `DriverScenario`
- `DriverScenarioFactory`
- `RuntimeCostSchedule`
- `RuntimeCostAdapter`

This layer provides path-level and quarter-level modulations for:
- market
- sales
- timing
- costs
- financing

## 3. Data Inputs

### 3.1 Project input

The hedonic runtime starts from one project payload:
- project id / name
- region key
- coordinates
- class / construction type / finishing
- lots, sellable area, average unit area
- start sales quarter
- planned RVZ / delivery quarter
- optional representative unit
- optional known geo / quality features

### 3.2 Market reference data

Used by the runtime market block and kNN features.

### 3.3 Hedonic artifacts

Loaded from disk only:
- geo model
- quality model
- PCA / scaler for geo features
- H3 encoder
- residual LightGBM model or fallback final model pipeline
- project premium table
- market index table

### 3.4 Cost and finance assumptions

These come from `ProjectConfig` and `CostSchedule`.

## 4. Layering of Uncertainty

The current architecture separates uncertainty into four levels:

### 4.1 Legacy scenario layer

This is the older `Scenario` structure:
- `GlobalScenario`
- `LocalScenarioPath`
- `ScenarioFactory`

It is still supported for backward compatibility.

### 4.2 Driver path layer

Path-level effects are fixed for the whole Monte Carlo path:
- market regime modulation
- macro modulation
- project premium modulation
- demand regime modulation
- project overrun multiplier
- RVZ delay
- cost inflation regime
- spread modulation
- key-rate modulation

### 4.3 Driver quarter layer

Quarter-level shocks vary by quarter:
- market shock
- price residual shock
- seasonal sales shock
- demand shock
- cost shock
- rate innovation
- spread shock

### 4.4 Runtime state layer

The engine stores quarter-wise realized values in `QuarterState` so each path can
be inspected after the run.

## 5. Market Block

The market block produces the market log-price path.

### 5.1 Modes

`MarketPathModel` supports:

- `replay`
  - deterministic lookup from `market_index_{REGION}.parquet`
- `trend`
  - extrapolation from the last fitted trend
- `stochastic`
  - AR-like quarterly evolution with shocks
- `scenario_stochastic`
  - stochastic market path with explicit scenario modulations
- `macro_guided`
  - trend plus macro-aware runtime modulation

### 5.2 What is modeled

Market price path is represented in log-space:

`market_log_price_t`

This is the main systematic price driver.

### 5.3 What is not modeled

No training happens in Monte Carlo.
No future market data is inferred from future simulated project sales.

### 5.4 Important implementation note

If market mode is `replay` or `trend`, driver-level market perturbations are
ignored. This keeps the market block deterministic in those modes.

## 6. Hedonic Price Block

The current hedonic formula is:

```text
final_log_price_sqm_t =
    market_log_price_t
    + geo_score_t
    + quality_score_t
    + project_premium_base
    + project_premium_path_modulation
    + residual_ml_boost_t
    + final_price_residual_shock_t
```

Then:

```text
price_sqm_t = exp(final_log_price_sqm_t)
```

### 6.1 Geo score

Computed from:
- distances
- log-distances
- accessibility features
- PCA features
- H3-related features if available

The geo model is inference-only and is loaded from artifacts.

### 6.2 Quality score

Computed from:
- area
- floor
- floor relative position
- rooms
- ceiling height
- project age
- months to RVZ
- lot counts
- class / construction / finishing

This is also inference-only.

### 6.3 Residual ML boost

The LightGBM residual model is kept in the stack.
It consumes a runtime feature frame built from:
- geo features
- quality features
- spatial context features
- H3 encodings
- kNN historical features

If the residual model is missing, a fallback pipeline can be used.

### 6.4 Premium modulation

The project premium comes from the premium table when available.
If the project is missing from the table, fallback is `0.0`.

The runtime driver can also add a path-level premium modulation.

## 7. Sales Block

The sales model is inference-only.

### 7.1 Baseline

The fitted NB sales model estimates a baseline expected count `mu`.

### 7.2 Runtime modulations

The sales block then applies:
- demand modulation
- seasonal quarter shock
- delay penalty from RVZ delay
- optional legacy local sales shock

### 7.3 Final realization

The final mean is transformed to a count distribution.
The model can sample a count stochastically, or return a rounded value in
deterministic mode.

### 7.4 Price-to-market ratio

The sales model can consume:
- current project price
- current market price
- price-to-market ratio
- remaining inventory

### 7.5 Lot area handling

Current cashflow logic uses a single average unit area:

`sold_area_sqm = sold_lots * avg_unit_area_sqm`

This is the implemented production simplification.

No full unit-mix revenue split is used yet.

## 8. Timing Block

The timing block currently contains:
- RVZ delay
- cost timing shift for the cost schedule

The active timing effect in the engine is the RVZ delay.

The cost timing adapter can move the cost curve forward or backward by quarters.

There is also a reserved sales start timing field in the driver state for
compatibility, but it is not wired into the current runtime flow.

## 9. Cost Block

The cost block takes the deterministic cost schedule and turns it into an
effective runtime cost schedule.

### 9.1 Base cost schedule

From `CostSchedule`:
- land
- SMR
- design
- other OPEX
- post-completion cost
- property tax
- VAT

### 9.2 Runtime modulations

The runtime adapter can apply:
- construction cost inflation path
- project overrun multiplier
- quarterly cost shock
- timing shift of the cost curve

### 9.3 Effective cost formula

```text
effective_cost_t =
    base_cost_t
    * cost_index_t
    * overrun_multiplier
```

The cost index is itself a path of multiplicative adjustments.

## 10. Financing Block

The financing block uses:
- base key rate
- key-rate modulation
- spread modulation
- delay impact via longer debt duration

### 10.1 Effective rate

```text
full_rate_t =
    key_rate_t
    + spread_t
```

### 10.2 Debt and fees

Each quarter the engine computes:
- operating debt draw
- financing debt draw
- interest payment
- usage fee
- reserve fee
- repayment

### 10.3 Delay linkage

If RVZ is delayed, the project carries debt longer.
This automatically increases interest burden and can worsen DSCR.

## 11. Quarterly Execution Order

For each quarter:

1. build quarter context
2. generate market log-price
3. build time-varying quality features
4. compute geo score
5. compute quality score
6. compute residual ML boost
7. add premium modulation and residual shock
8. convert to final price per sqm
9. pass price and market price into sales model
10. realize sales count
11. convert sold lots into sold area
12. compute revenue
13. route revenue through escrow rules
14. load and modulate costs
15. compute debt draw and financing cost
16. compute DSCR / ISCR / net cash flow
17. persist full diagnostic quarter state

## 12. Diagnostics

The engine stores diagnostic values in `QuarterState`, including:
- market log-price
- geo score
- quality score
- project premium components
- residual ML boost
- final log-price
- market shock
- demand modulation
- seasonal sales shock
- delay penalty
- realized sales count
- cost index
- overrun
- quarterly cost shock
- effective key rate
- spread shock
- final full rate

This is the main place to inspect what actually happened in one quarter.

`PathSummary` also stores a path-level driver summary.

## 13. Legacy Compatibility

The old scenario layer is still kept.
This means the following can coexist:
- deterministic scenario multipliers
- local random price / sales shocks
- new driver layer
- hedonic inference stack

The new architecture does not retrain anything and does not require removing the
legacy scenario layer.

## 14. Current Simplifications

These are accepted simplifications in the current implementation:

- one average unit area per project
- kNN built from historical reference deals only
- H3 features are static by project coordinates
- geo score can be cached when geo features are static
- project premium is a project-level constant
- market replay ignores driver market perturbations
- no future-neighbor simulation for kNN

## 15. How to Verify Logic

When reviewing a run, inspect the following in order:

1. `MarketPathModel` output
2. hedonic price decomposition
3. sales mean and count realization
4. cost schedule after runtime adaptation
5. key rate and spread path
6. debt draw / repayment
7. DSCR and ISCR
8. path summary

If one layer looks wrong, check the corresponding component in
`QuarterState.notes`.

## 16. Terminology Note

Some code fields still use legacy `shift` names for backward compatibility.
Architecturally, they should be read as:

- modulation
- deviation
- residual shock
- timing delay

The runtime logic is organized around modulation, not around a single global
multiplier.

