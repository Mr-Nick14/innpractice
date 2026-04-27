# Monte Carlo Model Artifacts

Canonical runtime parameters for `cashflow/monte_carlo`.

Files in this folder are loaded by default:
- `geo_model_Moscow.pkl`
- `quality_model_Moscow.pkl`
- `pca_geo_Moscow.pkl`
- `h3_encoder_Moscow.pkl`
- `lgb_residual_model_Moscow.pkl`
- `final_model_Moscow.pkl`
- `project_premium_Moscow.parquet`
- `market_index_Moscow.parquet`
  - if `city_log_price_median` is present, rows where it is `NaN` are treated as a diagnostic forward tail and ignored by runtime market anchoring
- `geo_coefs_Moscow.csv`
- `nb_sales_models.csv`

Runtime lookup prefers this folder first.
Legacy fallback remains available in `old_vers/cashflow_project/data/model_artifacts/`
for compatibility with older notebooks and scripts.
