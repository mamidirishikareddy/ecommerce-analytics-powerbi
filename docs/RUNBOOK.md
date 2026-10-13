# Product-Pulse-AI — Project Runbook

## 1. Project Overview

Product-Pulse-AI is a governed product intelligence and decision-support platform built on the Olist Brazilian E-Commerce dataset.

The project includes:
- Data quality validation and governed staging
- Dimensional warehouse and analytical marts
- Customer and product performance analytics
- Revenue decomposition and forecasting
- Customer voice and sentiment analysis
- AI-assisted decision reasoning and evidence
- Power BI reporting

## 2. Environment Setup

Run commands from the project root in the `product_pulse` Conda environment.

Requirements:
- Python 3.11
- Project dependencies installed in the active environment

Run the test suite:

```powershell
python -m pytest -q
```

**Last verified result:** 539 tests passed.

## 3. Core Pipeline Execution Order

Run the stages in dependency order. These scripts can regenerate existing artifacts, so check their outputs before rerunning them.

### Step 1 — Data Quality

```powershell
python scripts/run_current_data_quality.py
```

Inputs:
- `data/raw/`
- `configs/dq_execution.yaml`
- `configs/dq_rules.yaml`

Outputs:
- `data/processed/quality/dq_results.csv`
- `data/processed/quality/dq_summary.json`

The latest recorded run contains 44 rules: 38 passed, 6 warning-level failures, and 0 critical failures. Investigate failed rules rather than weakening validations or changing source data to force a pass.

### Step 2 — Staging

```powershell
python scripts/run_staging.py
```

Reads raw datasets and `configs/staging_mapping.yaml`.

Output directory: `data/staging/`

The last verified run staged 9 datasets.

### Step 3 — Warehouse

```powershell
python scripts/run_warehouse.py
```

Reads staged datasets and `configs/warehouse_mapping.yaml`.

Output directory: `data/processed/warehouse/`

The last verified run persisted 10 warehouse tables.

### Step 4 — Analytical Marts

Run the following commands after the warehouse is available:

```powershell
python scripts/run_customer_performance_mart.py
python scripts/run_market_performance_mart.py
python scripts/run_product_performance_mart.py
python scripts/run_product_market_performance_monthly.py
```

Output directory: `data/processed/marts/`

The mart scripts validate their expected grain. The last verified runs reported zero duplicate-grain rows.

### Step 5 — Revenue Decomposition

```powershell
python scripts/run_revenue_decomposition.py
python scripts/validate_revenue_decomposition.py
```

Output:

`data/processed/analytics/revenue_decomposition_monthly.parquet`

The last verified validation passed with 24 monthly rows and no duplicate months. September 2018 is a partial month; interpret its results accordingly.

## 4. Customer Voice and Sentiment

The Customer Voice build expects these existing inputs:

- `data/processed/nlp/customer_voice_input.parquet`
- `data/processed/nlp/sentiment_checkpoints/checkpoint_*.parquet`

After confirming those inputs are valid, run:

```powershell
python scripts/build_customer_voice.py
```

Expected output:

`data/processed/nlp/customer_voice.parquet`

The builder validates review counts, keys, sentiment labels, confidence values, and merge integrity. It deliberately refuses to overwrite an existing final output.

**Important:** The existing Customer Voice artifact was present during the last attempted build. The builder stopped at its overwrite safeguard. Do not delete or replace that artifact unless you explicitly intend to rebuild it and have preserved the existing version.

Sentiment inference may require substantial computation or API usage. Do not rerun inference just to validate an existing output.

## 5. AI Decision Reasoning

The project contains AI reasoning, grounding, semantic-audit, and persistence modules.

Avoid rerunning live LLM inference during routine validation. Regenerate reasoning only when explicitly intended, with awareness of API usage and the impact on persisted outputs.

## 6. Power BI

Existing report:

`powerbi/Product_Pulse_AI.pbix`

Open the report in Power BI Desktop to inspect its data sources, refresh status, relationships, measures, and visuals.

Before presenting refreshed results, verify that the report points to the intended project outputs and that its data refresh completes successfully.

The PBIX file is intentionally excluded from the current Git commit.

## 7. Preserved Data-Quality Baseline

Historical baseline:

`outputs/data_quality/dq_baseline_summary.json`

This is separate from the current results under `data/processed/quality/`.

Do not overwrite the historical baseline when running current data-quality checks.

## 8. Final Verification

Run the automated tests:

```powershell
python -m pytest -q
```

Check repository status and the latest commit:

```powershell
git status --short
git log -1 --oneline
```

Check unstaged and staged whitespace issues:

```powershell
git diff --check
git diff --cached --check
```

Latest verified code commit:

`6dc5772 — Build Product Pulse AI analytics and decision pipeline`

The historical baseline and existing Power BI report were verified as preserved during the previous checks.

## 9. Reproducibility Notes

- Execute commands from the repository root.
- Follow the dependency order; downstream stages require their upstream outputs.
- Review existing outputs before rerunning persistence scripts.
- Preserve historical baselines and existing report files.
- Treat validation failures as findings to investigate, not errors to suppress.
- Do not claim a complete end-to-end refresh until every intended stage and the Power BI refresh have been verified.
