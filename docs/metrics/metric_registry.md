# Product Pulse AI — Metric Registry

**Status:** Version 1.0 — Initial governed definitions
**Project:** Product Pulse AI
**Dataset:** Olist Brazilian E-Commerce Public Dataset
**Purpose:** Canonical definitions for SQL analytics, semantic models, DAX, Power BI, and downstream AI investigations.

---

## 1. Governance Principles

1. Every metric must have a defined business meaning, formula, grain, and source.
2. Warehouse and mart definitions are authoritative. Power BI measures must not silently redefine them.
3. Distinct counts must be recalculated at the requested reporting grain and must not be blindly summed across groups.
4. Item lines are not equivalent to units sold. The dataset does not provide a verified quantity field.
5. Customer identity must retain the distinction between `customer_id` and `customer_unique_id`.
6. Customer ZIP-code prefixes are geographic market proxies, not verified city or state identifiers.
7. Missing, unknown, or unclassified values must not be silently converted into zero or a definitive classification.
8. Metrics describe observed data and do not independently establish causation.
9. All rates must explicitly define their numerator, denominator, and population.
10. Data-quality warnings and reconciliation failures must remain visible to downstream consumers.

---

## 2. Metric Definition Template

Each metric should be documented with:

- **Metric ID:** Stable identifier.
- **Metric name:** Human-readable name.
- **Business meaning:** What the metric describes.
- **Formula:** Exact calculation.
- **Source:** Authoritative mart and columns.
- **Native grain:** Grain of the source mart.
- **Aggregation:** Rules for reporting at another grain.
- **Interpretation:** What can and cannot be concluded.

---

## 3. Product Performance Metrics

**Source mart:** Product Daily Performance
**Native grain:** `date + product_id`

| Metric ID | Metric | Definition | Source |
|---|---|---|---|
| PROD-001 | Revenue | Sum of item prices for the product at the reporting grain. | `revenue` |
| PROD-002 | Freight Value | Sum of freight values associated with the product's item lines. | `freight_value` |
| PROD-003 | Item Lines | Count of order-item rows. This is not units sold. | `item_line_count` |
| PROD-004 | Order Count | Distinct orders containing the product. | `order_count` |
| PROD-005 | Customer Count | Distinct `customer_unique_id` values associated with the product. | `customer_count` |
| PROD-006 | Gross Merchandise Value (GMV) | Defined by this mart as equal to revenue. | `gross_merchandise_value` |
| PROD-007 | Average Item Price | Mean item price across item lines at the source grain. | `avg_item_price` |

### Product Metric Rules

- Revenue is based on item `price`; freight is reported separately.
- GMV is equal to revenue in this implementation and is not an independent financial measure.
- `item_line_count` counts rows, not physical units.
- `order_count` and `customer_count` are distinct counts and must be recalculated when aggregating across dates or products.
- Average item price must be recomputed as total revenue divided by the relevant item-line count when a weighted aggregate is required. Do not average daily averages without weighting.
- Payments and reviews are excluded from this mart to prevent join fanout.
- Category translations are left-joined; missing translations must remain identifiable.

---

## 4. Customer Performance Metrics

**Source mart:** Customer Monthly Performance
**Native grain:** `month + customer_unique_id`

| Metric ID | Metric | Definition | Source |
|---|---|---|---|
| CUST-001 | Order Count | Distinct orders for a customer in a month. | `order_count` |
| CUST-002 | Item Lines | Number of order-item rows associated with the customer in a month. | `item_line_count` |
| CUST-003 | Revenue | Sum of item prices associated with the customer in a month. | `revenue` |
| CUST-004 | Freight Value | Sum of freight values associated with the customer in a month. | `freight_value` |
| CUST-005 | Average Order Value (AOV) | Revenue divided by order count for the customer-month. | `avg_order_value` |
| CUST-006 | First Purchase Month | Earliest observed purchase month for the customer in the available dataset history. | `first_purchase_month` |
| CUST-007 | Customer Status | `NEW` if the current month equals the first observed purchase month; otherwise `REPEAT`. | `customer_status` |

### Customer Metric Rules

- Customer identity at this mart's grain is `customer_unique_id`.
- `NEW` means first observed purchase in the available dataset history. It does not establish lifetime acquisition.
- `REPEAT` means the customer has an earlier observed purchase month within the available history.
- A customer with multiple orders in one month contributes one customer-month record.
- Customer counts across months are not additive because the same customer may appear in multiple months.
- A customer-month AOV is revenue divided by that customer's order count in that month. Overall AOV must be recomputed from total revenue divided by total orders.
- `first_purchase_month` is derived from available order history before customer-month aggregation.

---

## 5. Market Performance Metrics

**Source mart:** Market Monthly Performance
**Native grain:** `month + customer_zip_code_prefix`

| Metric ID | Metric | Definition | Source |
|---|---|---|---|
| MKT-001 | Order Count | Distinct orders assigned to the ZIP-prefix market in a month. | `order_count` |
| MKT-002 | Item Lines | Count of order-item rows assigned to the market in a month. | `item_line_count` |
| MKT-003 | Customer Count | Distinct customers (`customer_unique_id`) associated with the market in a month. | `customer_count` |
| MKT-004 | Revenue | Sum of item prices associated with the market. | `revenue` |
| MKT-005 | Freight Value | Sum of freight values associated with the market. | `freight_value` |
| MKT-006 | Average Order Value (AOV) | Revenue divided by order count. | `avg_order_value` |
| MKT-007 | Revenue per Customer | Revenue divided by distinct customers in the market-month. | `revenue_per_customer` |
| MKT-008 | Average Freight per Order | Freight value divided by order count. | `avg_freight_per_order` |

### Market Metric Rules

- Market is represented by `customer_zip_code_prefix`, a geographic proxy.
- ZIP prefixes must not be presented as verified cities, states, or standardized regions.
- Orders are assigned to the customer's ZIP-prefix market. The implementation assumes one customer market per order.
- Monthly order counts can be summed across markets under this one-market-per-order mapping.
- Customer counts across months are not additive.
- AOV and revenue per customer must be recalculated from their base measures at the requested reporting grain.
- Freight is kept separate from item-price revenue.

---

## 6. Product-Market Performance Metrics

**Source mart:** Product-Market Monthly Performance
**Native grain:** `month + product_id + customer_zip_code_prefix`

| Metric ID | Metric | Definition | Source |
|---|---|---|---|
| PMKT-001 | Order Count | Distinct orders containing the product in the market-month. | `order_count` |
| PMKT-002 | Item Lines | Number of order-item rows for the product in the market-month. | `item_line_count` |
| PMKT-003 | Customer Count | Distinct customers associated with the product in the market-month. | `customer_count` |
| PMKT-004 | Revenue | Sum of item prices for the product in the market-month. | `revenue` |
| PMKT-005 | Freight Value | Sum of freight values for the product in the market-month. | `freight_value` |
| PMKT-006 | Average Order Value (AOV) | Product-market revenue divided by product-market order count. | `avg_order_value` |
| PMKT-007 | Revenue per Customer | Product-market revenue divided by distinct product-market customers. | `revenue_per_customer` |
| PMKT-008 | Market Contribution to Product Revenue (%) | Product-market revenue divided by total revenue for that product across markets in the same month, multiplied by 100. | `market_contribution_to_product_revenue_pct` |

### Product-Market Metric Rules

- Market contribution is calculated within each product-month across the available ZIP-prefix markets.
- Contributions should sum to approximately 100% for each product-month, subject to floating-point precision.
- Product-market customer counts may overlap across products and markets; do not sum them to obtain unique overall customers.
- Product-market order counts may overlap across products; do not sum them to obtain unique overall orders.
- Payments and reviews are excluded from this mart to prevent fanout.
- Unmapped category translations must not cause product rows to be dropped.

---

## 7. Revenue Decomposition Metrics

**Source:** Revenue Decomposition Analytics
**Native grain:** Month

| Metric ID | Metric | Definition |
|---|---|---|
| REV-001 | Revenue | Monthly item-price revenue derived from the product performance mart. |
| REV-002 | Freight Value | Monthly freight value derived from the product performance mart. |
| REV-003 | Order Count | Monthly order count derived from the market performance mart. |
| REV-004 | Customer Count | Monthly distinct customer count derived from the customer performance mart. |
| REV-005 | Average Order Value (AOV) | Revenue divided by order count. |
| REV-006 | Revenue per Customer | Revenue divided by distinct customers. |
| REV-007 | Order Volume Effect | Change in order count multiplied by the previous observed month's AOV. |
| REV-008 | AOV Effect | Current month's order count multiplied by the change in AOV from the previous observed month. |

### Revenue Decomposition Rules

For month `t` and previous observed month `t-1`:

- `AOV_t = Revenue_t / Orders_t`
- `Order Volume Effect = (Orders_t - Orders_(t-1)) * AOV_(t-1)`
- `AOV Effect = Orders_t * (AOV_t - AOV_(t-1))`
- `Revenue Change = Order Volume Effect + AOV Effect`

The decomposition bridge must reconcile to the observed revenue change within the implementation's numerical tolerance.

- The first observed month has no prior comparison; the implementation sets its change and effects to zero.
- Comparisons use the previous observed month, which may not be a contiguous calendar month.
- A partial final month remains in the output and must not be silently discarded.
- This decomposition is arithmetic, not proof of causal drivers.
- Customer effect, product mix, market mix, and product-market mix are documented diagnostics, not output columns of the current implementation.

---

## 8. Customer Voice Metrics

**Source mart:** Customer Voice Monthly
**Native grain:** Month, based on review creation date

| Metric ID | Metric | Definition | Source |
|---|---|---|---|
| VOICE-001 | Review Count | Number of review records at the validated `review_id + order_id` grain. | `review_count` |
| VOICE-002 | Text Review Count | Number of reviews with non-empty combined review text. | `text_review_count` |
| VOICE-003 | Text Coverage (%) | Text-bearing reviews divided by all reviews, multiplied by 100. | `text_coverage_pct` |
| VOICE-004 | Average Review Score | Mean numeric review score for the month's review records. | `average_review_score` |
| VOICE-005 | Positive Review Rate (%) | Positive sentiment predictions divided by all text-bearing reviews, multiplied by 100. | `positive_review_rate_pct` |
| VOICE-006 | Negative Review Rate (%) | Negative sentiment predictions divided by all text-bearing reviews, multiplied by 100. | `negative_review_rate_pct` |
| VOICE-007 | Neutral Review Rate (%) | Neutral sentiment predictions divided by all text-bearing reviews, multiplied by 100. | `neutral_review_rate_pct` |
| VOICE-008 | Complaint Volume | Count of reviews classified with `complaint_flag = true`. | `complaint_volume` |
| VOICE-009 | Complaint Rate (%) | Confirmed complaint volume divided by all text-bearing reviews, multiplied by 100. | `complaint_rate_pct` |

### Customer Voice Metric Rules

- Review grain is `review_id + order_id`; review IDs or order IDs alone are not guaranteed unique.
- Review month is derived from `review_creation_date`.
- Text coverage denominator is all reviews in the month.
- Sentiment rates use all text-bearing reviews as the denominator. Unknown sentiment classifications can therefore cause the three sentiment rates to sum to less than 100%.
- Complaint rate uses all text-bearing reviews as the denominator, including reviews with unknown complaint classification.
- Textless reviews remain without inferred sentiment or complaint classifications.
- When a month has no text-bearing reviews, sentiment and complaint rates are set to missing (NA), not zero.
- Complaint themes are available in the complaint-detection output but are not columns in the current monthly customer voice mart.
- Sentiment model confidence is a model score and is not established as calibrated probability.
- Complaint confidence is a deterministic evidence score, not a calibrated probability.
- Sentiment and complaint classifications are analytical outputs, not verified ground-truth labels.
- These metrics describe observed review content and do not independently establish product defects or causal drivers.

---

## 9. Cross-Metric Calculation and Aggregation Standards

1. **Ratios:** Recalculate from numerator and denominator at the target grain. Do not average percentages unless the intended calculation explicitly requires an unweighted mean.
2. **Distinct counts:** Recompute distinct entities at the target grain. Do not sum daily, monthly, product, or market distinct counts where entities can recur or overlap.
3. **Averages:** Recalculate weighted averages from base values where appropriate. Avoid averaging pre-aggregated averages without weights.
4. **Missing values:** Preserve missingness and define denominator exclusions explicitly. Do not automatically interpret null as zero.
5. **Time comparisons:** State the comparison period and whether it is the previous observed period or previous calendar period.
6. **Partial periods:** Identify incomplete periods and avoid presenting them as directly comparable to complete periods without qualification.
7. **Classification metrics:** Retain unknown/unclassified cases and disclose the method used to classify records.
8. **Reconciliation:** Derived metrics must reconcile to their governed source measures within documented tolerances.
9. **Causality:** Descriptive trends, decompositions, correlations, and classifications do not alone prove causation.
10. **Presentation:** Power BI labels and tooltips must preserve the definitions and caveats in this registry.

---

## 10. Deferred Metrics

The following concepts are in the broader project scope but are not certified as implemented metrics in the current marts:

- Customer retention and cohort retention rates
- Days between purchases and customer tenure
- Customer segment contribution and behavioral segments
- Product growth, decline streaks, acceleration, and volatility
- Complaint-theme monthly volume and share
- Product and seller concentration measures
- Market growth and national-versus-market divergence
- Forecast values, prediction intervals, and forecast evaluation metrics
- Anomaly and signal metrics
- Composite data-quality scores

These require separately specified formulas, source contracts, validation rules, and documented limitations before implementation. No deferred metric should be represented as currently available or certified.

---

## 11. Change Control

- Metric definitions must be version-controlled.
- Changes to formulas, grain, source fields, denominator, or interpretation require a documented reason and validation.
- SQL, semantic-model, and DAX implementations must be checked against this registry.
- Changes must not silently break historical comparability.
- Any metric that cannot be reconciled to its source contract must be marked uncertified until resolved.
