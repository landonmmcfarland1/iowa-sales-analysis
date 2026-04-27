# Iowa Liquor Sales Analysis

A memory-efficient analysis of Iowa's liquor sales data handling ~27 million transaction records (~7GB dataset) using lazy evaluation techniques. Originally developed as part of MIS501 (Python Fundamentals), this project has been significantly extended to include a full predictive modeling section comparing Logistic Regression and Random Forest classifiers for retail store performance classification.

## What's New (April 2026 Update)

The predictive modeling section has been redesigned to produce models that are actually actionable rather than trivially confirming what a sales analyst already knows. The original models were trained on all store-month records and asked whether a store was in the top quartile — a task so dominated by persistence (big stores stay big) that it provided no useful signal. Three changes fix this:

- **Breakout Detection Framing**: The dataset is now filtered to stores that were in the bottom 75% last month. The models now predict which of those lower-tier stores will surge into the top quartile this month — the ~4.9% "breakout" event — rather than confirming that already-dominant stores remain dominant
- **Monthly Dynamic Thresholds**: The 75th percentile cutoff for `is_top_quartile` is computed per Year-Month rather than globally, so seasonal effects don't dominate. A store must outperform its peers *that specific month*, not just benefit from December being a high-revenue month industry-wide
- **StandardScaler on Logistic Regression**: The logistic regression now trains on standardized features (scaler fit on train only, applied to test) to prevent large-magnitude revenue lag features from drowning out smaller-scale predictors like `month_of_year` during gradient-based optimization

**Updated model results under the new framing:**

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| AUC-ROC | 0.9668 | 0.9675 |
| Top Quartile Precision | 0.34 | 0.34 |
| Top Quartile Recall | 0.93 | 0.93 |
| Top Quartile F1 | 0.50 | 0.50 |

The lower precision relative to the original README figures reflects the harder task: the model is now targeting a 4.9% positive class rather than a straightforward 25% top-quartile label on the full panel. The 34% precision represents a ~7x lift over the baseline breakout rate, and 93% recall means 186 out of 199 actual breakout stores in the test set are correctly flagged.

**Earlier additions (March 2026), still in place:**

- **Predictive Modeling Section (Section 6)**: Store-level revenue classification pipeline using engineered lag and rolling features
- **Temporal Feature Engineering**: Store-month aggregation with strictly backward-looking lag features (1, 2, 3 months), 3-month rolling average, and expanding cumulative mean — all designed to prevent data leakage
- **Data Leakage Fix**: Store baseline features use an expanding cumulative mean (`cum_sum() / cum_count()`) rather than a full-series mean, ensuring no future revenue leaks into training features
- **Temporal Train/Test Split**: Trains on 2012–April 2023 (~180,000 store-month rows), tests on the final 3 months — no shuffling
- **Hugging Face Dataset Integration**: The full dataset is now hosted on [Hugging Face](https://huggingface.co/datasets/lmmcfarland1/iowa_liquor_data_csv) and downloads automatically on first run, replacing the previous Iowa Data Portal link which was unreliable for large file downloads
- **Streaming Aggregation**: Store-month aggregation uses `collect(engine="streaming")` to reduce peak RAM usage during the heavy group-by operation
- **Year-by-Year Summary**: Descriptive statistics section now includes annual revenue trends, YoY growth rates, and average transaction values across all 11 years
- **Weekday vs. Weekend Visualization**: Previously commented out, now active with the full multi-year dataset

---

## Project Context

As a data consultant for the State of Iowa, this analysis provides insights into liquor sales patterns across the state, including revenue trends, product preferences, geographic distribution, temporal patterns, and predictive store performance classification. The project emphasizes **production-ready code** that can handle datasets larger than available RAM through efficient lazy evaluation strategies.

## Key Technical Achievements

- **Memory-Efficient Processing**: Processes ~7GB dataset (27M+ rows) on machines with ~12GB RAM using Polars lazy evaluation and streaming aggregation
- **Automated Data Pipeline**: Streamlined cleaning with minimal code duplication following DRY principles
- **Dynamic Category Mapping**: Consolidates 100+ liquor categories using regex patterns instead of hardcoded mappings
- **Interactive Visualizations**: Eight production-ready Plotly charts exploring multiple analysis dimensions
- **Predictive Modeling**: Logistic Regression vs. Random Forest comparison on a temporal breakout detection task; AUC-ROC ~0.967 on held-out temporal test set
- **Leak-Free Feature Engineering**: All predictive features are strictly backward-looking using lag, rolling, and expanding window operations
- **Automatic Dataset Download**: Full dataset streams from Hugging Face on first run and caches locally — no manual download required

---

## Quick Start

This project uses [Pixi](https://pixi.sh/) for environment management. Pixi is a modern, cross-platform package manager that handles Python dependencies and virtual environments automatically.

### Installing Pixi

**macOS/Linux:**
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Windows (PowerShell):**
```powershell
iwr -useb https://pixi.sh/install.ps1 | iex
```

**Alternative (using Homebrew on macOS):**
```bash
brew install pixi
```

After installation, restart your terminal or run:
```bash
source ~/.bashrc  # Linux
source ~/.zshrc   # macOS
```

### Running the Analysis

1. **Clone this repository** and navigate to the project directory:
   ```bash
   git clone https://github.com/landonmmcfarland1/iowa-sales-analysis.git
   cd iowa-sales-analysis
   ```

2. **Install dependencies**:
   ```bash
   pixi install
   ```

3. **Run the interactive notebook**:
   ```bash
   pixi run marimo edit iowa-sales-analyses.py
   ```
   This will launch an interactive Marimo notebook in your web browser at `http://localhost:2718`

4. **Dataset**: The full ~7GB dataset will **download automatically from Hugging Face** on first run. You do not need to download it manually. Ensure you have approximately **8GB of free disk space** and **12GB of available RAM** for the full analysis including the predictive modeling section.

> ⚠️ **First run note**: The dataset download (~7GB) will take several minutes depending on your connection speed. The file is cached locally by Hugging Face after the first download — subsequent runs skip the download entirely.

> ✅ **Sample data**: The repository includes `Iowa_Liquor_Sales_2022.parquet` (2022 only, ~500MB) for quick exploration. The predictive modeling section produces more reliable results with the full 11-year dataset.

### Optional: Clear the Dataset Cache

After the first run, the dataset is cached at `~/.cache/huggingface/`. If you are done with this analysis and want to reclaim ~7GB of disk space, uncomment and run the cleanup cell at the bottom of the notebook. **Do not run this if you plan to use the notebook again.**

---

## Project Structure

```
.
├── iowa-sales-analyses.py          # Main Marimo notebook with full analysis
├── Iowa_Liquor_Sales_2022.parquet  # Sample dataset (2022 only, for quick exploration)
├── pixi.toml                        # Pixi dependency configuration
├── pixi.lock                        # Locked dependency versions
└── README.md                        # This file
```

---

## Dataset Overview

**Source**: Iowa Alcoholic Beverages Division  
**Hosted**: [Hugging Face — lmmcfarland1/iowa_liquor_data_csv](https://huggingface.co/datasets/lmmcfarland1/iowa_liquor_data_csv)  
**Original Source**: [Iowa Data Portal](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy)  
**Coverage**: 2012–2023 (11 years, full dataset); 2022 only (sample parquet)  
**Scale**: ~27 million transaction records from licensed retailers  
**File size**: ~7GB (CSV)

### Data Schema

The dataset includes:
- **Transaction details**: Invoice numbers, dates, transaction volumes
- **Product information**: Item descriptions, categories, bottle sizes, pack sizes
- **Pricing**: State bottle cost, state bottle retail, sale price
- **Geographic data**: Store location, address, city, county, zip code
- **Vendor information**: Vendor names and numbers

---

## Analysis Pipeline

### 1. Data Ingestion & Size Estimation

Uses Polars `LazyFrame` to estimate dataset size without loading into RAM:
- Automatically downloads from Hugging Face if not found locally
- Scans only metadata and small samples for size estimation
- Estimates total size from 100K row sample

### 2. Automated Data Cleaning

Single-chain cleaning pipeline that:
- **Analyzes missing values**: Identifies columns with >10% missing data (sampled from 2M rows)
- **Drops unnecessary columns**: Removes geographic redundancies and irrelevant fields programmatically
- **Type casting**: Automatically casts numeric and string columns using comprehensions
- **Date parsing**: Converts date strings to proper date types
- **Feature engineering**: Extracts year, month, quarter, weekday, and weekend indicators

### 3. Automated Category Mapping

Consolidates 100+ unique liquor categories into 12 major groups using regex patterns:
- Whiskey (includes bourbon, scotch, whisky)
- Vodka, Rum, Tequila & Mezcal, Gin
- Brandy & Cognac, Schnapps, Liqueurs & Cordials
- Specialty & Other Spirits, Ready-to-Drink, Craft/Local
- Administrative/Non-Product

### 4. Descriptive Statistics

Calculates comprehensive metrics including:
- Total revenue, volume, bottles sold, and transaction counts
- Year-by-year summary with YoY revenue growth rates and average transaction values
- Top 10 product categories by revenue

### 5. Visualization Gallery (8 Charts)

1. **Annual Revenue Trend** — Year-over-year revenue line chart (2012–2023)
2. **Top 20 Products by Revenue** — Individual product performance
3. **Top 20 Products by Volume** — Volume-based product ranking
4. **Top 15 Counties by Revenue** — Geographic revenue distribution
5. **Top 20 Cities by Sales** — Urban market analysis
6. **Weekday vs. Weekend Sales** — Average daily sales by day type
7. **Top 20 Cities by Sales Efficiency** — Average sale per transaction
8. **Random Forest Feature Importances** — Predictive signal by feature

### 6. Store Performance Classification (Logistic Regression vs. Random Forest)

**Business Question**: Given a store's recent sales history, can we predict whether a currently lower-tier store will break into the top quartile of revenue performers next month?

**Why this framing?** The original version of this model asked whether any store would be a top-quartile performer, which turned out to be a near-trivial task — stores that were already dominant tended to stay dominant, and the model was largely just learning persistence. The redesigned task filters the sample to stores that were in the bottom 75% last month and asks whether those stores will surge into the top quartile this month. That's the ~4.9% "breakout" event, and it's a question a distributor can actually act on.

**Why top quartile with monthly thresholds?** Each month, stores are ranked by total revenue and the top 25% are labeled as top performers. The threshold is computed month-by-month (not globally) so that seasonal effects don't dominate — a store must outperform its peers *that specific month*, not just benefit from December being a high-revenue month industry-wide.

**Pipeline:**
- Aggregate 27M transactions to store-month level (~185,000 rows)
- Engineer lag features (1, 2, 3 months), 3-month rolling average, month-over-month growth rate, expanding cumulative mean baseline, transaction count lag, SKU diversity lag, and month-of-year
- Label each store-month with a monthly 75th-percentile threshold, track prior-month status, and filter to only bottom-75% stores from last month
- Temporal train/test split: train on 2012–April 2023, test on final 3 months — no shuffling
- Logistic Regression (with StandardScaler, fit on train only) and Random Forest (300 trees, max depth 8, balanced class weights) trained and evaluated in parallel

**Results:**

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| AUC-ROC | 0.9668 | 0.9675 |
| Top Quartile Precision | 0.34 | 0.34 |
| Top Quartile Recall | 0.93 | 0.93 |
| Top Quartile F1 | 0.50 | 0.50 |

**What the metrics mean in practice:** Out of 199 actual breakout stores in the test set, both models identify 186 of them — a 93% recall rate. The 34% precision means there are false positives, but it represents roughly a 7x lift over the 4.9% baseline breakout rate. A distributor targeting only the ~540 stores flagged by the model gets far better coverage of rising stores than any random allocation across 4,000+ lower-tier locations.

**Key finding:** The near-identical AUC scores between the linear baseline and the 300-tree ensemble confirm that the mathematical signal of a store breakout is largely linear. Revenue lag features and rolling averages push stores across the quartile threshold in a way that logistic regression captures almost as well as a Random Forest — the additional computational cost of the ensemble is not required for this task.

---

## Technologies

### Core Stack

- **Python 3.14**: Latest Python release
- **Marimo**: Reactive Python notebooks that run as pure Python files
- **Polars**: High-performance DataFrame library with lazy evaluation and streaming support
- **Plotly Express**: Interactive visualization library
- **scikit-learn**: Logistic Regression, Random Forest, StandardScaler, classification metrics
- **Hugging Face Hub**: Automatic dataset download and local caching

### Why Polars over Pandas?

- **Memory efficiency**: Lazy evaluation and streaming process data without loading the entire dataset
- **Performance**: Written in Rust, significantly faster than Pandas
- **Modern API**: Cleaner syntax with method chaining
- **Query optimization**: Automatically optimizes query plans before execution

### Why Marimo?

- **Pure Python files**: Version control friendly (`.py` not `.ipynb`)
- **Reactive execution**: Automatically re-runs dependent cells
- **No hidden state**: Cell execution order matches file order

---

## Memory & Performance Notes

**RAM Requirements:**
- Descriptive statistics and visualizations: ~8GB peak
- Store-month aggregation (streaming): ~5–6GB peak
- Full notebook including predictive modeling: ~12GB peak

**Execution Time** (approximate, M-series Mac):
- Data ingestion: <1 second (metadata only)
- Cleaning pipeline setup: <1 second (still lazy)
- Each visualization: 5–30 seconds
- Store-month aggregation (streaming): 3–5 minutes
- Model training (LR + RF): 2–4 minutes
- Full notebook: ~15–20 minutes

---

## Key Findings

- **Category dominance**: Whiskey generates the highest revenue despite vodka having more individual SKUs
- **Geographic concentration**: Polk County (Des Moines) accounts for a disproportionate share of sales
- **Breakout signal is linear**: Both Logistic Regression and Random Forest achieve near-identical AUC (~0.967) on the breakout detection task, confirming the predictive signal is largely linear even for the harder within-tier surge prediction problem
- **Revenue history dominates**: The 3-month rolling average and store historical average are the strongest features; SKU diversity, transaction count, and seasonality contribute almost nothing once revenue trajectory is controlled for

---

## Course Context

**Original Course**: MIS501 — Python Fundamentals  
**Institution**: University of Alabama, Culverhouse College of Business  
**Extended as**: Business Analytics Portfolio Project (March–April 2026)

---

## License

This project is available for educational purposes. Dataset is provided by the Iowa Alcoholic Beverages Division via the Iowa Data Portal and is subject to their terms of use (CC BY 4.0).

## Acknowledgments

- **Dataset**: Iowa Alcoholic Beverages Division
- **Hosting**: Hugging Face (lmmcfarland1/iowa_liquor_data_csv)
- **Original Course**: MIS501 Python Fundamentals, University of Alabama
- **Tools**: Marimo, Polars, Plotly, scikit-learn development teams

---

**Questions or Issues?**  
If you encounter any problems running this analysis or have questions about the methodology, please open an issue on GitHub.
