# Iowa Liquor Sales Analysis

A memory-efficient analysis of Iowa's liquor sales data handling ~27 million transaction records (~8GB dataset) using lazy evaluation techniques. This project was developed as part of MIS501 (Python Fundamentals) to demonstrate large-scale data processing capabilities on standard consumer hardware.

## Project Context

As a data consultant for the State of Iowa, this analysis provides insights into liquor sales patterns across the state, including revenue trends, product preferences, geographic distribution, and temporal patterns. The project emphasizes **production-ready code** that can handle datasets larger than available RAM through efficient lazy evaluation strategies.

## Key Technical Achievements

- **Memory-Efficient Processing**: Processes ~8GB dataset (27M+ rows) on machines with <8GB RAM using Polars lazy evaluation
- **Automated Data Pipeline**: Streamlined cleaning with minimal code duplication following DRY principles
- **Dynamic Category Mapping**: Consolidates 100+ liquor categories using regex patterns instead of hardcoded mappings
- **Interactive Visualizations**: Seven production-ready Plotly charts exploring multiple analysis dimensions

## Quick Start

This project uses [Pixi](https://pixi.sh/) for environment management. Pixi is a modern, cross-platform package manager that handles Python dependencies and virtual environments automatically.

### Installing Pixi

Pixi is not yet an industry standard, so you'll need to install it first:

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

1. **Clone this repository** and navigate to the project directory

2. **Install dependencies** (Pixi will automatically create a virtual environment and install all required packages):
   ```bash
   pixi install
   ```

3. **Download the dataset**:
   
   The full dataset is not included in this repository due to GitHub's file size limitations. You have two options:
   
   **Option A - Use the sample data** (included in repository):
   - File: `Iowa_Liquor_Sales_2022.parquet`
   - Coverage: 2022 only (subset of full dataset)
   - Size: ~500MB
   - Note: Some visualizations may show limited patterns with single-year data
   
   **Option B - Download the full dataset** (recommended for complete analysis):
   - Source: [Iowa Data Portal](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy)
   - Download as CSV, then convert to parquet (instructions below)
   - Coverage: 2012-2026 (12 years)
   - Size: ~10GB (current version)

4. **Run the interactive notebook**:
   ```bash
   pixi run marimo edit iowa-sales-analyses.py
   ```
   
   This will launch an interactive Marimo notebook in your web browser at `http://localhost:2718`

### Converting CSV to Parquet (Optional)

If you download the full CSV dataset from Iowa Data Portal, convert it to parquet format for better performance:

```python
import polars as pl

# Read CSV and write to parquet (one-time conversion)
pl.read_csv('Iowa_Liquor_Sales.csv').write_parquet('Iowa_Liquor_Sales_2022.parquet')
```

Or use this command with Pixi:
```bash
pixi run python -c "import polars as pl; pl.read_csv('Iowa_Liquor_Sales.csv').write_parquet('Iowa_Liquor_Sales_2022.parquet')"
```

## Project Structure

```
.
├── iowa-sales-analyses.py          # Main Marimo notebook with full analysis
├── Iowa_Liquor_Sales_2022.parquet  # Sample dataset (2022 only)
├── pixi.toml                        # Pixi dependency configuration
├── pixi.lock                        # Locked dependency versions
└── README.md                        # This file
```

## Dataset Overview

**Source**: Iowa Alcoholic Beverages Division  
**Access**: [Iowa Data Portal](https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy)  
**Coverage**: Full dataset spans 2012-2026 (sample dataset: 2022 only)  
**Scale**: 26.8M transaction records from licensed retailers  

### Data Schema

The dataset includes:
- **Transaction details**: Invoice numbers, dates, transaction volumes
- **Product information**: Item descriptions, categories, bottle sizes, pack sizes
- **Pricing**: State bottle cost, state bottle retail, sale price
- **Geographic data**: Store location, address, city, county, zip code
- **Vendor information**: Vendor names and numbers

### Important Notes

⚠️ **GitHub Limitation**: The repository contains a 2022 snapshot of the data due to GitHub's file size limits. The full dataset (~10GB) must be downloaded separately from the Iowa Data Portal.

✅ **Memory Efficiency**: Despite the large dataset size, this analysis runs on machines with <8GB RAM thanks to Polars lazy evaluation, which processes data without loading the entire dataset into memory.

## Analysis Pipeline

### 1. Data Ingestion & Size Estimation

Uses Polars `LazyFrame` to estimate dataset size without loading into RAM:
- Scans only metadata and small samples
- Estimates total size from 100K row sample
- Displays dimensions and memory usage

### 2. Automated Data Cleaning

Single-chain cleaning pipeline that:
- **Analyzes missing values**: Identifies columns with >10% missing data
- **Drops unnecessary columns**: Removes geographic redundancies and irrelevant fields
- **Type casting**: Automatically casts numeric and string columns
- **Date parsing**: Converts date strings to proper date types
- **Feature engineering**: Extracts year, month, quarter, weekday, and weekend indicators

**Key automation improvements**:
- All operations in single lazy chain (no intermediate DataFrames)
- Programmatic column dropping (not hardcoded lists)
- Comprehension-based type casting
- LazyFrame stays lazy until visualization

### 3. Category Mapping

Consolidates 100+ unique liquor categories into 11 major groups using regex patterns:
- Whiskey (includes bourbon, scotch, whisky)
- Vodka
- Rum
- Tequila & Mezcal
- Gin
- Brandy & Cognac
- Liqueurs & Cordials
- Schnapps
- Beer
- Wine
- Other Spirits

**Advantages of regex approach**:
- Automatically catches spelling variations
- Case-insensitive matching
- Easier to maintain than hardcoded lists
- Handles new product names automatically

### 4. Descriptive Statistics

Calculates comprehensive metrics:
- Total sales revenue
- Total volume sold (liters and gallons)
- Unique products, stores, vendors
- Transaction counts
- Date range coverage

### 5. Interactive Visualizations

Seven Plotly visualizations exploring different dimensions:

1. **Top 20 Alcohol Categories by Revenue** - Horizontal bar chart showing revenue distribution
2. **Top 20 Products by Revenue** - Individual product performance
3. **Top 20 Products by Volume** - Volume-based product ranking
4. **Top 15 Counties by Revenue** - Geographic revenue distribution
5. **Top 20 Cities by Sales** - Urban market analysis
6. **Weekday vs Weekend Sales** (commented out for single-year data) - Temporal patterns
7. **Top 20 Cities by Sales Efficiency** - Average sale per transaction metric

All charts use consistent styling:
- Clean white background
- No gridlines for minimal visual clutter
- Centered titles
- Appropriate height for readability

## Technologies

### Core Stack

- **Python 3.13**: Latest Python release with performance enhancements
- **Marimo**: Reactive Python notebooks that run as pure Python files
- **Polars**: High-performance DataFrame library with lazy evaluation
  - Replaces Pandas for memory efficiency
  - Uses lazy evaluation (`LazyFrame`) to defer computation
  - Optimizes query plans before execution
- **Plotly Express**: Interactive visualization library

### Why Polars over Pandas?

Polars was chosen for this project because:
- **Memory efficiency**: Lazy evaluation processes data without loading entire dataset
- **Performance**: Written in Rust, significantly faster than Pandas
- **Modern API**: Cleaner syntax with method chaining
- **Query optimization**: Automatically optimizes query plans before execution
- **Better type system**: Strict typing prevents common errors

### Why Marimo?

Marimo notebooks offer advantages over Jupyter:
- **Pure Python files**: Version control friendly (`.py` not `.ipynb`)
- **Reactive execution**: Automatically re-runs dependent cells
- **No hidden state**: Cell execution order matches file order
- **Interactive widgets**: Built-in UI components
- **Fast startup**: No kernel management overhead

## Development Guide

### Understanding Lazy Evaluation

This project extensively uses Polars `LazyFrame` (`lf`) instead of `DataFrame` (`df`):

```python
# LazyFrame - builds query plan, doesn't execute
lf = pl.scan_parquet('data.parquet')
lf = lf.filter(pl.col('Year') == 2022)  # Still lazy

# Execution only happens on .collect()
df = lf.collect()  # Now data is loaded into memory
```

**Benefits**:
- Multiple operations are optimized together
- Only necessary data is loaded
- Predicate pushdown (filters applied before reading)
- Projection pushdown (only needed columns are read)

### Adding New Visualizations

To add a new visualization:

1. Create a new `@app.cell` block (optionally with `hide_code=True`)
2. Use the `df_with_categories` LazyFrame
3. Build your aggregation query
4. Call `.collect()` at the end to execute
5. Create Plotly figure and call `.show()`

Example:
```python
@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    my_analysis = (
        df_with_categories
        .group_by('Some_Column')
        .agg([pl.col('Sale (Dollars)').sum().alias('Total')])
        .sort('Total', descending=True)
        .collect()  # Execute here
    )
    
    fig = px.bar(my_analysis, x='Some_Column', y='Total')
    fig.show()
    return
```

### Marimo Cell Structure

Marimo uses `@app.cell` decorators to define notebook cells:

```python
@app.cell
def _(dependency1, dependency2):
    # Cell code here
    result = dependency1 + dependency2
    return result,  # Note the comma - returns tuple
```

- **Imports**: Return imported modules as tuple
- **Dependencies**: Referenced in function signature
- **Returns**: Always return tuple (even for single value)
- **Hide code**: Use `@app.cell(hide_code=True)` for cleaner presentation

## Key Findings

While this analysis was primarily a technical demonstration of handling large datasets, several business insights emerged:

- **Category dominance**: Whiskey products generate the highest revenue despite vodka having more individual SKUs
- **Geographic concentration**: Polk County (Des Moines) accounts for a disproportionate share of sales
- **Product concentration**: Top 20 products represent a significant portion of total revenue
- **Efficiency variations**: Smaller cities often show higher average transaction values (wholesale vs retail mix)

## Performance Notes

**Memory Usage**:
- Sample scanning: ~50MB for 2M row sample
- Full dataset processing: <2GB peak memory usage
- Lazy operations: Minimal memory footprint

**Execution Time** (approximate, on modern laptop):
- Data ingestion: <1 second (metadata only)
- Cleaning pipeline setup: <1 second (still lazy)
- Each visualization: 5-30 seconds (depends on aggregation complexity)
- Full notebook run: ~2-3 minutes

## Course Context

**Course**: MIS501 - Python Fundamentals  
**Institution**: University of Alabama Culverhouse College of Business  
**Learning Objectives Demonstrated**:
- Handling large-scale data that exceeds typical computer RAM
- Implementing production-ready code with DRY principles
- Using modern Python libraries for data analysis
- Creating automated, maintainable data pipelines
- Generating publication-quality visualizations

## Future Enhancements

Potential extensions to this analysis:

- **Time series forecasting**: Predict future sales trends using SARIMA or Prophet
- **Clustering analysis**: Identify similar stores or products using K-means
- **Geographic visualization**: Map sales density across Iowa counties
- **Price elasticity**: Analyze relationship between pricing and volume
- **Seasonality decomposition**: Separate trend, seasonal, and residual components
- **Anomaly detection**: Identify unusual sales patterns or data quality issues

## License

This project is available for educational purposes. Dataset is provided by the Iowa Alcoholic Beverages Division and subject to their terms of use.

## Acknowledgments

- **Dataset**: Iowa Alcoholic Beverages Division
- **Course**: MIS501 Python Fundamentals, University of Alabama
- **Tools**: Marimo, Polars, Plotly development teams

---

**Questions or Issues?**

If you encounter any problems running this analysis or have questions about the methodology, please open an issue on GitHub.
# iowa-sales-analysis
