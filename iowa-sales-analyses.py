import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    from huggingface_hub import hf_hub_download
    import shutil
    import numpy as np
    import plotly.express as px
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import roc_auc_score

    return (
        LogisticRegression,
        Path,
        RandomForestClassifier,
        StandardScaler,
        classification_report,
        confusion_matrix,
        hf_hub_download,
        mo,
        pl,
        px,
        roc_auc_score,
        shutil,
    )


@app.cell
def _(Path, hf_hub_download, shutil):
    #The website where the original data comes from hardly works, so I created my own repository that can handle a file of this size. Github cannot hold a data file 8GB in size, so I used Hugging Face
    data_file = 'Iowa_Liquor_Sales.csv'

    if Path(data_file).exists():
        print(f"✓ Data file found locally: {data_file}")
    else:
        print("Data file not found locally. Downloading from Hugging Face (~7GB)...")
        print("This will take several minutes depending on your connection speed.")
        print("The file will be cached and this step will be skipped on future runs.\n")
        data_file = hf_hub_download(
            repo_id="lmmcfarland1/iowa_liquor_data_csv",
            filename="Iowa_Liquor_Sales.csv",
            repo_type="dataset",
        )
        shutil.copy(data_file, 'Iowa_Liquor_sales.csv')

        print(f"✓ Download complete: {data_file}")
    return (data_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IOWA LIQUOR SALES ANALYSIS (ORIGINALLY A COURSE PROJECT)

    **Technical Highlights:**
    - **Automatic Download of Original File from Hugging Face**: GitHub limits users to files under 100MB in size for pushes, so to prove my code can run on your consumer hardware, data will download from Hugging Face to prove it to you :)
    - **Memory-Efficient Processing**: Analyzes ~8 GB dataset (~27 Million rows) using Polars lazy evaluation
    - **Automated Data Pipeline**: Streamlined cleaning with minimal code duplication
    - **Production-Ready Code**: Follows DRY principles with reusable patterns for easy future edits
    - **Descriptive Statistics & Visualizations**: Uses Plotly Express to create interactive visualizations
    - **Classification Analysis with Train/Test Split Logistic Regression (Logit Model) & Temporal Random Forest**: Estimating temporal models to identify top 25% of revenue generating retail locations

    **Technical Stack:** Python, Polars (LazyFrames), Marimo, Plotly

    ---

    ## Dataset Overview

    12 years of Iowa liquor sales (2012-2026) covering:
    - 26.8M transaction records from licensed retailers
    - Product details, pricing, geographic data
    - Sales metrics across counties and cities
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Ingestion & Size Estimation

    Using LazyFrame evaluation to estimate DataFrame size without loading into RAM.
    """)
    return


@app.cell(hide_code=True)
def _(data_file, pl):
    #This is the creation of a Lazy Frame. Since the dataset is large (~7GB, we don't want to load everything onto RAM.)

    lf = pl.scan_csv(data_file
                        )
    #, ignore_errors=True) If you run the .csv file, rather than .parquet, ignore errors needed

    #Get dimensions efficiently (only scans metadata/samples)
    num_rows = lf.select(pl.len()).collect().item()
    num_cols = len(lf.collect_schema())

    #Estimate size from 1M row sample (avoids loading 6.99GB with ~27 million rows)
    sample = lf.head(100_000).collect()
    sample_bytes = sample.estimated_size()
    estimated_total_bytes = sample_bytes * (num_rows / 100_000)
    estimated_mb = estimated_total_bytes / (1024**2)

    print(f"Dataset: {num_rows:,} rows × {num_cols} columns")
    print(f"Estimated size: {estimated_mb:.2f} MB (~{estimated_mb/1024:.2f} GB)")
    print(f"Memory used for estimate: {sample_bytes / (1024**2):.2f} MB")
    return (lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Automated Data Cleaning Pipeline

    **Key automation improvements over original:**
    1. Use basic Polars operations to understand data types
    2. All column operations done in single chain (no intermediate DataFrames)
    3. Type casting automated with comprehensions
    4. Geographic columns dropped programmatically
    5. LazyFrame kept lazy until final visualization
    """)
    return


@app.cell(hide_code=True)
def _(lf):
    #Use lf.collect_schema() to collect metadata of dataframe (Polars reads only the CSV header, column names, data types, and infers data type from a tiny sample of rows)
    #Uses little to no RAM
    lf.collect_schema()
    return


@app.cell(hide_code=True)
def _(lf):
    #Use lf.head().collect() to return a sample of the dataframe to get a first glimpse of the values of each column
    lf.head(25).collect()
    return


@app.cell(hide_code=True)
def _(lf, pl):
    #Set threshold for dropping columns (10%)
    MISSING_VALUE_THRESHOLD = 0.10

    #Calculate missing value proportions for each column
    #(using a small sample to avoid loading full dataset)
    sample_for_nulls = lf.head(2_000_000).collect()

    missing_value_proportions = (
        sample_for_nulls.null_count() / len(sample_for_nulls)
    ).to_dicts()[0]

    #Display missing value analysis
    missing_df = pl.DataFrame({
        'Column': list(missing_value_proportions.keys()),
        'Missing_Proportion': list(missing_value_proportions.values())
    }).sort('Missing_Proportion', descending=True)
    pl.Config.set_tbl_rows(100)
    print("="*80)
    print("MISSING VALUE ANALYSIS ")
    print("="*80)
    print(missing_df.head(29))

    #Identify columns to drop programmatically
    columns_to_drop = [
        col for col, prop in missing_value_proportions.items()
        if prop >= MISSING_VALUE_THRESHOLD
    ]

    print(f"Columns to drop (>{MISSING_VALUE_THRESHOLD*100}% missing): {len(columns_to_drop)}")
    print(columns_to_drop)
    return (columns_to_drop,)


@app.cell(hide_code=True)
def _(columns_to_drop, lf, pl):
    #Additional columns to drop for analysis reasons (not missing values)
    ADDITIONAL_DROPS = [
        'Address',           #Redundant with lat/long
        'Vendor Number',     #Not needed for aggregate analysis
        'Item Number',       #Not needed for aggregate analysis
        'Pack',              #Not relevant for revenue analysis
        'Bottle Volume (ml)', #Have Volume in Liters/Gallons
        'State Bottle Cost',  #Focus on final sale price
        'State Bottle Retail',#Focus on actual sale price
    ]

    #Now to combine the columns with too many nulls with columns that are not needed due to not being required for the analysis
    all_columns_to_drop = list(set(columns_to_drop + ADDITIONAL_DROPS))

    print(f"Total columns to drop: {len(all_columns_to_drop)}")
    print(f"  - Due to missing values: {len(columns_to_drop)}")
    print(f"  - Due to business logic: {len(ADDITIONAL_DROPS)}")

    #Rest of your configuration...
    NUMERIC_COLUMNS = {
        'Bottles Sold': pl.Int16,
        'Sale (Dollars)': pl.Float64,
        'Volume Sold (Liters)': pl.Float64,
        'Volume Sold (Gallons)': pl.Float64,}

    STRING_COLUMNS = [
        'Store Name', 'City', 'Zip Code', 'County', 
        'Category', 'Category Name', 'Vendor Name', 'Item Description',]

    #Single cleaning pipeline
    df_cleaned = (
        lf
        # Drop columns programmatically (not hardcoded!)
        .drop([col for col in all_columns_to_drop if col in lf.collect_schema()])
        # Cast numeric columns
        .with_columns([
            pl.col(col).cast(dtype) 
            for col, dtype in NUMERIC_COLUMNS.items()
        ])
        # Cast string columns
        .with_columns([
            pl.col(col).cast(pl.Utf8) 
            for col in STRING_COLUMNS if col in lf.collect_schema()
        ])
        # Parse date column
        .with_columns([
            pl.col('Date').str.strptime(pl.Date, '%m/%d/%Y').alias('Date')
        ])
        # Engineer temporal features
        .with_columns([
            pl.col('Date').dt.year().alias('Year'),
            pl.col('Date').dt.month().alias('Month'),
            pl.col('Date').dt.quarter().alias('Quarter'),
            pl.col('Date').dt.weekday().alias('Weekday'),
            (pl.col('Date').dt.weekday() >= 5).alias('is_weekend'),
        ])
    )

    print("Cleaning pipeline configured (still lazy)")
    return (df_cleaned,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Automated Category Mapping

    Using regex patterns to consolidate 100+ liquor descriptions into major categories.
    This approach automatically catches variations and is easier to maintain than hardcoded lists.
    """)
    return


@app.cell(hide_code=True)
def _(df_cleaned):
    #First, let's see what categories actually exist in the data
    unique_categories = (
        df_cleaned
        .select('Category')
        .unique()
        .sort('Category')
        .collect()
    )

    print("="*80)
    print(f"UNIQUE CATEGORIES IN DATASET: {len(unique_categories)}")
    print("="*60)
    print(unique_categories.head(20))
    print(f"... and {len(unique_categories) - 20} more")
    return


@app.cell(hide_code=True)
def _(df_cleaned, pl):
    #Map different alcohol types to major categories using regex patterns
    #(?i) makes the pattern case-insensitive
    df_with_categories = df_cleaned.with_columns([
        pl.when(
            pl.col("Category Name").str.contains(
                "(?i)WHISKEY|WHISKIES|WHISKY|BOURBON|SCOTCH")
        )
        .then(pl.lit("WHISKEY"))
        .when(pl.col("Category Name").str.contains("(?i)VODKA"))
        .then(pl.lit("VODKA"))
        .when(pl.col("Category Name").str.contains("(?i)RUM"))
        .then(pl.lit("RUM"))
        .when(pl.col("Category Name").str.contains("(?i)TEQUILA|MEZCAL"))
        .then(pl.lit("TEQUILA & MEZCAL"))
        .when(pl.col("Category Name").str.contains("(?i)GIN"))
        .then(pl.lit("GIN"))
        .when(pl.col("Category Name").str.contains("(?i)BRANDY|BRANDIES|COGNAC"))
        .then(pl.lit("BRANDY & COGNAC"))
        .when(pl.col("Category Name").str.contains("(?i)SCHNAPPS"))
        .then(pl.lit("SCHNAPPS"))
        .when(pl.col("Category Name").str.contains(
            "(?i)AMARETTO|CORDIAL|LIQUEUR|ANISETTE|CREME|ROCK & RYE|TRIPLE SEC")
        )
        .then(pl.lit("LIQUEURS & CORDIALS"))
        .when(pl.col("Category Name").str.contains(
            "(?i)AMERICAN ALCOHOL|AMERICAN DISTILLED SPIRITS|DISTILLED SPIRITS SPECIALTY|IMPORTED DISTILLED SPIRITS|NEUTRAL GRAIN SPIRITS")
        )
        .then(pl.lit("SPECIALTY & OTHER SPIRITS"))
        .when(pl.col("Category Name").str.contains("(?i)COCKTAIL|RTD"))
        .then(pl.lit("READY-TO-DRINK"))
        .when(pl.col("Category Name").str.contains("(?i)IOWA DISTILLER"))
        .then(pl.lit("CRAFT/LOCAL"))
        .when(pl.col("Category Name").str.contains(
            "(?i)DECANTERS|DELISTED|SPECIAL ORDER|HIGH PROOF BEER|HOLIDAY VAP|TEMPORARY")
        )
        .then(pl.lit("ADMINISTRATIVE/NON-PRODUCT"))
        .otherwise(pl.lit("UNCATEGORIZED"))
        .alias("Major_Category")
    ])

    #Verifying the mapping
    print("="*80)
    print("DISTRIBUTION OF MAJOR CATEGORIES")
    print("="*60)
    category_distribution = (
        df_with_categories
        .group_by('Major_Category')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .collect()
    )
    print(category_distribution)

    #Checking to make sure what ended up to UNCATEGORIZED
    print("="*80)
    print("UNCATEGORIZED ITEMS (if any)")
    print("="*80)
    uncategorized = (
        df_with_categories
        .filter(pl.col('Major_Category') == 'UNCATEGORIZED')
        .group_by('Category Name')
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .collect()
    )
    if len(uncategorized) > 0:
        print(uncategorized.head(10))
    else:
        print("All categories successfully mapped!")

    #Showing some sample mappings
    print("\n" + "="*60)
    print("SAMPLE MAPPINGS (Category Name -> Major_Category)")
    print("="*80)
    sample_mappings = (
        df_with_categories
        .select(['Category Name', 'Major_Category'])
        .unique()
        .sort(['Major_Category', 'Category Name'])
        .collect()
    )
    print(sample_mappings.head(20))

    print("Category mapping completed successfully")
    return (df_with_categories,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Descriptive Statistics

    Computing high-level metrics. Note: `.collect()` is called here because we need
    the actual results. LazyFrame evaluation ensures only necessary data is loaded.
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl):
    #Now to compute summary statistics (whole dataframe with columns specified is collected)
    summary_stats = (
        df_with_categories
        .select([
            pl.col('Sale (Dollars)').sum().alias('Total Revenue'),
            pl.col('Volume Sold (Gallons)').sum().alias('Total Volume (Gallons)'),
            pl.col('Bottles Sold').sum().alias('Total Bottles Sold'),
            pl.len().alias('Total Transactions'),
        ])
        .collect()  # ← First time this dataframe is actually loaded into memory (don't run if you have low RAM)
    )

    #Format for display
    total_revenue = summary_stats['Total Revenue'][0]
    total_volume = summary_stats['Total Volume (Gallons)'][0]
    total_bottles = summary_stats['Total Bottles Sold'][0]
    total_transactions = summary_stats['Total Transactions'][0]

    print("="*80)
    print("OVERALL DATASET SUMMARY")
    print("="*80)
    print(f"Total Revenue:       ${total_revenue:,.2f}")
    print(f"Total Volume:        {total_volume:,.2f} gallons")
    print(f"Total Bottles:       {total_bottles:,}")
    print(f"Total Transactions:  {total_transactions:,}")
    print("="*80)

    yearly_stats = (
       df_with_categories
      .group_by('Year').agg([
           pl.col('Sale (Dollars)').sum().alias('Revenue'), 
           pl.col('Volume Sold (Gallons)').sum().alias('Volume_Gallons'),
           pl.col('Bottles Sold').sum().alias('Bottles'),
           pl.len().alias('Transactions'), 

       ]).sort('Year').with_columns([
           ((pl.col('Revenue') - pl.col('Revenue').shift(1)) / pl.col('Revenue').shift(1) * 100).alias('Revenue_Growth_%'), 
           (pl.col('Revenue') / pl.col('Transactions')).alias('Avg_Transaction_Value'),

       ]).collect()
    )

    print("\n" + "="*80)
    print("YEAR-BY-YEAR SUMMARY")
    print("="*80)

    for row in yearly_stats.iter_rows(named=True):
           year = row['Year']
           revenue = row['Revenue']
           volume = row['Volume_Gallons']
           bottles = row['Bottles']
           transactions = row['Transactions']
           growth = row['Revenue_Growth_%']
           avg_txn = row['Avg_Transaction_Value']

           print(f"\n{year}:")
           print(f"  Revenue:              ${revenue:,.2f}")
           if growth is not None:
               print(f"  YoY Growth:           {growth:+.2f}%")
           print(f"  Volume:               {volume:,.2f} gallons")
           print(f"  Bottles Sold:         {bottles:,}")
           print(f"  Transactions:         {transactions:,}")
           print(f"  Avg Transaction:      ${avg_txn:,.2f}")

    print("\n" + "="*80)
    return (yearly_stats,)


@app.cell(hide_code=True)
def _(px, yearly_stats):
    #Create multi-metric chart
    #This only works if you download a verison of the dataset with multiple years. My parquet file only has 1 year to go under GitHubs data file size limit.
    fig_yearly = px.line(
       yearly_stats,
       x='Year',
       y='Revenue',
       title='Annual Revenue Trend (2012-2023)',
       labels={'Revenue': 'Total Revenue ($)', 'Year': 'Year'},
       markers=True
    )

    fig_yearly.update_layout(
       plot_bgcolor='white',
       paper_bgcolor='white',
       xaxis=dict(showgrid=True, dtick=1), 
       yaxis=dict(showgrid=True),
       title_x=0.5,
       height=500
    )

    fig_yearly.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Top 10 Product Categories by Revenue
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl):
    top_categories = (
        df_with_categories
        .group_by('Major_Category')
        .agg([
            pl.col('Sale (Dollars)').sum().alias('Total Revenue')
        ])
        .sort('Total Revenue', descending=True)
        .head(10)
        .collect()
    )

    # Format for display
    top_categories_display = top_categories.with_columns([
        pl.col('Total Revenue').map_elements(
            lambda x: f"${x:,.2f}",
            return_dtype=pl.Utf8
        ).alias('Total Revenue')
    ])

    print("\n" + "="*60)
    print("TOP 10 PRODUCT CATEGORIES BY REVENUE")
    print("="*60)
    print(top_categories_display)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Visualization Gallery

    All visualizations below use the same automation pattern:
    - Lazy evaluation until .collect()
    - Reusable plotting configuration from Plotly Express
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 1: Quarterly Sales Trends (Only available with multi-year data)
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl):
    #Again, only works for a file with multiple years
    quarterly_sales = (
       df_with_categories
       .group_by(['Year', 'Quarter'])
       .agg([
           pl.col('Sale (Dollars)').sum().alias('Total Revenue')
       ])
       .sort(['Year', 'Quarter'])
       .collect()
    )

    #Create a combined Year-Quarter label for plotting
    quarterly_sales = quarterly_sales.with_columns([
       (pl.col('Year').cast(pl.Utf8) + ' Q' + pl.col('Quarter').cast(pl.Utf8)).alias('Period')
    ])

    print(f"Total quarters: {len(quarterly_sales)}")
    return (quarterly_sales,)


@app.cell(hide_code=True)
def _(px, quarterly_sales):
    fig1 = px.line(
       quarterly_sales,
       x='Period',
       y='Total Revenue',
       title='Quarterly Sales Trends (2012-2023)',
       labels={'Total Revenue': 'Total Revenue ($)', 'Period': 'Quarter'}
    )

    fig1.update_layout(
       plot_bgcolor='white',
       paper_bgcolor='white',
       xaxis=dict(showgrid=True, tickangle=-45),
       yaxis=dict(showgrid=True),
       title_x=0.5,
       height=500
    )

    fig1.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 2: Top 20 Products by Revenue
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    top_products_revenue = (
        df_with_categories
        .group_by('Item Description')
        .agg([
            pl.col('Sale (Dollars)').sum().alias('Total Revenue')
        ])
        .sort('Total Revenue', descending=True)
        .head(20)
        .collect()
    )

    fig2 = px.bar(
        top_products_revenue,
        x='Total Revenue',
        y='Item Description',
        orientation='h',
        title='Top 20 Products by Revenue',
        labels={'Total Revenue': 'Total Revenue ($)', 'Item Description': 'Product'}
    )

    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        title_x=0.5,
        height=600
    )

    fig2.show()
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 3: Top 20 Products by Volume
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    top_products_volume = (
        df_with_categories
        .group_by('Item Description')
        .agg([
            pl.col('Volume Sold (Liters)').sum().alias('Total Volume')
        ])
        .sort('Total Volume', descending=True)
        .head(20)
        .collect()
    )

    fig3 = px.bar(
        top_products_volume,
        x='Total Volume',
        y='Item Description',
        orientation='h',
        title='Top 20 Products by Volume (Liters)',
        labels={'Total Volume': 'Total Volume (L)', 'Item Description': 'Product'}
    )

    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        title_x=0.5,
        height=600
    )

    fig3.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 4: Top 15 Counties by Revenue
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    top_counties = (
        df_with_categories
        .group_by('County')
        .agg([
            pl.col('Sale (Dollars)').sum().alias('Total Revenue')
        ])
        .sort('Total Revenue', descending=True)
        .head(15)
        .collect()
    )

    fig4 = px.bar(
        top_counties,
        x='County',
        y='Total Revenue',
        title='Top 15 Counties by Revenue',
        labels={'Total Revenue': 'Total Revenue ($)', 'County': 'County'}
    )

    fig4.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        title_x=0.5
    )

    fig4.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 5: Top 20 Cities by Sales
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    top_cities = (
        df_with_categories
        .group_by('City')
        .agg([
            pl.col('Sale (Dollars)').sum().alias('Total Sales')
        ])
        .sort('Total Sales', descending=True)
        .head(20)
        .collect()
    )

    fig5 = px.bar(
        top_cities,
        x='Total Sales',
        y='City',
        orientation='h',
        title='Top 20 Cities by Sales',
        labels={'Total Sales': 'Total Sales ($)', 'City': 'City'}
    )

    fig5.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        title_x=0.5,
        height=600
    )

    fig5.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 6: Weekday vs Weekend Sales (Standardized) (Only available with multi-year data)
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    #Again, only works when we have multi-year dataset
    weekday_weekend = (
       df_with_categories
       .group_by('is_weekend')
       .agg([
           pl.col('Sale (Dollars)').sum().alias('Total Sales'),
           pl.col('Date').n_unique().alias('Number of Days')
       ])
       .with_columns([
           (pl.col('Total Sales') / pl.col('Number of Days')).alias('Avg Daily Sales'),
           pl.when(pl.col('is_weekend'))
               .then(pl.lit('Weekend'))
               .otherwise(pl.lit('Weekday'))
               .alias('Day Type')
       ])
       .collect()
    )

    fig6 = px.bar(
       weekday_weekend,
       x='Day Type',
       y='Avg Daily Sales',
       title='Weekday vs Weekend Sales (Average Daily)',
       labels={'Avg Daily Sales': 'Average Daily Sales ($)', 'Day Type': ''},
       text='Avg Daily Sales'
    )

    fig6.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')

    fig6.update_layout(
       plot_bgcolor='white',
       paper_bgcolor='white',
       xaxis=dict(showgrid=False),
       yaxis=dict(showgrid=False),
       title_x=0.5
    )

    fig6.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 7: Top 20 Cities by Sales Efficiency
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl, px):
    city_efficiency = (
        df_with_categories
        .group_by('City')
        .agg([
            pl.col('Sale (Dollars)').sum().alias('Total Sales'),
            pl.col('Invoice/Item Number').n_unique().alias('Num Transactions')
        ])
        .with_columns([
            (pl.col('Total Sales') / pl.col('Num Transactions')).alias('Avg Sale per Txn')
        ])
        .sort('Avg Sale per Txn', descending=True)
        .head(20)
        .collect()
    )

    fig7 = px.bar(
        city_efficiency,
        x='Avg Sale per Txn',
        y='City',
        orientation='h',
        title='Top 20 Cities by Sales Efficiency (Avg Sale per Transaction)',
        labels={'Avg Sale per Txn': 'Average Sale per Transaction ($)', 'City': 'City'}
    )

    fig7.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        title_x=0.5,
        height=600
    )

    fig7.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Store Performance Classification: Logistic Regression vs Random Forest

    **Business Question:** Given a store's sales history, can we predict whether it will break out into the top tier of revenue performers next month, and which modeling approach does this best?

    **What "top quartile" means here:** Each month, stores are ranked by total revenue. The top 25 percent of stores for that month are labeled as top performers (label = 1). The threshold is computed dynamically per month rather than globally, so a store is not simply rewarded for being in a high revenue city, it has to outperform its peers that specific month. A store in a small Iowa town can be a top tier performer just as easily as a store in Des Moines.

    **The Breakout Detector Shift:** Predicting if a massive supermarket will stay massive is trivial. Instead, this model filters the dataset to only evaluate stores that were in the bottom 75 percent last month. The model attempts to predict the rare ~4.9 percent of stores that will surge into the top quartile in the very next month.

    **Why this question matters:** In retail distribution, identifying which mid tier stores are about to become high performers has direct operational value. It informs proactive inventory allocation, promotional targeting, and sales rep prioritization. Getting this wrong in either direction has a cost. Missing a rising star (false negative) means lost revenue opportunity, while over allocating to a store that underperforms (false positive) ties up inventory and promotional budget.

    **Why compare two models?**

    We evaluate two classification approaches on the same features, training data, and test window to understand whether the added complexity of a Random Forest is justified over a simpler linear baseline:

    * **Logistic Regression** is the standard industry baseline for binary classification. It assumes a linear relationship between features and the log odds of being a top performer. It is fast, interpretable, and requires feature standardization since it is sensitive to differences in scale across variables.
    * **Random Forest** is an ensemble of decision trees that captures nonlinear relationships and interactions between features automatically. It is scale invariant. Unlike logistic regression, it does not require standardization because it makes decisions based on splits rather than distances. The cost is reduced interpretability relative to logistic regression, which is offset here by the feature importance output.

    The key question the comparison answers is whether the nonlinear structure of store revenue data requires a nonlinear model to capture, or if the relationship is linear enough that logistic regression performs comparably.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Aggregate Transactions to Store-Month Level
    """)
    return


@app.cell(hide_code=True)
def _(df_with_categories, pl):
    #Aggregate from transaction-level to store-month level.
    #This is how to bridge between the raw time-series data and the tabular feature matrix that scikit-learn expects.
    store_month = (
        df_with_categories
        .group_by(['Store Name', 'City', 'County', 'Year', 'Month'])
        .agg([
            pl.col('Sale (Dollars)').sum().alias('monthly_revenue'),
            pl.col('Sale (Dollars)').mean().alias('avg_transaction_value'),
            pl.col('Invoice/Item Number').n_unique().alias('num_transactions'),
            pl.col('Item Description').n_unique().alias('unique_skus'),
            pl.col('Bottles Sold').sum().alias('total_bottles'),
            pl.col('Volume Sold (Liters)').sum().alias('total_liters'),
        ])
        .sort(['Store Name', 'Year', 'Month'])
        .collect(engine="streaming")
    )

    print(f"Store-month panel: {len(store_month):,} rows")
    print(f"Unique stores:     {store_month['Store Name'].n_unique():,}")
    print(f"Date range:        {store_month['Year'].min()}-{store_month['Month'].min():02d} "
          f"to {store_month['Year'].max()}-{store_month['Month'].max():02d}")
    print()
    print(store_month.head(5))
    return (store_month,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Feature Engineering (Lag and Rolling Features)

    All features use only information available before the prediction period.
    Lag features carry forward a store's own recent history, while rolling features
    smooth short term noise. This is the key design choice that makes the model
    usable in practice, because at prediction time you only know the past.
    """)
    return


@app.cell(hide_code=True)
def _(pl, store_month):
    #Build lag and rolling features within each store's time series.
    #pl.Expr.shift(n) looks back n rows within the group — strictly past data.
    store_features = (
        store_month
        .sort(['Store Name', 'Year', 'Month'])
        .with_columns([
            # Lag features: revenue from 1, 2, and 3 months ago
            pl.col('monthly_revenue')
              .shift(1)
              .over('Store Name')
              .alias('revenue_lag1'),
            pl.col('monthly_revenue')
              .shift(2)
              .over('Store Name')
              .alias('revenue_lag2'),
            pl.col('monthly_revenue')
              .shift(3)
              .over('Store Name')
              .alias('revenue_lag3'),

            #Rolling 3-month average revenue (uses lags 1-3, not current month)
            pl.col('monthly_revenue')
              .shift(1)
              .rolling_mean(window_size=3)
              .over('Store Name')
              .alias('revenue_rolling3'),

            # Month-over-month growth rate (lag1 vs lag2)
            (
                (pl.col('monthly_revenue').shift(1) - pl.col('monthly_revenue').shift(2))
                / (pl.col('monthly_revenue').shift(2) + 1e-9)  # avoid div/0
            )
            .over('Store Name')
            .alias('mom_growth_rate'),

            (
        pl.col('monthly_revenue').cum_sum().over('Store Name')
        / pl.col('monthly_revenue').cum_count().over('Store Name')
            )
            .shift(1)
            .over('Store Name')
            .alias('store_avg_revenue'),

            # Lag of transaction count and SKU diversity
            pl.col('num_transactions').shift(1).over('Store Name').alias('txn_lag1'),
            pl.col('unique_skus').shift(1).over('Store Name').alias('skus_lag1'),

            #Month-of-year as a numeric feature (captures seasonality)
            pl.col('Month').alias('month_of_year'),
        ])
        #Drop rows where lag features are null (first 1-3 months per store have no history)
        .drop_nulls(subset=['revenue_lag1', 'revenue_lag2', 'revenue_lag3'])
    )

    print(f"Rows after dropping null lags: {len(store_features):,}")
    print(f"Features engineered: revenue_lag1/2/3, revenue_rolling3, mom_growth_rate, "
          f"store_avg_revenue, txn_lag1, skus_lag1, month_of_year")
    return (store_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Define Target and Temporal Train Test Split

    The target is top tier store performance for a given month. Using a
    dynamic monthly threshold (rather than a global one) controls for seasonality, ensuring
    a store that performs well in December is not artificially inflated because December
    is simply a high revenue month industry wide.

    The train and test split is strictly temporal. The last 3 months of data are held
    out as the test set. **No shuffling.** Shuffling a time series dataset leaks
    future rows into training, inflating reported accuracy in a way that would
    never replicate in production.
    """)
    return


@app.cell
def _(pl, store_features):
    #Month-specific top-quartile threshold (75th percentile per month)
    monthly_thresholds = (
        store_features
        .group_by(['Year', 'Month'])
        .agg(
            pl.col('monthly_revenue').quantile(0.75).alias('q75')
        )
    )

    labeled = (
        store_features
        .join(monthly_thresholds, on=['Year', 'Month'], how='left')
        .with_columns([
            (pl.col('monthly_revenue') >= pl.col('q75'))
            .cast(pl.Int8)
            .alias('is_top_quartile')
        ])
        .sort(['Store Name', 'Year', 'Month'])
        .with_columns([
            # Identify where the store stood in the previous month
            pl.col('is_top_quartile')
            .shift(1)
            .over('Store Name')
            .alias('was_top_quartile_last_month')
        ])
        # BUSINESS LOGIC SHIFT: 
        # We only want to analyze stores that were in the bottom 75% last month.
        # We are predicting if they will "break out" into the top 25% this month.
        .filter(pl.col('was_top_quartile_last_month') == 0)
    )

    # Temporal split: hold out the last 3 calendar months as test
    all_periods = (
        labeled
        .select(['Year', 'Month'])
        .unique()
        .sort(['Year', 'Month'])
    )
    cutoff_idx = len(all_periods) - 3
    cutoff_row = all_periods.row(cutoff_idx)
    cutoff_year, cutoff_month = cutoff_row[0], cutoff_row[1]

    train = labeled.filter(
        (pl.col('Year') < cutoff_year) |
        ((pl.col('Year') == cutoff_year) & (pl.col('Month') < cutoff_month))
    )
    test = labeled.filter(
        (pl.col('Year') > cutoff_year) |
        ((pl.col('Year') == cutoff_year) & (pl.col('Month') >= cutoff_month))
    )

    print(f"Temporal cutoff:  {cutoff_year}-{cutoff_month:02d}")
    print(f"Training rows:    {len(train):,}  ({train['is_top_quartile'].mean():.2%} positive breakouts)")
    print(f"Test rows:        {len(test):,}  ({test['is_top_quartile'].mean():.2%} positive breakouts)")
    return test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: Train Random Forest & Evaluate
    """)
    return


@app.cell
def _(
    LogisticRegression,
    RandomForestClassifier,
    StandardScaler,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    test,
    train,
):


    FEATURE_COLS = [
        'revenue_lag1', 'revenue_lag2', 'revenue_lag3',
        'revenue_rolling3', 'mom_growth_rate',
        'store_avg_revenue',
        'txn_lag1', 'skus_lag1', 'month_of_year',
    ]
    TARGET_COL = 'is_top_quartile'

    #Convert Polars → NumPy for scikit-learn
    X_train = train.select(FEATURE_COLS).to_numpy()
    y_train = train.select(TARGET_COL).to_numpy().ravel()
    X_test  = test.select(FEATURE_COLS).to_numpy()
    y_test  = test.select(TARGET_COL).to_numpy().ravel()

    #Now we're building the logistic regression model
    lr = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1] 


    print("=" * 60)
    print("LOGISTIC REGRESSION — CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred_lr,
          target_names=['Bottom 75%', 'Top Quartile']))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_lr):.4f}")

    #Now we're building the random forest model
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,           #Limits overfitting on smaller datasets
        min_samples_leaf=10,   #Each leaf needs at least 10 store-months
        class_weight='balanced',  #Corrects for class imbalance (25% positive by design)
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    print("=" * 60)
    print("RANDOM FOREST — CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred,
                                target_names=['Bottom 75%', 'Top Quartile']))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  {'':15s}  Pred: Bottom  Pred: Top")
    print(f"  {'Actual: Bottom':15s}  {cm[0,0]:>12,}  {cm[0,1]:>9,}")
    print(f"  {'Actual: Top':15s}  {cm[1,0]:>12,}  {cm[1,1]:>9,}")

    #Feature importances for visualization in next cell
    feature_importances = list(zip(FEATURE_COLS, rf.feature_importances_))

    #Get predicted probabilities (column 1 = probability of top quartile)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print()
    print(f"AUC-ROC: {auc:.4f}")
    return (feature_importances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualization 8: Feature Importances

    Feature importance from a Random Forest measures how much each variable
    reduces impurity (Gini) across all trees, averaged over all splits where
    that feature was used. Higher = more predictive signal for classifying
    top-quartile stores.
    """)
    return


@app.cell(hide_code=True)
def _(feature_importances, pl, px):
    #Build a tidy Polars DataFrame for plotting (consistent with rest of notebook)
    importance_df = (
        pl.DataFrame({
            'Feature': [f for f, _ in feature_importances],
            'Importance': [float(i) for _, i in feature_importances],
        })
        .sort('Importance', descending=True)
    )

    # Human-readable feature labels
    label_map = {
        'revenue_lag1':      'Revenue: 1 Month Ago',
        'revenue_lag2':      'Revenue: 2 Months Ago',
        'revenue_lag3':      'Revenue: 3 Months Ago',
        'revenue_rolling3':  '3-Month Rolling Avg Revenue',
        'mom_growth_rate':   'Month-over-Month Growth Rate',
        'store_avg_revenue': 'Store Historical Avg Revenue',
        'store_std_revenue': 'Store Revenue Volatility (Std)',
        'txn_lag1':          'Transaction Count: 1 Month Ago',
        'skus_lag1':         'Unique SKUs: 1 Month Ago',
        'month_of_year':     'Month of Year (Seasonality)',
    }
    importance_df = importance_df.with_columns([
        pl.col('Feature').replace(label_map).alias('Feature Label')
    ])

    fig8 = px.bar(
        importance_df,
        x='Importance',
        y='Feature Label',
        orientation='h',
        title='Random Forest Feature Importances<br>'
              '<sup>Predicting Top-Quartile Store Revenue Performance</sup>',
        labels={'Importance': 'Mean Gini Importance', 'Feature Label': ''},
        color='Importance',
        color_continuous_scale='Blues',
    )

    fig8.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange='reversed'),
        title_x=0.5,
        height=500,
        coloraxis_showscale=False,
        margin=dict(l=220),
    )

    fig8.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Results & Interpretation: Predicting Revenue Breakouts

    **LR — AUC-ROC: 0.9668 | Precision: 0.34 | Recall: 0.93 | F1: 0.50**
    **RF — AUC-ROC: 0.9675 | Precision: 0.34 | Recall: 0.93 | F1: 0.50**

    By filtering the dataset to only include stores that were in the bottom 75% last month, we shifted the model from a trivial "status predictor" to an actionable "breakout detector." The models are now attempting to predict the ~4.9% of stores that surge into the top quartile in a single month.

    **What the metrics mean in practice:**

    - **Exceptional Recall (93%):** Out of the 199 actual breakout stores in the test set, the models successfully identified 186 of them. It rarely misses a rising star.
    - **High Strategic Lift (34% Precision):** While a 34% precision means there are false positives (predicting a breakout that doesn't quite materialize), it represents a ~7x lift over the baseline breakout rate of 4.9%.
    - **Operational Value:** If a distributor wants to proactively allocate inventory or marketing budget to rising stores, they only need to target the 540 stores flagged by the model, rather than blanketing all 4,000+ lower-tier stores.

    **Methodological Conclusion:**

    The near-identical performance between the linear baseline (Logistic Regression) and the complex ensemble (Random Forest) is a key finding. It proves that the mathematical signals of a store breakout are highly linear. A store's recent lag features and rolling averages linearly push it across the quartile threshold, meaning the heavy computational cost of a Random Forest is not required to accurately forecast retail revenue surges in this market.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
