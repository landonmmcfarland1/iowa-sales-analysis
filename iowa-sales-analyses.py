import marimo

__generated_with = "0.19.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import polars as pl
    import numpy as np
    import plotly.express as px

    return Path, mo, pl, px


@app.cell(hide_code=True)
def _(Path):
    data_file = 'Iowa_Liquor_Sales_2022.parquet'

    #Check if data exists
    if not Path(data_file).exists():
        print("Data file not found!")
        print(f"Please download the dataset from:")
        print("https://data.iowa.gov/Sales-Distribution/Iowa-Liquor-Sales/m3tr-qhgy")
        print(f"And save it to: {data_file}")
        print("OR check the repository for Iowa_Liquor_Sales_2022.parquet. :)")
    else:
        print(f"✓ Data file found: {data_file}")
    return (data_file,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # IOWA LIQUOR SALES ANALYSIS (ORIGINALLY A COURSE PROJECT)

    **Technical Highlights:**
    - **Memory-Efficient Processing**: Analyzes ~8 GB dataset (~27 Million rows) using Polars lazy evaluation
    - **Automated Data Pipeline**: Streamlined cleaning with minimal code duplication
    - **Production-Ready Code**: Follows DRY principles with reusable patterns for easy future edits
    - **Descriptive Statistics & Visualizations**: Uses Plotly Express to create interactive visualizations

    **Technical Stack:** Python, Polars (LazyFrames), Marimo, Plotly

    ---

    ## Dataset Overview

    12 years of Iowa liquor sales (2012-2026) covering:
    - 26.8M transaction records from licensed retailers
    - Product details, pricing, geographic data
    - Sales metrics across counties and cities

    **Data Source:** Iowa Alcoholic Beverages Division
    ##**Important Note (Please Read):**
    - The data attached to my GitHub post is a snapshot of the original data file, as this file is too large for GitHub to handle. While this data is incomplete for your replication, if you were to download the most up-to-date version of the Iowa Liquor dataset, my work would be able to run on a computer with less than 4GB of ram, even though the dataset this was configured for is ~8GB in size and most up to date version of data is ~10GB in size.
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
    #For the parquet file, this is inaccurate, but this is the accurate estimation for a .csv file. 

    lf = pl.scan_parquet(data_file
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

    #yearly_stats = (
     #   df_with_categories
     #  .group_by('Year').agg([
     #       pl.col('Sale (Dollars)').sum().alias('Revenue'), 
     #       pl.col('Volume Sold (Gallons)').sum().alias('Volume_Gallons'),
     #       pl.col('Bottles Sold').sum().alias('Bottles'),
     #       pl.len().alias('Transactions'), 

      #  ]).sort('Year').with_columns([
     #       ((pl.col('Revenue') - pl.col('Revenue').shift(1)) / pl.col('Revenue').shift(1) * 100).alias('Revenue_Growth_%'), 
    #        (pl.col('Revenue') / pl.col('Transactions')).alias('Avg_Transaction_Value'),

      #  ]).collect()
    #)

    #print("\n" + "="*80)
    #print("YEAR-BY-YEAR SUMMARY")
    #print("="*80)

    #for row in yearly_stats.iter_rows(named=True):
    #        year = row['Year']
    #        revenue = row['Revenue']
    #        volume = row['Volume_Gallons']
    #        bottles = row['Bottles']
    #        transactions = row['Transactions']
    #        growth = row['Revenue_Growth_%']
    #        avg_txn = row['Avg_Transaction_Value']
    #
    #        print(f"\n{year}:")
    #        print(f"  Revenue:              ${revenue:,.2f}")
    #        if growth is not None:
    #            print(f"  YoY Growth:           {growth:+.2f}%")
    #        print(f"  Volume:               {volume:,.2f} gallons")
    #        print(f"  Bottles Sold:         {bottles:,}")
    #        print(f"  Transactions:         {transactions:,}")
    #        print(f"  Avg Transaction:      ${avg_txn:,.2f}")
    #
    #print("\n" + "="*80)
    return


@app.cell(hide_code=True)
def _():
    #Create multi-metric chart
    #This only works if you download a verison of the dataset with multiple years. My parquet file only has 1 year to go under GitHub's data file size limit.
    #fig_yearly = px.line(
    #    yearly_stats,
    #    x='Year',
    #    y='Revenue',
    #    title='Annual Revenue Trend (2012-2023)',
    #    labels={'Revenue': 'Total Revenue ($)', 'Year': 'Year'},
    #    markers=True
    #)

    #fig_yearly.update_layout(
    #    plot_bgcolor='white',
    #    paper_bgcolor='white',
    #    xaxis=dict(showgrid=True, dtick=1), 
    #    yaxis=dict(showgrid=True),
    #    title_x=0.5,
    #    height=500
    #)

    #fig_yearly.show()
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
def _():
    #Again, only works for a file with multiple years
    #quarterly_sales = (
    #    df_with_categories
    #    .group_by(['Year', 'Quarter'])
    #    .agg([
    #        pl.col('Sale (Dollars)').sum().alias('Total Revenue')
    #    ])
    #    .sort(['Year', 'Quarter'])
    #    .collect()
    #)
    #
    #Create a combined Year-Quarter label for plotting
    #quarterly_sales = quarterly_sales.with_columns([
    #    (pl.col('Year').cast(pl.Utf8) + ' Q' + pl.col('Quarter').cast(pl.Utf8)).alias('Period')
    #])

    #print(f"Total quarters: {len(quarterly_sales)}")
    return


@app.cell(hide_code=True)
def _():
    #fig1 = px.line(
    #    quarterly_sales,
    #    x='Period',
     #   y='Total Revenue',
    #    title='Quarterly Sales Trends (2012-2023)',
    #    labels={'Total Revenue': 'Total Revenue ($)', 'Period': 'Quarter'}
    #)

    #fig1.update_layout(
    #    plot_bgcolor='white',
    #    paper_bgcolor='white',
    #    xaxis=dict(showgrid=True, tickangle=-45),
    #    yaxis=dict(showgrid=True),
    #    title_x=0.5,
    #    height=500
    #)

    #fig1.show()
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
def _():
    #Again, only works when we have multi-year dataset
    #weekday_weekend = (
    #    df_with_categories
    #    .group_by('is_weekend')
    #    .agg([
    #        pl.col('Sale (Dollars)').sum().alias('Total Sales'),
    #        pl.col('Date').n_unique().alias('Number of Days')
    #    ])
    #    .with_columns([
    #        (pl.col('Total Sales') / pl.col('Number of Days')).alias('Avg Daily Sales'),
    #        pl.when(pl.col('is_weekend'))
    #            .then(pl.lit('Weekend'))
    #            .otherwise(pl.lit('Weekday'))
    #            .alias('Day Type')
    #    ])
    #    .collect()
    #)

    #fig6 = px.bar(
    #    weekday_weekend,
    #    x='Day Type',
    #    y='Avg Daily Sales',
    #    title='Weekday vs Weekend Sales (Average Daily)',
    #    labels={'Avg Daily Sales': 'Average Daily Sales ($)', 'Day Type': ''},
    #    text='Avg Daily Sales'
    #)

    #fig6.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')

    #fig6.update_layout(
    #    plot_bgcolor='white',
    #    paper_bgcolor='white',
    #    xaxis=dict(showgrid=False),
    #    yaxis=dict(showgrid=False),
    #    title_x=0.5
    #)

    #fig6.show()
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
