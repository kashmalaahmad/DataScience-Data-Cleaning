import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis, zscore
from sklearn.preprocessing import StandardScaler
import warnings
from scipy import stats  

warnings.filterwarnings("ignore")

# Load the merged dataset
merged_file = "merged_electricity_weather_data.csv"
df = pd.read_csv(merged_file)

# Time Series Preparation
def prepare_time_series(df):
    """Handle datetime conversion and data quality checks"""
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Data quality checks
    print(f"Missing timestamps: {df['timestamp'].isnull().sum()}")
    print(f"Duplicate timestamps: {df.duplicated(subset=['timestamp']).sum()}")
    
    # Sort and set index
    df = df.sort_values('timestamp').dropna(subset=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        print(f"\nImputing {df.isnull().sum().sum()} missing values...")
        df = df.interpolate(method='time').ffill().bfill()
    
    return df

# 1. Enhanced Statistical Summary
def statistical_summary(df):
    """Compute and interpret statistical metrics"""
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Basic statistics
    stats = df[numerical_cols].agg(['mean', 'median', 'std', 'min', 'max'])
    stats.loc['skewness'] = df[numerical_cols].apply(skew)
    stats.loc['kurtosis'] = df[numerical_cols].apply(kurtosis)
    
    # Interpretation
    print("\nStatistical Summary & Interpretation:")
    print(stats.round(2))
    
    # Distribution analysis
    print("\nDistribution Characteristics:")
    for col in numerical_cols:
        skew_val = skew(df[col])
        kurt_val = kurtosis(df[col])
        
        skew_int = "symmetric" if abs(skew_val) < 0.5 else \
                   "moderately skewed" if abs(skew_val) < 1 else \
                   "highly skewed"
        kurt_int = "mesokurtic" if abs(kurt_val) < 0.5 else \
                   "leptokurtic" if kurt_val > 0.5 else \
                   "platykurtic"
        
        print(f"\n{col}:")
        print(f"- Distribution: {skew_int} ({skew_val:.2f}), {kurt_int} ({kurt_val:.2f})")
        print(f"- Range: {df[col].max()-df[col].min():.2f} (IQR: {df[col].quantile(0.75)-df[col].quantile(0.25):.2f})")
        print(f"- Potential outliers: {len(df[col][np.abs(zscore(df[col])) > 3])}")

# 2. Enhanced Time Series Analysis
def time_series_analysis(df):
    """Advanced time series visualization"""
    plt.figure(figsize=(16, 8))
    
    # Raw data
    plt.plot(df.index, df['value'], label='Hourly Demand', alpha=0.5)
    
    # Trend line (30-day moving average)
    if 'value' in df.columns:
        df['30D_MA'] = df['value'].rolling(window=24*30, min_periods=1).mean()
        plt.plot(df.index, df['30D_MA'], 'r-', label='30-Day Trend')
        
        # Annotate anomalies
        z_scores = np.abs(zscore(df['value']))
        anomalies = df[z_scores > 3]
        plt.scatter(anomalies.index, anomalies['value'], color='black', 
                   label='Anomalies (Z>3)')
        
        # Seasonal annotation
        if 'season' not in df.columns:
            df['season'] = df.index.month.map({12:1,1:1,2:1, 3:2,4:2,5:2, 
                                              6:3,7:3,8:3, 9:4,10:4,11:4}).map(
                {1:'Winter', 2:'Spring', 3:'Summer', 4:'Autumn'})
            
        for season, color in zip(['Winter', 'Spring', 'Summer', 'Autumn'],
                                ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e']):
            season_mask = df['season'] == season
            plt.fill_between(df.index, 0, df['value'].max(), 
                            where=season_mask, color=color, alpha=0.1)
        
        plt.title("Electricity Demand Analysis\n(Trend, Seasonality & Anomalies)")
        plt.xlabel("Date")
        plt.ylabel("Demand (MW)")
        plt.legend()
        plt.tight_layout()
        plt.show()

# 3. Enhanced Univariate Analysis
def univariate_analysis(df):
    """Detailed distribution analysis"""
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(14, 8))
        
        # Create grid
        grid = plt.GridSpec(3, 3, hspace=0.5, wspace=0.3)
        main_ax = plt.subplot(grid[:2, :2])
        box_ax = plt.subplot(grid[0, 2])
        qq_ax = plt.subplot(grid[1, 2])
        stats_ax = plt.subplot(grid[2, :])
        
        # Time series view
        main_ax.plot(df.index, df[col], alpha=0.7)
        main_ax.set_title(f"{col} - Temporal Distribution")
        main_ax.set_ylabel(col)
        
        # Boxplot
        sns.boxplot(y=df[col], ax=box_ax, color='orange')
        box_ax.set_title("Boxplot")
        
        # QQ-Plot
        stats.probplot(df[col].dropna(), plot=qq_ax)
        qq_ax.set_title("QQ-Plot")
        
        # Statistics display
        stats_text = f"""
        Distribution Statistics:
        - Mean: {df[col].mean():.2f}
        - Median: {df[col].median():.2f}
        - Std Dev: {df[col].std():.2f}
        - Skewness: {skew(df[col].dropna()):.2f}
        - Kurtosis: {kurtosis(df[col].dropna()):.2f}
        - IQR: {df[col].quantile(0.75) - df[col].quantile(0.25):.2f}
        - Outliers (Z>3): {len(df[col][np.abs(zscore(df[col])) > 3])}
        """
        stats_ax.text(0.1, 0.5, stats_text, ha='left', va='center')
        stats_ax.axis('off')
        
        plt.suptitle(f"Univariate Analysis: {col}", y=1.02)
        plt.show()

# 4. Enhanced Correlation Analysis
def correlation_analysis(df):
    """Advanced correlation assessment"""
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if len(numerical_cols) > 1:
        # Calculate correlations
        corr_matrix = df[numerical_cols].corr(method='spearman')
        
        # Cluster features
        g = sns.clustermap(corr_matrix, cmap='coolwarm', 
                          annot=True, figsize=(12, 10), vmin=-1, vmax=1)
        plt.title("Correlation Cluster Map")
        plt.show()
        
        # Multicollinearity report
        high_corr = corr_matrix[(corr_matrix.abs() > 0.8) & (corr_matrix < 1)]
        if not high_corr.isnull().all().all():
            print("Critical Multicollinearity Issues:")
            print(high_corr.dropna(how='all').dropna(axis=1, how='all'))
            print("\nImplications:")
            print("- May cause instability in regression models")
            print("- Can inflate standard errors of coefficients")
            print("- Consider feature elimination or dimensionality reduction")
        else:
            print("No critical multicollinearity detected (|r| > 0.8)")

# 5. Enhanced Advanced Time Series
def advanced_time_series_analysis(df):
    """Comprehensive time series diagnostics"""
    if 'value' not in df.columns:
        return
    
    # Stationarity test
    print("\nStationarity Analysis:")
    adf_result = adfuller(df['value'].dropna())
    print(f"ADF Statistic: {adf_result[0]:.2f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key}: {value:.2f}")
        
    if adf_result[1] < 0.05:
        print("Conclusion: Stationary (reject null hypothesis)")
    else:
        print("Conclusion: Non-Stationary (cannot reject null hypothesis)")
    
    # Time series decomposition
    print("\nDecomposition Analysis:")
    try:
        # Auto-detect seasonal period
        freq = df.asfreq('H').index.inferred_freq  # Auto-detect frequency
        period = 24 if freq == 'H' else 7*24 if freq == 'D' else None
        
        if period:
            decomposition = seasonal_decompose(df['value'], model='additive', period=period)
            decomposition.plot()
            plt.suptitle("Time Series Decomposition")
            plt.tight_layout()
            plt.show()
        else:
            print("Could not automatically determine seasonal period")
    except ValueError as e:
        print(f"Decomposition error: {str(e)}")

# Main Execution
if __name__ == "__main__":
    # Prepare data
    df = prepare_time_series(df)
    
    # Perform analyses
    statistical_summary(df)
    time_series_analysis(df)
    univariate_analysis(df)
    correlation_analysis(df)
    advanced_time_series_analysis(df)