import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, skew, kurtosis
import matplotlib.dates as mdates
from scipy import stats 

# Load the dataset
df = pd.read_csv("merged_electricity_weather_data.csv")

# Ensure numerical columns are present
numerical_cols = df.select_dtypes(include=['number']).columns
if len(numerical_cols) == 0:
    raise ValueError("No numerical columns found for outlier detection.")

# 1. Enhanced IQR-Based Outlier Detection
def detect_outliers_iqr(df, column):
    """Detect outliers using the IQR method with dynamic thresholding"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Print rationale
    print(f"\nIQR Method Rationale for '{column}':")
    print(f"- 25th percentile: {Q1:.2f}, 75th percentile: {Q3:.2f}")
    print(f"- IQR range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"- Non-parametric approach suitable for non-normal distributions")
    
    return outliers, lower_bound, upper_bound

# 2. Enhanced Z-Score Method
def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-scores with distribution analysis"""
    z_scores = zscore(df[column])
    outliers = df[np.abs(z_scores) > threshold]
    
    # Print rationale
    print(f"\nZ-Score Method Rationale for '{column}':")
    print(f"- Assumes approximately normal distribution")
    print(f"- Threshold |Z| > {threshold} (covers {100 * (1 - 2 * (1 - stats.norm.cdf(threshold))):.1f}% of data)")
    print(f"- Sensitive to sample size (n={len(df)})")
    
    return outliers

# 3. Enhanced Outlier Handling Strategy
def handle_outliers(df, column, method='cap'):
    """
    Handle outliers with method comparison documentation
    Returns both cleaned data and modification report
    """
    original_stats = {
        'mean': df[column].mean(),
        'std': df[column].std(),
        'skew': skew(df[column]),
        'kurtosis': kurtosis(df[column])
    }
    
    if method == 'cap':
        _, lower_bound, upper_bound = detect_outliers_iqr(df, column)
        df[column] = np.clip(df[column], lower_bound, upper_bound)
        print(f"Capped {column} to IQR bounds")
    elif method == 'remove':
        outliers, _, _ = detect_outliers_iqr(df, column)
        df = df.drop(outliers.index)
        print(f"Removed {len(outliers)} outliers from {column}")
    elif method == 'transform':
        df[column] = np.log1p(df[column])
        print(f"Applied log transformation to {column}")
    else:
        raise ValueError("Invalid method. Choose 'cap', 'remove', or 'transform'.")
    
    new_stats = {
        'mean': df[column].mean(),
        'std': df[column].std(),
        'skew': skew(df[column]),
        'kurtosis': kurtosis(df[column])
    }
    
    return df, pd.DataFrame([original_stats, new_stats], index=['Original', 'Cleaned'])

# 4. Enhanced Visualization with Temporal Analysis
def visualize_outliers(df, column):
    """Comprehensive outlier visualization suite"""
    plt.figure(figsize=(18, 12))
    
    # Temporal Analysis
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df[column], label='Original')
    outliers_iqr, _, _ = detect_outliers_iqr(df, column)
    plt.scatter(outliers_iqr.index, outliers_iqr[column], color='red', label='Outliers')
    plt.title(f"Temporal Distribution of {column}")
    plt.xlabel("Time Index")
    plt.ylabel(column)
    plt.legend()
    
    # Distribution Comparison
    plt.subplot(2, 2, 2)
    sns.histplot(df[column], kde=True, color='blue', label='Original')
    df_capped, _ = handle_outliers(df.copy(), column, 'cap')
    sns.histplot(df_capped[column], kde=True, color='green', label='Capped', alpha=0.5)
    plt.title("Distribution Before/After Capping")
    plt.legend()
    
    # Boxplot Comparison
    plt.subplot(2, 2, 3)
    sns.boxplot(data=pd.DataFrame({
        'Original': df[column],
        'Capped': df_capped[column]
    }))
    plt.title("Boxplot Comparison")
    
    # QQ-Plot
    plt.subplot(2, 2, 4)
    stats.probplot(df[column].dropna(), plot=plt)
    plt.title("QQ-Plot of Original Data")
    
    plt.tight_layout()
    plt.show()

# 5. Comprehensive Impact Evaluation
def evaluate_impact(df, column):
    """Enhanced impact analysis with multiple methods"""
    methods = ['cap', 'remove', 'transform']
    results = []
    
    plt.figure(figsize=(15, 5))
    for i, method in enumerate(methods, 1):
        df_clean, stats_df = handle_outliers(df.copy(), column, method)
        
        # Store results
        stats_df['Method'] = method
        results.append(stats_df)
        
        # Plot distributions
        plt.subplot(1, 3, i)
        sns.histplot(df_clean[column], kde=True)
        plt.title(f"{method.capitalize()} Method")
    
    plt.suptitle("Outlier Handling Method Comparison")
    plt.tight_layout()
    plt.show()
    
    # Statistical comparison
    full_results = pd.concat(results)
    print("\nStatistical Comparison:")
    print(full_results.groupby('Method').mean().T.round(2))
    
    return full_results

# Main Execution
if __name__ == "__main__":
    # Configuration
    column = 'value'  # Target variable
    handling_method = 'cap'  # Preferred handling method
    
    # Initial Analysis
    print("="*60)
    print(f"Initial Outlier Analysis for '{column}'")
    print("="*60)
    
    # Detect outliers using both methods
    outliers_iqr, _, _ = detect_outliers_iqr(df, column)
    outliers_z = detect_outliers_zscore(df, column)
    
    print(f"\nIQR Outliers Detected: {len(outliers_iqr)} ({len(outliers_iqr)/len(df):.2%})")
    print(f"Z-Score Outliers Detected: {len(outliers_z)} ({len(outliers_z)/len(df):.2%})")
    
    # Visual Analysis
    visualize_outliers(df, column)
    
    # Impact Evaluation
    impact_report = evaluate_impact(df, column)
    
    # Final Processing
    df_clean, _ = handle_outliers(df, column, handling_method)
    df_clean.to_csv("cleaned_dataset.csv", index=False)
    
    print("\nFinal Dataset Summary:")
    print(f"Original records: {len(df)}")
    print(f"Cleaned records: {len(df_clean)}")
    print(f"Percentage retained: {len(df_clean)/len(df):.2%}")
    print("\n✅ Outlier processing complete!")