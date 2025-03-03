import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set data folder paths
data_folders = ["raw/electricity_raw_data", "raw/weather_raw_data"]

def load_file(file_path):
    """Load a file and standardize timestamps from different sources."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Try loading as JSON (Electricity Data)
        try:
            data = json.loads(content)
            df = pd.json_normalize(data['response']['data'])
            if 'period' in df.columns:
                df.rename(columns={'period': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%dT%H", errors='coerce')
            return df
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Try loading as CSV (Weather Data)
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            
            # Fix weather column name from 'date' to 'timestamp'
            if 'date' in df.columns:
                df.rename(columns={'date': 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            return df
        except pd.errors.ParserError:
            pass
        
        print(f"⚠️ Unsupported or unreadable file format: {file_path}")
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
    return None

def load_data(folders):
    """Load all files from specified folders and merge into a DataFrame."""
    dataframes = []
    for folder in folders:
        all_files = glob.glob(os.path.join(folder, "*"))
        for file in all_files:
            df = load_file(file)
            if df is not None:
                print(f"✅ Loaded {file} with {df.shape[0]} rows and {df.shape[1]} columns.")
                dataframes.append(df)

    if not dataframes:
        print("❌ No valid data found. Exiting...")
        return None

    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df

def analyze_missing_data(df):
    """Analyze missing data patterns to determine MCAR, MAR, or MNAR."""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    print("\n🔍 Missing Data Analysis:")
    print(missing_percent[missing_percent > 0].sort_values(ascending=False))

    # Visualizing missing values
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title("Missing Data Heatmap")
    plt.show()

    # Checking dependency with time (MAR or MNAR)
    if 'timestamp' in df.columns:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                plt.figure(figsize=(12, 5))
                df.set_index('timestamp')[col].isnull().astype(int).plot(title=f"Missing Pattern Over Time - {col}")
                plt.ylabel("Missing (1=Yes)")
                plt.show()

    # Checking MCAR using chi-square test
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            observed = df[col].isnull().astype(int)
            expected = np.random.choice([0, 1], size=len(df), p=[1 - observed.mean(), observed.mean()])
            contingency_table = pd.crosstab(observed, expected)
            _, p_value, _, _ = chi2_contingency(contingency_table)
            
            print(f"\n🔎 MCAR Test for {col}: p-value = {p_value:.5f}")
            if p_value > 0.05:
                print(f"✅ {col} is likely MCAR (random missing data).")
            else:
                print(f"⚠️ {col} may be MAR or MNAR (depends on other variables).")

def handle_missing_data(df, threshold=50):
    """Analyze missing data patterns and decide to drop or impute."""
    if df is None or df.empty:
        print("❌ Error: DataFrame is empty or None. Skipping missing data handling.")
        return df

    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    cols_to_drop = []
    for col in df.columns:
        if missing_percent[col] > 0:
            print(f"\n🔍 Analyzing missing data for column: '{col}'")
            print(f"   - Total rows: {len(df)}")
            print(f"   - Missing count: {missing_data[col]}")
            print(f"   - Missing percentage: {missing_percent[col]:.2f}%")

            if missing_percent[col] > threshold:
                print(f"🔹 Decision: Dropping column '{col}' because missing percentage ({missing_percent[col]:.2f}%) exceeds threshold of {threshold}%.")
                cols_to_drop.append(col)
            else:
                # Handle imputation
                if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                    # Categorical data
                    if df[col].nunique() == 0:
                        print(f"⚠️ Column '{col}' has no valid values. Dropping.")
                        cols_to_drop.append(col)
                    else:
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            mode_val = mode_val[0]
                            count = df[col].value_counts().get(mode_val, 0)
                            print(f"🔹 Decision: Imputing {missing_data[col]} missing values in categorical column '{col}' with mode '{mode_val}' (appears {count} times).")
                            df[col].fillna(mode_val, inplace=True)
                        else:
                            print(f"⚠️ Cannot impute '{col}' - no mode found. Dropping column.")
                            cols_to_drop.append(col)
                else:
                    # Numerical data
                    median_val = df[col].median()
                    print(f"🔹 Decision: Imputing {missing_data[col]} missing values in numerical column '{col}' with median {median_val:.2f}.")
                    df[col].fillna(median_val, inplace=True)

    # Drop columns after iteration to avoid modifying the DataFrame during iteration
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def add_feature_columns(df):
    """Add temporal feature columns (hour, day, month, season) based on timestamp."""
    # Create timestamp if missing
    if 'timestamp' not in df.columns:
        print("⚠️ Warning: 'timestamp' column missing. Creating placeholder with NaT.")
        df['timestamp'] = pd.NaT
    
    # Ensure timestamp is datetime type
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Extract temporal features with Int64 dtype to handle NaNs
    df['hour'] = df['timestamp'].dt.hour.astype('Int64')
    df['day'] = df['timestamp'].dt.day.astype('Int64')
    df['month'] = df['timestamp'].dt.month.astype('Int64')
    
    # Map month to season
    seasons = [
        (12, 1, 2, 'Winter'),
        (3, 4, 5, 'Spring'),
        (6, 7, 8, 'Summer'),
        (9, 10, 11, 'Autumn')
    ]
    
    def get_season(month):
        if pd.isna(month):
            return None
        for group in seasons:
            if month in group[:3]:
                return group[3]
        return None  # Fallback for invalid months
    
    df['season'] = df['month'].apply(get_season).astype('category')
    
    print("✅ Added temporal features: hour, day, month, season.")
    return df

def clean_data(df):
    """Clean and preprocess data."""
    if df is None or df.empty:
        print("❌ Error: DataFrame is empty or None. Skipping cleaning.")
        return df
    
    df = add_feature_columns(df)
    df = handle_missing_data(df)
    df.drop_duplicates(inplace=True)
    
    # Convert object columns to category
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category')
    
    return df

def merge_data(electricity_df, weather_df):
    """Merge electricity and weather data on timestamp."""
    if electricity_df is None or weather_df is None:
        print("❌ Error: One or both DataFrames are None. Cannot merge.")
        return None

    # Convert columns to datetime
    electricity_df['timestamp'] = pd.to_datetime(electricity_df['timestamp'], errors='coerce')
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'], errors='coerce')

    # Remove timezone info if present
    electricity_df['timestamp'] = electricity_df['timestamp'].dt.tz_localize(None)
    weather_df['timestamp'] = weather_df['timestamp'].dt.tz_localize(None)

    # Merge dataframes
    merged_df = pd.merge(electricity_df, weather_df, on='timestamp', how='inner')
    print(f"✅ Merged data shape: {merged_df.shape}")

    return merged_df

def detect_and_handle_outliers(df, numerical_columns):
    """Detect and handle outliers using IQR."""
    if not numerical_columns:
        print("⚠️ Warning: No numerical columns found. Skipping outlier detection.")
        return df

    for col in numerical_columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            print(f"🔍 Outliers detected in '{col}': {len(outliers)} rows ({(len(outliers)/len(df))*100:.2f}%)")
            df[col] = np.clip(df[col], lower, upper)
        else:
            print(f"✅ No outliers detected in '{col}'.")

    return df

def normalize_data(df, numerical_columns, method='standard'):
    """Normalize or standardize numerical columns."""
    if not numerical_columns:
        print("⚠️ Warning: No numerical columns found. Skipping normalization.")
        return df

    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print(f"✅ Normalized columns: {numerical_columns} using {method} scaling.")
    return df

# **Main Execution**
if __name__ == "__main__":
    # Load data
    electricity_data = load_data(["raw/electricity_raw_data"])
    weather_data = load_data(["raw/weather_raw_data"])
    
    if electricity_data is None or weather_data is None:
        print("❌ Error: Failed to load data. Exiting program.")
    else:
        # Merge data
        merged_data = merge_data(electricity_data, weather_data)
        
        if merged_data is not None and not merged_data.empty:
            # Clean data
            merged_data = clean_data(merged_data)
            
            # Detect and handle outliers
            numerical_cols = merged_data.select_dtypes(include=['number']).columns.tolist()
            merged_data = detect_and_handle_outliers(merged_data, numerical_cols)
            
            
            
            # Save cleaned data
            merged_data.to_csv("merged_electricity_weather_data.csv", index=False)
            print("✅ Data merging, cleaning, and preprocessing complete. Saved to merged_electricity_weather_data.csv")
        else:
            print("❌ No data available after merging. Exiting program.")