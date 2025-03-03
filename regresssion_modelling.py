import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
try:
    df = pd.read_csv("merged_electricity_weather_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("✅ Dataset loaded with shape:", df.shape)
except Exception as e:
    print("❌ Data loading error:", e)
    exit()

# Feature Engineering
def create_features(df):
    """Enhanced temporal feature engineering"""
    # Basic temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    
    # Lag features
    df['prev_hour_demand'] = df['value'].shift(1)
    
    # Rolling averages
    df['24h_avg'] = df['value'].rolling(24, min_periods=1).mean()
    
    # Temperature interactions
    df['temp_squared'] = df['temperature_2m'] ** 2
    df['working_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    # Handle NaNs from lag features
    df = df.dropna()
    return df

try:
    df = create_features(df)
    print("✅ Features created:", list(df.filter(regex='hour|day|month|temp|avg').columns))
except Exception as e:
    print("❌ Feature engineering failed:", e)
    exit()

# Model Configuration
target = 'value'
features = ['hour', 'day_of_week', 'month', 'temperature_2m', 
           'prev_hour_demand', '24h_avg', 'working_hour', 'temp_squared']

# Data Validation
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    exit()

X = df[features]
y = df[target]

# Time Series Split
tscv = TimeSeriesSplit(n_splits=5)
print("\n⏳ Time Series Cross-Validation...")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}: Train {len(train_idx)} days, Test {len(test_idx)} days")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42)

# Model Training
def train_evaluate_model(X_train, X_test, y_train, y_test):
    """Enhanced modeling with diagnostics"""
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    
    results = {}
    for name, model in models.items():
        # Training
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'coefs': dict(zip(features, model.coef_)) if hasattr(model, 'coef_') else None
        }
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.title(f"{name} - Actual vs Predicted")
        plt.xlabel("Actual Demand")
        plt.ylabel("Predicted Demand")
        plt.show()
        
    return results

print("\n🔨 Training Models...")
results = train_evaluate_model(X_train, X_test, y_train, y_test)

# Model Diagnostics
# ... (keep all previous imports and data loading code)

# Model Diagnostics (Updated)
def model_diagnostics(X_train, results, y_test):
    """Advanced model analysis with residual handling"""
    # Multicollinearity Check
    vif = pd.DataFrame()
    vif["Feature"] = X_train.columns
    vif["VIF"] = [variance_inflation_factor(X_train.values, i) 
                 for i in range(X_train.shape[1])]
    print("\n🔍 Multicollinearity Analysis:")
    print(vif.sort_values('VIF', ascending=False))
    
    # Feature Importance
    plt.figure(figsize=(10, 6))
    coefs = pd.DataFrame(results['Linear Regression']['coefs'].items(), 
                        columns=['Feature', 'Coefficient'])
    coefs['abs'] = coefs['Coefficient'].abs()
    sns.barplot(x='abs', y='Feature', data=coefs.sort_values('abs', ascending=False))
    plt.title("Feature Importance (Linear Regression)")
    plt.show()
    
    # Residual Analysis and Outlier Detection
    residual_stats = {}
    for name, res in results.items():
        residuals = y_test - res['model'].predict(X_test)
        residual_stats[name] = residuals
        
        # Calculate outlier threshold
        residual_std = np.std(residuals)
        outlier_threshold = 3 * residual_std
        num_outliers = np.sum(np.abs(residuals) > outlier_threshold)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        sns.histplot(residuals, kde=True)
        plt.title(f"{name} Residual Distribution\n({num_outliers} outliers)")
        
        plt.subplot(1, 2, 2)
        plt.scatter(res['model'].predict(X_test), residuals, alpha=0.3)
        plt.axhline(0, color='red')
        plt.axhline(outlier_threshold, color='orange', linestyle='--')
        plt.axhline(-outlier_threshold, color='orange', linestyle='--')
        plt.title(f"{name} Residuals vs Fitted")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.tight_layout()
        plt.show()
    
    return residual_stats

# Run diagnostics and get residual statistics
residual_stats = model_diagnostics(X_train, results, y_test)

# Performance Report with Residual Analysis
print("\n📊 Final Performance Report:")
report = pd.DataFrame.from_dict({k: v for k,v in results.items()}, orient='index')
print(report[['mse', 'rmse', 'r2']].sort_values('r2', ascending=False))

# Expert Recommendations with Residual Analysis
print("\n💡 Expert Recommendations:")
best_model = report['r2'].idxmax()
best_residuals = residual_stats[best_model]
print(f"1. Best performing model: {best_model} (R²={report.loc[best_model]['r2']:.2f})")

if report['r2'].max() < 0.7:
    print("2. Consider adding these features:")
    print("   - Holiday calendar data\n   - Economic indicators\n   - Longer-term weather forecasts")
    
# Outlier detection using best model's residuals
outlier_threshold = 3 * np.std(best_residuals)
if any(np.abs(best_residuals) > outlier_threshold):
    num_outliers = np.sum(np.abs(best_residuals) > outlier_threshold)
    print(f"3. Significant outliers detected ({num_outliers} points > {outlier_threshold:.2f})")
    print("   - Consider implementing robust scaling or outlier handling")
else:
    print("3. No significant outliers detected in best model residuals")

print("4. Next steps: Test tree-based models (Random Forest, XGBoost) for potential performance gains")