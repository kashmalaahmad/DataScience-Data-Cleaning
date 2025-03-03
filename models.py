import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Models from scikit-learn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



# Function to evaluate and plot model performance
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    try:
        # Fit the model
        model.fit(X_train, y_train)
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluation Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== {model_name} ===")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")
        
        # Plot Actual vs. Predicted
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted", color="blue")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit")
        plt.xlabel("Actual Electricity Demand")
        plt.ylabel("Predicted Electricity Demand")
        plt.title(f"Actual vs. Predicted Electricity Demand ({model_name})")
        plt.legend()
        plt.show()
        
        # Residual Analysis
        residuals = y_test - y_pred
        
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True, color="green")
        plt.xlabel("Residuals")
        plt.title(f"Residuals Distribution ({model_name})")
        plt.show()
        
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, alpha=0.5, color="purple")
        plt.xlabel("Predicted Electricity Demand")
        plt.ylabel("Residuals")
        plt.title(f"Residuals vs. Predicted ({model_name})")
        plt.axhline(0, color="red", linestyle="--")
        plt.show()
        
        print("Residuals Summary Statistics:")
        print(residuals.describe())
    except Exception as e:
        print(f"❌ Error evaluating {model_name}: {e}")

# Load the processed dataset
try:
    df = pd.read_csv("merged_electricity_weather_data.csv")
    print("✅ Successfully loaded the processed dataset.")
except Exception as e:
    print("❌ Error loading processed dataset:", e)
    exit()

# Ensure the timestamp column is in datetime format
if 'timestamp' in df.columns:
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        print("❌ Error converting 'timestamp' to datetime:", e)
        exit()
else:
    print("❌ 'timestamp' column is missing from the dataset.")
    exit()

# Create day_of_week feature if not present
if 'day_of_week' not in df.columns:
    try:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        print("✅ 'day_of_week' column created from 'timestamp'.")
    except Exception as e:
        print("❌ Error creating 'day_of_week' column:", e)
        exit()

# Define target variable and predictor features
# Here, we use 'value' as the electricity demand target
target_column = 'value'
predictor_features = ['hour', 'day', 'month', 'day_of_week', 'temperature_2m']

# Check if all required columns exist
required_columns = [target_column] + predictor_features
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print("❌ The following required columns are missing from the dataset:", missing_cols)
    exit()

# Select predictors and target
X = df[predictor_features]
y = df[target_column]

# Split data into training and testing sets (80% train, 20% test)
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("✅ Data split into training and testing sets.")
except Exception as e:
    print("❌ Error during train-test split:", e)
    exit()

# --- Model 1: Decision Tree Regression ---
dt_model = DecisionTreeRegressor(random_state=42)
evaluate_model(dt_model, X_train, y_train, X_test, y_test, model_name="Decision Tree Regression")

# --- Model 2: Random Forest Regression ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
evaluate_model(rf_model, X_train, y_train, X_test, y_test, model_name="Random Forest Regression")



# --- Model 5: Neural Network Regression (MLPRegressor) ---
# A simple multilayer perceptron with one hidden layer.
mlp_model = MLPRegressor(hidden_layer_sizes=(50,), activation='relu', solver='adam',
                         max_iter=500, random_state=42)
evaluate_model(mlp_model, X_train, y_train, X_test, y_test, model_name="Neural Network Regression (MLP)")
