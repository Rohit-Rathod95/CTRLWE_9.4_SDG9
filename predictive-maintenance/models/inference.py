import joblib
import pandas as pd
import numpy as np
import os

# ==============================
# Load Model Package and Scaler
# ==============================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "predictive_maintenance_model.pkl"
)

SCALER_PATH = os.path.join(
    os.path.dirname(__file__),
    "feature_scaler.pkl"
)

def load_model():
    return joblib.load(MODEL_PATH)

def load_scaler():
    return joblib.load(SCALER_PATH)

model_package = load_model()
scaler = load_scaler()

REQUIRED_FEATURES = model_package["feature_names"]
TARGETS = model_package["target_names"]

# ==============================
# Main Prediction Function (FULLY FIXED)
# ==============================
def predict_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs ensemble prediction for all rows in dataframe.
    Returns dataframe with 4 prediction columns (0–100 scale).
    
    FULLY FIXED: 
    - Scales input features using saved scaler
    - No broken normalization
    - Proper column name mapping for Streamlit compatibility
    """

    # Check for missing required features
    missing_features = set(REQUIRED_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}. "
                        f"Required features are: {REQUIRED_FEATURES}")

    # Ensure correct feature order
    df = df[REQUIRED_FEATURES].copy()

    # Handle NaN / infinite values safely
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median()).fillna(0)

    # ✅ CRITICAL FIX: Scale features using the saved scaler
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=REQUIRED_FEATURES)

    results = {target: [] for target in TARGETS}

    for _, row in df_scaled.iterrows():
        row_df = pd.DataFrame([row])

        for target in TARGETS:
            target_block = model_package["all_models"][target]
            models = target_block["models"]
            weights = target_block["weights"]

            preds = np.array([
                models["xgboost"].predict(row_df)[0],
                models["random_forest"].predict(row_df)[0],
                models["gradient_boosting"].predict(row_df)[0],
                models["ridge"].predict(row_df)[0]
            ])

            ensemble_pred = np.dot(weights, preds)

            # ✅ Just clip to 0-100
            score_0_100 = float(np.clip(ensemble_pred, 0, 100))
            
            results[target].append(score_0_100)

    results_df = pd.DataFrame(results)
    
    # ✅ Rename columns to match Streamlit expectations
    results_df = results_df.rename(columns={
        'vibration_health': 'vibration_index',
        'thermal_health': 'thermal_index',
        'efficiency_index': 'efficiency_index',
        'failure_risk': 'failure_risk'
    })
    
    return results_df

# ==============================
# CLI Test
# ==============================
if __name__ == "__main__":
    # Test with healthy pump (RAW VALUES - will be scaled automatically)
    test_df = pd.DataFrame([{
        "air_temperature_k": 299.1,
        "process_temperature_k": 304.5,
        "rotational_speed_rpm": 2500,
        "torque_nm": 28.5,
        "tool_wear_min": 42,
        "temperature": 25.9,
        "humidity": 54,
        "rainfall": 0
    }])

    print("\n🔍 Test Prediction Output (Healthy Pump - RAW INPUT):")
    print("Input values (before scaling):")
    print(test_df.iloc[0])
    
    print("\n📊 Predictions:")
    result = predict_from_dataframe(test_df)
    print(result)
    
    print("\n✅ Expected: High efficiency (>60), Low failure risk (<40)")
    print("📋 Column names: vibration_index, thermal_index, efficiency_index, failure_risk")