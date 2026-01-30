
import joblib
import pandas as pd
import numpy as np

def predict_maintenance(input_data):
    """
    Predict maintenance indicators from sensor data.
    
    Parameters:
    -----------
    input_data : dict
        Dictionary with keys:
        - air_temperature_k
        - process_temperature_k
        - rotational_speed_rpm
        - torque_nm
        - tool_wear_min
        - temperature
        - humidity
        - rainfall
    
    Returns:
    --------
    dict with keys:
        - vibration_health
        - thermal_health
        - efficiency_index
        - failure_risk
    """
    # Load model
    model_package = joblib.load('predictive_maintenance_model.pkl')
    
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Ensure correct feature order
    df = df[model_package['feature_names']]
    
    # Make predictions for each target
    predictions = {}
    for target in model_package['target_names']:
        target_models = model_package['all_models'][target]
        weights = target_models['weights']
        
        # Get predictions from each model
        xgb_pred = target_models['models']['xgboost'].predict(df)[0]
        rf_pred = target_models['models']['random_forest'].predict(df)[0]
        gb_pred = target_models['models']['gradient_boosting'].predict(df)[0]
        ridge_pred = target_models['models']['ridge'].predict(df)[0]
        
        # Weighted ensemble
        ensemble_pred = (
            weights[0] * xgb_pred +
            weights[1] * rf_pred +
            weights[2] * gb_pred +
            weights[3] * ridge_pred
        )
        
        predictions[target] = round(float(ensemble_pred), 2)
    
    return predictions


# Example usage
if __name__ == "__main__":
    sample_input = {
        "air_temperature_k": 298.5,
        "process_temperature_k": 310.2,
        "rotational_speed_rpm": 1500,
        "torque_nm": 45.3,
        "tool_wear_min": 120,
        "temperature": 25.3,
        "humidity": 65.0,
        "rainfall": 2.5
    }
    
    result = predict_maintenance(sample_input)
    print("Predictions:", result)
    # Expected output format:
    # {
    #     "vibration_health": 0.32,
    #     "thermal_health": 0.41,
    #     "efficiency_index": 0.67,
    #     "failure_risk": 0.81
    # }
