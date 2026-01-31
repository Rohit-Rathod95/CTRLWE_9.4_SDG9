import joblib
import pandas as pd
import numpy as np

print("="*80)
print("DIAGNOSTIC TEST - CHECKING MODEL OUTPUT RANGES")
print("="*80)

# Load model
model_package = joblib.load('predictive_maintenance_model.pkl')

# Perfect machine (should get HIGH health, LOW risk)
perfect = pd.DataFrame([{
    "air_temperature_k": 298.0,
    "process_temperature_k": 300.0,  # Only 2K diff
    "rotational_speed_rpm": 1500,
    "torque_nm": 20.0,  # Low stress
    "tool_wear_min": 10,  # Almost new
    "temperature": 25.0,
    "humidity": 50.0,
    "rainfall": 0
}])

# Terrible machine (should get LOW health, HIGH risk)  
terrible = pd.DataFrame([{
    "air_temperature_k": 298.0,
    "process_temperature_k": 330.0,  # 32K diff - overheating!
    "rotational_speed_rpm": 5000,
    "torque_nm": 90.0,  # Extreme stress
    "tool_wear_min": 300,  # Nearly destroyed
    "temperature": 35.0,
    "humidity": 85.0,
    "rainfall": 10
}])

# Your healthy pump
healthy_pump = pd.DataFrame([{
    "air_temperature_k": 299.1,
    "process_temperature_k": 304.5,
    "rotational_speed_rpm": 2500,
    "torque_nm": 28.5,
    "tool_wear_min": 42,
    "temperature": 25.9,
    "humidity": 54,
    "rainfall": 0
}])

print("\n" + "="*80)
print("RAW MODEL OUTPUTS (BEFORE NORMALIZATION)")
print("="*80)

for scenario_name, data in [("PERFECT MACHINE", perfect), 
                             ("TERRIBLE MACHINE", terrible),
                             ("HEALTHY PUMP (PMP-001)", healthy_pump)]:
    print(f"\n{scenario_name}:")
    print("-" * 60)
    
    for target in model_package['target_names']:
        target_block = model_package['all_models'][target]
        models = target_block['models']
        weights = target_block['weights']
        
        preds = np.array([
            models["xgboost"].predict(data)[0],
            models["random_forest"].predict(data)[0],
            models["gradient_boosting"].predict(data)[0],
            models["ridge"].predict(data)[0]
        ])
        
        ensemble_pred = np.dot(weights, preds)
        
        print(f"  {target:25s}: {ensemble_pred:10.4f}")

print("\n" + "="*80)
print("INTERPRETATION GUIDE:")
print("="*80)
print("""
For PERFECT machine:
  - vibration/thermal/efficiency should be HIGH (>70)
  - failure_risk should be LOW (<30)

For TERRIBLE machine:
  - vibration/thermal/efficiency should be LOW (<30)
  - failure_risk should be HIGH (>70)

For HEALTHY PUMP:
  - Should look similar to PERFECT machine
  - Efficiency should be >60
""")

print("\n📊 Now check: Do the values make sense?")
print("   If not, share this output and I'll tell you the fix!")