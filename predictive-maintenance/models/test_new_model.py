import joblib
import pandas as pd
import numpy as np

print("="*80)
print("TESTING NEW MODEL - REALISTIC SCENARIOS")
print("="*80)

model_package = joblib.load('predictive_maintenance_model.pkl')

# Test 1: PERFECT MACHINE (should be healthy)
perfect = pd.DataFrame([{
    "air_temperature_k": 298.0,
    "process_temperature_k": 300.0,  # Only 2K diff
    "rotational_speed_rpm": 1500,
    "torque_nm": 20.0,
    "tool_wear_min": 10,  # Almost new
    "temperature": 25.0,
    "humidity": 50.0,
    "rainfall": 0
}])

# Test 2: TERRIBLE MACHINE (should be failing)
terrible = pd.DataFrame([{
    "air_temperature_k": 298.0,
    "process_temperature_k": 330.0,  # 32K diff - overheating!
    "rotational_speed_rpm": 5000,
    "torque_nm": 90.0,
    "tool_wear_min": 280,  # Nearly destroyed
    "temperature": 35.0,
    "humidity": 85.0,
    "rainfall": 5.0
}])

# Test 3: YOUR HEALTHY PUMP (from new dataset)
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
print("TEST PREDICTIONS (0-100 SCALE)")
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
        clipped = np.clip(ensemble_pred, 0, 100)
        
        print(f"  {target:20s}: {clipped:6.1f}")
    
    # Calculate overall health
    vib = np.clip(np.dot(weights, [
        models["xgboost"].predict(data)[0] for models in 
        [model_package['all_models']['vibration_health']['models']]
    ][0].values()), 0, 100)
    
    therm = np.clip(np.dot(
        model_package['all_models']['thermal_health']['weights'],
        [m.predict(data)[0] for m in model_package['all_models']['thermal_health']['models'].values()]
    ), 0, 100)
    
    eff = np.clip(np.dot(
        model_package['all_models']['efficiency_index']['weights'],
        [m.predict(data)[0] for m in model_package['all_models']['efficiency_index']['models'].values()]
    ), 0, 100)
    
    risk = np.clip(np.dot(
        model_package['all_models']['failure_risk']['weights'],
        [m.predict(data)[0] for m in model_package['all_models']['failure_risk']['models'].values()]
    ), 0, 100)
    
    health = (vib + therm + eff) / 3
    
    print(f"\n  Overall Health Score: {health:.1f}/100")
    if health >= 70:
        print(f"  Risk Assessment: ✅ LOW RISK")
    elif health >= 50:
        print(f"  Risk Assessment: ⚠️  MEDIUM RISK")
    else:
        print(f"  Risk Assessment: 🚨 HIGH RISK")

print("\n" + "="*80)
print("INTERPRETATION:")
print("="*80)
print("""
EXPECTED RESULTS:
- Perfect Machine: Health ~85-95, All metrics >70, LOW RISK ✅
- Terrible Machine: Health ~15-30, All metrics <40, HIGH RISK 🚨
- Healthy Pump: Health ~70-80, All metrics >60, LOW/MEDIUM RISK ✅

If you see these patterns, YOUR MODEL IS WORKING PERFECTLY!
""")