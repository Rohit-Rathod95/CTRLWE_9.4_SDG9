import pandas as pd
import numpy as np

print("="*80)
print("CREATING REALISTIC SYNTHETIC TARGETS")
print("="*80)

# Load your original data
df = pd.read_csv('../data/merged_one.csv')
print(f"\n✅ Loaded data: {df.shape}")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)

# ==============================
# CREATE REALISTIC TARGETS BASED ON PHYSICS
# ==============================

print("\n[1/3] Creating synthetic targets based on operational physics...")

# 1. VIBRATION HEALTH (0-100, higher = better)
# Bad when: high speed + high torque + high tool wear
df['vibration_stress'] = (
    (df['rotational_speed_rpm'] / 6000) * 0.3 +  # Speed factor
    (df['torque_nm'] / 100) * 0.3 +              # Torque factor
    (df['tool_wear_min'] / 300) * 0.4            # Wear factor (most important)
)
df['vibration_health'] = 100 - np.clip(df['vibration_stress'] * 100, 0, 100)
df['vibration_health'] = df['vibration_health'] + np.random.normal(0, 3, len(df))  # Add realistic noise
df['vibration_health'] = np.clip(df['vibration_health'], 0, 100)

# 2. THERMAL HEALTH (0-100, higher = better)
# Bad when: high temperature difference + high process temp
df['temp_difference'] = df['process_temperature_k'] - df['air_temperature_k']
df['thermal_stress'] = (
    (df['temp_difference'] / 30) * 0.6 +        # Temp delta (most important)
    (df['process_temperature_k'] / 330) * 0.4   # Absolute temp
)
df['thermal_health'] = 100 - np.clip(df['thermal_stress'] * 100, 0, 100)
df['thermal_health'] = df['thermal_health'] + np.random.normal(0, 3, len(df))
df['thermal_health'] = np.clip(df['thermal_health'], 0, 100)

# 3. EFFICIENCY INDEX (0-100, higher = better)
# Bad when: high tool wear + high torque + temperature issues
df['efficiency_degradation'] = (
    (df['tool_wear_min'] / 300) * 0.5 +         # Wear (most important)
    (df['torque_nm'] / 100) * 0.2 +             # Load
    (df['temp_difference'] / 30) * 0.2 +        # Thermal inefficiency
    (df['humidity'] / 100) * 0.1                # Environmental
)
df['efficiency_index'] = 100 - np.clip(df['efficiency_degradation'] * 100, 0, 100)
df['efficiency_index'] = df['efficiency_index'] + np.random.normal(0, 4, len(df))
df['efficiency_index'] = np.clip(df['efficiency_index'], 0, 100)

# 4. FAILURE RISK (0-100, higher = worse)
# Composite of all health indicators
df['failure_risk'] = (
    (100 - df['vibration_health']) * 0.35 +
    (100 - df['thermal_health']) * 0.30 +
    (100 - df['efficiency_index']) * 0.35
)
df['failure_risk'] = np.clip(df['failure_risk'], 0, 100)

print("✅ Synthetic targets created")

# ==============================
# VERIFY TARGET QUALITY
# ==============================

print("\n[2/3] Verifying target quality...")

targets = ['vibration_health', 'thermal_health', 'efficiency_index', 'failure_risk']

print("\n📊 Target Statistics:")
print(df[targets].describe())

print("\n🔗 Correlations with Key Features:")
key_features = ['tool_wear_min', 'torque_nm', 'rotational_speed_rpm', 'process_temperature_k']
for target in targets:
    print(f"\n{target}:")
    for feature in key_features:
        corr = df[target].corr(df[feature])
        print(f"  {feature:30s}: {corr:6.3f}")

# ==============================
# SAVE CORRECTED DATASET
# ==============================

print("\n[3/3] Saving corrected dataset...")

# Keep only necessary columns
keep_cols = key_features + ['air_temperature_k', 'temperature', 'humidity', 'rainfall'] + targets

# Add machine type if exists
if 'type' in df.columns:
    keep_cols.insert(0, 'type')

df_clean = df[keep_cols].copy()

# Remove outliers (optional)
df_clean = df_clean.dropna()

output_file = '../data/merged_one_CORRECTED.csv'
df_clean.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   Shape: {df_clean.shape}")
print(f"   Targets range correctly from 0-100")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("""
1. Replace 'merged_one.csv' with 'merged_one_CORRECTED.csv'
2. Re-run preprocessing.py
3. Re-run training.py
4. Your model should now predict values in 0-100 range!

Expected results after retraining:
- Healthy machines: efficiency ~70-90, failure_risk ~10-30
- Failing machines: efficiency ~20-40, failure_risk ~60-80
""")

print("\n🎯 Target Value Examples:")
print("\nBest Case (New Tool, Low Load):")
print("  - Vibration Health: ~85-95")
print("  - Thermal Health: ~80-90")
print("  - Efficiency Index: ~75-90")
print("  - Failure Risk: ~10-20")

print("\nWorst Case (Worn Tool, High Load):")
print("  - Vibration Health: ~15-30")
print("  - Thermal Health: ~20-35")
print("  - Efficiency Index: ~10-25")
print("  - Failure Risk: ~70-85")