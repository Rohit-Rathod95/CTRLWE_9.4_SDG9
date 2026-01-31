import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

np.random.seed(42)

print("="*80)
print("CREATING COMPLETE SYNTHETIC TRAINING DATASET")
print("="*80)

# ==============================
# 1. GENERATE REALISTIC SENSOR DATA
# ==============================

print("\n[1/5] Generating 10,000 synthetic machine samples...")

n_samples = 10000

# Define realistic ranges for each machine type
machine_types = ['motor', 'pump', 'compressor', 'turbine', 'conveyor', 'fan', 'generator', 'mixer']

data = []

for i in range(n_samples):
    # Random machine type
    machine_type = np.random.choice(machine_types)
    
    # Base parameters vary by machine type
    if machine_type == 'motor':
        base_speed = np.random.uniform(1200, 1800)
        base_torque = np.random.uniform(30, 50)
    elif machine_type == 'pump':
        base_speed = np.random.uniform(2300, 2700)
        base_torque = np.random.uniform(25, 40)
    elif machine_type == 'compressor':
        base_speed = np.random.uniform(2900, 3300)
        base_torque = np.random.uniform(45, 60)
    elif machine_type == 'turbine':
        base_speed = np.random.uniform(5000, 6000)
        base_torque = np.random.uniform(60, 75)
    elif machine_type == 'conveyor':
        base_speed = np.random.uniform(700, 900)
        base_torque = np.random.uniform(20, 35)
    elif machine_type == 'fan':
        base_speed = np.random.uniform(1500, 1700)
        base_torque = np.random.uniform(15, 25)
    elif machine_type == 'generator':
        base_speed = np.random.uniform(1700, 1850)
        base_torque = np.random.uniform(65, 85)
    else:  # mixer
        base_speed = np.random.uniform(850, 1000)
        base_torque = np.random.uniform(35, 50)
    
    # Generate features with realistic correlations
    tool_wear = np.random.uniform(5, 280)  # minutes
    
    # Temperature increases with wear and load
    air_temp = np.random.uniform(296, 302)  # K
    temp_increase = (tool_wear / 100) * 3 + (base_torque / 50) * 5
    process_temp = air_temp + temp_increase + np.random.normal(0, 2)
    
    # Environmental
    temperature = air_temp - 273.15  # Convert to Celsius
    humidity = np.random.uniform(45, 80)
    rainfall = np.random.choice([0, 0, 0, 0, 0, 0.5, 1.0, 2.5, 5.0], p=[0.7, 0.1, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])
    
    # Add some noise to speed and torque
    speed = base_speed + np.random.normal(0, base_speed * 0.05)
    torque = base_torque + np.random.normal(0, base_torque * 0.1)
    
    data.append({
        'air_temperature_k': air_temp,
        'process_temperature_k': process_temp,
        'rotational_speed_rpm': speed,
        'torque_nm': torque,
        'tool_wear_min': tool_wear,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    })

df = pd.DataFrame(data)
print(f"✅ Created {len(df)} samples with 8 features")

# ==============================
# 2. CREATE REALISTIC TARGETS (PHYSICS-BASED)
# ==============================

print("\n[2/5] Generating realistic targets based on physics...")

# VIBRATION HEALTH (0-100, higher = better)
df['vibration_stress'] = (
    (df['rotational_speed_rpm'] / 6000) * 0.25 +
    (df['torque_nm'] / 100) * 0.30 +
    (df['tool_wear_min'] / 300) * 0.45
)
df['vibration_health'] = 100 - np.clip(df['vibration_stress'] * 100, 0, 100)
df['vibration_health'] += np.random.normal(0, 4, len(df))
df['vibration_health'] = np.clip(df['vibration_health'], 0, 100)

# THERMAL HEALTH (0-100, higher = better)
df['temp_diff'] = df['process_temperature_k'] - df['air_temperature_k']
df['thermal_stress'] = (
    (df['temp_diff'] / 25) * 0.70 +
    ((df['process_temperature_k'] - 300) / 30) * 0.30
)
df['thermal_health'] = 100 - np.clip(df['thermal_stress'] * 100, 0, 100)
df['thermal_health'] += np.random.normal(0, 3.5, len(df))
df['thermal_health'] = np.clip(df['thermal_health'], 0, 100)

# EFFICIENCY INDEX (0-100, higher = better)
df['efficiency_degradation'] = (
    (df['tool_wear_min'] / 300) * 0.50 +
    (df['torque_nm'] / 100) * 0.20 +
    (df['temp_diff'] / 25) * 0.20 +
    (df['humidity'] / 100) * 0.10
)
df['efficiency_index'] = 100 - np.clip(df['efficiency_degradation'] * 100, 0, 100)
df['efficiency_index'] += np.random.normal(0, 5, len(df))
df['efficiency_index'] = np.clip(df['efficiency_index'], 5, 100)

# FAILURE RISK (0-100, higher = worse)
df['failure_risk'] = (
    (100 - df['vibration_health']) * 0.35 +
    (100 - df['thermal_health']) * 0.30 +
    (100 - df['efficiency_index']) * 0.35
)
df['failure_risk'] = np.clip(df['failure_risk'], 0, 100)

print("✅ All 4 targets generated")

# Drop intermediate columns
df = df.drop(['vibration_stress', 'temp_diff', 'thermal_stress', 'efficiency_degradation'], axis=1)

# ==============================
# 3. VERIFY DATA QUALITY
# ==============================

print("\n[3/5] Verifying data quality...")

features = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 
            'torque_nm', 'tool_wear_min', 'temperature', 'humidity', 'rainfall']
targets = ['vibration_health', 'thermal_health', 'efficiency_index', 'failure_risk']

print("\n📊 Feature Statistics:")
print(df[features].describe())

print("\n📊 Target Statistics:")
print(df[targets].describe())

# ==============================
# 4. TRAIN-TEST SPLIT
# ==============================

print("\n[4/5] Creating train-test split (80-20)...")

X = df[features]
y = df[targets]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"✅ Train: X={X_train.shape}, y={y_train.shape}")
print(f"✅ Test:  X={X_test.shape}, y={y_test.shape}")

# ==============================
# 5. SCALE FEATURES AND SAVE SCALER
# ==============================

print("\n[5/5] Scaling features and saving scaler...")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# ✅ CRITICAL: Save the scaler for inference
joblib.dump(scaler, 'feature_scaler.pkl')
print("✅ Features scaled (targets kept in 0-100 range)")
print("✅ Scaler saved to: feature_scaler.pkl")

# ==============================
# 6. SAVE FILES
# ==============================

print("\n" + "="*80)
print("SAVING FILES")
print("="*80)

X_train_scaled.to_csv('X_train.csv', index=False)
X_test_scaled.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print(f"\n✅ Saved training files:")
print(f"   X_train.csv: {X_train_scaled.shape}")
print(f"   X_test.csv:  {X_test_scaled.shape}")
print(f"   y_train.csv: {y_train.shape}")
print(f"   y_test.csv:  {y_test.shape}")
print(f"   feature_scaler.pkl: RobustScaler object")

# ==============================
# 7. QUALITY CHECKS
# ==============================

print("\n" + "="*80)
print("QUALITY CHECKS")
print("="*80)

print("\n✅ Data Quality:")
print(f"   No NaN in X_train: {not X_train_scaled.isna().any().any()}")
print(f"   No NaN in y_train: {not y_train.isna().any().any()}")
print(f"   All targets 0-100: {(y_train >= 0).all().all() and (y_train <= 100).all().all()}")
print(f"   Scaler saved: feature_scaler.pkl")

print("\n📈 Expected Model Performance After Training:")
print("   vibration_health:  R² ~ 0.75-0.85")
print("   thermal_health:    R² ~ 0.70-0.80")
print("   efficiency_index:  R² ~ 0.80-0.90")
print("   failure_risk:      R² ~ 0.65-0.75")

print("\n" + "="*80)
print("✅ DATASET READY FOR TRAINING!")
print("="*80)
print("\nNext steps:")
print("1. Run: python model.py")
print("2. Verify all 4 targets train successfully")
print("3. Replace inference.py with version that loads feature_scaler.pkl")
print("4. Test with your Streamlit app")