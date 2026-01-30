import pandas as pd
import numpy as np

# Load both datasets
print("Loading datasets...")
turbojet_df = pd.read_csv('turbojet_merged_features.csv')
ai4i_df = pd.read_csv('ai4i_with_weather.csv')

print(f"Turbojet dataset shape: {turbojet_df.shape}")
print(f"AI4I dataset shape: {ai4i_df.shape}")

# Method: Horizontal merge by repeating AI4I data to match turbojet length
# This combines all features from both datasets side by side
print("\nMerging datasets horizontally...")

n_turbojet = len(turbojet_df)
n_ai4i = len(ai4i_df)

# Cycle through AI4I data to match turbojet length
ai4i_repeated_indices = np.tile(np.arange(n_ai4i), (n_turbojet // n_ai4i) + 1)[:n_turbojet]
ai4i_repeated = ai4i_df.iloc[ai4i_repeated_indices].reset_index(drop=True)

# Combine horizontally - all original columns preserved
merged_df = pd.concat([turbojet_df.reset_index(drop=True), ai4i_repeated], axis=1)

print(f"\nMerged dataset shape: {merged_df.shape}")
print(f"Total columns: {len(merged_df.columns)}")
print(f"Missing values: {merged_df.isnull().sum().sum()}")

# Save the merged dataset
output_path = 'merged_dataset.csv'
merged_df.to_csv(output_path, index=False)
print(f"\n✓ Successfully saved merged dataset to: {output_path}")
print("\nColumns in merged dataset:")
print(merged_df.columns.tolist())