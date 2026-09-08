import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# defining paths
Raw_data_dir = "data/raw"
Processed_dir = "data/processed"

# define sensors to process
sensors = [ 
    'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6',
    'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 
    'VS1', 'CE', 'CP', 'SE']

def extract_features(df, sensor_name):
    """
    Extracts the qualities of data.
    Includes the describing qualities of 
    the distribution's shape
    """
    features = pd.DataFrame()
    # The usual features
    features[f'{sensor_name}_mean'] = df.mean(axis=1)
    features[f'{sensor_name}_std'] = df.std(axis=1)
    features[f'{sensor_name}_max'] = df.max(axis=1)
    # Special qualities
    features[f'{sensor_name}_median'] = df.median(axis=1)
    features[f'{sensor_name}_skew'] = df.skew(axis=1)
    features[f'{sensor_name}_kurt'] = df.kurtosis(axis=1)
    return features

def main():
    all_features = []
    master_df = pd.DataFrame()

    # Loop over sensors and "pull" qualities
    for sensor in sensors:
        #Find file path
        file_path = os.path.join(Raw_data_dir, f"{sensor}.txt")

        # Check whether path exists
        if os.path.exists(file_path):
            # Load raw data (tab-separated)
            df = pd.read_csv(file_path, sep='\t', header=None)

            # Extract statistical features
            sensor_features = extract_features(df, sensor)

            # Combine into master table
            master_df = pd.concat([master_df, sensor_features], axis=1)

        # else the file is unreadable
        else:
            print(f"Warning: {sensor}.txt not found in {Raw_data_dir}")


    # Pull Target Labels (from profile.txt)
    # Column 0: Cooler condition (3: failure, 20: reduced, 100:full)
    # Create path
    profile_path = os.path.join(Raw_data_dir, "profile.txt")

    # Check whether path exists
    if os.path.exists(profile_path):
        profile = pd.read_csv(profile_path, sep='\t', header=None)
        # Add data to master
        master_df['Cooler_Condition'] = profile[0]
        master_df['Valve_Condition'] = profile[1]
        master_df['Pump_Leakage'] = profile[2]
        master_df['Accumulator_Condition'] = profile[3]
        master_df['Stable_Flag'] = profile[4]

    # Define all the labels
    all_labels = ['Cooler_Condition', 'Valve_Condition', 'Pump_Leakage',
                   'Accumulator_Condition', 'Stable_Flag']
    
    #Separate qualities and labels:
    feature_cols = master_df.drop(all_labels, axis=1)
    label_cols = master_df[all_labels]


    print("Standardizing features")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(feature_cols)

    # Build the original Dataframe back
    final_df = pd.DataFrame(scaled_array, columns=feature_cols.columns)
    final_df = pd.concat([final_df, label_cols], axis=1)

    # Save results
    os.makedirs(Processed_dir, exist_ok=True)
    output_path = os.path.join(Processed_dir, "master_data.csv")
    final_df.to_csv(output_path, index=False)

    #Checkers
    print(f"Success! Processed {len(final_df)} cycles.")
    print(f"Final file saved at: {output_path}")


if __name__ == "__main__":
    main()

