import pandas as pd
import os
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

# Define the path to the directory containing the combined features CSV files
combined_features_directory = 'epochs_combined'

# Define the output directory for the selected features
selected_features_directory = 'selected_features'
os.makedirs(selected_features_directory, exist_ok=True)

# Define the emotion labels
emotion_labels = ['Distracted', 'Focused', 'neutral']

# Iterate through each emotion to process the combined features CSV files
for emotion in emotion_labels:
    # Load the combined features CSV file
    features_filename_combined = os.path.join(combined_features_directory, emotion, f'new_features_{emotion}_combined.csv')
    
    # Check if the file exists
    if not os.path.isfile(features_filename_combined):
        print(f'File not found: {features_filename_combined}')
        continue
    
    # Load the data
    features_df = pd.read_csv(features_filename_combined)

    # Check the first few rows to understand the data
    print(f"First few rows of the data for {emotion}:")
    print(features_df.head())

    # Ensure that all columns are numeric
    X = features_df.select_dtypes(include=[np.number])
    if X.empty:
        print(f"No numeric data available for {emotion}. Skipping...")
        continue

    # Generate a dummy target variable for feature selection (use actual target if available)
    y_dummy = np.random.randint(0, 2, len(X))  # Dummy target variable for binary classification

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Experiment with different alpha values
    alpha_values = [0.01, 0.1, 1, 10]
    for alpha in alpha_values:
        print(f"Applying Lasso with alpha={alpha} for {emotion}...")
        lasso = Lasso(alpha=alpha, max_iter=10000)

        try:
            lasso.fit(X_scaled, y_dummy)
        except Exception as e:
            print(f"Error during Lasso fitting for {emotion} with alpha={alpha}: {e}")
            continue

        # Get the coefficients and identify non-zero features
        coef = lasso.coef_
        if len(coef) != X.shape[1]:
            print(f"Coefficient length mismatch for {emotion}. Skipping...")
            continue

        non_zero_indices = coef != 0
        selected_features = X.columns[non_zero_indices]

        # Check if any features are selected
        if len(selected_features) > 0:
            # Filter the data to include only selected features
            filtered_df = features_df[selected_features]
            
            # Save the filtered data to a CSV file
            filtered_data_filename = os.path.join(selected_features_directory, f'filtered_data_{emotion}_alpha_{alpha}.csv')
            filtered_df.to_csv(filtered_data_filename, index=False)
            print(f'Filtered data for {emotion} with alpha={alpha} saved as {filtered_data_filename}')
        else:
            print(f"No significant features found for {emotion} with alpha={alpha}.")
