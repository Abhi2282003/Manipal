import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Load your EEG data from the CSV file
eeg_data = pd.read_csv('preprocessed.csv')  # Replace with the actual file path

# Define the emotion labels for which you want to create epochs
emotion_labels = ['Stressed', 'relaxed', 'neutral']

# Directory containing the feature CSV files
output_directory = 'epochs_combined'
os.makedirs(output_directory, exist_ok=True)

# Initialize a dictionary to store the DataFrames for each emotion
emotion_features_combined = {}

# Iterate over each emotion to load the corresponding features file
for emotion in emotion_labels:
    emotion_subfolder = os.path.join(output_directory, emotion)
    features_filename_combined = os.path.join(emotion_subfolder, f'new_features_{emotion}_combined.csv')

    # Check if the file exists before trying to load it
    if os.path.exists(features_filename_combined):
        emotion_features_combined[emotion] = pd.read_csv(features_filename_combined)
    else:
        print(f"File for emotion '{emotion}' not found: {features_filename_combined}")

# Plotting mean and standard deviation for each emotion
selected_features = ['Mean', 'Std']

# Make sure that the DataFrame for each emotion contains the selected features
for feature in selected_features:
    plt.figure(figsize=(10, 6))
    for emotion, features_df in emotion_features_combined.items():
        if feature in features_df.columns:
            plt.plot(features_df.index, features_df[feature], label=emotion, marker='o')
        else:
            print(f"Feature '{feature}' not found in the data for '{emotion}'")

    plt.title(f'{feature} for Different Emotions')
    plt.xlabel('Epoch Index')
    plt.ylabel(feature)
    plt.legend()
    plt.show()
