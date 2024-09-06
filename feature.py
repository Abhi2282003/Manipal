import pandas as pd
import os
import numpy as np
from scipy.stats import entropy
from scipy.signal import welch
from nolds import sampen
from tsfresh.feature_extraction import feature_calculators
from hurst import compute_Hc

# Load your EEG data from the CSV file
eeg_data = pd.read_csv('preprocessed.csv')  # Replace with the actual file path

# Define the emotion labels for which you want to create epochs
emotion_labels = ['Distracted', 'Focused', 'neutral']

# Define the duration of each epoch in seconds (adjust as needed)
epoch_duration_seconds = 1  # 1 second, for example

# Calculate the number of rows per epoch based on the sample rate
sample_rate = 256  # Replace with your actual sample rate if known

rows_per_epoch = int(epoch_duration_seconds * sample_rate)

# Create a directory to store the subfolders
output_directory = 'epochs_combined'
os.makedirs(output_directory, exist_ok=True)

# Initialize a dictionary to store features for each emotion
emotion_features_combined = {emotion: [] for emotion in emotion_labels}

# Function to calculate MSE
def calculate_mse(eeg_data):
    return sampen(eeg_data)

# Function to calculate time domain features
def calculate_time_domain_features(eeg_data):
    mean = np.mean(eeg_data)
    std = np.std(eeg_data)
    skewness = np.mean((eeg_data - mean) ** 3) / (std ** 3)
    kurtosis = np.mean((eeg_data - mean) ** 4) / (std ** 4)
    return mean, std, skewness, kurtosis

# Function to calculate Shannon entropy
def calculate_shannon_entropy(eeg_data):
    return entropy(eeg_data)

# Function to calculate Hjorth parameters
def calculate_hjorth_parameters(eeg_data):
    activity = np.var(eeg_data)
    mobility = np.sqrt(np.var(np.diff(eeg_data))) / np.std(eeg_data)
    complexity = np.sqrt(np.var(np.diff(np.diff(eeg_data))) / mobility)
    return activity, mobility, complexity

# Function to calculate PSD
def calculate_psd(eeg_data, fs):
    f, Pxx = welch(eeg_data, fs=fs)
    return f, Pxx

# Function to calculate Hurst exponent
def calculate_hurst_exponent(time_series):
    H, c, data_range = compute_Hc(time_series, kind='change', simplified=True)
    return H

# Function to calculate time series features
def calculate_time_series_features(eeg_data, m=2, r=0.2, tau=1, dimension=3):
    ts_features_dict = {}
    for column in eeg_data.columns:
        ts_features_dict[f'{column}_entropy'] = feature_calculators.sample_entropy(eeg_data[column])
        ts_features_dict[f'{column}_approximate_entropy'] = feature_calculators.approximate_entropy(eeg_data[column], m=m, r=r)
        ts_features_dict[f'{column}_permutation_entropy'] = feature_calculators.permutation_entropy(eeg_data[column], tau=tau, dimension=dimension)
    return ts_features_dict

# Function to calculate power features
def calculate_power_features(eeg_data, fs):
    f, Pxx = welch(eeg_data, fs=fs)

    # Calculate absolute power in different frequency bands
    abs_power_delta = np.trapz(Pxx[(f >= 0.5) & (f < 4)])
    abs_power_theta = np.trapz(Pxx[(f >= 4) & (f < 8)])
    abs_power_alpha = np.trapz(Pxx[(f >= 8) & (f < 14)])
    abs_power_beta = np.trapz(Pxx[(f >= 14) & (f < 30)])
    abs_power_gamma = np.trapz(Pxx[(f >= 30) & (f <= 40)])

    # Calculate relative power in different frequency bands
    total_power = np.trapz(Pxx)
    rel_power_delta = abs_power_delta / total_power
    rel_power_theta = abs_power_theta / total_power
    rel_power_alpha = abs_power_alpha / total_power
    rel_power_beta = abs_power_beta / total_power
    rel_power_gamma = abs_power_gamma / total_power

    return abs_power_delta, abs_power_theta, abs_power_alpha, abs_power_beta, abs_power_gamma, \
           rel_power_delta, rel_power_theta, rel_power_alpha, rel_power_beta, rel_power_gamma

# Function to calculate amplitude features
def calculate_amplitude_features(eeg_data):
    amplitude_mean = np.mean(np.abs(eeg_data))
    amplitude_std = np.std(np.abs(eeg_data))
    amplitude_skewness = np.mean((np.abs(eeg_data) - amplitude_mean) ** 3) / (amplitude_std ** 3)
    amplitude_kurtosis = np.mean((np.abs(eeg_data) - amplitude_mean) ** 4) / (amplitude_std ** 4)
    return amplitude_mean, amplitude_std, amplitude_skewness, amplitude_kurtosis

# Function to calculate asymmetry features
def calculate_asymmetry_features(eeg_data):
    asymmetry_mean = np.mean(np.sign(eeg_data) * np.abs(eeg_data))
    asymmetry_std = np.std(np.sign(eeg_data) * np.abs(eeg_data))
    asymmetry_skewness = np.mean((np.sign(eeg_data) * np.abs(eeg_data) - asymmetry_mean) ** 3) / (asymmetry_std ** 3)
    asymmetry_kurtosis = np.mean((np.sign(eeg_data) * np.abs(eeg_data) - asymmetry_mean) ** 4) / (asymmetry_std ** 4)
    return asymmetry_mean, asymmetry_std, asymmetry_skewness, asymmetry_kurtosis

# Iterate through each emotion label and create epochs
for emotion in emotion_labels:
    emotion_data = eeg_data[eeg_data['Emotion'] == emotion]

    # Initialize a list to store features for each epoch
    epoch_features_combined = []

    # Create a subfolder for each emotion
    emotion_subfolder = os.path.join(output_directory, emotion)
    os.makedirs(emotion_subfolder, exist_ok=True)

    for i in range(0, len(emotion_data), rows_per_epoch):
        epoch_data = emotion_data[i:i + rows_per_epoch]

        if len(epoch_data) < rows_per_epoch:
            continue  # Skip incomplete epochs

        # Select only the EEG data columns, excluding non-numeric columns
        eeg_columns = epoch_data[['TP9', 'AF7', 'AF8', 'TP10']]

        # Calculate time domain features
        mean, std, skewness, kurtosis = calculate_time_domain_features(eeg_columns)

        # Calculate Shannon entropy
        entropy_value = calculate_shannon_entropy(eeg_columns)

        # Calculate Hjorth parameters
        activity, mobility, complexity = calculate_hjorth_parameters(eeg_columns)

        # Calculate PSD
        f, Pxx = calculate_psd(eeg_columns, sample_rate)

        # Calculate MSE
        mse_values = calculate_mse(eeg_columns)

        # Calculate Hurst exponent
        hurst_exponent = calculate_hurst_exponent(eeg_columns.values.flatten())

        # Calculate time series features
        ts_features_dict = calculate_time_series_features(eeg_columns, m=2, r=0.2, tau=1, dimension=3)

        # Calculate power features
        power_features = calculate_power_features(eeg_columns.values.flatten(), fs=sample_rate)

        # Calculate amplitude features
        amplitude_features = calculate_amplitude_features(eeg_columns.values.flatten())

        # Calculate asymmetry features
        asymmetry_features = calculate_asymmetry_features(eeg_columns.values.flatten())

        # Flatten the dictionary values to handle variable length
        ts_feature_values = np.array(list(ts_features_dict.values()))
        ts_feature_flat = ts_feature_values.flatten()

        # Combine all features into a single list for this epoch
        combined_features = [
            mean, std, skewness, kurtosis, entropy_value, activity, mobility, complexity,
            mse_values, hurst_exponent, f, Pxx, *ts_feature_flat, *power_features, *amplitude_features, *asymmetry_features
        ]

        epoch_features_combined.append(combined_features)

        # Save the epoch in the corresponding subfolder
        epoch_filename = os.path.join(emotion_subfolder, f'epoch_{i + 1}.csv')
        epoch_data.to_csv(epoch_filename, index=False)

    # Append epoch features to the emotion_features_combined dictionary
    emotion_features_combined[emotion] = epoch_features_combined

# Create CSV files for each emotion with all features
for emotion in emotion_labels:
    emotion_subfolder = os.path.join(output_directory, emotion)
    features_filename_combined = os.path.join(emotion_subfolder, f'new_features_{emotion}_combined.csv')

    # Define column names for the combined features
    combined_columns = ['Mean', 'Std', 'Skewness', 'Kurtosis', 'Shannon_Entropy', 'Activity', 'Mobility', 'Complexity', 'MSE', 'Hurst_Exponent', 'Frequency', 'PSD']
    ts_feature_columns = [f'{column}_entropy' for column in eeg_columns.columns] + \
                         [f'{column}_approximate_entropy' for column in eeg_columns.columns] + \
                         [f'{column}_permutation_entropy' for column in eeg_columns.columns]

    power_feature_columns = ['Abs_Power_Delta', 'Abs_Power_Theta', 'Abs_Power_Alpha', 'Abs_Power_Beta', 'Abs_Power_Gamma',
                             'Rel_Power_Delta', 'Rel_Power_Theta', 'Rel_Power_Alpha', 'Rel_Power_Beta', 'Rel_Power_Gamma']

    amplitude_feature_columns = ['Amplitude_Mean', 'Amplitude_Std', 'Amplitude_Skewness', 'Amplitude_Kurtosis']

    asymmetry_feature_columns = ['Asymmetry_Mean', 'Asymmetry_Std', 'Asymmetry_Skewness', 'Asymmetry_Kurtosis']

    combined_columns += ts_feature_columns + power_feature_columns + amplitude_feature_columns + asymmetry_feature_columns

    # Convert the list of features to a DataFrame and save to CSV
    features_df_combined = pd.DataFrame(emotion_features_combined[emotion], columns=combined_columns)
    features_df_combined.to_csv(features_filename_combined, index=False)

    print(f'Combined features for {emotion} saved as {features_filename_combined}')
