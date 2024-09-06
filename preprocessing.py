import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# Butterworth filter functions for low-pass and high-pass filtering
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency is half the sampling rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_filter(data, fs, low_cutoff=None, high_cutoff=None, order=5):
    if low_cutoff:
        b, a = butter_lowpass(low_cutoff, fs, order=order)
        data = filtfilt(b, a, data)  # Apply low-pass filter
    if high_cutoff:
        b, a = butter_highpass(high_cutoff, fs, order=order)
        data = filtfilt(b, a, data)  # Apply high-pass filter
    return data

def preprocess_file(input_csv, output_csv, fs=256, low_cutoff=50, high_cutoff=0.5):
    """
    Preprocess EEG data from input CSV, applying missing value imputation, scaling, 
    and high-pass and low-pass filtering.
    
    Parameters:
    - input_csv: path to the input CSV file
    - output_csv: path to the output CSV file
    - fs: sampling rate (in Hz) of the EEG data (default = 256 Hz)
    - low_cutoff: cutoff frequency for low-pass filter (default = 50 Hz)
    - high_cutoff: cutoff frequency for high-pass filter (default = 0.5 Hz)
    """
    
    # Load the dataset into a DataFrame
    df = pd.read_csv(input_csv)

    # Save the "Emotion" column separately
    emotion_column = df["Emotion"]

    # Drop the "Emotion" column from the DataFrame
    df = df.drop(columns=["Emotion"], errors='ignore')

    # Handle missing values using mean imputation for numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

    # Apply high-pass and low-pass filters to numeric columns (EEG data)
    for column in numeric_columns:
        df[column] = apply_filter(df[column], fs, low_cutoff=low_cutoff, high_cutoff=high_cutoff)

    # Scale numeric features using StandardScaler
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Add the "Emotion" column back to the DataFrame
    df["Emotion"] = emotion_column

    # Save the preprocessed DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv_path = "main.csv"  # Replace with the path to your input file
    output_csv_path = "preprocessed.csv"  # Replace with the desired output path

    # Call the preprocess function with EEG sampling rate and desired cutoff frequencies
    preprocess_file(input_csv_path, output_csv_path, fs=256, low_cutoff=50, high_cutoff=0.5)