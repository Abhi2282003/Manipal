import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier

# Define the feature names as they were used during training
feature_names = [
    'Mean', 'MSE', 'AF8_entropy', 'TP9_permutation_entropy', 'AF7_permutation_entropy',
    'TP10_permutation_entropy', 'Abs_Power_Theta', 'Abs_Power_Gamma', 'Rel_Power_Theta',
    'Rel_Power_Beta', 'Amplitude_Mean', 'Asymmetry_Mean', 'Asymmetry_Kurtosis', 'Hurst_Exponent',
    'TP10_entropy', 'AF7_approximate_entropy', 'TP10_approximate_entropy', 'Abs_Power_Delta',
    'Rel_Power_Delta', 'Rel_Power_Alpha', 'Rel_Power_Gamma', 'Asymmetry_Skewness', 'TP9_entropy',
    'Abs_Power_Beta', 'Amplitude_Kurtosis'
]

def load_models_and_tools():
    try:
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('imputer.pkl')
        voting_classifier_ovr = joblib.load('final_weighted_voting_classifier_model_with_gb.pkl')
        return scaler, imputer, voting_classifier_ovr
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None

def predict_emotion(input_data):
    scaler, imputer, voting_classifier_ovr = load_models_and_tools()
    if not scaler or not imputer or not voting_classifier_ovr:
        print("Failed to load models or preprocessing tools.")
        return None
    
    # Convert the input data to a DataFrame with appropriate column names
    df = pd.DataFrame(input_data, columns=feature_names)
    
    # Check for infinite or very large values and replace them with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute NaN values
    feature_data_imputed = pd.DataFrame(imputer.transform(df), columns=feature_names)
    
    # Scale the features
    feature_data_scaled = scaler.transform(feature_data_imputed)
    
    # Make predictions using the trained model
    predictions = voting_classifier_ovr.predict(feature_data_scaled)
    
    return predictions

def get_user_input():
    num_samples = int(input("Enter the number of samples to predict: "))
    
    input_data = []
    
    for i in range(num_samples):
        print(f"\nEnter data for sample {i + 1}:")
        
        # Collect data for each feature
        try:
            features = []
            for feature in feature_names:
                value = float(input(f"{feature}: "))
                features.append(value)
            input_data.append(features)
        except ValueError:
            print("Invalid input. Please enter numeric values.")
            return None
    
    return input_data

def main():
    user_input_data = get_user_input()
    
    if user_input_data:
        predictions = predict_emotion(user_input_data)
        if predictions is not None:
            for i, pred in enumerate(predictions):
                print(f"Sample {i + 1}: Predicted Emotion: {pred}")
    else:
        print("Failed to get valid input data.")

if __name__ == "__main__":
    main()
