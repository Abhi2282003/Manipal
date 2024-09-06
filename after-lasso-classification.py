import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Load the extracted features
Distracted_features = pd.read_csv('selected_features/filtered_data_Distracted_alpha_0.01.csv')
Focused_features = pd.read_csv('selected_features/filtered_data_Focused_alpha_0.01.csv')
neutral_features = pd.read_csv('selected_features/filtered_data_neutral_alpha_0.01.csv')

# Combine all features into one dataset
all_features = pd.concat([Distracted_features, Focused_features, neutral_features], ignore_index=True)

# Define the target labels
labels = ['Distracted'] * len(Distracted_features) + ['Focused'] * len(Focused_features) + ['neutral'] * len(neutral_features)

# Extract the numeric columns (drop non-numeric columns)
feature_data = all_features.select_dtypes(include=['float64'])

# Check for infinite or very large values and replace them with NaN
feature_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute NaN values with the mean of the column
imputer = SimpleImputer(strategy='mean')
feature_data_imputed = pd.DataFrame(imputer.fit_transform(feature_data), columns=feature_data.columns)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(feature_data_imputed, labels, test_size=0.2, random_state=42)
print("Unique classes in y_train:", np.unique(y_train))

# Save training and testing datasets
X_train.to_csv('train_features.csv', index=False)
X_test.to_csv('test_features.csv', index=False)
pd.DataFrame(y_train, columns=['label']).to_csv('train_labels.csv', index=False)
pd.DataFrame(y_test, columns=['label']).to_csv('test_labels.csv', index=False)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for SVM
svm_param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
svm_grid = GridSearchCV(SVC(probability=True), svm_param_grid, cv=5, scoring='accuracy')
svm_grid.fit(X_train_scaled, y_train)
best_svm_model = svm_grid.best_estimator_

# Hyperparameter tuning for K-Nearest Neighbors
knn_param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train_scaled, y_train)
best_knn_model = knn_grid.best_estimator_

# Hyperparameter tuning for Random Forest
rf_param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 5], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train_scaled, y_train)
best_rf_model = rf_grid.best_estimator_

# Hyperparameter tuning for Gradient Boosting
gb_param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_param_grid, cv=5, scoring='accuracy')
gb_grid.fit(X_train_scaled, y_train)
best_gb_model = gb_grid.best_estimator_

# Define weights for each classifier
svm_weight = 1
knn_weight = 1
rf_weight = 1
gb_weight = 1

# Create a Voting Classifier with SVM, KNN, Random Forest, and Gradient Boosting with weights
voting_classifier = VotingClassifier(estimators=[
    ('svm', best_svm_model),
    ('knn', best_knn_model),
    ('rf', best_rf_model),
    ('gb', best_gb_model)
], voting='soft', weights=[svm_weight, knn_weight, rf_weight, gb_weight])

# Wrap the Voting Classifier with OneVsRestClassifier for multi-class classification
voting_classifier_ovr = OneVsRestClassifier(voting_classifier)

# Train the Wrapped Voting Classifier on the training data
voting_classifier_ovr.fit(X_train_scaled, y_train)

# Save the models, scaler, and imputer to files
joblib.dump(best_svm_model, 'final_best_svm_model.pkl')
joblib.dump(best_knn_model, 'final_best_knn_model.pkl')
joblib.dump(best_rf_model, 'final_random_forest_model.pkl')
joblib.dump(best_gb_model, 'final_best_gb_model.pkl')
joblib.dump(voting_classifier_ovr, 'final_weighted_voting_classifier_model_with_gb.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')

# Later, when you want to make predictions on new data:
# Load the trained models from files
loaded_weighted_voting_classifier_with_gb_ovr = joblib.load('final_weighted_voting_classifier_model_with_gb.pkl')
loaded_scaler = joblib.load('scaler.pkl')
loaded_imputer = joblib.load('imputer.pkl')

# Use the loaded models to make predictions on new data
X_test_scaled = loaded_scaler.transform(X_test)
X_test_imputed = pd.DataFrame(loaded_imputer.transform(X_test), columns=X_test.columns)
weighted_voting_predictions_with_gb = loaded_weighted_voting_classifier_with_gb_ovr.predict(X_test_scaled)

# Evaluate the models
print("Weighted Voting Classifier with Gradient Boosting Classification Report:")
print(classification_report(y_test, weighted_voting_predictions_with_gb))

# Save evaluation results
classifiers = {
    'Best SVM': joblib.load('final_best_svm_model.pkl'),
    'Best KNN': joblib.load('final_best_knn_model.pkl'),
    'Best Random Forest': joblib.load('final_random_forest_model.pkl'),
    'Best Gradient Boosting': joblib.load('final_best_gb_model.pkl'),
    'Weighted Voting Classifier with Gradient Boosting': joblib.load('final_weighted_voting_classifier_model_with_gb.pkl')
}

# Create a directory to save the images
output_directory = 'evaluation_images'
os.makedirs(output_directory, exist_ok=True)

for clf_name, clf_model in classifiers.items():
    predictions = clf_model.predict(X_test_scaled)
    
    # Print Classification Report
    print(f"{clf_name} Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'{clf_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Save the Confusion Matrix plot
    cm_filename = os.path.join(output_directory, f'{clf_name}_confusion_matrix.png')
    plt.savefig(cm_filename)
    plt.show()

    # Plot Multiclass ROC curve (if applicable)
    if hasattr(clf_model, 'predict_proba'):
        lb = label_binarize(y_test, classes=np.unique(y_test))
        y_score = clf_model.predict_proba(X_test_scaled)

        plt.figure()
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(lb[:, 1], y_score[:, 1])
            plt.plot(fpr, tpr, label=f'{clf_name} ROC Curve (AUC = {auc(fpr, tpr):.2f})')
        else:
            for i in range(len(np.unique(y_test))):
                fpr, tpr, _ = roc_curve(lb[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{clf_name} (Class {i + 1}) - AUC = {roc_auc:.2f}')
                
        plt.title(f'ROC Curve - {clf_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        
        # Save the ROC curve plot
        roc_filename = os.path.join(output_directory, f'{clf_name}_roc_curve.png')
        plt.savefig(roc_filename)
        plt.show()
    else:
        print(f"{clf_name} does not support predict_proba for ROC curve plotting.")
