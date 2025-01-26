import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
import numpy as np
import joblib


class CrossValidator:
    def __init__(self, model, cv: int = 5):
       
        self.model = model
        self.cv = cv

    
    def calculate_performance(self,y_pred,y):

        with warnings.catch_warnings():  # Suppress warnings within this block
            warnings.filterwarnings("ignore", category=FutureWarning)
            
            accuracy = accuracy_score(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred)
            class_report = classification_report(y, y_pred, output_dict=True)
    
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
    
            # Return metrics as a dictionary
            metrics = {
                "accuracy": accuracy,
                "confusion_matrix": conf_matrix,
                "classification_report": class_report,
            }
            
        return metrics

    
    def validate(self, X: pd.DataFrame, y: pd.Series):
       
        with warnings.catch_warnings():  # Suppress warnings within this block
            warnings.filterwarnings("ignore", category=FutureWarning)

            # Use Stratified K-Folds for consistent class distribution across folds
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

            # Perform cross-validation and predict
            y_pred = cross_val_predict(self.model, X, y, cv=skf)

            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            conf_matrix = confusion_matrix(y, y_pred)
            class_report = classification_report(y, y_pred, output_dict=True)

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(classification_report(y, y_pred))

        # Return metrics as a dictionary
        metrics = {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }
        return metrics
