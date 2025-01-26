import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import warnings

class ParkinsonsClassifier:
    def __init__(self):
        self.model = None


    
    def train(self, X: pd.DataFrame, y: pd.Series, k: int = 3):
        """
        Train the KNN model on aggregated data.
        """

        with warnings.catch_warnings(): # Suppress warnings within this block
            warnings.filterwarnings("ignore", category=FutureWarning) 
            
            # Train KNN classifier
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.model.fit(X, y)


    
    def predict(self, input_data: pd.DataFrame):
        """
        Predict whether the subject is a patient or control.
        """
        with warnings.catch_warnings(): # Suppress warnings within this block
            warnings.filterwarnings("ignore", category=FutureWarning) 
            
            if not self.model:
                raise ValueError("Model is not trained. Please train the model first.")

            predictions = self.model.predict(input_data).tolist()

        return predictions


    
    def save_model(self, model_path: str):
        """
        Save the trained model.
        """
        joblib.dump(self.model, model_path)


    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        """
        self.model = joblib.load(model_path)