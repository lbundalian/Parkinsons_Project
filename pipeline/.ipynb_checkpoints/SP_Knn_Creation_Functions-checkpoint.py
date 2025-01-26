import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class ParkinsonsClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, dataframe: pd.DataFrame, grouping_variable: str, k: int = 3):
        """
        Train the KNN model on aggregated data.
        """
        # 1. Aggregate data
        numeric_cols = dataframe.select_dtypes(include=["float64", "int"]).columns
        aggregated_df = dataframe.groupby(grouping_variable)[numeric_cols].mean().reset_index()

        # 2. Extract target and features
        y = dataframe.groupby(grouping_variable)["status"].first().reset_index()["status"].astype(int)
        X = aggregated_df.drop(columns=[grouping_variable])

        # 3. Normalize data
        self.scaler = StandardScaler()
        X_normalized = self.scaler.fit_transform(X)

        # 4. Train KNN classifier
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X_normalized, y)

    def predict(self, input_data: pd.DataFrame):
        """
        Predict whether the subject is a patient or control.
        """
        if not self.model or not self.scaler:
            raise ValueError("Model is not trained. Please train the model first.")

        # Normalize the input data
        X_normalized = self.scaler.transform(input_data)
        return self.model.predict(X_normalized).tolist()

    def save_model(self, model_path: str, scaler_path: str):
        """
        Save the trained model and scaler.
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_path: str, scaler_path: str):
        """
        Load a previously trained model and scaler.
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
