import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load the actual dataset
df = pd.read_csv('final_dataset.csv')
Modified_df = df.copy()  # Create a copy of the original dataframe


# Define the ParkinsonsClassifier class
class ParkinsonsClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()  # Initialize the scaler here

    def train(self, X_train, y_train, k: int = 3):
        """
        Train the KNN model on the training data.
        """
        X_train_normalized = self.scaler.fit_transform(X_train)

        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(X_train_normalized, y_train)

    def validate(self, X_test, y_test):
        """
        Validate the trained model on the test data.
        """
        X_test_normalized = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_normalized)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy on test set: {accuracy * 100:.2f}%")
        print("Classification Report:\n", classification_report(y_test, y_pred))

    def cross_validate(self, X, y, k: int = 3):
        """
        Perform k-fold cross-validation and print the results.
        """
        knn_model = KNeighborsClassifier(n_neighbors=k)
        cross_val_scores = cross_val_score(knn_model, X, y, cv=5, scoring='accuracy')

        print(f"Cross-validation scores: {cross_val_scores}")
        print(f"Mean cross-validation score: {cross_val_scores.mean():.4f}")

# Usage
if __name__ == "__main__":
    # Features and target variable
    numeric_cols = Modified_df.select_dtypes(include=["float64", "int"]).columns
    X = Modified_df[numeric_cols].drop(columns=['Status'])  # Drop only 'Status' from the features
    y = Modified_df['Status']

    # Step 1: Split the data into training (70%) and testing (30%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize classifier
    classifier = ParkinsonsClassifier()

    # Step 2: Train the model using the training data
    classifier.train(X_train, y_train)

    # Step 3: Validate the model using the test data
    classifier.validate(X_test, y_test)

    # Step 4: Perform cross-validation
    classifier.cross_validate(X, y)
