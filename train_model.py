import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model():

    print("Loading dataset...")

    data = pd.read_csv("students_dropout_academic_success.csv")

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of features:", len(data.columns) - 1)

    # Separate input features and target
    X = data.drop("target", axis=1)
    y = data["target"]

    print("\nTarget classes:")
    print(y.value_counts())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records :", len(X_test))

    # Create ML pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    print("\nTraining model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # Save trained model
    joblib.dump(model, "student_result_model.pkl")

    print("\nModel saved as student_result_model.pkl")

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    train_model()
