import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("students_dropout_academic_success.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        data = pd.read_csv("students_dropout_academic_success.csv")

        X = data.drop("target", axis=1)

        sample = X.iloc[[0]]

        prediction = model.predict(sample)[0]

        self.assertIn(
            prediction,
            ["INVALID_RESULT"]
        )

    def test_model_prediction_valid_class(self):
        model = joblib.load("student_result_model.pkl")

        data = pd.read_csv("students_dropout_academic_success.csv")

        X = data.drop("target", axis=1)

        sample = X.iloc[[1]]

        prediction = model.predict(sample)[0]

        self.assertIn(
            prediction,
            ["Dropout", "Enrolled", "Graduate"]
        )

    def test_model_can_predict_multiple_students(self):
        model = joblib.load("student_result_model.pkl")

        data = pd.read_csv("students_dropout_academic_success.csv")

        X = data.drop("target", axis=1)

        samples = X.iloc[:5]

        predictions = model.predict(samples)

        self.assertEqual(len(predictions), 5)


if __name__ == "__main__":
    unittest.main()
