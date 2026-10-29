
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from werkzeug.exceptions import HTTPException


# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        model = joblib.load(MODEL_PATH)
        logger.info("event=model_loaded status=success")
        return model

    except Exception:
        logger.exception("event=model_load_failed")
        raise


@app.before_request
def log_request_start():
    g.request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()

    logger.info(
        "event=request_started request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


@app.after_request
def log_request_end(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    response.headers["X-Request-ID"] = request_id

    log_method = (
        logger.warning
        if response.status_code >= 400
        else logger.info
    )

    log_method(
        "event=request_completed request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    return response


# Handle nonexistent routes without converting 404 into 500
@app.errorhandler(404)
def handle_not_found(error):
    logger.warning(
        "event=route_not_found request_id=%s path=%s",
        getattr(g, "request_id", "unknown"),
        request.path,
    )

    return jsonify({
        "error": "Not found",
        "request_id": getattr(g, "request_id", "unknown"),
    }), 404


# Handle unexpected errors while preserving normal HTTP errors
@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        return error

    logger.exception(
        "event=unhandled_error request_id=%s",
        getattr(g, "request_id", "unknown"),
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(g, "request_id", "unknown"),
    }), 500


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=invalid_request request_id=%s reason=invalid_json",
            g.request_id,
        )
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=invalid_request request_id=%s reason=missing_fields "
            "fields=%s",
            g.request_id,
            missing_fields,
        )
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
       average_score = (
            float(data["attendance"])
            + float(data["internal_marks"])
            + float(data["assignment_marks"])
            + float(data["previous_score"])
            ) / 4

        if average_score >= 60:
            prediction_code = 1
            prediction = "PASS"
        else:
            prediction_code = 0
            prediction = "FAIL"

        logger.info(
            "event=prediction_completed request_id=%s result=%s",
            g.request_id,
            prediction,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        })

    except (TypeError, ValueError):
        logger.warning(
            "event=invalid_request request_id=%s "
            "reason=invalid_feature_value",
            g.request_id,
        )
        return jsonify({
            "error": "Feature values must be numeric"
        }), 400


@app.route("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    logger.info("event=application_starting")
    app.run(host="0.0.0.0", port=5000)
