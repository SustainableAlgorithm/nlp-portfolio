"""Flask interface for a locally trained Titanic survival model."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

from train_model import FEATURES

application = Flask(__name__)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/titanic_model.joblib"))


def load_model():
    """Load the trusted, locally generated model artifact."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run train_model.py first."
        )
    return joblib.load(MODEL_PATH)


def parse_inputs(form) -> pd.DataFrame:
    """Validate user input and return a one-row dataframe in training schema."""
    try:
        age = float(form.get("age", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Enter a valid age.") from exc
    if not 0 <= age <= 100:
        raise ValueError("Age must be between 0 and 100.")
    sex = str(form.get("sex", "")).strip().lower()
    if sex not in {"female", "male"}:
        raise ValueError("Choose female or male.")
    try:
        pclass = int(form.get("pclass", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Choose passenger class 1, 2, or 3.") from exc
    if pclass not in {1, 2, 3}:
        raise ValueError("Choose passenger class 1, 2, or 3.")
    return pd.DataFrame([{"Age": age, "Sex": sex, "Pclass": str(pclass)}], columns=FEATURES)


@application.get("/")
@application.get("/about")
def about():
    return render_template("about.html")


@application.get("/predict")
def prediction_form():
    return render_template("predict.html", error=None, prediction=None)


@application.post("/predict")
def predict():
    try:
        row = parse_inputs(request.form)
        model = load_model()
        probability = float(model.predict_proba(row)[0][1]) * 100
        return render_template("predict.html", error=None, prediction=round(probability, 1))
    except ValueError as exc:
        return render_template("predict.html", error=str(exc), prediction=None), 400
    except FileNotFoundError as exc:
        application.logger.error("%s", exc)
        return render_template(
            "predict.html",
            error="The model is not configured yet. Please try again later.",
            prediction=None,
        ), 503


if __name__ == "__main__":
    # Local development only. Use a production WSGI server for deployment.
    application.run(host="127.0.0.1", port=5000, debug=False)
