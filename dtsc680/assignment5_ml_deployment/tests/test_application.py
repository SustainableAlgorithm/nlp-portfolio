import pandas as pd
import pytest

import application as app_module
from train_model import build_pipeline, train


class FakeModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == ["Age", "Sex", "Pclass"]
        return [[0.25, 0.75]]


@pytest.fixture()
def client(monkeypatch):
    app_module.application.config.update(TESTING=True)
    monkeypatch.setattr(app_module, "load_model", lambda: FakeModel())
    with app_module.application.test_client() as test_client:
        yield test_client


def test_about_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"From trained model to usable application" in response.data


def test_prediction_page(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Estimate a model probability" in response.data


def test_valid_prediction(client):
    response = client.post("/predict", data={"age": "30", "sex": "female", "pclass": "1"})
    assert response.status_code == 200
    assert b"75.0%" in response.data


@pytest.mark.parametrize("form", [
    {"age": "abc", "sex": "female", "pclass": "1"},
    {"age": "120", "sex": "female", "pclass": "1"},
    {"age": "30", "sex": "other", "pclass": "1"},
    {"age": "30", "sex": "female", "pclass": "5"},
])
def test_invalid_inputs_are_rejected(client, form):
    response = client.post("/predict", data=form)
    assert response.status_code == 400
    assert b"alert" in response.data


def test_train_pipeline_on_tiny_local_dataset(tmp_path):
    csv = tmp_path / "tiny.csv"
    pd.DataFrame({
        "Survived": [0, 1, 0, 1, 0, 1],
        "Age": [22, 38, 26, 35, None, 28],
        "Sex": ["male", "female", "female", "female", "male", "male"],
        "Pclass": [3, 1, 3, 1, 3, 2],
    }).to_csv(csv, index=False)
    output = train(csv, tmp_path / "model.joblib")
    assert output.is_file()
    assert build_pipeline() is not None
