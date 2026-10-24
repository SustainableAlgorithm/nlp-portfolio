# Machine Learning Model Deployment: Titanic Survival Probability

An independently written Flask application that serves a logistic-regression model for an educational Titanic survival dataset. It demonstrates the end-to-end workflow of training a model, saving a model artifact locally, accepting validated form inputs, and presenting a probability with clear limitations.

## Scope and data handling

The dataset and trained model artifact are intentionally not included in this repository. Place a permitted copy of the Titanic CSV at a local path, then train the model. Check the dataset's license and terms before using or redistributing it.

Expected columns: `Survived`, `Age`, `Sex`, `Pclass`.

## Setup

```bash
python -m venv .venv
# Activate the environment
python -m pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py --data path/to/titanic.csv
```

This writes `artifacts/titanic_model.joblib` locally. The artifact directory is git-ignored.

## Run the web app

```bash
python application.py
```

Open `http://127.0.0.1:5000`. The model artifact must exist first. Set `MODEL_PATH` to use a different local artifact path.

## Tests

```bash
python -m pytest -q
```

Tests use a fake model and do not require a dataset or trained artifact.

## Model and limitations

The model uses age, sex, and passenger class. Its output is a model-estimated probability based on historical data, not a causal estimate, guarantee, or statement about an individual. Historical passenger data is not representative of current populations, and the features include sensitive personal characteristics. This project is for educational demonstration only, not for real-world decisions about people.
