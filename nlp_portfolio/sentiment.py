"""Reusable text-classification helpers for a licensed sentiment dataset."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]
    report: dict


def make_classifier(kind: str = "logistic", *, max_features: int = 50_000) -> Pipeline:
    """Create a TF-IDF text classifier. Supported kinds: logistic, naive_bayes, perceptron."""
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", max_features=max_features, ngram_range=(1, 2), min_df=1)
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "naive_bayes": MultinomialNB(),
        "perceptron": Perceptron(random_state=42, class_weight="balanced"),
    }
    if kind not in classifiers:
        raise ValueError(f"Unsupported classifier {kind!r}; choose one of {sorted(classifiers)}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifiers[kind])])


def evaluate_classifier(model: Pipeline, texts: Iterable[str], labels: Iterable[str]) -> ClassificationMetrics:
    """Evaluate predictions against labels and return serializable metrics."""
    y_true = list(labels)
    y_pred = model.predict(list(texts))
    labels_sorted = sorted(set(y_true) | set(y_pred))
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels_sorted).tolist(),
        report=classification_report(y_true, y_pred, labels=labels_sorted, output_dict=True, zero_division=0),
    )
