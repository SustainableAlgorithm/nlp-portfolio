"""Generic non-person image classification with HOG features and an SVM."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_to_features(path: str | Path, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Convert an image to a fixed-size HOG feature vector."""
    path = Path(path)
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize(size)
            pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Could not read image: {path}") from exc
    return hog(pixels, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def load_image_folder(root: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load class-labeled images from root/class_name/image files."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if len(classes) < 2:
        raise ValueError("Expected at least two class subdirectories")
    features, labels = [], []
    for label, class_name in enumerate(classes):
        paths = sorted(p for p in (root / class_name).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for path in paths:
            features.append(image_to_features(path))
            labels.append(label)
    if not features:
        raise ValueError(f"No supported images found under {root}")
    return np.vstack(features), np.asarray(labels, dtype=np.int64), classes


def make_classifier() -> object:
    """Build a reproducible SVM classifier for HOG feature vectors."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", random_state=42))


def evaluate_classifier(model: object, features: np.ndarray, labels: np.ndarray, class_names: Iterable[str]) -> dict:
    """Return common classification metrics for a held-out set."""
    names = list(class_names)
    predicted = model.predict(features)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=list(range(len(names))), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(len(names)))).tolist(),
        "classification_report": classification_report(
            labels, predicted, labels=list(range(len(names))), target_names=names,
            output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, help="Folder of training images, organized by class")
    parser.add_argument("--test-dir", required=True, help="Folder of held-out test images, organized by class")
    args = parser.parse_args()
    x_train, y_train, train_classes = load_image_folder(args.train_dir)
    x_test, y_test, test_classes = load_image_folder(args.test_dir)
    if train_classes != test_classes:
        raise ValueError("Training and test folders must have the same sorted class names")
    model = make_classifier()
    model.fit(x_train, y_train)
    results = evaluate_classifier(model, x_test, y_test, train_classes)
    print(f"Classes: {train_classes}")
    print(f"Training images: {len(y_train)} | Test images: {len(y_test)}")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key}: {results[key]:.4f}")
    print("Confusion matrix:")
    print(np.asarray(results["confusion_matrix"]))


if __name__ == "__main__":
    main()
