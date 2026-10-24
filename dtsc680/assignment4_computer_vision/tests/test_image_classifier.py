from pathlib import Path

import numpy as np
from PIL import Image

from image_classifier import image_to_features, load_image_folder, make_classifier, evaluate_classifier


def test_image_to_features_is_fixed_length(tmp_path: Path):
    image_path = tmp_path / "shape.png"
    pixels = np.zeros((64, 64), dtype=np.uint8)
    pixels[16:48, 16:48] = 255
    Image.fromarray(pixels).save(image_path)
    features = image_to_features(image_path)
    assert features.ndim == 1
    assert features.size > 0
    assert np.isfinite(features).all()


def test_load_folder_and_train_classifier(tmp_path: Path):
    for class_name, value in (("dark", 20), ("light", 230)):
        folder = tmp_path / class_name
        folder.mkdir()
        for i in range(3):
            pixels = np.full((64, 64), value, dtype=np.uint8)
            pixels[i * 5:(i + 1) * 5, :] = 255 - value
            Image.fromarray(pixels).save(folder / f"{i}.png")
    x, y, classes = load_image_folder(tmp_path)
    assert classes == ["dark", "light"]
    assert x.shape[0] == 6
    assert set(y.tolist()) == {0, 1}
    model = make_classifier().fit(x, y)
    results = evaluate_classifier(model, x, y, classes)
    assert np.asarray(results["confusion_matrix"]).shape == (2, 2)
    assert sum(sum(row) for row in results["confusion_matrix"]) == 6
    assert 0.0 <= results["accuracy"] <= 1.0
    assert {"macro_precision", "macro_recall", "macro_f1"} <= results.keys()
