# Computer Vision: Image Classification

An independently written computer-vision demonstration inspired by general skills practiced in an applied machine-learning course. This project uses generic, non-person image categories to demonstrate image loading, resizing, HOG feature extraction, supervised classification, and evaluation.

**Scope boundary:** This is not a face-recognition system and does not identify people. Original course notebooks, named-player face images, the pretrained VGGFace model, and course submission artifacts are not included.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Dataset format

Provide a local image folder with one subfolder per non-person object category:

```text
data/
  train/
    category_a/
    category_b/
  test/
    category_a/
    category_b/
```

Use images you own or are licensed to use. Do not use face or identity categories for this demonstration.

## Run

```bash
python image_classifier.py --train-dir data/train --test-dir data/test
```

The script reports accuracy, macro precision/recall/F1, and a confusion matrix. It does not bundle a dataset or claim benchmark performance.

## Method

1. Load supported image files and convert them to grayscale.
2. Resize each image to a consistent size.
3. Extract Histogram of Oriented Gradients (HOG) features.
4. Train an RBF-kernel Support Vector Classifier.
5. Evaluate on a separate labeled test folder.

This is a compact classical computer-vision baseline, not a modern deep-learning model. Results depend on the dataset, class balance, image quality, and train/test separation.
