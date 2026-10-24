# Code review: generic computer vision

## Scope

The original DTSC 680 Assignment 4 notebook asks students to detect and identify three named Philadelphia Phillies players using face images and a pretrained VGGFace model. This portfolio version intentionally does not reproduce biometric identification. It uses a newly written, generic non-person image classification pipeline instead.

## Implementation

- PIL loads and converts images to grayscale.
- HOG extracts fixed-length local gradient features.
- A scikit-learn SVM classifier provides a compact supervised baseline.
- Held-out evaluation reports accuracy, macro precision, macro recall, macro F1, and a confusion matrix.
- Images and datasets remain external/local; none are bundled.

## Validation

The included tests check fixed-length finite feature extraction and a small synthetic two-category classification workflow. Passing synthetic tests validates basic code behavior only; it does not establish performance on real-world image data.

## Limitations

- HOG features are hand-crafted and may perform poorly when classes differ by color, texture, viewpoint, scale, or background.
- The synthetic test set is tiny and is not a benchmark.
- A proper experiment requires licensed non-person images, an independent test set, and documented class balance.
- Do not use this demonstration to identify people or classify face images.
