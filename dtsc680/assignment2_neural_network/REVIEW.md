# Code Review: Neural Networks

## Scope

The original assignment notebook was reviewed for its general technical themes. This private repository contains a newly written implementation, not the notebook or its exported artifacts.

## Key lessons reflected in the implementation

1. The original notebook's unweighted model reported accuracy 0.9475, recall 0.4061, F1 0.5467, and precision 0.8362.
2. Its class-weighted model reported accuracy 0.9066, recall 0.7887, F1 0.5683, and precision 0.4442.
3. These figures are values recorded in the supplied assignment notebook; they have not been independently reproduced in this portfolio implementation.
4. The tradeoff is important: class weighting increased recall of the positive class but reduced precision and overall accuracy. This illustrates why accuracy alone is insufficient for an imbalanced classification task.
5. The data in the original assignment contains a very large majority class and a much smaller positive class. A confusion matrix and recall are particularly important for understanding missed positive cases.

## Engineering improvements

- Preprocessing is fitted only on training data, then applied to the test set.
- Categorical variables are one-hot encoded rather than assigned arbitrary ordinal values.
- The target's positive/negative meaning is explicit.
- Class weights use the inverse-frequency formula from the assignment.
- Metrics validate input lengths and probability ranges.
- TensorFlow is imported lazily so light-weight preprocessing tests do not require installing a large deep-learning package.

## Limitations

- The tests use a tiny synthetic fixture and do not evaluate generalization.
- The original dataset is not bundled; users must supply a dataset they are allowed to use.
- The one-layer sigmoid network is a baseline, not an optimized architecture.
- Threshold 0.5 is configurable but may not be appropriate for every use case.
- Class weighting does not eliminate the need for validation, calibration, or domain-specific review.
- This code is not a clinical tool and should not be used to diagnose or guide treatment.

## Suggested next steps

1. Run the test suite in CI.
2. Add a reproducible, licensed dataset experiment if terms permit.
3. Compare an unweighted baseline with class-weighted training using the same split and threshold.
4. Report the confusion matrix and precision-recall tradeoff, and document the threshold selection process.
