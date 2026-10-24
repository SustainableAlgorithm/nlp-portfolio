# Neural Networks for Imbalanced Binary Classification

Independent portfolio implementation inspired by general techniques studied in DTSC 680: Applied Machine Learning.

## What this demonstrates

- Target cleanup and exclusion of non-binary labels such as `other`.
- Stratified train/test splitting.
- Training-only preprocessing to avoid fitting transformations on the held-out test set.
- Median imputation and standardization for numeric features.
- Mode imputation and one-hot encoding for categorical features.
- A compact Keras binary classifier with a sigmoid output and binary cross-entropy.
- Inverse-frequency class weights and evaluation using precision, recall, F1, accuracy, and a confusion matrix.

The preprocessing and metric helpers can be tested without TensorFlow. To build and train the Keras network, install TensorFlow separately:

```bash
python -m pip install "tensorflow>=2.15,<3.0"
```

Example workflow using a CSV you are permitted to use:

```python
import pandas as pd
from neural_network import (
    prepare_binary_split,
    balanced_class_weights,
    build_keras_binary_classifier,
    evaluate_binary_predictions,
)

frame = pd.read_csv("your-permitted-dataset.csv")
data = prepare_binary_split(frame, target_column="label")
weights = balanced_class_weights(data.y_train)
model = build_keras_binary_classifier(data.x_train.shape[1])
model.fit(
    data.x_train,
    data.y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    class_weight=weights,
    verbose=1,
)
probabilities = model.predict(data.x_test, verbose=0).ravel()
print(evaluate_binary_predictions(data.y_test, probabilities))
```

Do not use a test set to fit preprocessing or select model settings. For imbalanced classification, accuracy alone can be misleading; review recall and precision together and inspect the confusion matrix.

## Data and safety

The original course notebook, coronavirus CSV, pickle exports, and saved H5 model are not included. This is a methodological demonstration, not a medical diagnostic system. No performance or clinical-use claims are made.
