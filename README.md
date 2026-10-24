# NLP Portfolio Lab

Independent, newly written implementations of core natural language processing techniques for a professional portfolio.

> These are original demonstrations of general NLP methods, not copies of university assignment solutions. Original course notebooks, datasets, and serialized models are intentionally excluded. Use only datasets whose license and redistribution terms permit your use.

## Topics

- **Text classification:** TF-IDF features, Naive Bayes, Logistic Regression, and classification metrics.
- **Language modeling:** unigram probabilities, smoothed n-gram probabilities, and next-token scoring.
- **Sequence tagging:** a generic Hidden Markov Model Viterbi decoder using log probabilities.
- **Syntactic parsing:** CKY chart parsing for a binary context-free grammar.

## Setup

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python -m pytest -q
```

The examples use small synthetic fixtures in tests. They do not bundle course datasets or claim benchmark performance.

## DTSC 680: Applied Machine Learning

The independent PCA/t-SNE portfolio implementation is in [`dtsc680/assignment1_pca_tsne/`](dtsc680/assignment1_pca_tsne/README.md). See its [review and local results](dtsc680/assignment1_pca_tsne/REVIEW.md). The original course notebook, CSV, pickle outputs, and environment files are intentionally excluded.

The independent neural-network implementation is in [`dtsc680/assignment2_neural_network/`](dtsc680/assignment2_neural_network/README.md). It demonstrates stratified splitting, training-only preprocessing, class weighting, a Keras binary classifier, and evaluation metrics. The original notebook, coronavirus dataset, pickle exports, and saved model are excluded.

## Project structure

```text
nlp_portfolio/
  sentiment.py
  language_models.py
  hmm_viterbi.py
  cky_parser.py
tests/
  test_nlp.py
```

## Review status

Eight smoke tests passed in a clean temporary Python workspace, including n-gram scoring and Viterbi edge cases. See [CODE_REVIEW.md](reports/CODE_REVIEW.md) for coverage, limitations, and next improvements. These tests demonstrate basic behavior only; they are not a benchmark or evidence of real-world model performance.

This repository is private while the code and documentation are reviewed. A separate public showcase summarizes the techniques without exposing course materials or private artifacts.
