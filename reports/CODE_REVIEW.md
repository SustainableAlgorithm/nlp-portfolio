# NLP code review

Review date: 2026-10-24

## Test run

The current committed source and test logic were run in a clean temporary Python workspace with the listed modules and installed dependencies.

**Result: 8 tests passed after the hardening changes below.**

Coverage includes:
- A tiny synthetic sentiment classification example for the Logistic Regression pipeline.
- Smoothed unigram probabilities for seen and unseen words.
- Smoothed n-gram probabilities for an unseen token.
- N-gram perplexity and next-token ranking.
- Viterbi rejection of impossible sequences and invalid probabilities.
- Viterbi decoding on a two-state toy example.
- CKY parsing for a binary grammar and rejection of a sentence without a complete parse.

These are smoke tests of core behavior, not a full benchmark or proof of generalization. The private repository also has a GitHub Actions workflow configured, but no hosted workflow run was visible at review time.

## Findings and limitations

### Sentiment classification
- The pipeline uses TF-IDF unigrams and bigrams and supports Logistic Regression, Multinomial Naive Bayes, and Perceptron.
- The test uses a very small synthetic dataset. It is only a functional test; it does not establish meaningful sentiment accuracy.
- `evaluate_classifier` assumes the supplied texts and labels are aligned and non-empty. A future hardening pass should validate this explicitly and include a stratified train/validation/test example using a licensed dataset.

### Unigram and n-gram models
- Additive smoothing gives non-zero probability to unseen tokens.
- The tokenizer is a baseline Unicode-aware regular expression, not a language-specific tokenizer. It should not be presented as sufficient Japanese segmentation.
- The n-gram model now includes basic perplexity and next-token ranking helpers. It still does not implement backoff/interpolation, and its tokenizer remains intentionally simple.
- Model probabilities assume the caller supplies a context in the same token form used during training.

### HMM Viterbi decoder
- Uses log probabilities and returns a highest-scoring path for the supplied start, transition, and emission probabilities.
- The caller must smooth missing emissions/transitions when appropriate.
- It now validates individual probability values are between zero and one and raises an error when no valid path exists. It does not validate that probability rows sum to one; use beyond a toy demonstration requires further tests.

### CKY parser
- Implements chart parsing for binary context-free grammars in Chomsky Normal Form.
- Unary productions, epsilon rules, weighted/probabilistic parsing, and returning all possible parse trees are not implemented.
- It deterministically returns one parse tree; this should be documented as a basic parser rather than a general-purpose parser.

## Publication status

This is an independent implementation of general NLP methods. The original DTSC 685 assignment notebooks, course datasets, and serialized models are not included. Do not add those materials without explicit permission. Before making any code public, add a licensed, reproducible experiment and review course policy, dataset terms, documentation, and test results.

## Recommended next improvements

1. Add input validation and tests for empty inputs and mismatched text/label lengths.
2. Add n-gram perplexity and optional next-token ranking.
3. Add a licensed sentiment dataset experiment with a fixed split and baseline metrics.
4. Add probability validation and no-path handling to Viterbi.
5. Expand CKY tests for ambiguous grammars and longer spans.
6. Rename the repository from `gitexample` to `nlp-portfolio` in GitHub repository settings; the connected GitHub actions available in this session do not expose a repository-rename operation.
