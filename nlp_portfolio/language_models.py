"""Small unigram and n-gram language models with additive smoothing."""
from __future__ import annotations

from collections import Counter
from math import log
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Simple Unicode-aware tokenization; language-specific tokenizers may be needed."""
    return TOKEN_PATTERN.findall(text.lower())


class UnigramModel:
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.counts: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.total = 0

    def fit(self, texts: Iterable[str]) -> "UnigramModel":
        self.counts.clear()
        for text in texts:
            self.counts.update(tokenize(text))
        self.vocabulary = set(self.counts) | {"<UNK>"}
        self.total = sum(self.counts.values())
        return self

    def probability(self, token: str) -> float:
        if not self.vocabulary:
            raise RuntimeError("fit the model before requesting probabilities")
        token = token.lower()
        count = self.counts.get(token, 0)
        return (count + self.alpha) / (self.total + self.alpha * len(self.vocabulary))

    def log_likelihood(self, text: str) -> float:
        tokens = tokenize(text)
        return sum(log(self.probability(token)) for token in tokens)


class NGramModel:
    """Conditional n-gram model; unseen contexts use additive smoothing."""
    def __init__(self, n: int = 3, alpha: float = 0.5):
        if n < 2:
            raise ValueError("n must be at least 2")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.n, self.alpha = n, alpha
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.ngram_counts: Counter[tuple[str, ...]] = Counter()
        self.vocabulary: set[str] = set()

    def fit(self, texts: Iterable[str]) -> "NGramModel":
        self.context_counts.clear(); self.ngram_counts.clear(); self.vocabulary.clear()
        for text in texts:
            tokens = ["<s>"] * (self.n - 1) + tokenize(text) + ["</s>"]
            self.vocabulary.update(tokens)
            for i in range(self.n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.n + 1:i + 1])
                self.ngram_counts[ngram] += 1
                self.context_counts[ngram[:-1]] += 1
        self.vocabulary.add("<UNK>")
        return self

    def probability(self, context: Iterable[str], token: str) -> float:
        if not self.vocabulary:
            raise RuntimeError("fit the model before requesting probabilities")
        context_tuple = tuple(context)
        if len(context_tuple) != self.n - 1:
            raise ValueError(f"context must contain exactly {self.n - 1} tokens")
        token = token.lower()
        count = self.ngram_counts[context_tuple + (token,)]
        denominator = self.context_counts[context_tuple] + self.alpha * len(self.vocabulary)
        return (count + self.alpha) / denominator
