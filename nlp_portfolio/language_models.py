"""Small unigram and n-gram language models with additive smoothing."""
from __future__ import annotations
from collections import Counter
from math import exp, log
import re
from typing import Iterable

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

def tokenize(text: str) -> list[str]:
    """Simple Unicode-aware tokenization; language-specific tokenizers may be needed."""
    return TOKEN_PATTERN.findall(text.lower())

class UnigramModel:
    def __init__(self, alpha: float = 1.0):
        if alpha <= 0: raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.counts: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.total = 0

    def fit(self, texts: Iterable[str]) -> "UnigramModel":
        self.counts.clear()
        for text in texts: self.counts.update(tokenize(text))
        self.vocabulary = set(self.counts) | {"<UNK>"}
        self.total = sum(self.counts.values())
        return self

    def probability(self, token: str) -> float:
        if not self.vocabulary: raise RuntimeError("fit the model before requesting probabilities")
        token = token.lower()
        return (self.counts.get(token, 0) + self.alpha) / (self.total + self.alpha * len(self.vocabulary))

    def log_likelihood(self, text: str) -> float:
        """Return the sum of token log probabilities for the text."""
        return sum(log(self.probability(token)) for token in tokenize(text))

class NGramModel:
    """Conditional n-gram model with additive smoothing and simple scoring helpers."""
    def __init__(self, n: int = 3, alpha: float = 0.5):
        if n < 2: raise ValueError("n must be at least 2")
        if alpha <= 0: raise ValueError("alpha must be positive")
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
        if not self.vocabulary: raise RuntimeError("fit the model before requesting probabilities")
        context_tuple = tuple(t.lower() for t in context)
        if len(context_tuple) != self.n - 1: raise ValueError(f"context must contain exactly {self.n - 1} tokens")
        count = self.ngram_counts[context_tuple + (token.lower(),)]
        denominator = self.context_counts[context_tuple] + self.alpha * len(self.vocabulary)
        return (count + self.alpha) / denominator

    def next_token_candidates(self, context: Iterable[str], *, limit: int = 10) -> list[tuple[str, float]]:
        """Return highest-probability observed next tokens for a context."""
        if limit < 1: raise ValueError("limit must be at least 1")
        context_tuple = tuple(t.lower() for t in context)
        if len(context_tuple) != self.n - 1: raise ValueError(f"context must contain exactly {self.n - 1} tokens")
        if not self.vocabulary: raise RuntimeError("fit the model before requesting predictions")
        candidates = {ngram[-1] for ngram in self.ngram_counts if ngram[:-1] == context_tuple}
        if not candidates: candidates = self.vocabulary - {"<s>"}
        ranked = [(token, self.probability(context_tuple, token)) for token in candidates]
        return sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit]

    def perplexity(self, texts: Iterable[str]) -> float:
        """Compute token-level perplexity, including end-of-sequence tokens."""
        if not self.vocabulary: raise RuntimeError("fit the model before requesting perplexity")
        log_probabilities = []
        token_count = 0
        for text in texts:
            tokens = ["<s>"] * (self.n - 1) + tokenize(text) + ["</s>"]
            for i in range(self.n - 1, len(tokens)):
                context, token = tokens[i - self.n + 1:i], tokens[i]
                log_probabilities.append(log(self.probability(context, token)))
                token_count += 1
        if token_count == 0: raise ValueError("perplexity requires at least one predicted token")
        return exp(-sum(log_probabilities) / token_count)
