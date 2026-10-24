"""Generic Hidden Markov Model Viterbi decoding using log probabilities."""
from __future__ import annotations

from math import log
from typing import Mapping, Sequence


def _safe_log(probability: float) -> float:
    if probability < 0:
        raise ValueError("probabilities cannot be negative")
    return log(probability) if probability > 0 else float("-inf")


def viterbi_decode(
    observations: Sequence[str],
    states: Sequence[str],
    start_prob: Mapping[str, float],
    transition_prob: Mapping[tuple[str, str], float],
    emission_prob: Mapping[tuple[str, str], float],
) -> tuple[list[str], float]:
    """Return the most likely state sequence and its log probability.

    Missing transition/emission entries are treated as zero probability. For
    unknown observations, callers should supply a smoothed emission probability.
    """
    if not observations:
        return [], 0.0
    if not states:
        raise ValueError("at least one state is required")
    score: dict[str, float] = {}
    backpointers: list[dict[str, str | None]] = []
    first_back: dict[str, str | None] = {}
    for state in states:
        score[state] = _safe_log(start_prob.get(state, 0.0)) + _safe_log(emission_prob.get((state, observations[0]), 0.0))
        first_back[state] = None
    backpointers.append(first_back)
    for observation in observations[1:]:
        next_score: dict[str, float] = {}
        back: dict[str, str | None] = {}
        for current in states:
            emission = _safe_log(emission_prob.get((current, observation), 0.0))
            candidates = [(score[previous] + _safe_log(transition_prob.get((previous, current), 0.0)) + emission, previous) for previous in states]
            best_score, best_previous = max(candidates, key=lambda item: item[0])
            next_score[current] = best_score
            back[current] = best_previous
        score, back = next_score, back
        backpointers.append(back)
    final_state = max(states, key=lambda state: score[state])
    best_log_probability = score[final_state]
    path = [final_state]
    for position in range(len(observations) - 1, 0, -1):
        previous = backpointers[position][path[-1]]
        if previous is None:
            raise RuntimeError("invalid backpointer while reconstructing path")
        path.append(previous)
    path.reverse()
    return path, best_log_probability
