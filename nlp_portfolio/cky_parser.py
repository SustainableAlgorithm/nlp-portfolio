"""CKY recognition and backtracking for a binary context-free grammar."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ParseNode:
    label: str
    children: tuple["ParseNode", ...] = ()
    token: str | None = None

    def as_tuple(self):
        if self.token is not None:
            return (self.label, self.token)
        return (self.label, *(child.as_tuple() for child in self.children))


def cky_parse(tokens: Iterable[str], binary_rules: Iterable[tuple[str, str, str]], lexical_rules: Iterable[tuple[str, str]]) -> ParseNode | None:
    """Return one parse tree, or None if no parse exists.

    Rules are supplied as (parent, left_child, right_child) and (preterminal, token).
    Grammar should be in Chomsky Normal Form; unary and epsilon rules are not handled.
    """
    words = list(tokens)
    n = len(words)
    if n == 0:
        return None
    binary_by_children: dict[tuple[str, str], list[str]] = defaultdict(list)
    for parent, left, right in binary_rules:
        binary_by_children[(left, right)].append(parent)
    lexical_by_word: dict[str, list[str]] = defaultdict(list)
    for tag, word in lexical_rules:
        lexical_by_word[word].append(tag)
    chart: dict[tuple[int, int], dict[str, ParseNode]] = {}
    for i, word in enumerate(words):
        chart[(i, i + 1)] = {tag: ParseNode(tag, token=word) for tag in lexical_by_word.get(word, [])}
    for span in range(2, n + 1):
        for start in range(n - span + 1):
            end = start + span
            cell: dict[str, ParseNode] = {}
            for split in range(start + 1, end):
                left_cell = chart.get((start, split), {})
                right_cell = chart.get((split, end), {})
                for left_label, left_node in left_cell.items():
                    for right_label, right_node in right_cell.items():
                        for parent in binary_by_children.get((left_label, right_label), []):
                            cell.setdefault(parent, ParseNode(parent, (left_node, right_node)))
            chart[(start, end)] = cell
    full_cell = chart.get((0, n), {})
    if not full_cell:
        return None
    # A grammar may have multiple roots; return a deterministic first root.
    root_label = sorted(full_cell)[0]
    return full_cell[root_label]
