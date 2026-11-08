"""Requests waiting for a backend, and the ones it's generating."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count


@dataclass(eq=False)  # the same sequence, not an equal one
class Sequence:
    id: int
    prompt: list[int]
    max_tokens: int
    tokens: list[int] = field(default_factory=list)  # generated so far
    done: bool = False


class Queue:
    def __init__(self, max_tokens: int, eos: set[int]):
        """``max_tokens``: for a request that doesn't say. ``eos``: tokens that end a sequence."""
        self.max_tokens, self.eos = max_tokens, eos
        self.ids = count()
        self.sequences: list[Sequence] = []  # every sequence not yet forgotten
        self.waiting: deque[Sequence] = deque()  # in order; backends take from the front

    def add(self, prompt: list[int], max_tokens: int | None = None) -> Sequence:
        seq = Sequence(next(self.ids), prompt, max_tokens or self.max_tokens)
        self.sequences.append(seq)
        self.waiting.append(seq)
        return seq

    def cancel(self, seq: Sequence) -> None:
        """Stop generating ``seq``: it leaves the line, or its backend drops it from here on."""
        if seq in self.waiting:
            self.waiting.remove(seq)
        seq.done = True

    def forget(self, seq: Sequence) -> None:
        """Stop keeping a finished sequence."""
        self.sequences.remove(seq)
