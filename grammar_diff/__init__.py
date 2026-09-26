"""Bounded grammar comparison for Lark grammars.

Given two Lark grammars, a shared finite token sample table and a maximum
token count, find the shortest token sequence that is accepted by exactly
one of the two grammars, together with the parse outcome on both sides.

The comparison is strictly bounded: only token sequences composed from the
sample table with length up to ``max_tokens`` are considered.  A verdict of
``equivalent-within-bound`` is NOT a proof that the two context-free
languages are equal; it only states that no distinguishing sequence exists
inside the bounded search space.
"""

from .core import (
    CompareResult,
    LalrMachine,
    SideReport,
    compare_grammars,
)

__all__ = [
    "CompareResult",
    "LalrMachine",
    "SideReport",
    "compare_grammars",
]

__version__ = "0.1.0"
