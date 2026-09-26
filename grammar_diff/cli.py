"""Command line interface for the bounded grammar comparison tool.

Usage:
    python -m grammar_diff.cli GRAMMAR_A GRAMMAR_B --tokens TOKENS.json \
        --max-tokens N [--budget N] [--start RULE] [--json] [--trees]

TOKENS.json maps terminal names to lists of sample lexemes, e.g.
    {"NUM": ["1", "42"], "PLUS": ["+"], "COMMA": [","]}
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .core import compare_grammars


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="grammar_diff",
        description=(
            "Bounded comparison of two Lark grammars over token sequences "
            "built from a finite sample table. Not a CFG equivalence proof."
        ),
    )
    parser.add_argument("grammar_a", help="path to the first Lark grammar")
    parser.add_argument("grammar_b", help="path to the second Lark grammar")
    parser.add_argument(
        "--tokens",
        required=True,
        help="JSON file mapping terminal names to sample lexemes",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="maximum number of tokens per candidate sequence (inclusive)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=200000,
        help="maximum number of search nodes to expand (default: 200000)",
    )
    parser.add_argument("--start", default=None, help="common start rule name")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument(
        "--trees",
        action="store_true",
        help="print parse trees for the distinguishing sequence",
    )
    args = parser.parse_args(argv)

    with open(args.grammar_a, encoding="utf-8") as f:
        grammar_a = f.read()
    with open(args.grammar_b, encoding="utf-8") as f:
        grammar_b = f.read()
    with open(args.tokens, encoding="utf-8") as f:
        token_samples = json.load(f)

    result = compare_grammars(
        grammar_a,
        grammar_b,
        token_samples,
        args.max_tokens,
        start=args.start,
        budget=args.budget,
    )

    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        print(result.summary())
        if args.trees and result.is_different:
            for name, side in (("A", result.side_a), ("B", result.side_b)):
                if side.tree:
                    print("side %s parse tree:" % name)
                    print(side.tree)
    return 0 if not result.is_different else 1


if __name__ == "__main__":
    sys.exit(main())

