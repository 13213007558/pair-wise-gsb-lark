"""Run the three bundled example comparisons and print the verdicts.

1. comma:    grammar B adds an optional comma  -> a distinguishing sequence
             exists and is reported.
2. assoc:    operator associativity changed (left vs right recursion) but
             the accepted language is unchanged -> no distinguishing
             sequence.  Associativity only changes the *shape of the parse
             trees*; it is not part of the accepted language and therefore
             not a comparison target of this tool.
3. rewrite:  a genuinely equivalent rewrite (EBNF repetition vs nullable
             recursive tail rule) -> no distinguishing sequence.
"""

import json
import os

from grammar_diff import compare_grammars

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


def run(title, note, base, max_tokens):
    print("=" * 72)
    print(title)
    print("-" * 72)
    result = compare_grammars(
        load(base + "_a.lark"),
        load(base + "_b.lark"),
        json.loads(load(base + "_tokens.json")),
        max_tokens,
    )
    print(result.summary())
    if result.is_different:
        for label, side in (("A", result.side_a), ("B", result.side_b)):
            if side.tree:
                print("side %s parse tree:" % label)
                print(side.tree)
    print(note)
    print()
    return result


def main():
    comma = run(
        "Example 1: optional comma added",
        "Grammar B accepts two items with no comma between them; grammar A "
        "does not. The shortest witness is reported.",
        "comma",
        max_tokens=4,
    )
    assert comma.is_different
    assert comma.distinguishing_tokens == ("NAME", "NAME")

    assoc = run(
        "Example 2: associativity changed, language unchanged",
        "Both grammars accept exactly NUM (PLUS NUM)*. The left- vs "
        "right-associative grouping only changes parse-tree shape, which is "
        "outside this tool's comparison target, so the bounded verdict is "
        "'equivalent-within-bound'.",
        "assoc",
        max_tokens=5,
    )
    assert assoc.status == "equivalent-within-bound" and assoc.exhausted

    rewrite = run(
        "Example 3: genuinely equivalent rewrite",
        "EBNF repetition and a nullable recursive tail rule derive the same "
        "language, so no distinguishing sequence exists in the bounded "
        "space.",
        "rewrite",
        max_tokens=5,
    )
    assert rewrite.status == "equivalent-within-bound" and rewrite.exhausted


if __name__ == "__main__":
    main()

