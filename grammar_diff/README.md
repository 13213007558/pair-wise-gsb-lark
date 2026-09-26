# grammar_diff — bounded grammar comparison for Lark

Given **two Lark grammars**, a **shared finite token sample table** and a
**maximum token count**, this tool searches for the **shortest token
sequence that is accepted by exactly one of the two grammars** and reports
the parse outcome of that sequence on both sides.

## Scope (read this first)

* The comparison is **strictly bounded**: the only candidates are token
  sequences composed from the sample table, with length up to
  `max_tokens`. A verdict of `equivalent-within-bound` means *no
  distinguishing sequence exists inside that bounded space*. It is **not**
  a proof that the two context-free languages are equal — that problem is
  undecidable in general, and this tool does not attempt it.
* The search runs at the level of **token types**; sample lexemes are only
  used to render the distinguishing text (the first lexeme of each type).
  A warning is emitted if a lexeme does not lex back to its declared type.
* The tool compares **accepted languages only**. Properties that live in
  the parse trees — operator associativity, precedence, rule factoring —
  are **not comparison targets**. Two grammars that accept the same
  language but group operators differently are reported as
  `equivalent-within-bound` (see example 2).
* Grammars are compiled with Lark's LALR parser; anything Lark/LALR can
  compile works, including **left recursion, nullable rules and mutually
  recursive rules**.

## Usage

```bash
python -m grammar_diff.cli A.lark B.lark --tokens tokens.json --max-tokens 6 \
    [--budget 200000] [--start RULE] [--json] [--trees]
```

`tokens.json` maps terminal names to sample lexemes:

```json
{"NUM": ["1", "42"], "PLUS": ["+"], "COMMA": [","]}
```

Exit code is `1` when a distinguishing sequence is found, `0` otherwise.

Library use:

```python
from grammar_diff import compare_grammars

result = compare_grammars(grammar_a, grammar_b,
                          {"NUM": ["1"], "PLUS": ["+"]}, max_tokens=6)
print(result.summary())
```

## How the search works

Instead of enumerating sequences and parsing each one from scratch, the
search walks the two compiled LALR automatons breadth-first over **shared
prefixes**:

* Each search node carries one immutable LALR state stack per grammar, so
  every sequence sharing a prefix reuses the same parser state. A node
  whose stack is `None` on one side means that side can no longer accept
  any extension of the prefix within the bound.
* **Derivable-length pruning**: for each stack, an exact *minimum number of
  tokens to acceptance* is computed by BFS over the LALR automaton (which
  encodes the grammar's derivable lengths, including nullable and recursive
  rules). If `prefix_length + min_completion > max_tokens`, the branch is
  pruned before any sequence is generated.
* **Deduplication**: nodes are keyed by the pair of stacks, so equivalent
  suffix behaviour is explored once, at the shortest prefix.
* Acceptance is checked at every node (empty sequence included); the first
  node accepted by exactly one side yields the **shortest** distinguishing
  sequence, whose both-sided parse results are then re-computed from
  scratch (accepted tree, or rejection position plus expected tokens).

## Budget and exhaustion

`budget` caps the number of expanded search nodes. The result always
reports `nodes_expanded`, `branches_pruned` and whether the bounded
range was **exhausted**:

* `exhausted=True` — the whole bounded space was checked; the verdict
  (`different` / `equivalent-within-bound`) is final for that space.
* `exhausted=False` with status `inconclusive-budget` — the budget ran
  out; a distinguishing sequence may still exist in the unexplored part.

## Examples

Run `PYTHONPATH=. python grammar_diff/examples/run_examples.py`:

1. **Optional comma added** (`comma_*.lark`) — grammar B accepts
   `NAME NAME` with no comma; grammar A rejects it. The tool reports the
   shortest witness `x x` with both parse outcomes.
2. **Associativity changed, language unchanged** (`assoc_*.lark`) —
   left- vs right-recursive addition both accept `NUM (PLUS NUM)*`.
   Verdict: `equivalent-within-bound`. The associativity difference only
   changes parse-tree grouping; it is not part of the accepted language
   and therefore not a comparison target of this tool.
3. **Genuinely equivalent rewrite** (`rewrite_*.lark`) — EBNF repetition
   `term (PLUS term)*` vs a nullable recursive tail rule
   `rest: PLUS term rest |`. Verdict: `equivalent-within-bound`.

## Tests

```bash
python -m pytest grammar_diff/tests
```

## Limitations

* Bounded and token-based: no claim beyond the sample-table token space.
* LALR only (Lark's `parser='lalr'`); lexer is bypassed, tokens are fed
  directly, so contextual-lexing ambiguities are out of scope.
* Extremely large `max_tokens` × sample-table sizes are kept in check by
  pruning, deduplication and the node budget, but the space is still
  exponential in the worst case.

