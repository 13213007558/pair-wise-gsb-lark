import pytest

from grammar_diff import LalrMachine, compare_grammars

COMMA_A = """
start: item (COMMA item)*
item: NAME
NAME: /[a-z]+/
COMMA: ","
%ignore " "
"""

COMMA_B = """
start: item (COMMA? item)*
item: NAME
NAME: /[a-z]+/
COMMA: ","
%ignore " "
"""

ASSOC_A = """
start: expr
expr: expr PLUS term | term
term: NUM
PLUS: "+"
NUM: /[0-9]+/
%ignore " "
"""

ASSOC_B = """
start: expr
expr: term PLUS expr | term
term: NUM
PLUS: "+"
NUM: /[0-9]+/
%ignore " "
"""

REWRITE_B = """
start: expr
expr: term rest
rest: PLUS term rest |
term: NUM
PLUS: "+"
NUM: /[0-9]+/
%ignore " "
"""

# Mutually recursive rules (a <-> b) and an equivalent inlined rewrite.
MUTUAL_A = """
start: a
a: b PLUS a | b
b: NUM | LP a RP
PLUS: "+"
NUM: /[0-9]+/
LP: "("
RP: ")"
"""

MUTUAL_B = """
start: a
a: NUM PLUS a | LP a RP PLUS a | NUM | LP a RP
PLUS: "+"
NUM: /[0-9]+/
LP: "("
RP: ")"
"""

EXPR_TOKENS = {"NUM": ["1"], "PLUS": ["+"]}


def test_optional_comma_is_distinguished():
    result = compare_grammars(
        COMMA_A, COMMA_B, {"NAME": ["x"], "COMMA": [","]}, max_tokens=4
    )
    assert result.status == "different"
    assert result.distinguishing_tokens == ("NAME", "NAME")
    assert result.distinguishing_text == "x x"
    assert not result.side_a.accepted
    assert result.side_b.accepted
    assert result.side_b.tree is not None
    assert "rejected at token 2" in result.side_a.detail


def test_distinguishing_sequence_is_shortest():
    # The comma difference needs two adjacent NAMEs; nothing shorter works.
    result = compare_grammars(
        COMMA_B, COMMA_A, {"NAME": ["x"], "COMMA": [","]}, max_tokens=4
    )
    assert result.distinguishing_tokens == ("NAME", "NAME")
    assert result.side_a.accepted  # sides follow argument order
    assert not result.side_b.accepted


def test_difference_beyond_bound_is_not_reported():
    # With max_tokens=1 the bounded space contains no witness; the tool
    # must say so honestly instead of claiming equivalence in general.
    result = compare_grammars(
        COMMA_A, COMMA_B, {"NAME": ["x"], "COMMA": [","]}, max_tokens=1
    )
    assert result.status == "equivalent-within-bound"
    assert result.exhausted


def test_associativity_change_keeps_language():
    result = compare_grammars(ASSOC_A, ASSOC_B, EXPR_TOKENS, max_tokens=5)
    assert result.status == "equivalent-within-bound"
    assert result.exhausted
    assert result.distinguishing_tokens is None


def test_equivalent_nullable_rewrite():
    ebnf = ASSOC_A.replace("expr: expr PLUS term | term", "expr: term (PLUS term)*")
    result = compare_grammars(ebnf, REWRITE_B, EXPR_TOKENS, max_tokens=5)
    assert result.status == "equivalent-within-bound"
    assert result.exhausted


def test_mutual_recursion_equivalent_inline():
    tokens = {"NUM": ["1"], "PLUS": ["+"], "LP": ["("], "RP": [")"]}
    result = compare_grammars(MUTUAL_A, MUTUAL_B, tokens, max_tokens=6)
    assert result.status == "equivalent-within-bound"
    assert result.exhausted


def test_mutual_recursion_real_difference():
    # Drop one inlined alternative: only A then accepts "(1)+1".
    broken = MUTUAL_B.replace(" | LP a RP PLUS a", "")
    tokens = {"NUM": ["1"], "PLUS": ["+"], "LP": ["("], "RP": [")"]}
    result = compare_grammars(MUTUAL_A, broken, tokens, max_tokens=6)
    assert result.status == "different"
    assert result.distinguishing_tokens == ("LP", "NUM", "RP", "PLUS", "NUM")
    assert result.side_a.accepted
    assert not result.side_b.accepted


def test_nullable_start_gives_empty_witness():
    tokens = {"A": ["a"], "B": ["b"]}
    result = compare_grammars(
        "start: (A B)?\nA: \"a\"\nB: \"b\"\n",
        "start: A B\nA: \"a\"\nB: \"b\"\n",
        tokens,
        max_tokens=2,
    )
    assert result.status == "different"
    assert result.distinguishing_tokens == ()
    assert result.side_a.accepted
    assert not result.side_b.accepted


def test_budget_exhaustion_is_reported():
    result = compare_grammars(ASSOC_A, ASSOC_B, EXPR_TOKENS, max_tokens=7, budget=1)
    assert result.status == "inconclusive-budget"
    assert not result.exhausted
    assert result.nodes_expanded == 1


def test_unknown_token_type_rejected():
    with pytest.raises(ValueError, match="not a terminal"):
        compare_grammars(ASSOC_A, ASSOC_B, {"NOPE": ["x"]}, max_tokens=2)


def test_min_completion_distance():
    machine = LalrMachine(ASSOC_A)
    # From the start, at least one NUM is needed to finish a parse.
    assert machine.min_completion(machine.start_stack, cap=5) == 1
    # A cap below the true distance proves nothing and reports None.
    assert machine.min_completion(machine.start_stack, cap=0) is None

