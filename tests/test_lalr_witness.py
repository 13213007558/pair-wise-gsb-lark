# -*- coding: utf-8 -*-
"""Tests for the LALR conflict witness diagnostics (lark/parsers/lalr_witness.py).

Strict-mode LALR build errors carry a witness: a terminal sequence that
reaches the conflict state (verified by replaying the parser state machine),
the two competing actions, and - when both actions extend to full parses of
the same input - the two derivations proving the ambiguity.
"""
from __future__ import absolute_import

import unittest

from lark import Lark, LexError
from lark.exceptions import GrammarError
from lark.parsers import lalr_witness


# Truly ambiguous: the classic dangling-else grammar.
AMBIGUOUS_GRAMMAR = r"""
%ignore " "
start: "if" start "else" start
     | "if" start
     | "a"
"""

# LR(1) but not LALR(1): the LALR construction merges the states reached
# after "a" c and "b" c, creating a reduce/reduce conflict that does not
# exist in canonical LR(1). The grammar itself is unambiguous.
LALR_ONLY_CONFLICT_GRAMMAR = r"""
%ignore " "
start: "a" x "d" | "b" y "d" | "a" y "e" | "b" x "e"
x: "c"
y: "c"
"""

# The path to the conflict state crosses a nullable nonterminal.
NULLABLE_GRAMMAR = r"""
start: nb a "."
nb: "b"|
a: "."+
"""

# Lexer regex collision: a lexer diagnostic, not a grammar conflict.
REGEX_COLLISION_GRAMMAR = r"""
start: A | B
A: /e?rez/
B: /erez?/
"""


def build_strict(grammar, **kwargs):
    return Lark(grammar, parser='lalr', strict=True, **kwargs)


def distinct_tree_count(grammar, text):
    "Number of distinct parse trees for 'text', via the earley parser."
    from lark.tree import Tree
    parser = Lark(grammar, parser='earley', ambiguity='explicit', keep_all_tokens=True)
    forest = parser.parse(text)

    def expand(node):
        if not isinstance(node, Tree):
            return [node]
        if node.data == '_ambig':
            return [t for child in node.children for t in expand(child)]
        trees = [Tree(node.data, [])]
        for child in node.children:
            if isinstance(child, Tree) and child.data == '_ambig':
                trees = [Tree(t.data, t.children + [e]) for t in trees for e in expand(child)]
            else:
                trees = [Tree(t.data, t.children + [c]) for t in trees for c in expand(child)]
        return trees

    return len(set(expand(forest)))


class TestLalrWitness(unittest.TestCase):
    def test_ambiguous_grammar_proven(self):
        with self.assertRaises(GrammarError) as cm:
            build_strict(AMBIGUOUS_GRAMMAR)
        msg = str(cm.exception)
        self.assertIn('Shift/Reduce conflict for terminal ELSE', msg)
        self.assertIn('Conflict state is reachable', msg)
        self.assertIn('replaying the parser state machine', msg)
        self.assertIn('shift ELSE', msg)
        self.assertIn('reduce <start : IF start>', msg)
        self.assertIn('AMBIGUITY PROVEN', msg)
        self.assertIn('IF IF A ELSE A', msg)
        self.assertIn('Derivation 1 (shift ELSE first)', msg)
        self.assertIn('Derivation 2 (reduce <start : IF start> first)', msg)

    def test_ambiguous_witness_is_real(self):
        # Cross-check the reported witness: the input really has two trees.
        self.assertEqual(distinct_tree_count(AMBIGUOUS_GRAMMAR, 'if if a else a'), 2)

    def test_lalr_only_conflict_not_proven(self):
        with self.assertRaises(GrammarError) as cm:
            build_strict(LALR_ONLY_CONFLICT_GRAMMAR)
        msg = str(cm.exception)
        self.assertIn('Reduce/Reduce collision', msg)
        self.assertIn('Conflict state is reachable', msg)
        self.assertIn('replaying the parser state machine', msg)
        self.assertIn('NO two-tree witness found', msg)
        self.assertIn('exhaustive', msg)
        self.assertNotIn('AMBIGUITY PROVEN', msg)

    def test_lalr_only_grammar_is_unambiguous(self):
        # The LALR-only conflict grammar is unambiguous: every input has
        # exactly one tree, which is why no two-tree witness exists.
        for text in ('a c d', 'b c d', 'a c e', 'b c e'):
            self.assertEqual(distinct_tree_count(LALR_ONLY_CONFLICT_GRAMMAR, text), 1)

    def test_nullable_rules_not_treated_as_input(self):
        with self.assertRaises(GrammarError) as cm:
            build_strict(NULLABLE_GRAMMAR)
        msg = str(cm.exception)
        # nb is nullable; the witness must contain terminals only
        self.assertIn('on input: DOT\n', msg)
        self.assertNotIn('nb', msg.split('Competing actions')[0])

    def test_import_and_alias_in_report(self):
        grammar = '%import .test_witness_dangling_else.start'
        with self.assertRaises(GrammarError) as cm:
            build_strict(grammar, source_path=__file__)
        msg = str(cm.exception)
        # the original rule (with its alias) is identifiable in the report
        self.assertIn('alias: test_witness_dangling_else__ifonly', msg)
        self.assertIn('AMBIGUITY PROVEN', msg)

    def test_length_budget_exceeded(self):
        old_budget = lalr_witness.DEFAULT_BUDGET
        lalr_witness.DEFAULT_BUDGET = lalr_witness.WitnessBudget(max_input_len=3)
        try:
            with self.assertRaises(GrammarError) as cm:
                build_strict(AMBIGUOUS_GRAMMAR)
        finally:
            lalr_witness.DEFAULT_BUDGET = old_budget
        msg = str(cm.exception)
        self.assertIn('Conflict state is reachable', msg)
        self.assertIn('NO two-tree witness found', msg)
        self.assertIn('input length budget exceeded', msg)
        self.assertNotIn('AMBIGUITY PROVEN', msg)

    def test_state_budget_exceeded(self):
        old_budget = lalr_witness.DEFAULT_BUDGET
        lalr_witness.DEFAULT_BUDGET = lalr_witness.WitnessBudget(max_states=1)
        try:
            with self.assertRaises(GrammarError) as cm:
                build_strict(AMBIGUOUS_GRAMMAR)
        finally:
            lalr_witness.DEFAULT_BUDGET = old_budget
        msg = str(cm.exception)
        self.assertIn('state budget exceeded', msg)

    def test_regex_collision_is_not_a_grammar_conflict(self):
        # Lexer regex collisions are a lexer diagnostic; they must not show
        # up as parser state machine (grammar) conflicts.
        with self.assertRaises(LexError) as cm:
            build_strict(REGEX_COLLISION_GRAMMAR)
        self.assertNotIn('Witness search', str(cm.exception))

    def test_non_strict_still_resolves_as_shift(self):
        # Non-strict mode is unaffected: the conflict resolves as shift.
        parser = Lark(AMBIGUOUS_GRAMMAR, parser='lalr')
        tree = parser.parse('if if a else a')
        self.assertEqual(tree.data, 'start')


if __name__ == '__main__':
    unittest.main()
