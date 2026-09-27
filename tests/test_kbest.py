from __future__ import absolute_import

import unittest
from itertools import islice, product

from lark import Lark, Tree, Token
from lark.exceptions import ConfigurationError
from lark.grammar import Rule, NonTerminal, Terminal
from lark.lexer import Token as LexerToken
from lark.parsers.earley_kbest import KBestIterator
from lark.parsers.earley_forest import StableSymbolNode, TokenNode, ForestVisitor, SymbolNode


GROUPING_GRAMMAR = r'''
start: s
s: item+
item: WORD -> w
    | WORD WORD -> ww
%import common.WORD
%ignore " "
'''

EXPR_GRAMMAR = r'''
start: e
e: e "+" e -> add
 | e "*" e -> mul
 | "(" e ")" -> par
 | CNAME -> var
%import common.CNAME
%ignore " "
'''


def tree_cost(tree, costs):
    return sum(costs.get(node.data, 0) for node in tree.iter_subtrees())


def enumerate_explicit(tree):
    """Enumerate every tree of an ambiguity='explicit' parse."""
    if tree.data == '_ambig':
        for child in tree.children:
            for t in enumerate_explicit(child):
                yield t
    else:
        child_options = [list(enumerate_explicit(c)) if isinstance(c, Tree) else [c]
                         for c in tree.children]
        for combo in product(*child_options):
            yield Tree(tree.data, combo)


class TestKBest(unittest.TestCase):
    def test_small_grammar_full_order(self):
        # All 5 parses of 'a b c d', costs 4, 6, 6, 6, 8
        costs = {'w': 1, 'ww': 4}
        rule_cost = lambda rule: costs.get(rule.alias or rule.origin.name, 0)
        parser = Lark(GROUPING_GRAMMAR, parser='earley')
        it = parser.parse_kbest('a b c d', rule_cost=rule_cost)
        trees = list(it)

        self.assertEqual(len(trees), 5)
        self.assertEqual([tree_cost(t, costs) for t in trees], [4, 6, 6, 6, 8])
        # No duplicates
        self.assertEqual(len({t for t in trees}), 5)

        # Cross-check against brute-force enumeration of the explicit forest
        explicit = Lark(GROUPING_GRAMMAR, parser='earley', ambiguity='explicit').parse('a b c d')
        brute_force = sorted(tree_cost(t, costs) for t in enumerate_explicit(explicit))
        self.assertEqual([tree_cost(t, costs) for t in trees], brute_force)

    def test_stable_tie_order(self):
        # Two runs over identical parsers must produce identical sequences
        costs = {'w': 1, 'ww': 4}
        rule_cost = lambda rule: costs.get(rule.alias or rule.origin.name, 0)
        runs = []
        for _ in range(2):
            parser = Lark(GROUPING_GRAMMAR, parser='earley')
            runs.append(list(parser.parse_kbest('a b c d e', rule_cost=rule_cost)))
        self.assertEqual(len(runs[0]), 8)
        self.assertEqual(runs[0], runs[1])

    def test_token_positions_preserved(self):
        parser = Lark(GROUPING_GRAMMAR, parser='earley')
        tree = next(iter(parser.parse_kbest('a b', rule_cost=lambda rule: 0)))
        tokens = list(tree.scan_values(lambda v: isinstance(v, Token)))
        self.assertEqual([t.start_pos for t in tokens], [0, 2])
        self.assertEqual([t.end_pos for t in tokens], [1, 3])
        self.assertEqual([t.line for t in tokens], [1, 1])
        self.assertEqual([t.column for t in tokens], [1, 3])

    def test_cost_callback_memoized(self):
        costs = {'w': 1, 'ww': 4}
        calls = {}

        def rule_cost(rule):
            calls[rule] = calls.get(rule, 0) + 1
            return costs.get(rule.alias or rule.origin.name, 0)

        parser = Lark(GROUPING_GRAMMAR, parser='earley')
        list(parser.parse_kbest('a b c d', rule_cost=rule_cost))
        self.assertTrue(calls)
        for rule, n in calls.items():
            self.assertEqual(n, 1, 'callback for %r ran %d times' % (rule, n))

    def test_cost_mapping(self):
        parser = Lark(GROUPING_GRAMMAR, parser='earley')
        trees = list(parser.parse_kbest('a b c', rule_cost={'w': 1, 'ww': 4}))
        self.assertEqual([tree_cost(t, {'w': 1, 'ww': 4}) for t in trees], [3, 5, 5])

    def test_k_parameter(self):
        parser = Lark(GROUPING_GRAMMAR, parser='earley')
        it = parser.parse_kbest('a b c d', rule_cost=lambda rule: 0, k=3)
        self.assertEqual(len(list(it)), 3)
        self.assertEqual(it.trees_yielded, 3)

    def test_dynamic_lexer(self):
        parser = Lark(GROUPING_GRAMMAR, parser='earley', lexer='dynamic')
        trees = list(parser.parse_kbest('a b c', rule_cost={'w': 1, 'ww': 4}))
        self.assertEqual(len(trees), 3)

    def test_dedup_intermediate_nodes(self):
        # Hand-built SPPF where the same tree is represented twice:
        # s(0,1) -> a(0,1) b(1,1) directly, and via an intermediate node.
        rule_s = Rule(NonTerminal('s'), [NonTerminal('a'), NonTerminal('b')])
        rule_a = Rule(NonTerminal('a'), [Terminal('X')])
        rule_b = Rule(NonTerminal('b'), [])

        tok = TokenNode(LexerToken('X', 'x', start_pos=0, end_pos=1), None)
        a_sym = StableSymbolNode(NonTerminal('a'), 0, 1)
        a_sym.add_family(NonTerminal('a'), rule_a, 0, None, tok)
        b_sym = StableSymbolNode(NonTerminal('b'), 1, 1)
        b_sym.add_family(NonTerminal('b'), rule_b, 1, None, None)
        inter = StableSymbolNode((rule_s, 1), 0, 1)
        inter.add_family((rule_s, 1), rule_s, 0, None, a_sym)
        s_sym = StableSymbolNode(NonTerminal('s'), 0, 1)
        s_sym.add_family(NonTerminal('s'), rule_s, 0, a_sym, b_sym)
        s_sym.add_family(NonTerminal('s'), rule_s, 0, inter, b_sym)

        it = KBestIterator(s_sym, rule_cost=lambda rule: 0)
        trees = list(it)
        self.assertEqual(trees, [Tree('s', [Tree('a', ['x']), Tree('b', [])])])
        self.assertEqual(it.duplicates_skipped, 1)

    def test_unit_cycle_no_hang(self):
        # s: s | "a" generates infinitely many derivations of cost 0
        parser = Lark('start: s\ns: s | "a"\n', parser='earley')
        it = parser.parse_kbest('a', rule_cost=lambda rule: 0)
        trees = list(islice(it, 5))
        self.assertEqual(len(trees), 5)
        self.assertEqual(len({t for t in trees}), 5)

    def test_nullable_cycle_no_hang(self):
        # Infinitely ambiguous nullable grammar; zero-cost cycles must not
        # hang the iterator, and duplicate trees are still filtered out.
        parser = Lark('start: s\ns: s s | "a" |\n', parser='earley')
        it = parser.parse_kbest('a', rule_cost=lambda rule: 0)
        trees = list(islice(it, 10))
        self.assertEqual(len(trees), 10)
        self.assertEqual(len({t for t in trees}), 10)

    def test_cycle_with_costs_terminates(self):
        # Positive cost on the cyclic rule: cheapest trees come first,
        # and each next() still returns promptly.
        parser = Lark('start: s\ns: s -> wrap | "a" -> leaf\n', parser='earley')
        costs = {'wrap': 1, 'leaf': 0}
        rule_cost = lambda rule: costs.get(rule.alias or rule.origin.name, 0)
        it = parser.parse_kbest('a', rule_cost=rule_cost)
        trees = list(islice(it, 4))
        self.assertEqual([tree_cost(t, costs) for t in trees], [0, 1, 2, 3])

    def test_big_grammar_lazy(self):
        costs = {'add': 1, 'mul': 2}
        rule_cost = lambda rule: costs.get(rule.alias or rule.origin.name, 0)
        text = 'a+b*c+d*e+f*g+h+i*j+k'

        parser = Lark(EXPR_GRAMMAR, parser='earley')
        it = parser.parse_kbest(text, rule_cost=rule_cost, k=3)
        trees = list(it)
        self.assertEqual(len(trees), 3)
        self.assertEqual([tree_cost(t, costs) for t in trees], [14] * 3)

        # Laziness: only a few hundred derivation nodes were expanded,
        # out of 16796 possible derivations (Catalan(10)).
        self.assertLess(it.nodes_expanded, 1000)

        # Expanding everything costs orders of magnitude more
        it_full = parser.parse_kbest(text, rule_cost=rule_cost)
        total = sum(1 for _ in it_full)
        self.assertEqual(total, 16796)
        self.assertGreater(it_full.nodes_expanded, it.nodes_expanded * 100)

    def test_forest_root_interface(self):
        # KBestIterator also works directly on the SPPF of ambiguity='forest'
        parser = Lark(GROUPING_GRAMMAR, parser='earley', ambiguity='forest')
        forest = parser.parse('a b')
        it = KBestIterator(forest, rule_cost={'w': 1, 'ww': 4})
        trees = list(it)
        self.assertEqual(len(trees), 2)
        self.assertEqual(it.stats()['trees_yielded'], 2)

    def test_errors(self):
        parser = Lark(GROUPING_GRAMMAR, parser='earley')

        # Negative cost
        with self.assertRaises(ValueError):
            list(parser.parse_kbest('a b', rule_cost=lambda rule: -1))
        # NaN cost
        with self.assertRaises(ValueError):
            list(parser.parse_kbest('a b', rule_cost=lambda rule: float('nan')))
        # Bad rule_cost type
        with self.assertRaises(TypeError):
            parser.parse_kbest('a b', rule_cost=42)
        # Negative k
        with self.assertRaises(ValueError):
            parser.parse_kbest('a b', rule_cost=lambda rule: 0, k=-1)
        # Not an Earley parser
        lalr = Lark(GROUPING_GRAMMAR, parser='lalr')
        with self.assertRaises(ConfigurationError):
            lalr.parse_kbest('a b', rule_cost=lambda rule: 0)


if __name__ == '__main__':
    unittest.main()
