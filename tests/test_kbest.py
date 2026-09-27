"""Tests for the k-best Earley interface (Lark.parse_kbest / KBestForestIterator)."""

from unittest import TestCase, main
from itertools import product

from lark import Lark, Token, Tree
from lark.exceptions import ConfigurationError
from lark.parsers.earley_kbest import KBestForestIterator
from lark.parsers.earley_forest import ForestVisitor


class _ForestNodeCounter(ForestVisitor):
    """Counts the nodes of an SPPF (each node counted once)."""
    def __init__(self):
        super().__init__(single_visit=True)
        self.count = 0

    def visit_symbol_node_in(self, node):
        self.count += 1
        return node.children

    def visit_packed_node_in(self, node):
        self.count += 1
        return node.children

    def visit_token_node(self, token):
        self.count += 1


def _tree_key(value):
    if isinstance(value, Token):
        return (value.type, str(value), value.start_pos)
    return (value.data, tuple(_tree_key(c) for c in value.children))


def _expand_ambig(tree):
    """Expand an ambiguity='explicit' tree into the list of all trees it represents."""
    if isinstance(tree, Token):
        return [tree]
    if tree.data == '_ambig':
        return [t for child in tree.children for t in _expand_ambig(child)]
    expanded = [_expand_ambig(c) for c in tree.children]
    return [Tree(tree.data, list(children)) for children in product(*expanded)]


def _tree_cost(tree, costs):
    """Total rule cost of a fully expanded tree."""
    if isinstance(tree, Token):
        return 0
    return costs.get(tree.data, 0) + sum(_tree_cost(c, costs) for c in tree.children)


class TestKBest(TestCase):
    def test_small_grammar_full_ordering(self):
        """Enumerate all trees of a small grammar and check the cost ordering
        against a brute-force expansion of the explicit forest."""
        g = """
        start: seg
        seg: A seg   -> cons_a
           | A       -> one_a
           | AA seg  -> cons_aa
           | AA      -> one_aa
        A: "a"
        AA: "aa"
        """
        costs = {'cons_a': 1, 'one_a': 1, 'cons_aa': 1, 'one_aa': 1, 'seg': 0, 'start': 0}
        text = 'aaaaa'

        parser = Lark(g, parser='earley')
        it = parser.parse_kbest(text, rule_cost=costs)
        got = [(t, c) for t, c in it.iter_with_cost()]

        # costs are non-decreasing
        got_costs = [c for _, c in got]
        self.assertEqual(got_costs, sorted(got_costs))

        # compare with brute-force enumeration of all derivations
        explicit = Lark(g, parser='earley', ambiguity='explicit').parse(text)
        all_trees = _expand_ambig(explicit)
        expected = sorted((_tree_cost(t, costs), _tree_key(t)) for t in all_trees)
        self.assertEqual(got_costs, [c for c, _ in expected])
        # ties may be yielded in any stable order: compare as multisets
        self.assertEqual(
            sorted(_tree_key(t) for t, _ in got),
            sorted(k for _, k in expected),
        )

        # every yielded cost matches an independent recomputation
        for tree, cost in got:
            self.assertEqual(cost, _tree_cost(tree, costs))

    def test_ambiguous_expr(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]/
        """
        parser = Lark(g, parser='earley')
        trees = list(parser.parse_kbest('1+2+3'))
        explicit = Lark(g, parser='earley', ambiguity='explicit').parse('1+2+3')
        self.assertEqual(len(trees), 2)
        self.assertEqual(
            {_tree_key(t) for t in trees},
            {_tree_key(t) for t in _expand_ambig(explicit)},
        )

    def test_rule_cost_called_once_per_rule(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]/
        """
        calls = []

        def rule_cost(rule):
            calls.append(rule)
            return 1

        parser = Lark(g, parser='earley')
        list(parser.parse_kbest('1+2+3+4', rule_cost=rule_cost))
        self.assertTrue(calls)
        self.assertEqual(len(calls), len(set(calls)), "rule_cost called more than once per rule")

    def test_rule_cost_mapping_uses_alias(self):
        g = """
        start: seg
        seg: A seg   -> cons_a
           | A       -> one_a
           | AA seg  -> cons_aa
           | AA      -> one_aa
        A: "a"
        AA: "aa"
        """
        parser = Lark(g, parser='earley')
        # 'cons_a' etc. are aliases, not rule origin names
        cheap_aa = {'cons_a': 5, 'one_a': 5, 'cons_aa': 0, 'one_aa': 0}
        trees = list(parser.parse_kbest('aaaa', rule_cost=cheap_aa, k=1))
        self.assertEqual(len(trees), 1)
        # all-aa segmentation: start -> cons_aa(AA, cons_aa(AA, one_aa(AA)))
        self.assertEqual(trees[0].children[0].data, 'cons_aa')

    def test_negative_cost_rejected(self):
        parser = Lark('start: "a" | "a" "a"', parser='earley')
        self.assertRaises(ValueError, lambda: list(parser.parse_kbest('aa', rule_cost={'start': -1})))

    def test_nan_cost_rejected(self):
        parser = Lark('start: "a" | "a" "a"', parser='earley')
        self.assertRaises(ValueError, lambda: list(parser.parse_kbest('aa', rule_cost=lambda rule: float('nan'))))

    def test_bad_rule_cost_type_rejected(self):
        parser = Lark('start: "a"', parser='earley')
        self.assertRaises(TypeError, parser.parse_kbest, 'a', rule_cost=42)

    def test_negative_k_rejected(self):
        parser = Lark('start: "a"', parser='earley')
        self.assertRaises(ValueError, parser.parse_kbest, 'a', k=-1)

    def test_k_limits_output(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]/
        """
        parser = Lark(g, parser='earley')
        self.assertEqual(len(list(parser.parse_kbest('1+2+3+4+5', k=2))), 2)
        self.assertEqual(len(list(parser.parse_kbest('1+2+3+4+5', k=0))), 0)

    def test_duplicate_derivations_yielded_once(self):
        # The single "x" can be generated by any of the three 'a' positions,
        # and the SPPF represents these through different intermediate nodes.
        # With the anonymous token filtered out, all derivations look the same.
        g = 'start: a a a\na: "x"?\n'
        parser = Lark(g, parser='earley')
        it = parser.parse_kbest('x')
        trees = list(it)
        self.assertEqual(len(trees), 1)
        self.assertEqual(it.duplicates_skipped, 2)

        # With the token kept, the three positions are genuine distinct trees
        parser = Lark(g, parser='earley', keep_all_tokens=True)
        trees = list(parser.parse_kbest('x'))
        self.assertEqual(len(trees), 3)
        positions = []
        for t in trees:
            toks = list(t.scan_values(lambda v: isinstance(v, Token)))
            self.assertEqual(len(toks), 1)
            positions.append(toks[0].start_pos)
        self.assertEqual(positions, [0, 0, 0])  # same token, different tree positions

    def test_token_positions_preserved(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]+/
        """
        parser = Lark(g, parser='earley', propagate_positions=True)
        it = parser.parse_kbest('12+34', k=1)
        tree = next(iter(it))
        tokens = list(tree.scan_values(lambda v: isinstance(v, Token)))
        self.assertEqual([(t.value, t.start_pos, t.end_pos) for t in tokens],
                         [('12', 0, 2), ('+', 2, 3), ('34', 3, 5)])
        self.assertEqual((tree.meta.start_pos, tree.meta.end_pos), (0, 5))

    def test_nullable_cycle_terminates(self):
        # b derives itself: infinitely many derivations, all discarded
        g = 'start: b\nb: b | "x" | \n'
        parser = Lark(g, parser='earley')
        for text in ('x', ''):
            it = parser.parse_kbest(text, rule_cost={'b': 0, 'start': 0})
            trees = list(it)  # must terminate
            self.assertEqual(len(trees), 1)
            self.assertGreater(it.cycles_pruned, 0)

    def test_zero_cost_cycle_terminates(self):
        # A finite zero-cost cycle: b => b costs nothing
        g = 'start: b\nb: b | "x"\n'
        parser = Lark(g, parser='earley')
        it = parser.parse_kbest('x', rule_cost={'b': 0, 'start': 0})
        trees = list(it)
        self.assertEqual(len(trees), 1)
        self.assertGreater(it.cycles_pruned, 0)

    def test_recursive_nullable_cycle_terminates(self):
        g = 'start: s\ns: s s | "x" | \n'
        parser = Lark(g, parser='earley')
        it = parser.parse_kbest('x', rule_cost={'s': 0, 'start': 0})
        trees = list(it)
        self.assertTrue(trees)

    def test_stable_order_for_ties(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]/
        """
        # uniform cost: every derivation ties, order must be deterministic
        runs = []
        for _ in range(3):
            parser = Lark(g, parser='earley')
            trees = list(parser.parse_kbest('1+2+3+4', rule_cost=lambda rule: 1))
            runs.append([_tree_key(t) for t in trees])
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])
        # Catalan(3) == 5 trees
        self.assertEqual(len(runs[0]), 5)

    def test_lazy_expansion_stats(self):
        # The cheap derivation is a chain; the fancy one is a bushy O(n^2)
        # family. Taking only the best tree must not expand the fancy part.
        g = """
        start: cheap | fancy
        cheap: A cheap  -> cheap_cons
             | A        -> cheap_one
        fancy: fancy fancy -> fancy_cat
             | A           -> fancy_one
        A: "a"
        """
        n = 30
        text = 'a' * n
        costs = {'cheap_cons': 0, 'cheap_one': 0, 'fancy_cat': 10, 'fancy_one': 10}

        forest_root = Lark(g, parser='earley', ambiguity='forest').parse(text)
        counter = _ForestNodeCounter()
        counter.visit(forest_root)

        parser = Lark(g, parser='earley')
        it = parser.parse_kbest(text, rule_cost=costs, k=1)
        trees = list(it)
        self.assertEqual(len(trees), 1)
        self.assertEqual(it.last_cost, 0)
        stats = it.stats
        self.assertEqual(set(stats), {'nodes_expanded', 'candidates_popped',
                                      'cycles_pruned', 'duplicates_skipped'})
        # laziness: only a small part of the forest was expanded
        self.assertLess(stats['nodes_expanded'], counter.count // 3)

    def test_iter_with_cost_and_last_cost(self):
        parser = Lark('start: "a" "a" | "a"', parser='earley')
        it = parser.parse_kbest('a', rule_cost={'start': 3})
        [(tree, cost)] = list(it.iter_with_cost())
        self.assertEqual(cost, 3)
        self.assertEqual(it.last_cost, 3)

    def test_default_costs(self):
        # rule_cost=None: every rule costs 1, fewer rule applications first
        parser = Lark('start: "a" | "a" "a"', parser='earley')
        trees = list(parser.parse_kbest('a'))
        self.assertEqual(len(trees), 1)

    def test_basic_lexer(self):
        g = """
        start: expr
        expr: expr PLUS expr | NUM
        PLUS: "+"
        NUM: /[0-9]/
        """
        parser = Lark(g, parser='earley', lexer='basic')
        trees = list(parser.parse_kbest('1+2+3'))
        self.assertEqual(len(trees), 2)

    def test_dynamic_complete_lexer(self):
        g = """
        start: ab+
        ab: "a" "b"?
        """
        parser = Lark(g, parser='earley', lexer='dynamic_complete')
        trees = list(parser.parse_kbest('ab'))
        self.assertTrue(trees)

    def test_lalr_rejected(self):
        parser = Lark('start: "a"', parser='lalr')
        self.assertRaises(ConfigurationError, parser.parse_kbest, 'a')

    def test_forest_ambiguity_rejected(self):
        parser = Lark('start: "a"', parser='earley', ambiguity='forest')
        self.assertRaises(ConfigurationError, parser.parse_kbest, 'a')

    def test_iterator_protocol(self):
        parser = Lark('start: "a" | "b"', parser='earley')
        it = parser.parse_kbest('a')
        self.assertIsInstance(it, KBestForestIterator)
        self.assertIs(iter(it), it)
        tree = next(it)
        self.assertEqual(tree.data, 'start')
        self.assertRaises(StopIteration, next, it)


if __name__ == '__main__':
    main()
