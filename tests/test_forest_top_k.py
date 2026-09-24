from __future__ import absolute_import

import unittest
from itertools import product

from lark import Lark
from lark.tree import Tree
from lark.parsers.earley_forest import (
    ForestTopKIterator, InfiniteAmbiguityError, SymbolNode, TokenNode,
)


def forest_rule_rank(root):
    """Rank every rule in the forest by its stable identity: the origin
    name and the order amongst same-named rules."""
    rules = {}
    seen = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for packed in node.children:
            rules.setdefault(id(packed.rule), packed.rule)
            for child in packed.children:
                if isinstance(child, SymbolNode):
                    stack.append(child)
    ordered = sorted(rules.values(), key=lambda r: (r.origin.name, r.order))
    return {id(rule): rank for rank, rule in enumerate(ordered)}


def brute_force_top(root, cost_fn):
    """Reference implementation: fully enumerate all derivations of an
    (acyclic) forest and sort them by (cost, shape).

    Returns a list of (cost, tree), cheapest first.
    """
    rule_rank = forest_rule_rank(root)

    def enum(node):
        derivations = []
        for branch, packed in enumerate(node.children):
            weight = 0 if node.is_intermediate else cost_fn(packed.rule)
            child_lists = []
            for child in packed.children:
                if isinstance(child, TokenNode):
                    child_lists.append([(0, (), child.token)])
                else:
                    child_lists.append(enum(child))
            for combo in product(*child_lists):
                cost = weight + sum(item[0] for item in combo)
                shape = (branch, rule_rank[id(packed.rule)])
                shape += tuple(s for item in combo for s in item[1])
                children = []
                for item in combo:
                    if isinstance(item[2], list):
                        children.extend(item[2])
                    else:
                        children.append(item[2])
                if node.is_intermediate:
                    tree = children
                else:
                    tree = Tree(packed.rule.origin.name, children)
                derivations.append((cost, shape, tree))
        derivations.sort(key=lambda d: (d[0], d[1]))
        return derivations

    return [(cost, tree) for cost, shape, tree in enum(root)]


def parse_forest(grammar, text):
    return Lark(grammar, parser='earley', ambiguity='forest').parse(text)


class TestForestTopK(unittest.TestCase):

    def test_full_enumeration_small_grammar(self):
        grammar = """
        start: expr
        expr: expr "+" expr | term
        term: "a" | "b"
        """
        forest = parse_forest(grammar, "a+b+a+b")
        cost_fn = lambda rule: 2 if str(rule) == '<expr : expr + expr>' else 1
        expected = brute_force_top(forest, cost_fn)
        self.assertEqual(len(expected), 5)

        iterator = ForestTopKIterator(forest, rule_cost=cost_fn)
        self.assertEqual(iterator.expansions, 0)
        got = list(iterator.with_costs())
        self.assertEqual([cost for tree, cost in got], [cost for cost, tree in expected])
        self.assertEqual([tree for tree, cost in got], [tree for cost, tree in expected])

        # costs must be non-decreasing
        costs = [cost for tree, cost in got]
        self.assertEqual(costs, sorted(costs))

        # enumeration is deterministic across independent parses
        forest2 = parse_forest(grammar, "a+b+a+b")
        got2 = list(ForestTopKIterator(forest2, rule_cost=cost_fn).with_costs())
        self.assertEqual([tree for tree, cost in got], [tree for tree, cost in got2])

    def test_nullable_rules(self):
        grammar = 'start: a a a\na: "x"? "y"?'
        forest = parse_forest(grammar, "xy")
        expected = brute_force_top(forest, lambda rule: 1)
        self.assertEqual(len(expected), 6)
        got = list(ForestTopKIterator(forest).with_costs())
        self.assertEqual([tree for tree, cost in got], [tree for cost, tree in expected])

    def test_shared_subforest_costs(self):
        # x(0,1) and x(1,2) are shared by the four derivations of p; each use
        # must be charged separately.
        grammar = """
        start: p
        p: x x
        x: "a" | "a" e
        e:
        """
        forest = parse_forest(grammar, "aa")
        costs = {'<start : p>': 1, '<p : x x>': 1, '<x : A>': 0, '<x : A e>': 3, '<e : >': 0}
        cost_fn = lambda rule: costs[str(rule)]
        got = list(ForestTopKIterator(forest, rule_cost=cost_fn).with_costs())
        self.assertEqual([cost for tree, cost in got], [2, 5, 5, 8])

        expected = brute_force_top(forest, cost_fn)
        self.assertEqual([tree for tree, cost in got], [tree for cost, tree in expected])

    def test_dedup_is_configurable(self):
        # Two derivations of the same text: kept by default, and kept by
        # structural dedup (the trees differ); collapsed by a custom key.
        grammar = 'start: ab ab\nab: "a"?'
        forest = parse_forest(grammar, "a")

        trees = list(ForestTopKIterator(forest))
        self.assertEqual(len(trees), 2)
        self.assertNotEqual(trees[0], trees[1])

        trees = list(ForestTopKIterator(forest, dedup=True))
        self.assertEqual(len(trees), 2)

        key = lambda tree: tuple(child.data for child in tree.children)
        trees = list(ForestTopKIterator(forest, dedup=key))
        self.assertEqual(len(trees), 1)

    def test_zero_cost_cycle_raises(self):
        grammar = """
        start: s
        s: s | "x"
        """
        forest = parse_forest(grammar, "x")
        self.assertRaises(InfiniteAmbiguityError, ForestTopKIterator, forest, rule_cost=lambda rule: 0)
        # a zero-cost cycle exists as soon as the cycling rule is free
        cost_fn = lambda rule: 0 if str(rule) == '<s : s>' else 1
        self.assertRaises(InfiniteAmbiguityError, ForestTopKIterator, forest, rule_cost=cost_fn)

    def test_zero_cost_cycle_nullable_raises(self):
        grammar = """
        start: s
        s: s e | "x"
        e:
        """
        forest = parse_forest(grammar, "x")
        self.assertRaises(InfiniteAmbiguityError, ForestTopKIterator, forest, rule_cost=lambda rule: 0)

    def test_positive_cost_cycle(self):
        grammar = """
        start: s
        s: s | "x"
        """
        forest = parse_forest(grammar, "x")
        cost_fn = lambda rule: 1 if str(rule) == '<s : s>' else 0
        got = list(ForestTopKIterator(forest, rule_cost=cost_fn, k=5).with_costs())
        self.assertEqual([cost for tree, cost in got], [0, 1, 2, 3, 4])
        self.assertEqual(len({tree.pretty() for tree, cost in got}), 5)
        for i, (tree, cost) in enumerate(got):
            self.assertEqual(sum(1 for _ in tree.iter_subtrees()), i + 2)

    def test_lazy_expansion_count(self):
        grammar = """
        start: s
        s: s s s | s s | "a"
        """
        forest = parse_forest(grammar, "a" * 8)
        expected = brute_force_top(forest, lambda rule: 1)
        self.assertEqual(len(expected), 2871)

        iterator = ForestTopKIterator(forest, k=3)
        self.assertEqual(iterator.expansions, 0)
        got = list(iterator.with_costs())
        self.assertEqual(len(got), 3)
        self.assertEqual([tree for tree, cost in got], [tree for cost, tree in expected[:3]])
        # only a small fraction of the represented trees is ever expanded
        self.assertLess(iterator.expansions, len(expected) // 5)

    def test_k_limits(self):
        forest = parse_forest('start: "a" "b"', "ab")
        self.assertEqual(len(list(ForestTopKIterator(forest, k=10))), 1)
        self.assertEqual(len(list(ForestTopKIterator(forest, k=0))), 0)
        self.assertRaises(ValueError, ForestTopKIterator, forest, k=-1)

    def test_cost_validation(self):
        forest = parse_forest('start: "a"', "a")
        self.assertRaises(ValueError, ForestTopKIterator, forest, rule_cost=lambda rule: -1)
        self.assertRaises(TypeError, ForestTopKIterator, "not a forest")

    def test_dict_costs(self):
        grammar = """
        start: s
        s: s s | "a"
        """
        forest = parse_forest(grammar, "aaa")
        # rules missing from the dict default to a cost of 1
        got = list(ForestTopKIterator(forest, rule_cost={}).with_costs())
        self.assertEqual([cost for tree, cost in got], [6, 6])

    def test_tie_break_is_stable(self):
        # equal-cost derivations are ordered by branch index and rule
        # identity, deterministically and without relying on object addresses
        grammar = """
        start: s
        s: t | u
        t: "a"
        u: "a"
        """
        forest = parse_forest(grammar, "a")
        got = list(ForestTopKIterator(forest))
        self.assertEqual(got, [
            Tree('start', [Tree('s', [Tree('t', ['a'])])]),
            Tree('start', [Tree('s', [Tree('u', ['a'])])]),
        ])
        forest2 = parse_forest(grammar, "a")
        got2 = list(ForestTopKIterator(forest2))
        self.assertEqual(got, got2)


if __name__ == '__main__':
    unittest.main()
