from __future__ import absolute_import

import unittest
from collections import Counter

from lark import Lark
from lark.exceptions import InfiniteAmbiguityError
from lark.parsers.earley_forest import TokenNode
from lark.parsers.earley_topk import TopKForestEnumerator, ScoredTree
from lark.tree import Tree


def uniform_cost(rule):
    return 1


def brute_force_enumerate(root, rule_cost=uniform_cost):
    """Reference implementation: eagerly enumerates every derivation of a
    small, acyclic forest. Returns a list of (cost, tree)."""
    memo = {}

    def enum_symbol(node):
        if id(node) in memo:
            return memo[id(node)]
        result = []
        for packed in node.children:
            for cost, children in enum_packed(packed):
                if node.is_intermediate:
                    result.append((cost, children))
                else:
                    rule = packed.rule
                    name = rule.alias or rule.options.template_source or rule.origin.name
                    result.append((cost, Tree(name, children)))
        memo[id(node)] = result
        return result

    def enum_packed(packed):
        weight = 0 if packed.parent.is_intermediate else rule_cost(packed.rule)
        combos = [(weight, [])]
        for child in (packed.left, packed.right):
            if child is None:
                continue
            if isinstance(child, TokenNode):
                combos = [(c, ch + [child.token]) for c, ch in combos]
            elif child.is_intermediate:
                combos = [(c1 + c2, ch1 + ch2) for c1, ch1 in combos for c2, ch2 in enum_symbol(child)]
            else:
                combos = [(c1 + c2, ch1 + [tree]) for c1, ch1 in combos for c2, tree in enum_symbol(child)]
        return combos

    return enum_symbol(root)


class TestTopKForestEnumerator(unittest.TestCase):

    def parse_forest(self, grammar, text):
        return Lark(grammar, parser='earley', ambiguity='forest').parse(text)

    def test_full_enumeration_matches_brute_force(self):
        grammar = """
        start: e
        e: e "+" e | e "*" e | "a" | "b"
        """
        forest = self.parse_forest(grammar, "a+a*b+a")
        lazy = list(TopKForestEnumerator(forest))
        reference = brute_force_enumerate(forest)
        # Same derivations with the same costs, multiplicity included.
        self.assertEqual(Counter(lazy), Counter(reference))
        self.assertEqual(len(lazy), 5)
        # Emitted in non-decreasing cost order.
        costs = [cost for cost, tree in lazy]
        self.assertEqual(costs, sorted(costs))
        # Every emitted item is a ScoredTree(cost, tree) pair.
        for scored in lazy:
            self.assertIsInstance(scored, ScoredTree)
            self.assertIsInstance(scored.tree, Tree)

    def test_take_is_prefix_of_full_enumeration(self):
        grammar = """
        start: e
        e: e "+" e | e "*" e | "a" | "b"
        """
        forest = self.parse_forest(grammar, "a+a*b+a")
        full = list(TopKForestEnumerator(forest))
        taken = TopKForestEnumerator(forest).take(3)
        self.assertEqual(taken, full[:3])
        # Asking for more trees than exist yields whatever is available.
        self.assertEqual(TopKForestEnumerator(forest).take(100), full)
        self.assertEqual(TopKForestEnumerator(forest).take(0), [])

    def test_weighted_costs_change_order(self):
        grammar = """
        start: s
        s: x | y
        x: "a" "b"
        y: x
        """
        forest = self.parse_forest(grammar, "ab")
        # Uniform cost: the shorter derivation (fewer rule applications) wins.
        result = list(TopKForestEnumerator(forest))
        self.assertEqual([cost for cost, tree in result], [3, 4])
        derivations = [tree.children[0].children[0].data for cost, tree in result]
        self.assertEqual(derivations, ['x', 'y'])
        # Weighting rules changes the totals but not the correctness of the order.
        # Both derivations apply the x rule once, so both pay for it.
        result = list(TopKForestEnumerator(forest, rule_cost={'x': 10}))
        self.assertEqual([cost for cost, tree in result], [10, 10])
        self.assertEqual([tree.children[0].children[0].data for cost, tree in result], ['x', 'y'])
        # Only the longer derivation applies the y rule.
        result = list(TopKForestEnumerator(forest, rule_cost={'y': 10}))
        self.assertEqual([cost for cost, tree in result], [0, 10])
        self.assertEqual([tree.children[0].children[0].data for cost, tree in result], ['x', 'y'])

    def test_tie_break_uses_stable_rule_identity(self):
        grammar = """
        start: a | b
        a: "x"
        b: "x"
        """
        forest = self.parse_forest(grammar, "x")
        # Equal costs: the rule defined first in the grammar wins.
        result = list(TopKForestEnumerator(forest))
        self.assertEqual([cost for cost, tree in result], [2, 2])
        self.assertEqual([tree.children[0].data for cost, tree in result], ['a', 'b'])
        # Costs dominate the tie-break.
        result = list(TopKForestEnumerator(forest, rule_cost={'a': 5}))
        self.assertEqual([cost for cost, tree in result], [0, 5])
        self.assertEqual([tree.children[0].data for cost, tree in result], ['b', 'a'])

    def test_order_is_deterministic_across_instances(self):
        grammar = """
        start: e
        e: e "+" e | e "*" e | "a" | "b"
        """
        forest1 = self.parse_forest(grammar, "a+a*b+a")
        forest2 = self.parse_forest(grammar, "a+a*b+a")
        result1 = [(cost, tree.pretty()) for cost, tree in TopKForestEnumerator(forest1)]
        result2 = [(cost, tree.pretty()) for cost, tree in TopKForestEnumerator(forest2)]
        self.assertEqual(result1, result2)

    def test_shared_subforest_counts_each_use(self):
        # t(0,2) and t(2,4) are each shared by two of the four complete
        # derivations, but every tree must pay for its own rule applications.
        grammar = """
        start: t t
        t: "aa" | "a" "a"
        """
        forest = self.parse_forest(grammar, "aaaa")
        result = list(TopKForestEnumerator(forest))
        self.assertEqual(len(result), 4)
        self.assertEqual([cost for cost, tree in result], [3, 3, 3, 3])
        reference = brute_force_enumerate(forest)
        self.assertEqual(Counter(result), Counter(reference))

    def test_dedup_is_configurable(self):
        grammar = """
        start: t t
        t: "aa" | "a" "a"
        """
        forest = self.parse_forest(grammar, "aaaa")

        class DataOnlyTree(Tree):
            def __eq__(self, other):
                return isinstance(other, DataOnlyTree) and self.data == other.data

            def __hash__(self):
                return hash(self.data)

        # By default every derivation is kept, even when two derivations
        # produce the same tree.
        kept = list(TopKForestEnumerator(forest, tree_class=DataOnlyTree))
        self.assertEqual(len(kept), 4)
        # With dedup=True, duplicate trees are emitted only once.
        deduped = list(TopKForestEnumerator(forest, tree_class=DataOnlyTree, dedup=True))
        self.assertEqual(len(deduped), 1)
        # Costs stay in order even when duplicates are dropped.
        self.assertEqual(deduped[0].cost, 3)

    def test_zero_cost_cycle_is_reported(self):
        grammar = """
        start: s
        s: s s | "a" |
        """
        forest = self.parse_forest(grammar, "a")
        # With zero-cost rules the nullable recursion yields infinitely many
        # derivations of the same minimal cost: no complete ordering exists.
        with self.assertRaises(InfiniteAmbiguityError):
            TopKForestEnumerator(forest, rule_cost=lambda rule: 0)
        with self.assertRaises(InfiniteAmbiguityError):
            forest.iter_topk(rule_cost=lambda rule: 0)

    def test_positive_cost_cycle_enumerates_lazily(self):
        grammar = """
        start: s
        s: s s | "a" |
        """
        forest = self.parse_forest(grammar, "a")
        # The forest is cyclic, but every cycle has a positive cost, so the
        # cheapest derivations are well-defined and enumeration terminates.
        enumerator = TopKForestEnumerator(forest)
        result = enumerator.take(5)
        self.assertEqual([cost for cost, tree in result], [2, 4, 4, 6, 6])

    def test_only_needed_combinations_are_expanded(self):
        grammar = """
        start: e
        e: e "+" e | "a"
        """
        forest = self.parse_forest(grammar, "a+a+a+a+a+a")
        full = TopKForestEnumerator(forest)
        all_trees = list(full)
        self.assertEqual(len(all_trees), 42)

        top3 = TopKForestEnumerator(forest)
        three = top3.take(3)
        self.assertEqual(three, all_trees[:3])
        # Taking 3 trees out of 42 expands only a fraction of the combinations.
        self.assertLess(top3.expanded * 5, full.expanded)
        self.assertLess(top3.generated * 5, full.generated)

    def test_iter_topk_convenience_method(self):
        grammar = """
        start: a | b
        a: "x"
        b: "x"
        """
        forest = self.parse_forest(grammar, "x")
        taken = forest.iter_topk(2)
        self.assertEqual(len(taken), 2)
        self.assertEqual([tree.children[0].data for cost, tree in taken], ['a', 'b'])
        enumerator = forest.iter_topk()
        self.assertIsInstance(enumerator, TopKForestEnumerator)
        self.assertEqual(len(list(enumerator)), 2)

    def test_negative_cost_is_rejected(self):
        grammar = """
        start: a
        a: "x"
        """
        forest = self.parse_forest(grammar, "x")
        with self.assertRaises(ValueError):
            TopKForestEnumerator(forest, rule_cost=lambda rule: -1)


if __name__ == '__main__':
    unittest.main()
