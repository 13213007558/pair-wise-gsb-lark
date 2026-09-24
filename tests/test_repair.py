# -*- coding: utf-8 -*-
"""Tests for InteractiveParser.repair_search() and apply_repair() (LALR only)."""
from __future__ import absolute_import

import unittest

from lark import Lark, Tree
from lark.exceptions import UnexpectedToken
from lark.parsers.lalr_interactive_parser import RepairEdit
from lark.visitors import Transformer


LIST_GRAMMAR = r'''
    start: "[" num ("," num)* "]"
    ?num: SIGNED_NUMBER
    %import common.SIGNED_NUMBER
    %ignore " "
'''


def _Lark(grammar, **kwargs):
    kwargs.setdefault('parser', 'lalr')
    return Lark(grammar, **kwargs)


def tree_shape(tree):
    """Structure of a tree, comparing token types only (inserted tokens have no value)."""
    if isinstance(tree, Tree):
        return (tree.data, [tree_shape(c) for c in tree.children])
    return tree.type


def parse_error_ip(parser, text, **kwargs):
    """Parse text, return (UnexpectedToken, InteractiveParser) at the error point."""
    try:
        parser.parse(text, **kwargs)
    except UnexpectedToken as e:
        return e, e.interactive_parser
    raise AssertionError("Expected UnexpectedToken for %r" % text)


class TestRepairSearch(unittest.TestCase):
    def test_insert_missing_rpar_at_eof(self):
        # The missing-right-bracket case: the error is at EOF, the fix is an insertion
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0, 1, 2")
        self.assertEqual(e.token.type, '$END')

        res = ip.repair_search(first_token=e.token, k=1)
        self.assertTrue(res.optimal)
        self.assertEqual(len(res.plans), 1)
        plan = res.best
        self.assertEqual(plan.cost, 1)
        self.assertEqual(plan.edits, (RepairEdit('insert', 0, 'RSQB'),))
        self.assertEqual(plan.resume_index, 0)

        tree = ip.apply_repair(plan, res.tokens)
        self.assertEqual(tree, parser.parse("[0, 1, 2]"))

    def test_insert_missing_comma_mid_input(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1, 2]")
        self.assertEqual(e.token.type, 'SIGNED_NUMBER')

        res = ip.repair_search(first_token=e.token, k=3, delete_cost=2)
        self.assertTrue(res.optimal)
        plan = res.best
        self.assertEqual(plan.cost, 1)
        self.assertEqual(plan.edits, (RepairEdit('insert', 0, 'COMMA'),))
        self.assertEqual(plan.resume_index, 0)

        tree = ip.apply_repair(plan, res.tokens)
        self.assertEqual(tree, parser.parse("[0, 1, 2]"))

    def test_delete_extra_token(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0,, 1]")
        res = ip.repair_search(first_token=e.token, k=1, insert_cost=5)
        self.assertEqual(res.best.edits, (RepairEdit('delete', 0, 'COMMA'),))
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("[0, 1]"))

    def test_replace_token(self):
        parser = _Lark(r'''
            start: "a" "b" "c"
        ''')
        e, ip = parse_error_ip(parser, "aac")
        res = ip.repair_search(first_token=e.token, k=1,
                               insert_cost=5, delete_cost=5, replace_cost=1)
        self.assertEqual(res.best.edits, (RepairEdit('replace', 0, 'B', 'A'),))
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("abc"))

    def test_costs_determine_plan(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")

        # Deleting is cheaper than inserting a comma
        res = ip.repair_search(first_token=e.token, k=1, insert_cost=5, delete_cost=1)
        self.assertEqual(res.best.edits, (RepairEdit('delete', 0, 'SIGNED_NUMBER'),))

        # Inserting is cheaper than deleting
        res = ip.repair_search(first_token=e.token, k=1, insert_cost=1, delete_cost=5)
        self.assertEqual(res.best.edits, (RepairEdit('insert', 0, 'COMMA'),))

    def test_multiple_plans_ordered_by_cost(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")
        res = ip.repair_search(first_token=e.token, k=4)
        self.assertTrue(res.optimal)
        costs = [p.cost for p in res.plans]
        self.assertEqual(costs, sorted(costs))
        # The only non-dominated plans cost 1: insert COMMA, or delete the "1"
        self.assertEqual(costs, [1, 1])
        kinds = {p.edits[0].kind for p in res.plans}
        self.assertEqual(kinds, {'insert', 'delete'})

    def test_multiple_plans_two_errors(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1 2]")
        res = ip.repair_search(first_token=e.token, k=3)
        self.assertTrue(res.optimal)
        costs = [p.cost for p in res.plans]
        self.assertEqual(costs, sorted(costs))
        # Cheapest: replacing "1" with a comma yields the valid "[0, 2]"
        self.assertEqual(costs[0], 1)
        self.assertEqual(res.best.edits, (RepairEdit('replace', 0, 'COMMA', 'SIGNED_NUMBER'),))
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("[0, 2]"))
        # The runner-up deletes both stray numbers
        self.assertEqual(res.plans[1].edits,
                         (RepairEdit('delete', 0, 'SIGNED_NUMBER'), RepairEdit('delete', 1, 'SIGNED_NUMBER')))

    def test_budget_exhausted_marks_not_optimal(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")
        res = ip.repair_search(first_token=e.token, k=1, budget=0)
        self.assertFalse(res.optimal)
        self.assertEqual(res.plans, [])
        self.assertEqual(res.expanded, 0)

    def test_zero_cost_cycles_terminate(self):
        # Zero-cost inserts of "a" can loop forever growing the stack;
        # the search must still terminate (via dedup + budget) and find the plan.
        parser = _Lark(r'''
            start: A* B
            A: "a"
            B: "b"
        ''')
        e, ip = parse_error_ip(parser, "aaa")
        self.assertEqual(e.token.type, '$END')
        res = ip.repair_search(first_token=e.token, k=1,
                               insert_cost=0, delete_cost=0, replace_cost=0, budget=1000)
        self.assertEqual(res.best.cost, 0)
        self.assertEqual(res.best.edits, (RepairEdit('insert', 0, 'B'),))
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree_shape(tree), tree_shape(parser.parse("aaab")))

    def test_empty_production(self):
        parser = _Lark(r'''
            start: "a" maybe_b "c"
            maybe_b: "b" |
        ''')
        # Missing "c" at EOF, past an empty reduction
        e, ip = parse_error_ip(parser, "ab")
        res = ip.repair_search(first_token=e.token, k=1)
        self.assertEqual(res.best.cost, 1)
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("abc"))

        # The empty alternative is taken: "a" then EOF
        e, ip = parse_error_ip(parser, "a")
        res = ip.repair_search(first_token=e.token, k=1)
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("ac"))

    def test_multiple_start_rules(self):
        parser = _Lark(r'''
            s1: "a" "b"
            s2: "x" "y"
        ''', start=['s1', 's2'])

        e, ip = parse_error_ip(parser, "x", start='s2')
        res = ip.repair_search(first_token=e.token, k=1)
        self.assertTrue(res.optimal)
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("xy", start='s2'))

        e, ip = parse_error_ip(parser, "a", start='s1')
        res = ip.repair_search(first_token=e.token, k=1)
        tree = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(tree, parser.parse("ab", start='s1'))

    def test_candidates_restrict_edits(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")
        res = ip.repair_search(first_token=e.token, k=2, candidates={'COMMA'}, delete_cost=5)
        # Only COMMA may be inserted or used as a replacement
        for plan in res.plans:
            for edit in plan.edits:
                if edit.kind != 'delete':
                    self.assertEqual(edit.token_type, 'COMMA')
        self.assertEqual(res.best.edits, (RepairEdit('insert', 0, 'COMMA'),))

    def test_no_plan_exists(self):
        parser = _Lark(r'start: "a" "b"?')
        e, ip = parse_error_ip(parser, "b")
        # No candidates to insert, and deleting "b" leaves an invalid empty input
        res = ip.repair_search(first_token=e.token, candidates=set())
        self.assertEqual(res.plans, [])
        self.assertTrue(res.optimal)  # space exhausted: proven, there is no plan

    def test_invalid_arguments(self):
        parser = _Lark(LIST_GRAMMAR)
        ip = parser.parse_interactive("[0")
        self.assertRaises(ValueError, ip.repair_search, insert_cost=-1)
        self.assertRaises(ValueError, ip.repair_search, k=0)
        self.assertRaises(ValueError, ip.repair_search, budget=-1)

    def test_search_does_not_run_callbacks(self):
        # A transformer that counts every callback invocation
        class Counter(Transformer):
            def __init__(self):
                self.count = 0

            def __default__(self, data, children, meta):
                self.count += 1
                return Tree(data, children)

        grammar = r'''
            start: "[" item ("," item)* "]"
            item: SIGNED_NUMBER
            %import common.SIGNED_NUMBER
            %ignore " "
        '''
        counter = Counter()
        parser = _Lark(grammar, transformer=counter)

        # Baseline: callbacks fired by a normal parse of the corrected input
        normal = parser.parse("[0, 1, 2]")
        normal_count = counter.count

        counter.count = 0
        e, ip = parse_error_ip(parser, "[0 1, 2]")
        count_after_error = counter.count
        res = ip.repair_search(first_token=e.token, k=3, delete_cost=2)
        # The search itself must not fire any callback, not even for unchosen plans
        self.assertEqual(counter.count, count_after_error)

        # Applying the chosen plan fires exactly the remaining callbacks,
        # so the total matches a normal parse of the corrected input
        result = ip.apply_repair(res.best, res.tokens)
        self.assertEqual(counter.count, normal_count)
        self.assertEqual(result, normal)

    def test_search_does_not_mutate_parser(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")
        before = ip.copy()
        ip.repair_search(first_token=e.token, k=2)
        self.assertEqual(ip.parser_state, before.parser_state)
        self.assertEqual(ip.lexer_thread.state, before.lexer_thread.state)

    def test_resume_index(self):
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0, 1,, 2]")
        res = ip.repair_search(first_token=e.token, k=1, insert_cost=5)
        plan = res.best
        self.assertEqual(plan.edits, (RepairEdit('delete', 0, 'COMMA'),))
        self.assertEqual(plan.resume_index, 1)
        tree = ip.apply_repair(plan, res.tokens)
        self.assertEqual(tree, parser.parse("[0, 1, 2]"))

    def test_immutable_interactive_parser_search(self):
        from lark.parsers.lalr_interactive_parser import ImmutableInteractiveParser
        parser = _Lark(LIST_GRAMMAR)
        e, ip = parse_error_ip(parser, "[0 1]")
        imm = ip.as_immutable()
        self.assertIsInstance(imm, ImmutableInteractiveParser)
        res = imm.repair_search(first_token=e.token, k=1, delete_cost=2)
        self.assertEqual(res.best.edits, (RepairEdit('insert', 0, 'COMMA'),))


if __name__ == '__main__':
    unittest.main()
