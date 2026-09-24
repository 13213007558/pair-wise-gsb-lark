from unittest import TestCase

from lark import Lark, Transformer, UnexpectedToken
from lark.tree import Tree
from lark.repair import find_repairs, Insert, Delete, Replace


PARENS_GRAMMAR = r"""
    start: LPAR item RPAR
    item: CNAME
    LPAR: "("
    RPAR: ")"
    RBRACK: "]"
    %import common.CNAME
    %import common.WS
    %ignore WS
    %declare RBRACK
"""


def shape(node):
    """Tree/token structure, ignoring the values of (possibly fabricated) tokens."""
    if isinstance(node, Tree):
        return node.data, [shape(child) for child in node.children]
    return node.type


def parse_error(parser, text, start=None):
    try:
        parser.parse(text, start=start)
    except UnexpectedToken as e:
        return e
    raise AssertionError("Expected UnexpectedToken for input: %r" % text)


class TestRepair(TestCase):
    def test_missing_closer_at_eof(self):
        parser = Lark(PARENS_GRAMMAR, parser='lalr')
        e = parse_error(parser, "(foo")
        self.assertEqual(e.token.type, '$END')

        result = find_repairs(e)
        self.assertTrue(result.proven_optimal)
        self.assertEqual(result.tokens, [])
        self.assertEqual(len(result.plans), 1)

        plan = result.plans[0]
        self.assertEqual(plan.edits, (Insert(position=0, token_type='RPAR'),))
        self.assertEqual(plan.cost, 1)
        self.assertEqual(plan.resume_index, 0)
        self.assertEqual(shape(result.apply(plan)), shape(parser.parse("(foo)")))

    def test_missing_closer_from_interactive_parser(self):
        # find_repairs also accepts an InteractiveParser positioned at the error
        parser = Lark(PARENS_GRAMMAR, parser='lalr')
        ip = parser.parse_interactive("(foo")
        ip.exhaust_lexer()

        result = find_repairs(ip)
        self.assertTrue(result.proven_optimal)
        plan = result.plans[0]
        self.assertEqual(plan.edits, (Insert(position=0, token_type='RPAR'),))
        self.assertEqual(shape(result.apply(plan)), shape(parser.parse("(foo)")))

    def test_delete_extra_token(self):
        parser = Lark("start: CNAME CNAME\n%import common.CNAME\n%import common.WS\n%ignore WS\n", parser='lalr')
        e = parse_error(parser, "a b c")

        result = find_repairs(e)
        self.assertTrue(result.proven_optimal)
        self.assertEqual(len(result.plans), 1)

        plan = result.plans[0]
        self.assertEqual(plan.edits, (Delete(position=0, token=e.token),))
        self.assertEqual(plan.cost, 1)
        self.assertEqual(plan.resume_index, 1)
        self.assertEqual(result.apply(plan), parser.parse("a b"))

    def test_replace_is_cheaper_than_delete_insert(self):
        parser = Lark(PARENS_GRAMMAR, parser='lalr', lexer='basic')
        e = parse_error(parser, "(foo]")

        result = find_repairs(e, max_plans=3)
        self.assertTrue(result.proven_optimal)
        costs = [plan.cost for plan in result.plans]
        self.assertEqual(costs, sorted(costs))

        plan = result.plans[0]
        self.assertEqual(plan.edits, (Replace(position=0, token_type='RPAR', token=e.token),))
        self.assertEqual(plan.cost, 1)
        self.assertEqual(plan.resume_index, 1)
        self.assertEqual(shape(result.apply(plan)), shape(parser.parse("(foo)")))
        # The next cheapest repair is delete + insert
        self.assertEqual(result.plans[1].cost, 2)

    def test_plans_are_verified_against_remaining_input(self):
        # Inserting B at the error position is accepted locally, but only deleting
        # or replacing the bad token lets the rest of the input parse.
        grammar = '\n'.join([
            'start: A B C',
            'A: "a"',
            'B: "b"',
            'C: "c"',
            'X: "x"',
            '%declare X',
            '%import common.WS',
            '%ignore WS',
        ])
        parser = Lark(grammar, parser='lalr', lexer='basic')
        e = parse_error(parser, "a x c")
        self.assertEqual(e.interactive_parser.accepts(), {"B"})

        result = find_repairs(e, max_plans=5)
        self.assertTrue(result.proven_optimal)
        best = result.plans[0]
        self.assertEqual(best.cost, 1)
        self.assertIsInstance(best.edits[0], (Delete, Replace))
        # A plan that inserts B still has to deal with the 'x' afterwards
        insert_plans = [p for p in result.plans if isinstance(p.edits[0], Insert)]
        self.assertTrue(insert_plans)
        self.assertTrue(all(p.cost >= 2 for p in insert_plans))
        self.assertEqual(shape(result.apply(best)), shape(parser.parse("a b c")))

    def test_custom_costs_and_candidates(self):
        parser = Lark(PARENS_GRAMMAR, parser='lalr', lexer='basic')
        e = parse_error(parser, "(foo]")

        # With replacement made expensive, delete+insert wins
        result = find_repairs(e, candidates=['RPAR'], replace_cost=5, max_plans=2)
        self.assertEqual(result.plans[0].cost, 2)
        self.assertEqual([type(edit) for edit in result.plans[0].edits], [Delete, Insert])

        # Disabling deletion (infinite cost) leaves replacement as the cheapest
        result = find_repairs(e, candidates=['RPAR'], delete_cost=float('inf'), max_plans=1)
        self.assertEqual(result.plans[0].edits, (Replace(position=0, token_type='RPAR', token=e.token),))

    def test_budget_exhaustion_is_reported(self):
        parser = Lark(PARENS_GRAMMAR, parser='lalr', lexer='basic')
        e = parse_error(parser, "(foo]")

        result = find_repairs(e, budget=0)
        self.assertFalse(result.proven_optimal)
        self.assertEqual(result.plans, [])

        result = find_repairs(e, budget=1)
        self.assertFalse(result.proven_optimal)

    def test_zero_cost_cycle_terminates(self):
        # 'cs' is nullable and recursive: inserting C at zero cost cycles back to
        # the same parser state. The search must terminate anyway.
        grammar = '\n'.join([
            'start: A cs B',
            'cs: cs C |',
            'A: "a"',
            'B: "b"',
            'C: "c"',
        ])
        parser = Lark(grammar, parser='lalr')
        e = parse_error(parser, "a")

        result = find_repairs(e, insert_cost=0)
        self.assertTrue(result.proven_optimal)
        plan = result.plans[0]
        self.assertEqual(plan.cost, 0)
        self.assertEqual(plan.edits, (Insert(position=0, token_type='B'),))
        self.assertEqual(shape(result.apply(plan)), shape(parser.parse("ab")))

    def test_empty_production(self):
        grammar = '\n'.join([
            'start: A maybe B',
            'maybe: C |',
            'A: "a"',
            'B: "b"',
            'C: "c"',
        ])
        parser = Lark(grammar, parser='lalr')
        e = parse_error(parser, "a")

        result = find_repairs(e)
        self.assertTrue(result.proven_optimal)
        plan = result.plans[0]
        self.assertEqual(plan.edits, (Insert(position=0, token_type='B'),))
        self.assertEqual(shape(result.apply(plan)), shape(parser.parse("ab")))

    def test_multiple_start_rules(self):
        grammar = '\n'.join([
            'rule_a: A B',
            'rule_b: A C',
            'A: "a"',
            'B: "b"',
            'C: "c"',
        ])
        parser = Lark(grammar, parser='lalr', start=['rule_a', 'rule_b'])
        e = parse_error(parser, "a", start='rule_b')

        result = find_repairs(e)
        self.assertTrue(result.proven_optimal)
        plan = result.plans[0]
        self.assertEqual(plan.edits, (Insert(position=0, token_type='C'),))
        tree = result.apply(plan)
        self.assertEqual(tree.data, 'rule_b')
        self.assertEqual(shape(tree), shape(parser.parse("ac", start='rule_b')))

    def test_search_does_not_run_transformer_callbacks(self):
        class Counter(Transformer):
            def __init__(self):
                self.count = 0

            def item(self, children):
                self.count += 1
                return children[0]

            def CNAME(self, token):
                self.count += 1
                return str(token)

        clean_counter = Counter()
        clean_parser = Lark(PARENS_GRAMMAR, parser='lalr', transformer=clean_counter)
        clean_parser.parse("(foo)")
        normal_count = clean_counter.count
        self.assertGreater(normal_count, 0)

        counter = Counter()
        parser = Lark(PARENS_GRAMMAR, parser='lalr', transformer=counter)
        e = parse_error(parser, "(foo")
        count_after_error = counter.count

        # The search itself must not fire any callbacks, for any candidate plan
        result = find_repairs(e)
        self.assertEqual(counter.count, count_after_error)

        # Applying the chosen plan fires exactly the remaining callbacks,
        # so the total matches a normal parse of the repaired input
        result.apply(result.plans[0])
        self.assertEqual(counter.count, normal_count)

    def test_invalid_arguments(self):
        parser = Lark(PARENS_GRAMMAR, parser='lalr')
        e = parse_error(parser, "(foo")
        self.assertRaises(ValueError, find_repairs, e, insert_cost=-1)
        self.assertRaises(ValueError, find_repairs, e, max_plans=0)
        self.assertRaises(TypeError, find_repairs, "not a parser")


if __name__ == '__main__':
    import unittest
    unittest.main()
