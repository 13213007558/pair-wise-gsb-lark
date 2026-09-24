# -*- coding: utf-8 -*-
"""Tests for lark.incremental.IncrementalParseSession."""
from __future__ import absolute_import

import unittest

from lark import Lark, Token, Tree
from lark.exceptions import ConfigurationError, UnexpectedToken
from lark.incremental import IncrementalParseSession
from lark.lexer import Lexer, LexerThread
from lark.visitors import Transformer

GRAMMAR = r"""
    ?start: sum
    ?sum: product ("+" product)*
    ?product: atom ("*" atom)*
    ?atom: NUMBER
        | "(" sum ")"
    %import common.NUMBER
"""


class _ListLexer(Lexer):
    """Plays back a ready-made token list, so the stock (non-incremental)
    LALR parser can be used as a reference over arbitrary token sequences."""

    def __init__(self, tokens):
        self.tokens = tokens

    def lex(self, lexer_state, parser_state):
        return iter(self.tokens)


def full_parse(lark, tokens, start=None):
    """Reference: parse a token sequence with the stock LALR parser."""
    frontend = lark.parser
    start = frontend._verify_start(start)
    return frontend.parser.parse(LexerThread(_ListLexer(tokens), True), start)


def lex_all(lark, text):
    return list(lark.lex(text))


def assert_same_tree(tc, t1, t2):
    tc.assertIs(type(t1), type(t2))
    if isinstance(t1, Tree):
        tc.assertEqual(t1.data, t2.data)
        tc.assertEqual(len(t1.children), len(t2.children))
        for c1, c2 in zip(t1.children, t2.children):
            assert_same_tree(tc, c1, c2)
    else:
        tc.assertIsInstance(t1, Token)
        tc.assertEqual(t1.type, t2.type)
        tc.assertEqual(t1.value, t2.value)


def assert_same_positions(tc, t1, t2):
    attrs = ('start_pos', 'line', 'column', 'end_pos', 'end_line', 'end_column')
    if isinstance(t1, Tree):
        for a in attrs:
            tc.assertEqual(getattr(t1.meta, a, None), getattr(t2.meta, a, None), a)
        tc.assertEqual(len(t1.children), len(t2.children))
        for c1, c2 in zip(t1.children, t2.children):
            assert_same_positions(tc, c1, c2)
    else:
        for a in attrs:
            tc.assertEqual(getattr(t1, a), getattr(t2, a), a)


class CountingTransformer(Transformer):
    """Pure transformer: rebuilds the tree, counting rule invocations."""

    def __init__(self):
        self.counts = {}

    def __default__(self, data, children, meta):
        self.counts[data] = self.counts.get(data, 0) + 1
        return Tree(data, children)

    def reset(self):
        self.counts = {}

    @property
    def total(self):
        return sum(self.counts.values())


class TestIncrementalParseSession(unittest.TestCase):
    def make_lark(self, **kwargs):
        kwargs.setdefault('parser', 'lalr')
        kwargs.setdefault('lexer', 'basic')
        return Lark(GRAMMAR, **kwargs)

    def test_full_parse_matches_reference(self):
        lark = self.make_lark()
        tokens = lex_all(lark, "1+2*(3+4)")
        session = IncrementalParseSession(lark)
        result = session.parse(tokens)
        assert_same_tree(self, result, full_parse(lark, tokens))

    def test_insert_token(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "1+2+3"))
        new_tokens = lex_all(lark, "1+2+9+3")
        # insert "9+" before the last token: "1+2+3" -> "1+2+9+3"
        result = session.update(4, 4, new_tokens[4:6])
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("1+2+9+3"))
        # parsing resumed at the edit point; the prefix was not re-fed
        self.assertEqual(session.last_stats.resumed_at, 4)
        self.assertEqual(session.last_stats.tokens_fed, 3)

    def test_delete_across_nested_boundaries(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "((1)+(2))"))
        # tokens: ( ( 1 ) + ( 2 ) )  ->  delete "+(2)" (tokens 4..7), a range
        # crossing the boundary between the two nested expressions.
        result = session.update(4, 8, [])
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("((1))"))
        self.assertEqual(session.last_stats.resumed_at, 4)

    def test_eof_error_then_repair(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        with self.assertRaises(UnexpectedToken) as cm:
            session.parse(lex_all(lark, "(1+2"))
        e = cm.exception
        # current diagnostics are reported for the erroneous input
        self.assertEqual(e.token.type, '$END')
        self.assertTrue(e.expected)
        self.assertIsNotNone(e.interactive_parser)
        self.assertTrue(e.accepts)
        # the same session continues after the input is fixed
        result = session.update(4, 4, lex_all(lark, ")"))
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("(1+2)"))
        self.assertEqual(session.last_stats.resumed_at, 4)

    def test_error_then_fix_same_session(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        with self.assertRaises(UnexpectedToken) as cm:
            session.parse(lex_all(lark, "1++2"))
        self.assertEqual(str(cm.exception.token), "+")
        # delete the extra '+' -> "1+2"
        result = session.update(2, 3, [])
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("1+2"))

    def test_update_before_parse(self):
        session = IncrementalParseSession(self.make_lark())
        with self.assertRaises(ConfigurationError):
            session.update(0, 0, [])

    def test_invalid_range(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "1+2"))
        with self.assertRaises(ValueError):
            session.update(2, 1, [])
        with self.assertRaises(ValueError):
            session.update(0, 99, [])
        with self.assertRaises(ValueError):
            session.update(-1, 0, [])

    def test_non_lalr_refused(self):
        lark = Lark(GRAMMAR, parser='earley')
        with self.assertRaises(ConfigurationError):
            IncrementalParseSession(lark)

    def test_transformer_must_be_declared_pure(self):
        lark = self.make_lark(transformer=CountingTransformer())
        with self.assertRaises(ConfigurationError):
            IncrementalParseSession(lark)
        # declared pure: allowed
        IncrementalParseSession(lark, assume_pure_callbacks=True)

    def test_sequential_edits(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "1+2+3"))
        edits = [
            (5, 5, lex_all(lark, "+4")),   # append   -> 1+2+3+4
            (0, 0, lex_all(lark, "9+")),   # prepend  -> 9+1+2+3+4
            (3, 5, []),                    # delete   -> 9+1+3+4
            (6, 7, lex_all(lark, "8")),    # replace  -> 9+1+3+8
        ]
        for start, end, new in edits:
            result = session.update(start, end, new)
            assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("9+1+3+8"))

    def test_positions_match_full_reparse(self):
        lark = self.make_lark(propagate_positions=True)
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "1+22+333"))
        # same-width replacement keeps the positions of the suffix stable
        result = session.update(2, 3, lex_all(lark, "1+99+333")[2:3])
        expected = lark.parse("1+99+333")
        assert_same_tree(self, result, expected)
        assert_same_positions(self, result, expected)
        # append at the end
        result = session.update(5, 5, lex_all(lark, "1+99+333+4")[5:7])
        expected = lark.parse("1+99+333+4")
        assert_same_tree(self, result, expected)
        assert_same_positions(self, result, expected)

    def test_rule_callbacks_not_rerun_for_unaffected_prefix(self):
        counter = CountingTransformer()
        lark = self.make_lark(transformer=counter)
        session = IncrementalParseSession(lark, assume_pure_callbacks=True)
        text = "+".join(str(i) for i in range(30))
        tokens = lex_all(lark, text)
        session.parse(tokens)
        full_rules = session.last_stats.rules_executed
        self.assertGreater(full_rules, 50)
        counter.reset()

        # replace the last operand (token index 58 out of 59 tokens)
        result = session.update(58, 59, lex_all(lark, "93"))
        stats = session.last_stats
        executed_total = counter.total
        star_count = counter.counts.get('__sum_star_0', 0)
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual(stats.resumed_at, 58)
        self.assertEqual(stats.tokens_fed, 1)
        # only the rules on the path from the edited token to the root re-run
        self.assertLessEqual(stats.rules_executed, 10)
        self.assertEqual(executed_total, 2)  # __sum_star_0 and sum
        self.assertEqual(star_count, 1)

        # same-width replacement of an operand in the middle (token 20)
        counter.reset()
        result = session.update(20, 21, lex_all(lark, "55"))
        stats = session.last_stats
        star_count = counter.counts.get('__sum_star_0', 0)
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual(stats.resumed_at, 20)
        # the 19 suffix operands are reused, not re-executed ...
        self.assertGreaterEqual(stats.rules_reused, 38)
        # ... and only the 20 chain nodes containing/above the edit re-run
        self.assertEqual(star_count, 20)

    def test_suffix_reuse_with_same_width_edit(self):
        counter = CountingTransformer()
        lark = self.make_lark(transformer=counter)
        session = IncrementalParseSession(lark, assume_pure_callbacks=True)
        session.parse(lex_all(lark, "1+2+3+4+5"))
        counter.reset()
        # replace "3" (token index 4) with "9"; the suffix keeps its identity
        result = session.update(4, 5, lex_all(lark, "9"))
        stats = session.last_stats
        star_count = counter.counts.get('__sum_star_0', 0)
        sum_count = counter.counts.get('sum', 0)
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual(stats.resumed_at, 4)
        # suffix tokens are still fed through the state machine ...
        self.assertEqual(stats.tokens_fed, 5)
        # ... but their rule callbacks are reused, not re-executed
        self.assertGreater(stats.rules_reused, 0)
        # the prefix chain node was reused; only chain nodes above the edit
        # (and the root) re-run
        self.assertEqual(star_count, 3)
        self.assertEqual(sum_count, 1)

    def test_maybe_placeholders_false(self):
        lark = self.make_lark(maybe_placeholders=False)
        session = IncrementalParseSession(lark)
        session.parse(lex_all(lark, "1+2+3"))
        result = session.update(2, 3, lex_all(lark, "9"))
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual(session.last_stats.resumed_at, 2)

    def test_edit_after_error_beyond_error_position(self):
        lark = self.make_lark()
        session = IncrementalParseSession(lark)
        with self.assertRaises(UnexpectedToken):
            session.parse(lex_all(lark, "1++2"))
        # an edit past the error position resumes from the last valid
        # checkpoint before it and re-feeds the tokens in between
        with self.assertRaises(UnexpectedToken):
            session.update(4, 4, lex_all(lark, "+5"))
        self.assertLess(session.last_stats.resumed_at, 4)
        # fixing the actual error recovers
        result = session.update(2, 3, [])
        assert_same_tree(self, result, full_parse(lark, session.tokens))
        self.assertEqual([str(t) for t in session.tokens], list("1+2+5"))

    def test_empty_input(self):
        lark = Lark('start: "a"*', parser='lalr', lexer='basic')
        session = IncrementalParseSession(lark)
        result = session.parse([])
        assert_same_tree(self, result, full_parse(lark, []))
        result = session.update(0, 0, lex_all(lark, "aa"))
        assert_same_tree(self, result, full_parse(lark, session.tokens))


if __name__ == '__main__':
    unittest.main()
