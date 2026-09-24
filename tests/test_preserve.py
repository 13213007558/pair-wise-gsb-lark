# coding=utf-8

import unittest
from unittest import TestCase

from lark import Lark, Token, Tree
from lark.exceptions import ConfigurationError
from lark.preserve import (
    MovedNodeError,
    PreservingReconstructor,
    SourcePreservationError,
)


CONFIG_GRAMMAR = r"""
start: pair+
pair: WORD "=" value ";"?
value: WORD
      | NUMBER
      | list
list: "[" [value ("," value)*] "]"

%import common.WS
%import common.WORD
%import common.NUMBER
%ignore WS
%ignore COMMENT
COMMENT: /#[^\n]*/
"""


def _semantic(node):
    "A position-independent view of a tree: (data, children) with typed tokens."
    if isinstance(node, Token):
        return (node.type, str(node))
    return (node.data, [_semantic(c) for c in node.children if c is not None])


def _find_token(node, value):
    if isinstance(node, Token):
        if node == value:
            return node
        return None
    for child in node.children:
        found = _find_token(child, value)
        if found is not None:
            return found
    return None


class TestPreservingReconstructor(TestCase):

    def make_reconstructor(self, grammar=CONFIG_GRAMMAR, term_subs=None, **options):
        options.setdefault('maybe_placeholders', False)
        parser = Lark(grammar, parser='lalr', propagate_positions=True, **options)
        return PreservingReconstructor(parser, term_subs=term_subs)

    def assert_semantic_roundtrip(self, pr, tree, output):
        "The reconstructed output must re-parse to the same semantic tree."
        reparsed = pr.parse(output)
        self.assertEqual(_semantic(reparsed), _semantic(tree))

    def test_requires_lalr(self):
        parser = Lark(CONFIG_GRAMMAR, parser='earley', propagate_positions=True)
        with self.assertRaises(ConfigurationError):
            PreservingReconstructor(parser)

    def test_requires_propagate_positions(self):
        parser = Lark(CONFIG_GRAMMAR, parser='lalr')
        with self.assertRaises(ConfigurationError):
            PreservingReconstructor(parser)

    def test_unmodified_tree_is_byte_identical(self):
        code = (
            "# header comment\n"
            "name   =   foo\n"
            "\n"
            "nums = [1,  2,3]  # trailing comment\n"
            "flag = yes;\n"
            "# eof comment\n"
        )
        pr = self.make_reconstructor()
        tree = pr.parse(code)
        self.assertEqual(pr.reconstruct(tree), code)

    def test_parse_associates_tokens_spans_and_ignored_text(self):
        code = "a = 1 # note\nb = 2\n"
        pr = self.make_reconstructor()
        tree = pr.parse(code)

        # original tokens, with offsets
        tok = _find_token(tree, "a")
        self.assertIsInstance(tok, Token)
        self.assertEqual(code[tok.start_pos:tok.end_pos], "a")

        # rule spans
        pair = tree.children[0]
        self.assertEqual(code[pair.meta.start_pos:pair.meta.end_pos], "a = 1")
        self.assertIs(pair.meta.source, code)

        # ignored text is attached to the tree
        self.assertIn("# note", "".join(tree.meta.ignored_gaps))
        self.assertEqual(len(tree.meta.child_spans), len(tree.children))

    def test_leaf_value_replacement_preserves_comments(self):
        code = "name = foo  # keep me\nother = bar\n"
        pr = self.make_reconstructor()
        tree = pr.parse(code)

        value = tree.children[0].children[1]
        self.assertEqual(value.data, 'value')
        value.children[0] = Token('WORD', 'baz')

        out = pr.reconstruct(tree)
        self.assertEqual(out, "name = baz  # keep me\nother = bar\n")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_adjacent_identical_tokens(self):
        grammar = r"""
        start: "row" NUM NUM NUM
        NUM: /\d+/
        %import common.WS
        %ignore WS
        """
        pr = self.make_reconstructor(grammar)
        tree = pr.parse("row 5 5 5")

        # replace only the second "5"
        tree.children[1] = Token('NUM', '7')
        out = pr.reconstruct(tree)
        self.assertEqual(out, "row 5 7 5")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_empty_rule(self):
        grammar = r"""
        start: BEGIN body END
        BEGIN: "begin"
        END: "end"
        body: item*
        item: WORD
        %import common.WS
        %import common.WORD
        %ignore WS
        """
        pr = self.make_reconstructor(grammar)

        # empty rule stays empty, spacing is preserved
        tree = pr.parse("begin   end")
        self.assertEqual(pr.reconstruct(tree), "begin   end")

        # adding items into the empty rule produces re-parseable text
        body = tree.children[1]
        self.assertEqual(body.data, 'body')
        self.assertEqual(body.children, [])
        body.children.append(Tree('item', [Token('WORD', 'x')]))
        body.children.append(Tree('item', [Token('WORD', 'y')]))
        out = pr.reconstruct(tree)
        self.assertEqual(out, "begin x y   end")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_subtree_replacement_preserves_surroundings(self):
        code = "first = 1\nsecond = [1, 2] # c\nthird = 3\n"
        pr = self.make_reconstructor()
        tree = pr.parse(code)

        new_pair = Tree('pair', [Token('WORD', 'replaced'), Tree('value', [Token('NUMBER', '42')])])
        tree.children[1] = new_pair

        out = pr.reconstruct(tree)
        self.assertEqual(out, "first = 1\nreplaced=42; # c\nthird = 3\n")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_nested_replacement(self):
        code = "nums = [1, 2,3]\n"
        pr = self.make_reconstructor()
        tree = pr.parse(code)

        value = tree.children[0].children[1]
        lst = value.children[0]
        self.assertEqual(lst.data, 'list')
        # deep leaf replacement: 2 -> 20
        lst.children[1].children[0] = Token('NUMBER', '20')
        # nested subtree replacement: value(3) -> value(99)
        lst.children[2] = Tree('value', [Token('NUMBER', '99')])

        out = pr.reconstruct(tree)
        self.assertEqual(out, "nums = [1, 20,99]\n")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_trailing_comment_at_eof(self):
        code = "a = 1\n# final comment, no newline"
        pr = self.make_reconstructor()
        tree = pr.parse(code)
        tree.children[0].children[1].children[0] = Token('NUMBER', '2')
        out = pr.reconstruct(tree)
        self.assertEqual(out, "a = 2\n# final comment, no newline")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_missing_terminal_generator_reports_position(self):
        grammar = r"""
        start: item+
        item: WORD "=" WORD _NL
        _NL: /(\r?\n)+/
        %import common.WS_INLINE
        %import common.WORD
        %ignore WS_INLINE
        """
        pr = self.make_reconstructor(grammar)
        code = "a = u\nb = v\n"
        tree = pr.parse(code)

        tree.children[1] = Tree('item', [Token('WORD', 'x'), Token('WORD', 'y')])
        with self.assertRaises(SourcePreservationError) as ctx:
            pr.reconstruct(tree)
        message = str(ctx.exception)
        self.assertIn("line 2", message)
        self.assertIn("'_NL'", message)
        self.assertIn("'item'", message)

    def test_term_subs_generator_for_new_nodes(self):
        grammar = r"""
        start: item+
        item: WORD "=" WORD _NL
        _NL: /(\r?\n)+/
        %import common.WS_INLINE
        %import common.WORD
        %ignore WS_INLINE
        """
        pr = self.make_reconstructor(grammar, term_subs={'_NL': lambda _: '\n'})
        code = "a = u\nb = v\n"
        tree = pr.parse(code)

        tree.children[1] = Tree('item', [Token('WORD', 'x'), Token('WORD', 'y')])
        out = pr.reconstruct(tree)
        self.assertEqual(out, "a = u\nx=y\n")
        self.assert_semantic_roundtrip(pr, tree, out)

    def test_moved_node_is_rejected(self):
        pr = self.make_reconstructor()
        tree = pr.parse("a = 1\nb = 2\n")

        # swapping two preserved subtrees is a move, not a replacement
        tree.children = [tree.children[1], tree.children[0]]
        with self.assertRaises(MovedNodeError):
            pr.reconstruct(tree)

    def test_reconstruct_rejects_foreign_tree(self):
        pr = self.make_reconstructor()
        with self.assertRaises(SourcePreservationError):
            pr.reconstruct(Tree('start', []))


if __name__ == '__main__':
    unittest.main()
