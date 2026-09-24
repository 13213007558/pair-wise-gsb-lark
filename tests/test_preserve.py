"""Tests for the text-preserving parse & reconstruct mode (lark.preserve)."""

from unittest import TestCase

from lark import Lark, Token, Tree
from lark.exceptions import ConfigurationError
from lark.preserve import TextPreservingReconstructor, MissingTerminalGeneratorError


def _semantic_equal(a, b):
    """Compare parse trees including terminal types (Token string comparison
    ignores the type)."""
    if isinstance(a, Token) or isinstance(b, Token):
        return isinstance(a, Token) and isinstance(b, Token) \
            and a.type == b.type and str(a) == str(b)
    if isinstance(a, Tree) or isinstance(b, Tree):
        return isinstance(a, Tree) and isinstance(b, Tree) \
            and a.data == b.data \
            and len(a.children) == len(b.children) \
            and all(_semantic_equal(x, y) for x, y in zip(a.children, b.children))
    return a == b


ASSIGN_GRAMMAR = r"""
start: stmt*
stmt: NAME "=" value ";"
?value: NAME | NUMBER
NAME: /[a-zA-Z_]\w*/
NUMBER: /\d+/
WS: /[ \t\n]+/
COMMENT: /#[^\n]*/
%ignore WS
%ignore COMMENT
"""


def make_parser(grammar=ASSIGN_GRAMMAR, **options):
    options.setdefault('parser', 'lalr')
    options.setdefault('maybe_placeholders', False)
    return Lark(grammar, **options)


class TestPreserve(TestCase):
    def roundtrip(self, grammar, code):
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)
        out = pr.reconstruct(tree)
        self.assertEqual(out, code)
        return parser, pr, tree

    def test_unchanged_is_byte_identical(self):
        code = '# header\na = 1; # set a\nb = x;\n'
        parser, pr, tree = self.roundtrip(ASSIGN_GRAMMAR, code)
        self.assertEqual(_semantic_equal(parser.parse(pr.reconstruct(tree)), tree), True)

    def test_leaf_value_replace_keeps_end_of_line_comment(self):
        code = '# header\na = 1; # set a\nb = x;\n'
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)

        tree.children[0].children[1] = Token('NUMBER', '42')
        out = pr.reconstruct(tree)
        self.assertEqual(out, '# header\na = 42; # set a\nb = x;\n')

        reparsed = parser.parse(out)
        self.assertTrue(_semantic_equal(reparsed, tree))

    def test_comment_between_replaced_token_and_operator_kept(self):
        code = 'a = # mid\n1;\n'
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)
        tree.children[0].children[1] = Token('NUMBER', '2')
        out = pr.reconstruct(tree)
        self.assertEqual(out, 'a = # mid\n2;\n')
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_in_place_token_update(self):
        code = 'a = 1; # c\n'
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)
        token = tree.children[0].children[1]
        tree.children[0].children[1] = token.update(value='7')
        out = pr.reconstruct(tree)
        self.assertEqual(out, 'a = 7; # c\n')
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_adjacent_same_value_tokens(self):
        grammar = r"""
        start: NAME+
        NAME: /\w+/
        %ignore /[ \t\n]+/
        """
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser)

        # Only the middle identical token is replaced; position (not value)
        # decides which original bytes to keep.
        tree = pr.parse('a a a\n')
        tree.children[1] = Token('NAME', 'bb')
        out = pr.reconstruct(tree)
        self.assertEqual(out, 'a bb a\n')
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

        # Replacing every token must still keep neighbouring identifiers apart.
        tree = pr.parse('a a a\n')
        tree.children = [Token('NAME', v) for v in ('x', 'y', 'z')]
        out = pr.reconstruct(tree)
        self.assertEqual(out, 'x y z\n')
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_empty_rule(self):
        grammar = r"""
        start: "[" maybe "]"
        maybe: | ITEM
        ITEM: /\w+/
        %ignore /\s+/
        """
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser)

        tree = pr.parse(' [ ]\n')
        self.assertEqual(pr.reconstruct(tree), ' [ ]\n')

        tree = pr.parse('[ a ]')
        item = tree.children[0].children[0]
        tree.children[0].children[0] = Token('ITEM', 'b')
        out = pr.reconstruct(tree)
        self.assertEqual(out, '[ b ]')
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_nested_replacement(self):
        grammar = r"""
        ?start: expr
        expr: expr "+" term | term
        term: NUMBER | "(" expr ")"
        NUMBER: /\d+/
        %ignore /[ \t]+/
        %ignore /#[^\n]*/
        %ignore /\n/
        """
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser)
        code = "( 1 + ( 2 + # deep comment\n   3 ) )  # outer\n"
        tree = pr.parse(code)

        # Replace the deeply nested NUMBER 2 -> 20.
        replaced = False
        for subtree in tree.iter_subtrees():
            for i, child in enumerate(subtree.children):
                if isinstance(child, Token) and child == '2':
                    subtree.children[i] = Token('NUMBER', '20')
                    replaced = True
        self.assertTrue(replaced)

        out = pr.reconstruct(tree)
        self.assertEqual(out, "( 1 + ( 20 + # deep comment\n   3 ) )  # outer\n")
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_subtree_replacement(self):
        code = 'a = 1; # first\nb = 2;\n'
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)

        fresh = parser.parse('c = 3;')
        tree.children[0] = fresh.children[0]
        out = pr.reconstruct(tree)
        self.assertTrue(_semantic_equal(parser.parse(out), tree))
        # The untouched second statement and its layout survive.
        self.assertIn('b = 2;', out)

    def test_inserted_subtree_is_reparseable(self):
        code = 'a = 1;\nb = 2;\n'
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse(code)
        tree.children.insert(1, parser.parse('cc = 33;').children[0])
        out = pr.reconstruct(tree)
        self.assertTrue(_semantic_equal(parser.parse(out), tree))

    def test_missing_generator_reports_position(self):
        grammar = r"""
        start: item*
        item: NAME _NL
        _NL: /\n+/
        NAME: /\w+/
        %ignore /[ \t]+/
        """
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser)
        tree = pr.parse('a\nb\n')
        tree.children.append(Tree('item', [Token('NAME', 'c')]))

        with self.assertRaises(MissingTerminalGeneratorError) as ctx:
            pr.reconstruct(tree)
        message = str(ctx.exception)
        self.assertIn('_NL', message)
        self.assertIn('item', message)
        self.assertIn('line 2', message)

    def test_term_subs_generator(self):
        grammar = r"""
        start: item*
        item: NAME _NL
        _NL: /\n+/
        NAME: /\w+/
        %ignore /[ \t]+/
        """
        parser = make_parser(grammar)
        pr = TextPreservingReconstructor(parser, term_subs={'_NL': lambda _t: '\n'})
        tree = pr.parse('a\nb\n')
        tree.children.append(Tree('item', [Token('NAME', 'c')]))
        out = pr.reconstruct(tree)
        self.assertTrue(_semantic_equal(parser.parse(out), tree))
        self.assertEqual(out, 'a\nb\nc\n')

    def test_requires_lalr(self):
        earley = Lark('start: "a"', parser='earley')
        with self.assertRaises(ConfigurationError):
            TextPreservingReconstructor(earley)

    def test_requires_maybe_placeholders_false(self):
        parser = Lark('start: "a"', parser='lalr')  # default: maybe_placeholders=True
        with self.assertRaises(ConfigurationError):
            TextPreservingReconstructor(parser)

    def test_reconstruct_without_parse(self):
        parser = make_parser()
        pr = TextPreservingReconstructor(parser)
        with self.assertRaises(ConfigurationError):
            pr.reconstruct(parser.parse('a = 1;'))
