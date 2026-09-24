import asyncio
import unittest
from unittest import TestCase

from lark import Tree, Token
from lark.exceptions import VisitError
from lark.visitors import AsyncTransformer, Discard, v_args


def run(coro):
    return asyncio.run(coro)


class TestAsyncTransformer(TestCase):
    def test_basic_async(self):
        class T(AsyncTransformer):
            async def number(self, children):
                await asyncio.sleep(0)
                return int(children[0])

            async def add(self, children):
                await asyncio.sleep(0)
                return children[0] + children[1]

        tree = Tree('add', [Tree('number', ['1']), Tree('number', ['2'])])
        self.assertEqual(run(T().transform(tree)), 3)

    def test_mixed_sync_and_async(self):
        class T(AsyncTransformer):
            def number(self, children):
                return int(children[0])

            async def add(self, children):
                await asyncio.sleep(0)
                return children[0] + children[1]

            def mul(self, children):
                return children[0] * children[1]

        tree = Tree('mul', [
            Tree('add', [Tree('number', ['1']), Tree('number', ['2'])]),
            Tree('number', ['3']),
        ])
        self.assertEqual(run(T().transform(tree)), 9)

    def test_default_copies_tree(self):
        tree = Tree('a', [Tree('b', [Token('T', 'x')]), 'leaf'])
        res = run(AsyncTransformer().transform(tree))
        self.assertEqual(res, tree)
        self.assertIsNot(res, tree)

    def test_v_args_inline(self):
        @v_args(inline=True)
        class T(AsyncTransformer):
            async def add(self, a, b):
                await asyncio.sleep(0)
                return a + b

            def number(self, n):
                return int(n)

        tree = Tree('add', [Tree('number', ['40']), Tree('number', ['2'])])
        self.assertEqual(run(T().transform(tree)), 42)

    def test_v_args_meta(self):
        @v_args(meta=True)
        class T(AsyncTransformer):
            async def a(self, meta, children):
                await asyncio.sleep(0)
                return (meta.line, list(children))

        tree = Tree('a', ['x', 'y'])
        tree.meta.line = 7
        self.assertEqual(run(T().transform(tree)), (7, ['x', 'y']))

    def test_v_args_tree(self):
        class T(AsyncTransformer):
            @v_args(tree=True)
            async def a(self, tree):
                await asyncio.sleep(0)
                return Tree(tree.data, list(reversed(tree.children)), tree.meta)

        tree = Tree('a', [Tree('b', [1]), Tree('c', [2])])
        res = run(T().transform(tree))
        self.assertEqual(res, Tree('a', [Tree('c', [2]), Tree('b', [1])]))

    def test_discard(self):
        class T(AsyncTransformer):
            async def junk(self, children):
                await asyncio.sleep(0)
                return Discard

            def keep(self, children):
                return list(children)

        tree = Tree('keep', [Tree('junk', []), Tree('keep', ['a']), 'b'])
        self.assertEqual(run(T().transform(tree)), [['a'], 'b'])

    def test_discard_root_returns_none(self):
        class T(AsyncTransformer):
            async def a(self, children):
                return Discard

        self.assertIs(run(T().transform(Tree('a', []))), None)

    def test_token_callbacks(self):
        class T(AsyncTransformer):
            async def NUM(self, token):
                await asyncio.sleep(0)
                return int(token) * 2

            def SKIP(self, token):
                return Discard

            def start(self, children):
                return list(children)

        tree = Tree('start', [Token('NUM', '21'), Token('SKIP', 'x'), Token('OTHER', 'y')])
        self.assertEqual(run(T().transform(tree)), [42, Token('OTHER', 'y')])

    def test_visit_tokens_false(self):
        class T(AsyncTransformer):
            async def NUM(self, token):
                raise AssertionError('should not be called')

            def start(self, children):
                return list(children)

        tree = Tree('start', [Token('NUM', '1')])
        self.assertEqual(run(T(visit_tokens=False).transform(tree)), [Token('NUM', '1')])

    def test_children_order_preserved(self):
        delays = [0.06, 0.04, 0.02, 0.0]

        class T(AsyncTransformer):
            async def leaf(self, children):
                i = children[0]
                await asyncio.sleep(delays[i])
                return i

            def root(self, children):
                return list(children)

        tree = Tree('root', [Tree('leaf', [i]) for i in range(4)])
        self.assertEqual(run(T().transform(tree)), [0, 1, 2, 3])

    def test_parent_called_after_children(self):
        events = []

        class T(AsyncTransformer):
            async def leaf(self, children):
                await asyncio.sleep(0.01)
                events.append(('leaf', children[0]))
                return children[0]

            async def root(self, children):
                events.append(('root', list(children)))
                return children

        tree = Tree('root', [Tree('leaf', [i]) for i in range(3)])
        self.assertEqual(run(T().transform(tree)), [0, 1, 2])
        self.assertEqual(events[-1], ('root', [0, 1, 2]))
        self.assertEqual(sorted(events[:-1]), [('leaf', 0), ('leaf', 1), ('leaf', 2)])

    def test_concurrency_unlimited(self):
        max_seen = self._measure_concurrency(None, 4)
        self.assertEqual(max_seen, 4)

    def test_concurrency_limited(self):
        max_seen = self._measure_concurrency(2, 4)
        self.assertEqual(max_seen, 2)

    def test_max_concurrency_validation(self):
        self.assertRaises(ValueError, AsyncTransformer, max_concurrency=0)

    def _measure_concurrency(self, max_concurrency, n):
        class T(AsyncTransformer):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.current = 0
                self.max_seen = 0

            async def leaf(self, children):
                self.current += 1
                self.max_seen = max(self.max_seen, self.current)
                try:
                    await asyncio.sleep(0.02)
                finally:
                    self.current -= 1
                return children[0]

            def root(self, children):
                return list(children)

        t = T(max_concurrency=max_concurrency)
        tree = Tree('root', [Tree('leaf', [i]) for i in range(n)])
        self.assertEqual(run(t.transform(tree)), list(range(n)))
        return t.max_seen

    def test_failure_cancels_siblings(self):
        finally_events = []

        class T(AsyncTransformer):
            async def bad(self, children):
                await asyncio.sleep(0.01)
                raise ValueError('boom')

            async def slow(self, children):
                try:
                    await asyncio.sleep(30)
                    return 'slow'
                finally:
                    finally_events.append('slow-finally')

            def root(self, children):
                return list(children)

        tree = Tree('root', [Tree('slow', []), Tree('bad', [])])
        with self.assertRaises(VisitError) as cm:
            run(T().transform(tree))
        self.assertEqual(cm.exception.rule, 'bad')
        self.assertIsInstance(cm.exception.orig_exc, ValueError)
        self.assertEqual(cm.exception.obj, Tree('bad', []))
        self.assertEqual(finally_events, ['slow-finally'])

    def test_token_failure_wrapped_in_visit_error(self):
        class T(AsyncTransformer):
            async def NUM(self, token):
                raise KeyError('nope')

        with self.assertRaises(VisitError) as cm:
            run(T().transform(Tree('start', [Token('NUM', '1')])))
        self.assertEqual(cm.exception.rule, 'NUM')
        self.assertIsInstance(cm.exception.orig_exc, KeyError)

    def test_shared_subtree_not_cached(self):
        calls = []

        class T(AsyncTransformer):
            async def leaf(self, children):
                await asyncio.sleep(0)
                calls.append(children[0])
                return children[0]

            def root(self, children):
                return list(children)

        shared = Tree('leaf', ['x'])
        tree = Tree('root', [shared, shared])
        self.assertEqual(run(T().transform(tree)), ['x', 'x'])
        self.assertEqual(sorted(calls), ['x', 'x'])

    def test_deep_tree(self):
        class T(AsyncTransformer):
            async def node(self, children):
                await asyncio.sleep(0)
                return sum(children) + 1

        tree = Tree('node', [Tree('node', [Tree('node', [0])])])
        self.assertEqual(run(T().transform(tree)), 3)


if __name__ == '__main__':
    unittest.main()
