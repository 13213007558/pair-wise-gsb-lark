import asyncio
import unittest
from unittest import TestCase

from lark import Tree, Token
from lark.exceptions import GrammarError, VisitError
from lark.visitors import AsyncTransformer, Discard, v_args


def run(coro):
    return asyncio.run(coro)


class TestAsyncTransformer(TestCase):
    def test_basic_mixed_sync_and_async(self):
        class T(AsyncTransformer):
            async def number(self, children):
                await asyncio.sleep(0)
                return int(children[0])

            def add(self, children):  # sync method mixed with async ones
                return sum(children)

            async def NUM(self, token):
                await asyncio.sleep(0)
                return token.value

        tree = Tree('add', [Tree('number', [Token('NUM', '1')]), Tree('number', [Token('NUM', '2')])])
        self.assertEqual(run(T().transform(tree)), 3)

    def test_parent_called_after_children_complete(self):
        completed = set()

        class T(AsyncTransformer):
            async def leaf(self, children):
                tag = children[0]
                await asyncio.sleep(0.01 * (3 - tag))  # finish out of order
                completed.add(tag)
                return tag

            def root(self, children):
                # By the time the parent runs, every child must have completed
                assert completed == {0, 1, 2}
                return list(children)

        tree = Tree('root', [Tree('leaf', [i]) for i in range(3)])
        self.assertEqual(run(T().transform(tree)), [0, 1, 2])

    def test_v_args_inline_meta_tree(self):
        @v_args(inline=True)
        class TInline(AsyncTransformer):
            async def add(self, left, right):
                await asyncio.sleep(0)
                return left + right

            def number(self, children):
                return int(children[0])

        tree = Tree('add', [Tree('number', ['1']), Tree('number', ['2'])])
        self.assertEqual(run(TInline().transform(tree)), 3)

        class TMeta(AsyncTransformer):
            @v_args(meta=True)
            async def root(self, meta, children):
                await asyncio.sleep(0)
                return (meta, children)

        tree = Tree('root', ['x'])
        tree.meta.line = 42
        meta, children = run(TMeta().transform(tree))
        self.assertEqual(meta.line, 42)
        self.assertEqual(children, ['x'])

        class TTree(AsyncTransformer):
            @v_args(tree=True)
            async def root(self, tree):
                await asyncio.sleep(0)
                return tree.data, list(tree.children)

        self.assertEqual(run(TTree().transform(Tree('root', ['a']))), ('root', ['a']))

    def test_v_args_meta_inline(self):
        class T(AsyncTransformer):
            @v_args(meta=True, inline=True)
            async def root(self, meta, child):
                await asyncio.sleep(0)
                return child * 2

        self.assertEqual(run(T().transform(Tree('root', [21]))), 42)

    def test_discard(self):
        class T(AsyncTransformer):
            async def ignore_me(self, children):
                await asyncio.sleep(0)
                return Discard

            def KEEP(self, token):
                return token

            async def DROP(self, token):
                return Discard

        tree = Tree('root', [
            Tree('ignore_me', [Token('KEEP', 'x')]),
            Tree('keep_too', [Token('DROP', 'y'), Token('KEEP', 'z')]),
        ])
        res = run(T().transform(tree))
        self.assertEqual(res, Tree('root', [Tree('keep_too', [Token('KEEP', 'z')])]))

    def test_discard_root_returns_none(self):
        class T(AsyncTransformer):
            async def root(self, children):
                return Discard

        self.assertIs(run(T().transform(Tree('root', []))), None)

    def test_visit_tokens_false(self):
        class T(AsyncTransformer):
            async def A(self, token):
                raise AssertionError("should not be called")

        tree = Tree('root', [Token('A', 'x')])
        res = run(T(visit_tokens=False).transform(tree))
        self.assertEqual(res, Tree('root', [Token('A', 'x')]))

    def test_default_copies_tree(self):
        tree = Tree('root', [Tree('child', [Token('A', 'x')])])
        res = run(AsyncTransformer().transform(tree))
        self.assertEqual(res, tree)
        self.assertIsNot(res, tree)

    def test_concurrency_limit(self):
        for max_concurrency, expected_max in [(None, 4), (2, 2), (1, 1)]:
            running = 0
            max_running = 0

            class T(AsyncTransformer):
                async def leaf(self, children):
                    nonlocal running, max_running
                    running += 1
                    max_running = max(max_running, running)
                    try:
                        # different delays: later siblings finish first
                        await asyncio.sleep(0.02 * (4 - children[0]))
                        return children[0]
                    finally:
                        running -= 1

            tree = Tree('root', [Tree('leaf', [i]) for i in range(4)])
            res = run(T(max_concurrency=max_concurrency).transform(tree))
            # Order of the children list matches the original tree, despite the delays
            self.assertEqual(res.children, [0, 1, 2, 3])
            self.assertEqual(max_running, expected_max)

    def test_failure_cancels_siblings_and_raises_visit_error(self):
        finalized = set()
        started = set()

        class T(AsyncTransformer):
            async def boom(self, children):
                await asyncio.sleep(0.01)
                raise ValueError("original failure")

            async def slow(self, children):
                tag = children[0]
                started.add(tag)
                try:
                    await asyncio.sleep(30)
                    return tag
                finally:
                    finalized.add(tag)

        tree = Tree('root', [
            Tree('slow', [0]),
            Tree('boom', []),
            Tree('slow', [1]),
        ])
        with self.assertRaises(VisitError) as cm:
            run(T().transform(tree))

        err = cm.exception
        self.assertEqual(err.rule, 'boom')
        self.assertIsInstance(err.orig_exc, ValueError)
        self.assertEqual(str(err.orig_exc), 'original failure')
        # The unfinished siblings were cancelled and awaited: their finally blocks ran
        self.assertEqual(started, {0, 1})
        self.assertEqual(finalized, {0, 1})

    def test_token_failure_raises_visit_error(self):
        class T(AsyncTransformer):
            async def BAD(self, token):
                await asyncio.sleep(0)
                raise KeyError("nope")

        tree = Tree('root', [Token('BAD', 'x')])
        with self.assertRaises(VisitError) as cm:
            run(T().transform(tree))
        self.assertEqual(cm.exception.rule, 'BAD')
        self.assertIsInstance(cm.exception.orig_exc, KeyError)
        self.assertIsInstance(cm.exception.obj, Token)

    def test_grammar_error_not_wrapped(self):
        class T(AsyncTransformer):
            async def root(self, children):
                raise GrammarError("grammar problem")

        with self.assertRaises(GrammarError):
            run(T().transform(Tree('root', [])))

    def test_shared_subtree_not_cached(self):
        calls = []

        class T(AsyncTransformer):
            async def leaf(self, children):
                await asyncio.sleep(0)
                calls.append(1)
                return len(calls)

        shared = Tree('leaf', [])
        tree = Tree('root', [shared, shared, shared])
        res = run(T().transform(tree))
        # Structural traversal: the shared subtree is transformed once per occurrence
        self.assertEqual(len(calls), 3)
        self.assertEqual(res.children, [1, 2, 3])

    def test_async_default_and_default_token(self):
        class T(AsyncTransformer):
            async def __default__(self, data, children, meta):
                await asyncio.sleep(0)
                return Tree(data + '_async', children, meta)

            async def __default_token__(self, token):
                await asyncio.sleep(0)
                return token.update(value=token.value.upper())

        tree = Tree('root', [Tree('child', [Token('A', 'x')])])
        res = run(T().transform(tree))
        self.assertEqual(res, Tree('root_async', [Tree('child_async', [Token('A', 'X')])]))

    def test_invalid_max_concurrency(self):
        self.assertRaises(ValueError, AsyncTransformer, max_concurrency=0)
        self.assertRaises(ValueError, AsyncTransformer, max_concurrency=-3)

    def test_deep_tree_with_limit_no_deadlock(self):
        class T(AsyncTransformer):
            async def node(self, children):
                await asyncio.sleep(0)
                return children[0] + 1 if children else 0

        tree = Tree('node', [])
        for _ in range(50):
            tree = Tree('node', [tree])
        self.assertEqual(run(T(max_concurrency=1).transform(tree)), 50)


if __name__ == '__main__':
    unittest.main()
