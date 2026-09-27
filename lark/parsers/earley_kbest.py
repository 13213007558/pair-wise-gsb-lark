"""Lazy k-best enumeration of parse trees from an Earley SPPF.

This module implements a k-best interface over the shared packed parse
forest (SPPF) produced by the Earley parser. The caller assigns every
grammar rule a non-negative cost, and KBestIterator yields parse trees
in non-decreasing total rule-cost order, smallest first.

The enumeration is lazy: derivations are expanded on demand, best-first,
so requesting a small k never expands the whole derivation space of the
forest. The number of derivation nodes actually expanded is reported on
the iterator's nodes_expanded attribute.

Semantics and guarantees:

- Costs. rule_cost may be a callable receiving a Rule object, or a
  mapping keyed by Rule or by rule name (rules absent from the mapping
  get cost 0). Costs must be non-negative numbers (inf is allowed and
  sorts last); a negative or NaN cost raises ValueError. The callback is
  memoized: it is called at most once per rule.

- Ordering. Trees are yielded in non-decreasing total cost. Ties are
  broken by a stable order derived from the forest (rule order, then
  insertion order), so two runs over the same parser produce the same
  sequence (use the default ordered_sets=True for full determinism).

- Duplicates. The same tree is never yielded twice. Identical structures
  reached through different intermediate (LR0) nodes of the SPPF are
  collapsed; tokens keep their usual position information.

- Cycles. Nullable or recursive rules can create cycles in the SPPF,
  which correspond to infinitely many derivations (e.g. a: a | "x").
  The iterator never hangs: every next() call performs a bounded amount
  of work. Cyclic derivation families that would require re-entering a
  node that is currently being expanded are skipped, so in the presence
  of cycles the enumeration is not guaranteed to be exhaustive - but it
  stays responsive, keeps yielding distinct trees in cost order, and
  terminates whenever the remaining derivations are all cyclic. Callers
  should bound the iteration (e.g. itertools.islice or the k parameter)
  when infinite derivations are possible.

Note: deep forests may require a higher sys.setrecursionlimit, as
derivations are expanded recursively.
"""

import heapq
from itertools import count
from collections.abc import Mapping

from ..tree import Tree
from ..lexer import Token
from .earley_forest import SymbolNode, TokenNode


class _NodeState:
    """Lazy k-best enumeration state for a single SPPF symbol node."""
    __slots__ = ('derivs', 'heap', 'used', 'pending', 'exhausted', 'tentative')

    def __init__(self):
        # Finalized derivations, in non-decreasing cost order.
        # Each entry is (cost, packed_node, left_deriv, right_deriv), where
        # a child deriv is a nested tuple of the same shape, a Token, or None.
        self.derivs = []
        # Candidate heap of (cost, seq, packed_node, left_index, right_index).
        self.heap = []
        # (id(packed_node), left_index, right_index) already pushed.
        self.used = set()
        # Candidates blocked by a cyclic dependency, retried on reseeding:
        # maps (id(packed_node), left_index, right_index) -> packed_node.
        self.pending = {}
        # True when no more derivations can ever appear.
        self.exhausted = False
        # True when exhaustion was observed while a cyclic dependency was
        # still being resolved; more derivations may appear later.
        self.tentative = False


# Sentinel returned by _has_deriv when a derivation might become available
# once the nodes currently being resolved grow.
_BLOCKED = object()


class KBestIterator:
    """Iterates the parse trees of an Earley SPPF, cheapest first.

    Parameters:
        root: The root SymbolNode of the parse forest, as returned by the
            Earley parser (e.g. via ambiguity='forest').
        rule_cost: A callable Rule -> non-negative number, or a mapping
            from Rule objects or rule names to costs (missing rules
            default to 0). Called at most once per rule.
        callbacks: Optional mapping of Rule to tree-building functions.
            Defaults to the parser's own callbacks; rules without a
            callback build tree_class(rule_name, children).
        tree_class: The tree class used for rules without a callback.
        k: If given, the iterator stops after yielding k unique trees.

    Attributes:
        nodes_expanded: Number of derivation nodes expanded so far. This
            is the laziness metric: for small k it stays tiny compared to
            the size of the forest / derivation space.
        trees_yielded: Number of unique trees yielded so far.
        duplicates_skipped: Derivations skipped because their tree was
            already yielded (e.g. same structure via different
            intermediate nodes).
    """

    def __init__(self, root, rule_cost, callbacks=None, tree_class=Tree, k=None):
        if not isinstance(root, SymbolNode):
            raise TypeError("KBestIterator expects the root SymbolNode of an Earley forest, got %r" % (root,))
        if k is not None and k < 0:
            raise ValueError("k must be non-negative, got %r" % (k,))
        self.root = root
        self._cost_fn = self._normalize_cost(rule_cost)
        self.callbacks = callbacks if callbacks is not None else {}
        self.tree_class = tree_class

        self._cost_memo = {}
        self._states = {}
        self._resolving = set()
        self._seq = count()

        self._root_index = 0
        self._remaining = k
        self._seen_trees = set()

        self.nodes_expanded = 0
        self.trees_yielded = 0
        self.duplicates_skipped = 0

    @staticmethod
    def _normalize_cost(rule_cost):
        if callable(rule_cost):
            return rule_cost
        if isinstance(rule_cost, Mapping):
            mapping = rule_cost

            def cost_fn(rule):
                try:
                    return mapping[rule]
                except KeyError:
                    pass
                if rule.alias is not None and rule.alias in mapping:
                    return mapping[rule.alias]
                return mapping.get(rule.origin.name, 0)
            return cost_fn
        raise TypeError("rule_cost must be a callable or a mapping, got %r" % (rule_cost,))

    def _rule_cost(self, rule):
        # Memoized, so the user callback runs at most once per rule.
        try:
            return self._cost_memo[rule]
        except KeyError:
            pass
        cost = self._cost_fn(rule)
        if not isinstance(cost, (int, float)) or cost != cost or cost < 0:
            raise ValueError("rule_cost must return a non-negative number, got %r for rule %s" % (cost, rule))
        self._cost_memo[rule] = cost
        return cost

    def _packed_base_cost(self, packed):
        # The rule cost is counted once per rule application, at the packed
        # node directly under the non-intermediate symbol node (the same
        # place where ForestToParseTree applies the rule callback).
        if packed.parent.is_intermediate:
            return 0
        return self._rule_cost(packed.rule)

    ### Lazy per-node k-best enumeration (best-first over the SPPF)

    def _state(self, node):
        try:
            return self._states[node]
        except KeyError:
            state = self._states[node] = _NodeState()
            return state

    def _has_deriv(self, node, index):
        state = self._state(node)
        if len(state.derivs) > index:
            return True
        if state.exhausted:
            return False
        if id(node) in self._resolving:
            # Cyclic dependency: only already-materialized derivations
            # (checked above) may be used.
            return _BLOCKED
        if self._deriv(node, index) is not None:
            return True
        return _BLOCKED if state.tentative else False

    def _push_candidate(self, state, packed, i, j):
        key = (id(packed), i, j)
        if key in state.used:
            return
        cost = self._packed_base_cost(packed)
        left, right = packed.left, packed.right
        if isinstance(left, SymbolNode):
            r = self._has_deriv(left, i)
            if r is not True:
                if r is _BLOCKED:
                    state.pending[key] = packed
                else:
                    state.pending.pop(key, None)
                return
            cost += self._state(left).derivs[i][0]
        elif i:
            return
        if isinstance(right, SymbolNode):
            r = self._has_deriv(right, j)
            if r is not True:
                if r is _BLOCKED:
                    state.pending[key] = packed
                else:
                    state.pending.pop(key, None)
                return
            cost += self._state(right).derivs[j][0]
        elif j:
            return
        state.used.add(key)
        state.pending.pop(key, None)
        heapq.heappush(state.heap, (cost, next(self._seq), packed, i, j))

    def _deriv(self, node, k):
        """Ensure node has k+1 derivations; return the k-th (or None)."""
        state = self._state(node)
        self._resolving.add(id(node))
        try:
            while len(state.derivs) <= k and not state.exhausted:
                # (Re)seed: candidates that were blocked by a cyclic
                # dependency or an empty child may have become available.
                for packed in node.children:
                    self._push_candidate(state, packed, 0, 0)
                for key, packed in list(state.pending.items()):
                    self._push_candidate(state, packed, key[1], key[2])
                if not state.heap:
                    state.exhausted = not state.pending
                    state.tentative = bool(state.pending)
                    break
                cost, _, packed, i, j = heapq.heappop(state.heap)
                entry = (cost, packed,
                         self._entry_at(packed.left, i),
                         self._entry_at(packed.right, j))
                state.derivs.append(entry)
                self.nodes_expanded += 1
                self._push_candidate(state, packed, i + 1, j)
                self._push_candidate(state, packed, i, j + 1)
        finally:
            self._resolving.discard(id(node))
        if len(state.derivs) > k:
            return state.derivs[k]
        return None

    def _entry_at(self, child, index):
        # Candidates are only pushed after the child entries exist.
        if child is None:
            return None
        if isinstance(child, TokenNode):
            return child.token
        return self._state(child).derivs[index]

    ### Tree materialization and dedup

    def _materialize(self, deriv):
        _, packed, left_deriv, right_deriv = deriv
        children = []
        left = packed.left
        if left is not None:
            result = left_deriv if isinstance(left, TokenNode) else self._materialize(left_deriv)
            if left.is_intermediate:
                children.extend(result)
            else:
                children.append(result)
        right = packed.right
        if right is not None:
            result = right_deriv if isinstance(right, TokenNode) else self._materialize(right_deriv)
            children.append(result)
        if packed.parent.is_intermediate:
            return children
        callback = self.callbacks.get(packed.rule)
        if callback is not None:
            return callback(children)
        return self.tree_class(packed.rule.origin.name, children)

    def _tree_key(self, obj):
        if isinstance(obj, Token):
            return (Token, obj.type, str(obj), obj.start_pos, obj.end_pos)
        if isinstance(obj, Tree):
            return (obj.data, tuple(self._tree_key(child) for child in obj.children))
        try:
            hash(obj)
        except TypeError:
            return repr(obj)
        return obj

    ### Iterator protocol

    def __iter__(self):
        return self

    def __next__(self):
        if self._remaining is not None and self._remaining <= 0:
            raise StopIteration
        while True:
            deriv = self._deriv(self.root, self._root_index)
            if deriv is None:
                raise StopIteration
            self._root_index += 1
            tree = self._materialize(deriv)
            key = self._tree_key(tree)
            if key not in self._seen_trees:
                self._seen_trees.add(key)
                self.trees_yielded += 1
                if self._remaining is not None:
                    self._remaining -= 1
                return tree
            self.duplicates_skipped += 1

    def stats(self):
        """Return a dict of expansion statistics (see class attributes)."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'trees_yielded': self.trees_yielded,
            'duplicates_skipped': self.duplicates_skipped,
            'forest_nodes_expanded': len(self._states),
        }
