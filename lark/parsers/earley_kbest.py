"""This module implements a lazy k-best iterator over an SPPF.

Given an Earley parse forest (SPPF) and a cost for each grammar rule,
KBestForestIterator yields parse trees in order of increasing total
rule cost, expanding as little of the forest as possible. Iteration is
lazy: requesting only a few trees does not expand the whole forest.

The enumeration is a best-first search over the derivation space, in the
spirit of AO* / k-best hypergraph enumeration. Each symbol node of the
forest gets a lazy enumerator that materializes its derivations one at a
time, in non-decreasing cost order. Candidates whose exact cost is not
yet known carry a lower bound and are refined on demand, so nodes are
only expanded while they remain competitive.
"""

from heapq import heappush, heappop
from collections.abc import Mapping

from ..tree import Tree
from ..lexer import Token
from .earley_forest import TokenNode


class _CyclicPath(Exception):
    """Internal signal: a derivation path re-entered a node that is
    already being expanded, i.e. the path loops through an SPPF cycle."""


class _Derivation:
    """A single materialized derivation of a forest node."""
    __slots__ = ('cost', 'value')

    def __init__(self, cost, value):
        self.cost = cost
        self.value = value


class _TokenEnumerator:
    """Trivial enumerator for token (leaf) nodes: exactly one derivation."""
    active = False

    def __init__(self, node):
        self.derivations = [_Derivation(0, node.token)]
        self.exhausted = False

    def get(self, index):
        if index == 0:
            return self.derivations[0]
        self.exhausted = True
        return None

    def lower_bound(self, index):
        return 0 if index == 0 else None


class _SymbolNodeEnumerator:
    """Lazily enumerates the derivations of one SPPF symbol (or
    intermediate) node, in non-decreasing cost order.

    Derivations are combinations (packed_node, left_index, right_index).
    Heap candidates carry a key that is a lower bound on the combination's
    cost; a candidate is materialized only when it is popped with an exact
    key, which guarantees derivations come out in non-decreasing cost
    order without expanding uncompetitive parts of the forest.
    """

    def __init__(self, kbest, node):
        self.kbest = kbest
        self.node = node
        self.derivations = []
        self.heap = []
        self.active = False
        self.exhausted = False
        for packed in node.children:
            self._push_candidate(packed, 0, 0)

    def get(self, index):
        """Return the index-th cheapest derivation, or None if there
        are fewer than index + 1 derivations."""
        if self.active:
            # Re-entrant expansion: this node is an ancestor of itself in
            # the current search path, so the path is cyclic.
            raise _CyclicPath()
        if index < len(self.derivations):
            return self.derivations[index]
        if self.exhausted:
            return None
        self.active = True
        try:
            while len(self.derivations) <= index and not self.exhausted:
                self._advance()
        finally:
            self.active = False
        if index < len(self.derivations):
            return self.derivations[index]
        return None

    def lower_bound(self, index):
        """A lower bound on the cost of the index-th cheapest
        derivation, or None if that derivation cannot exist."""
        if index < len(self.derivations):
            return self.derivations[index].cost
        if self.exhausted:
            return None
        bound = 0
        if self.derivations:
            bound = self.derivations[-1].cost
        if self.heap:
            bound = max(bound, self.heap[0][0])
        return bound

    def _push_candidate(self, packed, left_index, right_index):
        kbest = self.kbest
        exact = self._exact_cost(packed, left_index, right_index)
        if exact is _DEAD:
            return
        if exact is not None:
            kbest._push(self.heap, (exact, 0, packed, left_index, right_index))
            return
        cost = kbest._packed_cost(packed)
        if packed.left is not None:
            bound = kbest._lower_bound(packed.left, left_index)
            if bound is None:
                return
            cost += bound
        if packed.right is not None:
            bound = kbest._lower_bound(packed.right, right_index)
            if bound is None:
                return
            cost += bound
        kbest._push(self.heap, (cost, 1, packed, left_index, right_index))

    def _exact_cost(self, packed, left_index, right_index):
        """The exact cost of a combination, None if some child derivation
        is not materialized yet, or _DEAD if it can never exist."""
        kbest = self.kbest
        cost = kbest._packed_cost(packed)
        if packed.left is not None:
            child = kbest._peek_enumerator(packed.left)
            if child is None or child.active:
                return None
            if left_index < len(child.derivations):
                cost += child.derivations[left_index].cost
            elif child.exhausted:
                return _DEAD
            else:
                return None
        if packed.right is not None:
            child = kbest._peek_enumerator(packed.right)
            if child is None or child.active:
                return None
            if right_index < len(child.derivations):
                cost += child.derivations[right_index].cost
            elif child.exhausted:
                return _DEAD
            else:
                return None
        return cost

    def _refine(self, packed, left_index, right_index):
        """Resolve the child derivations of a combination, expanding the
        forest as needed, and return the combination's exact cost (None
        if it can never exist)."""
        kbest = self.kbest
        cost = kbest._packed_cost(packed)
        if packed.left is not None:
            child = kbest._enumerator(packed.left)
            if child.active:
                raise _CyclicPath()
            if left_index >= len(child.derivations):
                if child.exhausted:
                    return None
                if child.get(left_index) is None:
                    return None
            bound = child.lower_bound(left_index)
            if bound is None:
                return None
            cost += bound
        if packed.right is not None:
            child = kbest._enumerator(packed.right)
            if child.active:
                raise _CyclicPath()
            if right_index >= len(child.derivations):
                if child.exhausted:
                    return None
                if child.get(right_index) is None:
                    return None
            bound = child.lower_bound(right_index)
            if bound is None:
                return None
            cost += bound
        return cost

    def _advance(self):
        if not self.heap:
            self.exhausted = True
            return
        kbest = self.kbest
        key, _, _, packed, left_index, right_index = heappop(self.heap)
        kbest.candidates_popped += 1
        exact = self._exact_cost(packed, left_index, right_index)
        if exact is _DEAD:
            # A child derivation this combination relies on cannot exist.
            return
        if exact is not None and exact == key:
            self.derivations.append(kbest._build_derivation(packed, left_index, right_index, exact))
            if packed.left is not None:
                self._push_candidate(packed, left_index + 1, right_index)
            if left_index == 0 and packed.right is not None:
                self._push_candidate(packed, left_index, right_index + 1)
            return
        try:
            bound = self._refine(packed, left_index, right_index)
        except _CyclicPath:
            # This combination loops through an SPPF cycle. Derivations
            # through it form an infinite family and are discarded.
            kbest.cycles_pruned += 1
            return
        if bound is not None:
            # After a successful refinement all child derivations are
            # materialized, so the bound is the exact cost.
            kbest._push(self.heap, (bound, 0, packed, left_index, right_index))


class _DeadType:
    pass


_DEAD = _DeadType()


class KBestForestIterator:
    """Lazily iterate the k best parse trees of an Earley SPPF, ordered
    by increasing total rule cost.

    The forest is expanded on demand: requesting only a few trees does
    not expand the whole forest. Progress can be inspected through the
    nodes_expanded, candidates_popped, cycles_pruned and
    duplicates_skipped counters.

    Parameters:
        root: The root SymbolNode of an SPPF, as returned by the Earley
            parser when ambiguity='forest'.
        rule_cost: Provides the non-negative cost of each grammar rule.
            Either a callable rule -> cost, or a mapping of rule name
            (or alias) to cost; rules missing from the mapping default
            to 0. The callable is invoked at most once per rule; costs
            are cached.
            If None, every rule costs 1, so trees with fewer rule
            applications come first. A cost that is negative or NaN
            raises ValueError.
        callbacks: A dict mapping rules to tree-building callbacks, as
            produced by ParseTreeBuilder.create_callback (this is
            what parser_conf.callbacks contains). Used to build the
            output trees, so tokens keep their positions as usual.
        tree_class: The tree class of the output (only used to detect
            tree values for deduplication).
        k: Maximum number of trees to yield. None means no limit; since
            cyclic derivations are discarded, iteration always terminates.

    Notes:
        - Derivations that loop through an SPPF cycle (infinite
          ambiguity, typically caused by nullable recursive rules) are
          discarded, consistent with ambiguity='resolve'. What
          remains is always a finite set of trees.
        - Trees that are equal in structure and token positions but are
          represented through different intermediate nodes are yielded
          only once.
        - Trees of equal cost are yielded in a stable, deterministic
          order (this relies on the parser's default
          ordered_sets=True).
        - Deduplication compares the produced trees. If callbacks
          contain a transformer that returns non-tree values, comparison
          falls back to object identity and duplicates may be yielded.
        - Expansion recurses along the forest depth, so extremely deep
          forests may require a higher recursion limit.
    """

    def __init__(self, root, rule_cost=None, callbacks=None, tree_class=Tree, k=None):
        if k is not None and k < 0:
            raise ValueError("k must be non-negative or None")
        if rule_cost is None:
            def rule_cost(rule):
                return 1
        elif isinstance(rule_cost, Mapping):
            mapping = rule_cost
            def rule_cost(rule):
                return mapping.get(rule.alias or rule.origin.name, 0)
        elif not callable(rule_cost):
            raise TypeError(
                "rule_cost must be a callable, a mapping of rule names to costs, or None")
        self._rule_cost = rule_cost
        self._cost_cache = {}
        self.callbacks = callbacks if callbacks is not None else {}
        self.tree_class = tree_class
        self.k = k
        self._enumerators = {}
        self._seq = 0
        self._cursor = 0
        self._seen = set()
        self._returned = 0
        self.last_cost = None
        # Expansion statistics
        self.nodes_expanded = 0
        self.candidates_popped = 0
        self.cycles_pruned = 0
        self.duplicates_skipped = 0
        self._root_enum = self._enumerator(root)

    @property
    def stats(self):
        """A snapshot of the laziness/expansion counters."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'candidates_popped': self.candidates_popped,
            'cycles_pruned': self.cycles_pruned,
            'duplicates_skipped': self.duplicates_skipped,
        }

    def _enumerator(self, node):
        """Return the (memoized) lazy enumerator for a forest node."""
        key = id(node)
        enum = self._enumerators.get(key)
        if enum is None:
            if isinstance(node, TokenNode):
                enum = _TokenEnumerator(node)
            else:
                enum = _SymbolNodeEnumerator(self, node)
            self._enumerators[key] = enum
            self.nodes_expanded += 1
        return enum

    def _peek_enumerator(self, node):
        """Return the enumerator for a forest node if it was already
        created, else None. Never creates new enumerators."""
        return self._enumerators.get(id(node))

    def _lower_bound(self, node, index):
        """A lower bound on the cost of a node's index-th derivation,
        without expanding the node. Unexpanded nodes get the trivial
        bound 0 (costs are non-negative)."""
        enum = self._peek_enumerator(node)
        if enum is None:
            return 0
        return enum.lower_bound(index)

    def _push(self, heap, entry):
        heappush(heap, (entry[0], entry[1], self._seq) + entry[2:])
        self._seq += 1

    def _rule_cost_of(self, rule):
        try:
            return self._cost_cache[rule]
        except KeyError:
            cost = self._rule_cost(rule)
            if not isinstance(cost, (int, float)) or isinstance(cost, bool):
                raise ValueError(
                    "rule cost must be a number, got %r for rule %s" % (cost, rule))
            if cost != cost or cost < 0:
                raise ValueError(
                    "rule cost must be non-negative, got %r for rule %s" % (cost, rule))
            self._cost_cache[rule] = cost
            return cost

    def _packed_cost(self, packed):
        # The cost of a rule is counted once per rule application, at the
        # packed node directly under the completed symbol node (the same
        # convention as ForestSumVisitor). Intermediate nodes are free.
        if packed.parent.is_intermediate:
            return 0
        return self._rule_cost_of(packed.rule)

    def _build_derivation(self, packed, left_index, right_index, cost):
        children = []
        if packed.left is not None:
            left_value = self._enumerator(packed.left).derivations[left_index].value
            if packed.left.is_intermediate:
                children += left_value
            else:
                children.append(left_value)
        if packed.right is not None:
            children.append(self._enumerator(packed.right).derivations[right_index].value)
        if packed.parent.is_intermediate:
            value = children
        else:
            value = self.callbacks[packed.rule](children)
        return _Derivation(cost, value)

    def _value_key(self, value):
        """A hashable key identifying a produced tree by structure and
        token positions, used to suppress duplicate outputs."""
        if isinstance(value, Token):
            return (value.type, str(value), value.start_pos, value.end_pos)
        if isinstance(value, self.tree_class):
            return (value.data, tuple(self._value_key(child) for child in value.children))
        # Not a tree (e.g. a transformer's output): cannot compare
        # structurally, fall back to identity.
        return ('_obj', id(value))

    def __iter__(self):
        return self

    def __next__(self):
        if self.k is not None and self._returned >= self.k:
            raise StopIteration
        while True:
            derivation = self._root_enum.get(self._cursor)
            if derivation is None:
                raise StopIteration
            self._cursor += 1
            key = self._value_key(derivation.value)
            if key in self._seen:
                self.duplicates_skipped += 1
                continue
            self._seen.add(key)
            self._returned += 1
            self.last_cost = derivation.cost
            return derivation.value

    def next_with_cost(self):
        """Like next(), but also returns the total rule cost of the
        tree. Useful for verifying the cost ordering."""
        if self.k is not None and self._returned >= self.k:
            raise StopIteration
        tree = next(self)
        return tree, self.last_cost

    def iter_with_cost(self):
        """Iterate over (tree, cost) pairs, in increasing cost."""
        while True:
            try:
                yield self.next_with_cost()
            except StopIteration:
                return
