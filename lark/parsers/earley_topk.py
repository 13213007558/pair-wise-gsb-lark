"""Lazy top-k enumeration of an Earley SPPF, ordered by rule cost.

The shared packed parse forest (SPPF) produced by the Earley parser
(``ambiguity='forest'``) may represent thousands of parse trees, so
expanding it fully and sorting the trees afterwards is not feasible.
``TopKForestEnumerator`` instead enumerates complete parse trees lazily,
in order of increasing rule cost, expanding only the combinations that are
needed to produce the requested trees.

The cost of a tree is the sum of the costs of its rule applications. Rule
costs are non-negative, and every application counts, even when a sub-forest
is shared between several derivations. Trees with equal cost are ordered
deterministically, using the stable identity of the rules (their position
in the grammar) and the branch index of each derivation - never object
addresses.

Forests that contain a cycle of zero-cost derivations (created by nullable
recursion) have infinitely many parse trees of the same minimal cost. In
that case a complete cost-ordered enumeration cannot be provided, and
``InfiniteAmbiguityError`` is raised. Cycles whose rules have a positive
cost are supported: only finitely many derivations fit below any cost
bound, so the lazy enumeration still terminates.
"""

from heapq import heappush, heappop
from itertools import count
from collections import namedtuple

from ..exceptions import InfiniteAmbiguityError
from ..tree import Tree
from .earley_forest import SymbolNode, TokenNode

#: A parse tree paired with its total rule cost.
ScoredTree = namedtuple('ScoredTree', ['cost', 'tree'])


class _Candidate:
    """A (possibly partial) derivation of a symbol node.

    ``packed`` is None while this node has not been expanded yet.
    ``kids`` holds one candidate per symbol-node child of ``packed``
    (token children need no expansion).
    """
    __slots__ = ('sym', 'packed', 'kids')

    def __init__(self, sym, packed, kids):
        self.sym = sym
        self.packed = packed
        self.kids = kids


def _symbol_children(packed):
    return [child for child in (packed.left, packed.right) if isinstance(child, SymbolNode)]


def _find_unexpanded(cand):
    """Return the path (tuple of kid indices) of the first unexpanded leaf
    of the candidate, in pre-order, or None if it is complete."""
    stack = [(cand, ())]
    while stack:
        node, path = stack.pop()
        if node.packed is None:
            return path
        for index in range(len(node.kids) - 1, -1, -1):
            stack.append((node.kids[index], path + (index,)))
    return None


def _candidate_at(cand, path):
    for index in path:
        cand = cand.kids[index]
    return cand


def _replace(cand, path, new):
    """Return a copy of ``cand`` with the sub-candidate at ``path`` replaced."""
    if not path:
        return new
    index = path[0]
    kids = list(cand.kids)
    kids[index] = _replace(kids[index], path[1:], new)
    return _Candidate(cand.sym, cand.packed, tuple(kids))


class TopKForestEnumerator:
    """Lazily enumerates the parse trees of an Earley forest, cheapest first.

    The enumerator is a one-shot iterator of ``ScoredTree(cost, tree)``.
    It only expands the combinations needed to produce the trees that are
    actually requested, so taking the first k trees of a highly ambiguous
    forest is cheap even when the forest represents exponentially many
    derivations.

    Parameters:
        root: The root ``SymbolNode`` of the forest, as returned by parsing
            with ``ambiguity='forest'``.
        rule_cost: The cost of each rule application. Either ``None`` (every
            rule costs 1), a callable ``rule -> cost``, or a mapping keyed by
            ``Rule``, rule alias, or rule name. Costs must be non-negative.
        default_cost: The cost of rules missing from a ``rule_cost`` mapping.
        tree_class: The tree class used to build the output trees.
        dedup: If True, derivations that produce an identical tree are
            emitted only once. If False, every derivation is emitted, even
            when it yields the same text and tree as another one.

    Attributes:
        expanded: How many candidates have been expanded so far.
        generated: How many candidates have been generated so far.
        yielded: How many trees have been emitted so far.

    Raises:
        InfiniteAmbiguityError: If the forest contains a cycle of zero-cost
            derivations, in which case infinitely many trees share the same
            minimal cost and the complete ordering cannot be provided.
        ValueError: If a rule cost is negative.
    """

    def __init__(self, root, rule_cost=None, default_cost=0, tree_class=Tree, dedup=False):
        if not isinstance(root, SymbolNode):
            raise TypeError("Expected a SymbolNode forest root, got %r" % (root,))
        self._root = root
        self._tree_class = tree_class
        self._dedup = dedup
        self._cost_of = self._make_cost_fn(rule_cost, default_cost)
        self.expanded = 0
        self.generated = 0
        self.yielded = 0
        self._symbol_nodes = []
        self._alternatives = {}     # id(symbol node) -> [PackedNode, ...] in stable branch order
        self._mincost = {}          # id(symbol node) -> cost of its cheapest derivation
        self._packed_weight = {}    # id(packed node) -> rule cost + mincost of its symbol children
        self._collect(root)
        self._compute_mincost()
        self._check_zero_cost_cycles()
        self._heap = None
        self._counter = None
        self._seen = None

    @staticmethod
    def _make_cost_fn(rule_cost, default_cost):
        if rule_cost is None:
            fn = lambda rule: 1
        elif callable(rule_cost):
            fn = rule_cost
        else:
            def fn(rule):
                for key in (rule, rule.alias, rule.origin.name):
                    if key in rule_cost:
                        return rule_cost[key]
                return default_cost

        def checked(rule):
            cost = fn(rule)
            if cost < 0:
                raise ValueError("Rule costs must be non-negative, got %r for rule %r" % (cost, rule))
            return cost
        return checked

    @staticmethod
    def _alternative_key(packed):
        """A deterministic, address-independent ordering for the packed
        children of a symbol node. The position of a packed node in this
        order is its branch index."""
        def span(node):
            if node is None:
                return (-1, -1)
            if isinstance(node, TokenNode):
                token = node.token
                start = token.start_pos if token.start_pos is not None else -1
                end = token.end_pos if token.end_pos is not None else -1
                return (start, end)
            return (node.start, node.end)
        return (packed.is_empty, -packed.priority, packed.rule.order, span(packed.left), span(packed.right))

    def _packed_rule_cost(self, packed):
        # The rules of packed nodes below intermediate nodes are counted at
        # their completing packed node, so each rule application counts once.
        if packed.parent.is_intermediate:
            return 0
        return self._cost_of(packed.rule)

    def _collect(self, root):
        seen = set()
        stack = [root]
        while stack:
            node = stack.pop()
            if isinstance(node, TokenNode) or id(node) in seen:
                continue
            seen.add(id(node))
            self._symbol_nodes.append(node)
            alternatives = sorted(node.children, key=self._alternative_key)
            self._alternatives[id(node)] = alternatives
            for packed in alternatives:
                for child in (packed.left, packed.right):
                    if child is not None:
                        stack.append(child)

    def _compute_mincost(self):
        # Least fixpoint of: mincost(n) = min over alternatives of
        # (rule cost + sum of children's mincost). Iterating upwards from 0
        # is exact for non-negative costs, even in the presence of cycles,
        # because a cheapest derivation never needs to traverse a cycle.
        mincost = self._mincost = {id(node): 0 for node in self._symbol_nodes}
        weight = self._packed_weight
        while True:
            changed = False
            for node in self._symbol_nodes:
                best = None
                for packed in self._alternatives[id(node)]:
                    w = self._packed_rule_cost(packed)
                    for child in _symbol_children(packed):
                        w += mincost[id(child)]
                    weight[id(packed)] = w
                    if best is None or w < best:
                        best = w
                if best is not None and best > mincost[id(node)]:
                    mincost[id(node)] = best
                    changed = True
            if not changed:
                break

    def _zero_weight_successors(self, node):
        for packed in self._alternatives[id(node)]:
            if self._packed_weight[id(packed)] == 0:
                for child in _symbol_children(packed):
                    yield child

    def _check_zero_cost_cycles(self):
        # A cycle made of zero-weight packed nodes can be traversed over and
        # over at zero additional cost, producing infinitely many derivations
        # of the same minimal cost.
        white, gray, black = 0, 1, 2
        color = {id(node): white for node in self._symbol_nodes}
        for root in self._symbol_nodes:
            if color[id(root)] != white:
                continue
            color[id(root)] = gray
            stack = [(root, iter(self._zero_weight_successors(root)))]
            while stack:
                node, successors = stack[-1]
                for successor in successors:
                    successor_color = color[id(successor)]
                    if successor_color == gray:
                        raise InfiniteAmbiguityError(
                            "The forest contains a cycle of zero-cost derivations (nullable recursion), "
                            "so it has infinitely many parse trees of the same minimal cost and a "
                            "complete cost-ordered enumeration cannot be provided. "
                            "Assign a positive cost to the rules involved in the cycle to enumerate "
                            "this forest.")
                    if successor_color == white:
                        color[id(successor)] = gray
                        stack.append((successor, iter(self._zero_weight_successors(successor))))
                        break
                else:
                    color[id(node)] = black
                    stack.pop()

    def __iter__(self):
        return self

    def __next__(self):
        if self._heap is None:
            self._heap = [(self._mincost[id(self._root)], (), 0, _Candidate(self._root, None, ()))]
            self._counter = count(1)
            self._seen = set()
        heap = self._heap
        while heap:
            bound, seq, _, cand = heappop(heap)
            path = _find_unexpanded(cand)
            if path is None:
                tree = self._build_tree(cand)
                if self._dedup:
                    if tree in self._seen:
                        continue
                    self._seen.add(tree)
                self.yielded += 1
                return ScoredTree(bound, tree)
            self.expanded += 1
            leaf = _candidate_at(cand, path)
            for branch_index, packed in enumerate(self._alternatives[id(leaf.sym)]):
                kids = tuple(_Candidate(child, None, ()) for child in _symbol_children(packed))
                new_cand = _replace(cand, path, _Candidate(leaf.sym, packed, kids))
                new_bound = bound - self._mincost[id(leaf.sym)] + self._packed_weight[id(packed)]
                new_seq = seq + ((packed.rule.order, branch_index),)
                heappush(heap, (new_bound, new_seq, next(self._counter), new_cand))
                self.generated += 1
        raise StopIteration

    def take(self, k):
        """Return the next ``k`` cheapest ``ScoredTree(cost, tree)`` pairs.

        Fewer than ``k`` pairs are returned when the forest has fewer
        remaining derivations.
        """
        if k < 0:
            raise ValueError("k must be non-negative, got %r" % (k,))
        result = []
        for _ in range(k):
            try:
                result.append(next(self))
            except StopIteration:
                break
        return result

    def _build_tree(self, cand):
        # Returns a tree for symbol nodes, a flat list of children for
        # intermediate nodes.
        packed = cand.packed
        kids = iter(cand.kids)
        children = []
        for child in (packed.left, packed.right):
            if child is None:
                continue
            if isinstance(child, TokenNode):
                children.append(child.token)
            elif child.is_intermediate:
                children.extend(self._build_tree(next(kids)))
            else:
                children.append(self._build_tree(next(kids)))
        if cand.sym.is_intermediate:
            return children
        rule = packed.rule
        name = rule.alias or rule.options.template_source or rule.origin.name
        return self._tree_class(name, children)
