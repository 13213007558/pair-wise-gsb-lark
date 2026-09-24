Working with the SPPF
=====================

When parsing with Earley, Lark provides the ``ambiguity='forest'`` option
to obtain the shared packed parse forest (SPPF) produced by the parser as
an alternative to it being automatically converted to a tree.

Lark provides a few tools to facilitate working with the SPPF. Here are some
things to consider when deciding whether or not to use the SPPF.

**Pros**

- Efficient storage of highly ambiguous parses
- Precise handling of ambiguities
- Custom rule prioritizers
- Ability to handle infinite ambiguities
- Directly transform forest -> object instead of forest -> tree -> object

**Cons**

- More complex than working with a tree
- SPPF may contain nodes corresponding to rules generated internally
- Loss of Lark grammar features:

  - Rules starting with '_' are not inlined in the SPPF
  - Rules starting with '?' are never inlined in the SPPF
  - All tokens will appear in the SPPF

SymbolNode
----------

.. autoclass:: lark.parsers.earley_forest.SymbolNode
   :members: is_ambiguous, children, iter_topk

PackedNode
----------

.. autoclass:: lark.parsers.earley_forest.PackedNode
   :members: children

ForestVisitor
-------------

.. autoclass:: lark.parsers.earley_forest.ForestVisitor
   :members: visit, visit_symbol_node_in, visit_symbol_node_out,
             visit_packed_node_in, visit_packed_node_out,
             visit_token_node, on_cycle, get_cycle_in_path

ForestTransformer
-----------------

.. autoclass:: lark.parsers.earley_forest.ForestTransformer
   :members: transform, transform_symbol_node, transform_intermediate_node,
             transform_packed_node, transform_token_node

TreeForestTransformer
---------------------

.. autoclass:: lark.parsers.earley_forest.TreeForestTransformer
   :members: __default__, __default_token__, __default_ambig__

handles_ambiguity
-----------------

.. autofunction:: lark.parsers.earley_forest.handles_ambiguity

TopKForestEnumerator
--------------------

The forest may represent thousands of parse trees, so expanding it fully
and sorting the trees afterwards is not always feasible.
``TopKForestEnumerator`` enumerates complete parse trees lazily, in order
of increasing rule cost, expanding only the combinations needed to produce
the requested trees. The cost of a tree is the sum of the costs of its rule
applications.

If the forest contains a cycle of zero-cost derivations (created by nullable
recursion), it has infinitely many parse trees of the same minimal cost, and
``lark.exceptions.InfiniteAmbiguityError`` is raised.

.. autoclass:: lark.parsers.earley_topk.TopKForestEnumerator
   :members: take
