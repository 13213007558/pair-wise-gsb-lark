"""Source-preserving parsing and reconstruction.

This module implements a "preserve the original text" mode on top of an
unambiguous LALR parser. Where lark.reconstruct.Reconstructor regenerates
the whole text from the grammar (normalizing whitespace and dropping
%ignored text such as comments), the class here only regenerates the parts
of the tree that lost their correspondence to the original source. Every
other byte - including whitespace, blank lines and comments - is emitted
exactly as it appeared in the input.

Usage::

    parser = Lark(grammar, parser='lalr', propagate_positions=True, maybe_placeholders=False)
    pr = PreservingReconstructor(parser, term_subs={'WS_NL': lambda _: '\n'})

    tree = pr.parse(source)          # tree annotated with spans & ignored text
    # ... edit the tree: replace token values or whole subtrees ...
    new_source = pr.reconstruct(tree)

Limitations (first version):

- The parser must be an unambiguous LALR parser (parser='lalr') created
  with propagate_positions=True.
- Arbitrary tree *moves* are not inferred: a preserved subtree must stay
  under its original parent, in its original order. Detached nodes are
  fully regenerated; moved preserved nodes raise MovedNodeError.
- Tokens whose value was transformed by a lexer callback no longer match
  the source text, so they are re-emitted from their (transformed) value.
- Text of discarded terminals (e.g. string literals) is preserved through
  the spans of surrounding kept nodes; a tree with no kept tokens at all
  cannot be preserved and is regenerated from scratch.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import ConfigurationError, LarkError
from .grammar import Symbol
from .lark import Lark
from .lexer import Token
from .reconstruct import MissingTerminalGenerator, Reconstructor
from .tree import ParseTree, Tree
from .utils import is_id_continue

Span = Tuple[int, int]


class SourcePreservationError(LarkError):
    "Base class for errors raised by the source-preserving reconstructor."


class MovedNodeError(SourcePreservationError):
    "Raised when a preserved node appears at a position that doesn't match its original span."


def _span_of(node: Any) -> Optional[Span]:
    "Return the (start, end) character offsets of a parsed node, or None."
    if isinstance(node, Token):
        if node.start_pos is not None and node.end_pos is not None:
            return (node.start_pos, node.end_pos)
        return None
    if isinstance(node, Tree):
        start = getattr(node.meta, 'start_pos', None)
        end = getattr(node.meta, 'end_pos', None)
        if start is not None and end is not None:
            return (start, end)
    return None


def _is_preserved(node: Any, source: str) -> bool:
    "True if the node still carries the annotations of a parse of source."
    if isinstance(node, Token):
        return node.start_pos is not None and node.end_pos is not None
    if isinstance(node, Tree):
        return getattr(node.meta, 'source', None) is source
    return False


class PreservingReconstructor:
    """Parses text and reconstructs it, preserving untouched parts byte-for-byte.

    Parameters:
        parser: a Lark instance created with parser='lalr' and
            propagate_positions=True (maybe_placeholders=False is recommended,
            so that empty rules show up as empty trees).
        term_subs: a dictionary of [Terminal name as str] to a callable
            returning the text to generate for that terminal. Used only for
            terminals that have no explicit textual representation (e.g.
            discarded regexp terminals) when generating *new* nodes.
    """

    def __init__(self, parser: Lark, term_subs: Optional[Dict[str, Callable[[Symbol], str]]] = None) -> None:
        if parser.options.parser != 'lalr':
            raise ConfigurationError(
                "PreservingReconstructor requires an unambiguous LALR parser (parser='lalr'), "
                "got parser=%r" % parser.options.parser)
        if parser.options.propagate_positions is not True:
            raise ConfigurationError(
                "PreservingReconstructor requires the parser to be created with propagate_positions=True")
        self._parser = parser
        self._reconstructor = Reconstructor(parser, term_subs)

    def parse(self, text: str, start: Optional[str] = None) -> ParseTree:
        """Parse text and return a tree annotated for preserving reconstruction.

        Every rule node keeps, on its meta:

        - source: the original text (shared reference)
        - start_pos / end_pos: the character span of the rule
        - child_spans: the span of each child (None for zero-width children)
        - ignored_gaps: the ignored/discarded text found before each child
        - ignored_tail: the ignored/discarded text after the last child

        Tokens in the tree are the original lexer tokens, with their offsets.
        """
        tree = self._parser.parse(text, start=start)
        self._annotate(tree, text)
        return tree

    def _annotate(self, node: ParseTree, source: str) -> None:
        meta = node.meta
        meta.source = source
        cursor = getattr(meta, 'start_pos', None)
        spans: List[Optional[Span]] = []
        gaps: List[str] = []
        for child in node.children:
            span = _span_of(child)
            spans.append(span)
            if span is not None and cursor is not None:
                gaps.append(source[cursor:span[0]])
                cursor = span[1]
            else:
                gaps.append('')
            if isinstance(child, Tree):
                self._annotate(child, source)
        meta.child_spans = spans
        meta.ignored_gaps = gaps
        end = getattr(meta, 'end_pos', None)
        meta.ignored_tail = source[cursor:end] if cursor is not None and end is not None else ''

    def reconstruct(self, tree: ParseTree, insert_spaces: bool = True) -> str:
        """Rebuild the text for tree, preserving every byte that still
        corresponds to the original parse, and generating the rest.

        New nodes adjacent to preserved text are separated by a space when
        needed to keep the output re-parseable (unless insert_spaces is
        disabled).
        """
        source = getattr(tree.meta, 'source', None)
        if source is None:
            raise SourcePreservationError(
                "This tree was not produced by PreservingReconstructor.parse(); "
                "use lark.reconstruct.Reconstructor to generate text from scratch")
        root_span = _span_of(tree)
        if root_span is None:
            # The tree contains no kept tokens to anchor to (e.g. only
            # discarded literals and empty rules): regenerate from scratch.
            return self._reconstructor.reconstruct(tree)
        segments: List[Tuple[str, bool]] = [(source[:root_span[0]], False)]
        self._emit_tree(tree, source, segments, root_span[0])
        segments.append((source[root_span[1]:], False))
        return self._join(segments, insert_spaces)

    ### Implementation

    def _emit_tree(self, node: ParseTree, source: str, segments: List[Tuple[str, bool]], pos_hint: int) -> None:
        meta = node.meta
        span = _span_of(node)
        if span is None:
            # A preserved zero-width node (e.g. an empty rule) has no text of
            # its own, but the caller may have added new children to it.
            for child in node.children:
                if child is None:
                    continue
                if _is_preserved(child, source):
                    self._emit_tree(child, source, segments, pos_hint)
                else:
                    self._emit_generated(child, source, pos_hint, node, segments)
            return

        orig_spans: List[Optional[Span]] = meta.child_spans
        orig_span_set = {s for s in orig_spans if s is not None}

        # Classify the current children and detect which original spans they claim.
        claimed = set()
        plan = []  # (kind, child, span)
        for child in node.children:
            if child is None:
                continue  # maybe_placeholders empty slot: zero-width
            child_span = _span_of(child)
            if child_span is None:
                if _is_preserved(child, source):
                    plan.append(('empty', child, None))  # preserved empty rule: zero-width
                else:
                    plan.append(('generated', child, None))
            elif child_span in orig_span_set and _is_preserved(child, source):
                claimed.add(child_span)
                plan.append(('preserved', child, child_span))
            elif _is_preserved(child, source) and isinstance(child, Tree):
                raise MovedNodeError(
                    "Preserved subtree %r (originally at characters %d-%d) was moved; "
                    "arbitrary tree moves are not supported. Replace it with a fresh "
                    "Tree to have it regenerated." % (child.data, child_span[0], child_span[1]))
            else:
                # A token detached from its original position regenerates as its value.
                plan.append(('generated', child, None))

        deleted_spans = sorted(s for s in orig_span_set if s not in claimed)

        cursor = span[0]
        for index, (kind, child, child_span) in enumerate(plan):
            if kind == 'preserved':
                assert child_span is not None
                if child_span[0] < cursor:
                    raise MovedNodeError(
                        "Preserved node %r (characters %d-%d) appears out of order; "
                        "reordering preserved nodes is not supported."
                        % (getattr(child, 'data', child), child_span[0], child_span[1]))
                self._emit_gap(source, cursor, child_span[0], deleted_spans, segments)
                self._emit_preserved(child, source, segments)
                cursor = child_span[1]
            elif kind == 'empty':
                self._emit_tree(child, source, segments, cursor)
            else:
                # A new node. If it stands in for a deleted original child,
                # place it where the deleted child was: emit the ignored text
                # up to that spot first, then continue after it.
                next_preserved = next(
                    (s for _k, _c, s in plan[index + 1:] if _k == 'preserved'),
                    None)
                limit = next_preserved[0] if next_preserved is not None else span[1]
                target = next(
                    (d for d in deleted_spans if d[0] >= cursor and d[1] <= limit),
                    None)
                if target is not None:
                    self._emit_gap(source, cursor, target[0], deleted_spans, segments)
                    self._emit_generated(child, source, target[0], node, segments)
                    cursor = target[1]
                    deleted_spans.remove(target)
                else:
                    self._emit_generated(child, source, cursor, node, segments)
        self._emit_gap(source, cursor, span[1], deleted_spans, segments)

    @staticmethod
    def _emit_gap(source: str, start: int, end: int, deleted_spans: List[Span],
                  segments: List[Tuple[str, bool]]) -> None:
        "Emit source[start:end], skipping the spans of deleted children."
        pos = start
        for span_start, span_end in deleted_spans:
            if start <= span_start and span_end <= end and span_start >= pos:
                if span_start > pos:
                    segments.append((source[pos:span_start], False))
                pos = span_end
        if end > pos:
            segments.append((source[pos:end], False))

    def _emit_preserved(self, node: Any, source: str, segments: List[Tuple[str, bool]]) -> None:
        if isinstance(node, Token):
            span = _span_of(node)
            assert span is not None
            original = source[span[0]:span[1]]
            if original == node:
                segments.append((original, False))
            else:
                # The token value was modified in place: regenerate it.
                segments.append((str(node), True))
        else:
            child_span = _span_of(node)
            assert child_span is not None
            self._emit_tree(node, source, segments, child_span[0])

    def _emit_generated(self, node: Any, source: str, pos: int, parent: ParseTree,
                        segments: List[Tuple[str, bool]]) -> None:
        if isinstance(node, Token):
            segments.append((str(node), True))
            return
        if not isinstance(node, Tree):
            raise SourcePreservationError(
                "Cannot reconstruct child %r of rule %r: expected a Token or a Tree"
                % (node, parent.data))
        try:
            text = self._reconstructor.reconstruct(node)
        except MissingTerminalGenerator as e:
            line = source.count('\n', 0, pos) + 1
            line_start = source.rfind('\n', 0, pos) + 1
            column = pos - line_start + 1
            raise SourcePreservationError(
                "Cannot reconstruct new node for rule %r inside rule %r "
                "(inserted at line %d, column %d): terminal %r has no textual "
                "representation. Provide a generator for it via term_subs."
                % (node.data, parent.data, line, column, e.terminal)) from e
        segments.append((text, True))

    @staticmethod
    def _join(segments: List[Tuple[str, bool]], insert_spaces: bool) -> str:
        out = []
        prev = ''
        for text, _generated in segments:
            if not text:
                continue
            if insert_spaces and prev and is_id_continue(prev[-1]) and is_id_continue(text[0]):
                out.append(' ')
            out.append(text)
            prev = text
        return ''.join(out)
