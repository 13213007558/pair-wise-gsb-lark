"""Text-preserving parse and reconstruction.

Unlike :class:`~lark.reconstruct.Reconstructor`, which regenerates every byte of a
modified tree, this module keeps the original source text and only regenerates the
parts of the tree that no longer correspond to it.

Usage::

    pr = TextPreservingReconstructor(parser)
    tree = pr.parse(text)          # the parser records source positions
    tree.children[0] = ...        # replace a leaf or a whole subtree
    out = pr.reconstruct(tree)    # unchanged bytes (incl. comments) are kept

Only unambiguous LALR(1) parsing is supported. The first version does not try to
infer arbitrary tree moves; replacing values and subtrees is supported.
"""

from typing import Callable, Dict, List, Optional

from .lark import Lark
from .tree import ParseTree, Tree
from .lexer import Token, PatternStr
from .grammar import Terminal
from .exceptions import ConfigurationError, LarkError
from .utils import is_id_continue
from .reconstruct import Reconstructor, WriteTokensTransformer, is_iter_empty
from .tree_matcher import is_discarded_terminal, parse_rulename


class MissingTerminalGeneratorError(LarkError):
    """Raised when reconstruction needs to generate a terminal that has no known
    textual representation and no generator was supplied via ``term_subs``."""


class _Placeholder:
    """A discarded terminal that the (reduced) tree does not contain, which must
    be emitted during reconstruction. May later adopt an original token."""
    __slots__ = ('terminal', 'rule', 'line', 'column', 'index', 'text')

    def __init__(self, terminal: str, rule: str, line: Optional[int], column: Optional[int]) -> None:
        self.terminal = terminal
        self.rule = rule
        self.line = line
        self.column = column
        self.index: Optional[int] = None
        self.text: Optional[str] = None

    def adopt(self, index: int, token: Token) -> None:
        self.index = index
        self.text = str(token)


class _Piece:
    """A single emitted unit in the flat reconstruction stream.

    ``index`` is the index of the corresponding token in the recorded token
    stream, or ``None`` if this piece has no original counterpart."""
    __slots__ = ('index', 'text', 'terminal')

    def __init__(self, text: str, index: Optional[int], terminal: Optional[str] = None) -> None:
        self.text = text
        self.index = index
        self.terminal = terminal

    def adopt(self, index: int, token: Token) -> None:
        # A replacement token keeps its new text, but inherits the original
        # token's position so its surrounding bytes are preserved.
        self.index = index


class _PreservingWriteTokens(WriteTokensTransformer):
    """Like WriteTokensTransformer, but discarded terminals become _Placeholder
    objects instead of being resolved to text immediately."""

    def __default__(self, data, children, meta):
        if not getattr(meta, 'match_tree', False):
            return Tree(data, children)

        iter_args = iter(children)
        to_write = []
        for sym in meta.orig_expansion:
            if is_discarded_terminal(sym):
                to_write.append(_Placeholder(sym.name, data, getattr(meta, 'line', None),
                                             getattr(meta, 'column', None)))
            else:
                x = next(iter_args)
                if isinstance(x, list):
                    to_write += x
                else:
                    if isinstance(x, Token):
                        assert Terminal(x.type) == sym, x
                    else:
                        name, _args = parse_rulename(sym.name)
                        assert x.data == name, (sym, x)
                        x.meta.orig_rulename = sym.name
                    to_write.append(x)

        assert is_iter_empty(iter_args)
        return to_write


class TextPreservingReconstructor(Reconstructor):
    """Parse text and reconstruct it after tree edits, preserving all unchanged bytes.

    The reconstructor associates every parse result with the original tokens,
    the text the lexer ignored (``%ignore`` terminals such as whitespace and
    comments) and rule spans. When :meth:`reconstruct` is called after a leaf
    value or whole subtree has been replaced, only the parts without an original
    counterpart are generated; every other byte, including ignored text, is
    copied from the source verbatim. Newly generated nodes are separated from
    neighbouring original text so that the result stays a valid, re-parseable
    input.

    Parameters:
        parser: A Lark instance built with ``parser='lalr'``.
        term_subs: Maps terminal names to generators, ``Callable[[Terminal], str]``,
            used when a discarded terminal without a fixed textual representation
            (e.g. a regexp terminal) has to be generated for a node without an
            original counterpart.

    Raises:
        ConfigurationError: If the parser is not a LALR parser.
        MissingTerminalGeneratorError: If a terminal without a fixed
            representation must be generated and no generator was provided.
    """

    _source: Optional[str]
    _stream: List[Token]
    _token_ids: Dict[int, int]

    def __init__(self, parser: Lark,
                 term_subs: Optional[Dict[str, Callable[[Terminal], str]]] = None) -> None:
        if parser.options.parser != 'lalr':
            raise ConfigurationError(
                "TextPreservingReconstructor only supports unambiguous LALR(1) parsing "
                "(got parser=%r)." % parser.options.parser)
        if parser.options.maybe_placeholders:
            raise ConfigurationError(
                "TextPreservingReconstructor requires maybe_placeholders=False "
                "(pass maybe_placeholders=False to Lark).")
        super().__init__(parser, term_subs or {})
        self.write_tokens = _PreservingWriteTokens({t.name: t for t in self.tokens}, term_subs or {})
        self._source = None
        self._stream = []
        self._token_ids = {}

    def parse(self, text: str, start: Optional[str] = None) -> ParseTree:
        """Parse ``text``, recording every token (including terminals discarded
        from the tree) together with its source span, and return the tree.

        The returned tree can be edited in place; :meth:`reconstruct` will reuse
        the original bytes for everything that was left untouched.
        """
        interactive = self.parser.parse_interactive(text, start=start)
        stream: List[Token] = []
        for token in interactive.iter_parse():
            stream.append(token)
        tree = interactive.feed_eof(stream[-1] if stream else None)

        self._source = text
        self._stream = stream
        self._token_ids = {id(token): i for i, token in enumerate(stream)}
        tree.meta.preserve_state = (text, stream, self._token_ids)  # type: ignore[attr-defined]
        return tree

    def _piece_for_token(self, token: Token, state) -> _Piece:
        source, stream, token_ids = state
        index = token_ids.get(id(token))
        if index is None:
            # A genuinely new token, created by the caller.
            return _Piece(str(token), None, token.type)

        original = stream[index]
        if str(token) == str(original):
            # An untouched original token.
            return _Piece(source[original.start_pos:original.end_pos], index, token.type)

        # The original token object was mutated in place (e.g. Token.update).
        # Keep its index so its surrounding bytes (comments, whitespace) survive.
        return _Piece(str(token), index, token.type)

    def _emit(self, tree: ParseTree, state, orig_rulename: Optional[str] = None):
        unreduced_tree = self.match_tree(tree, orig_rulename or tree.data)
        res = self.write_tokens.transform(unreduced_tree)
        for item in res:
            if isinstance(item, Tree):
                yield from self._emit(item, state, getattr(item.meta, 'orig_rulename', None))
            elif isinstance(item, _Placeholder):
                yield item
            else:
                yield self._piece_for_token(item, state)

    def _adopt_run(self, pieces, candidates, offset) -> None:
        """Try to map each replacement piece onto an original token of the same
        terminal type, preserving the tokens' surrounding bytes. Greedy and
        in-order: a piece that has no matching candidate is left without an
        original counterpart."""
        ci = 0
        for piece in pieces:
            wanted = piece.terminal
            k = ci
            while k < len(candidates) and candidates[k].type != wanted:
                k += 1
            if k < len(candidates):
                piece.adopt(offset + k, candidates[k])
                ci = k + 1

    def _adopt(self, pieces, state) -> None:
        source, stream, token_ids = state
        anchors = [k for k, piece in enumerate(pieces)
                   if not isinstance(piece, _Placeholder) and piece.index is not None]
        if not anchors:
            # No original kept token survives (e.g. the tree consists only of
            # discarded terminals, or was rebuilt entirely). Let the pieces
            # adopt same-typed original tokens, so their layout is inherited.
            self._adopt_run(pieces, stream, 0)
            return

        # New pieces inserted before the first / after the last original token.
        first = anchors[0]
        last = anchors[-1]
        self._adopt_run(pieces[:first], stream[:pieces[first].index], 0)
        tail_start = pieces[last].index + 1
        self._adopt_run(pieces[last + 1:], stream[tail_start:], tail_start)

        for left, right in zip(anchors, anchors[1:]):
            i = pieces[left].index
            j = pieces[right].index
            if j <= i:
                # Out-of-order indices: the tree was rearranged, which is not
                # supported in this version.
                continue
            self._adopt_run(pieces[left + 1:right], stream[i + 1:j], i + 1)

    def _resolve_placeholder(self, placeholder: _Placeholder, pieces, pos: int, state) -> str:
        if placeholder.text is not None:
            return placeholder.text

        if placeholder.terminal in self.write_tokens.term_subs:
            return self.write_tokens.term_subs[placeholder.terminal](Terminal(placeholder.terminal))

        terminal_def = self.write_tokens.tokens.get(placeholder.terminal)
        if terminal_def is not None and isinstance(terminal_def.pattern, PatternStr):
            return terminal_def.pattern.value

        where = "in rule %r" % placeholder.rule
        if placeholder.line is not None:
            where += " at line %s, column %s" % (placeholder.line, placeholder.column)
        else:
            # Fall back to the position of the nearest original token emitted
            # around this placeholder.
            stream = state[1]
            for k in range(pos - 1, -1, -1):
                index = getattr(pieces[k], 'index', None)
                if index is not None:
                    token = stream[index]
                    where += ", right after %r at line %s, column %s" % (
                        str(token), token.line, token.column)
                    break
            else:
                for k in range(pos + 1, len(pieces)):
                    index = getattr(pieces[k], 'index', None)
                    if index is not None:
                        token = stream[index]
                        where += ", right before %r at line %s, column %s" % (
                            str(token), token.line, token.column)
                        break
        raise MissingTerminalGeneratorError(
            "Cannot generate terminal %r %s: it has no fixed textual representation "
            "and no generator was provided. Pass a generator as "
            "term_subs[%r]." % (placeholder.terminal, where, placeholder.terminal))

    def reconstruct(self, tree: ParseTree) -> str:
        """Rebuild the source text of ``tree``, copying every byte that still
        corresponds to the original parse and regenerating only replaced or
        inserted parts."""
        state = getattr(tree.meta, 'preserve_state', None)
        if state is None:
            state = (self._source, self._stream, self._token_ids)
        if state[0] is None:
            raise ConfigurationError(
                "Nothing to reconstruct from: call TextPreservingReconstructor.parse(text) "
                "on this tree first, or use Reconstructor for trees built from scratch.")
        source, stream, token_ids = state

        raw_pieces = list(self._emit(tree, state))
        self._adopt(raw_pieces, state)

        pieces: List[_Piece] = []
        for pos, piece in enumerate(raw_pieces):
            if isinstance(piece, _Placeholder):
                text = self._resolve_placeholder(piece, raw_pieces, pos, state)
                pieces.append(_Piece(text, piece.index, piece.terminal))
            else:
                pieces.append(piece)

        output: List[str] = []
        if pieces and pieces[0].index == 0:
            output.append(source[:stream[0].start_pos])

        previous: Optional[_Piece] = None
        for piece in pieces:
            if previous is not None:
                if previous.index is not None and piece.index is not None and piece.index == previous.index + 1:
                    output.append(source[stream[previous.index].end_pos:stream[piece.index].start_pos])
                elif previous.text and piece.text \
                        and is_id_continue(previous.text[-1]) and is_id_continue(piece.text[0]):
                    output.append(' ')
            output.append(piece.text)
            previous = piece

        if pieces and pieces[-1].index == len(stream) - 1:
            output.append(source[stream[-1].end_pos:])

        return ''.join(output)
