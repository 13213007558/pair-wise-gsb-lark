"""Incremental parsing sessions for LALR parsers.

This module provides IncrementalParseSession, an explicit incremental
parsing session on top of an existing parser='lalr' Lark instance.

The session is fed a token stream with *stable identities* (the same logical
token keeps the same Token object across edits) plus a contiguous replacement
range describing the edit. It keeps resumable parse checkpoints from the
previous parse, resumes from the earliest affected token, and reuses results
of the previous parse for any rule whose inputs (the parse stack context and
the identities of the tokens it spans) are unchanged.

.. note::
    Incremental *lexing* (mapping character edits to token edits) is out of
    scope. The caller is responsible for producing the new token sequence and
    for describing the edit as a replacement range. Every token that changed
    in any way (type, value, or position) must be covered by the replacement
    range; tokens outside the range are reused by identity.

Guarantees:

- The result (tree, transformed value, and positions) is identical to a full
  re-parse of the same token sequence.
- Rule callbacks are only re-executed for rules whose subtree spans the edit.
  session.last_stats exposes the counts.
- On invalid input the session raises the usual UnexpectedToken diagnostics,
  and the *same session* can be used to continue parsing after the input is
  fixed.

Limitations (first version):

- Only parser='lalr' is supported.
- Reuse requires the transformer/callbacks to be pure functions of their
  inputs. A session over a Lark with a transformer must explicitly declare
  purity via assume_pure_callbacks=True; otherwise construction is refused.
  (The default tree-building callbacks are pure.)
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

from .exceptions import ConfigurationError, UnexpectedInput, UnexpectedToken
from .lexer import Token, LexerThread
from . import parse_tree_builder as _ptb
from .parse_tree_builder import ChildFilterLALR_NoPlaceholders, ParseTreeBuilder
from .parsers.lalr_analysis import Shift
from .parsers.lalr_interactive_parser import InteractiveParser
from .parsers.lalr_parser_state import ParseConf, ParserState
from .visitors import Transformer_InPlace

if TYPE_CHECKING:
    from .lark import Lark

__all__ = ('IncrementalParseSession', 'IncrementalParseStats')

_MISSING = object()


@dataclass
class IncrementalParseStats:
    """Bookkeeping for the most recent parse()/update() call.

    Attributes:
        resumed_at: Index of the first token that was (re)fed. Tokens before
            it were neither fed nor had their callbacks re-run.
        tokens_fed: How many tokens were fed through the LALR state machine.
        rules_executed: Rule callbacks that were actually invoked.
        rules_reused: Rule results reused from the previous parse (callback
            not invoked).
    """
    resumed_at: int = 0
    tokens_fed: int = 0
    rules_executed: int = 0
    rules_reused: int = 0


class _Trace:
    """The recorded outcome of one parse (successful or failed)."""

    __slots__ = ('tokens', 'checkpoints', 'reduces', 'shift_values', 'result', 'error')

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        # checkpoints[i] = (state_stack, value_stack, span_stack) right before token i is fed.
        # span_stack entries are (start, end) token-index pairs of this trace.
        self.checkpoints: List[Tuple[tuple, tuple, tuple]] = []
        # (rule, span_start, span_end) -> (args, result), spans in token indices of this trace
        self.reduces: Dict[tuple, tuple] = {}
        # id(token) -> shifted value, for tokens with terminal callbacks
        self.shift_values: Dict[int, Any] = {}
        self.result: Any = None
        self.error: Optional[UnexpectedInput] = None


def _same_args(old_args: tuple, new_args: list) -> bool:
    return len(old_args) == len(new_args) and all(a is b for a, b in zip(old_args, new_args))


def _combined_old_span(spans: List[tuple]) -> Tuple[Optional[int], Optional[int]]:
    """Combine the old-stream spans of reduce arguments, if they form one
    contiguous span in the previous token stream."""
    start = spans[0][2]
    end = spans[0][3]
    if start is None:
        return None, None
    for s in spans[1:]:
        if s[2] is None or s[2] != end:
            return None, None
        end = s[3]
    return start, end


def _checked_tokens(tokens: Iterable[Token]) -> List[Token]:
    tokens = list(tokens)
    for t in tokens:
        if not isinstance(t, Token):
            raise TypeError("Expected a sequence of Token instances, got %r" % (t,))
    return tokens


class _ChildFilterLALR_Pure(ChildFilterLALR_NoPlaceholders):
    """ChildFilterLALR_NoPlaceholders without the left-recursion optimization.

    The stock optimization aliases the first expanded child's children list
    and extends it in place, mutating previously built trees. Incremental
    sessions keep references to those trees (checkpoints and the reuse log),
    so callbacks used by a session must never mutate their inputs.
    """
    def __call__(self, children):
        filtered = []
        for i, to_expand in self.to_include:
            if to_expand:
                filtered += children[i].children
            else:
                filtered.append(children[i])
        return self.node_builder(filtered)


def _build_pure_callbacks(lark: 'Lark') -> Dict[str, Any]:
    """Rebuild the parser callbacks of the given Lark instance, replacing the
    mutating LALR child filters with pure equivalents, so that values recorded
    by the session are never modified after the fact.

    The returned callbacks produce trees identical to the stock callbacks.
    """
    builder = lark._parse_tree_builder
    saved = _ptb.ChildFilterLALR, _ptb.ChildFilterLALR_NoPlaceholders
    _ptb.ChildFilterLALR = _ptb.ChildFilter
    _ptb.ChildFilterLALR_NoPlaceholders = _ChildFilterLALR_Pure
    try:
        pure_builder = ParseTreeBuilder(
            lark.rules,
            builder.tree_class,
            builder.propagate_positions,
            builder.ambiguous,
            builder.maybe_placeholders,
        )
        callbacks = pure_builder.create_callback(lark.options.transformer)
    finally:
        _ptb.ChildFilterLALR, _ptb.ChildFilterLALR_NoPlaceholders = saved
    # Terminal callbacks (from the transformer) are keyed by terminal name;
    # keep them, and override only the rule callbacks.
    merged = dict(lark._callbacks)
    merged.update(callbacks)
    return merged


class IncrementalParseSession:
    """An explicit incremental parsing session over a token stream.

    Parameters:
        lark: A Lark instance created with parser='lalr'.
        start: Start symbol. Required if the grammar declares several.
        assume_pure_callbacks: Must be set to True when the Lark instance has
            a transformer, to declare that the transformer (and any terminal
            callbacks) are pure functions of their inputs. Reuse with
            side-effecting callbacks is unsound and is refused.

    Example::

        session = IncrementalParseSession(lark)
        tree = session.parse(tokens)                    # full parse
        tree = session.update(4, 6, [tok_a, tok_b])     # replace tokens 4..5
    """

    def __init__(self, lark: 'Lark', start: Optional[str] = None, assume_pure_callbacks: bool = False):
        options = lark.options
        if options.parser != 'lalr':
            raise ConfigurationError(
                "IncrementalParseSession only supports parser='lalr' (got %r)" % (options.parser,))
        if getattr(options, 'transformer', None) is not None and not assume_pure_callbacks:
            raise ConfigurationError(
                "IncrementalParseSession reuses results computed by the parser callbacks, "
                "which is only sound when they are pure functions of their inputs. "
                "This Lark instance has a transformer; pass assume_pure_callbacks=True to "
                "declare it pure, or remove the transformer.")
        if isinstance(getattr(options, 'transformer', None), Transformer_InPlace):
            raise ConfigurationError(
                "IncrementalParseSession cannot be used with Transformer_InPlace, "
                "which mutates its input trees and is therefore never pure.")
        frontend = lark.parser
        inner = getattr(getattr(frontend, 'parser', None), 'parser', None)
        if not (hasattr(inner, 'parse_table') and hasattr(inner, 'callbacks')):
            raise ConfigurationError("IncrementalParseSession requires the standard LALR parser frontend")
        self._start = frontend._verify_start(start)
        self._parser = inner
        self._parse_conf = ParseConf(inner.parse_table, _build_pure_callbacks(lark), self._start)
        self._trace: Optional[_Trace] = None
        self.last_stats: Optional[IncrementalParseStats] = None

    @property
    def tokens(self) -> Optional[List[Token]]:
        """A copy of the token sequence of the most recent parse (or None)."""
        return None if self._trace is None else list(self._trace.tokens)

    def parse(self, tokens: Iterable[Token]) -> Any:
        """Parse a full token sequence, replacing any previous state.

        Raises UnexpectedToken on invalid input; the session stays usable and
        a following update() applies to the failed token sequence.
        """
        trace = self._run(_checked_tokens(tokens), None, 0, None)
        self._trace = trace
        if trace.error is not None:
            raise trace.error
        return trace.result

    def update(self, start: int, end: int, new_tokens: Iterable[Token]) -> Any:
        """Replace tokens [start, end) of the current sequence with
        new_tokens, and re-parse incrementally.

        Parsing resumes from the last checkpoint at or before start. Tokens
        outside the replaced range keep their identity (they are the same
        Token objects as before), which is what makes reuse of the unaffected
        prefix and suffix sound.

        Raises UnexpectedToken on invalid input; the session keeps the
        (failed) token sequence, so a fixing update() can follow.
        """
        old = self._trace
        if old is None:
            raise ConfigurationError("update() requires a previous parse(); call parse() first")
        n = len(old.tokens)
        if not 0 <= start <= end <= n:
            raise ValueError("Invalid replacement range [%d, %d) for a sequence of %d tokens" % (start, end, n))
        new_tokens = _checked_tokens(new_tokens)
        tokens = old.tokens[:start] + new_tokens + old.tokens[end:]
        resume_from = min(start, len(old.checkpoints) - 1)
        trace = self._run(tokens, old, resume_from, (start, end, len(new_tokens)))
        self._trace = trace
        if trace.error is not None:
            raise trace.error
        return trace.result

    def _run(self, tokens: List[Token], old_trace: Optional[_Trace], resume_from: int,
             edit: Optional[Tuple[int, int, int]]) -> _Trace:
        conf = self._parse_conf
        states = conf.states
        callbacks = conf.callbacks
        end_state = conf.end_state
        stats = IncrementalParseStats(resumed_at=resume_from)

        if old_trace is not None and resume_from > 0:
            state_stack, value_stack, span_stack = (list(s) for s in old_trace.checkpoints[resume_from])
            # Restored entries sit in the unchanged prefix, so their spans are
            # the same in the old and the new token stream.
            span_stack = [(s, e, s, e) for (s, e) in span_stack]
            checkpoints = list(old_trace.checkpoints[:resume_from])
            reduces = {k: v for k, v in old_trace.reduces.items() if k[2] <= resume_from}
        else:
            state_stack = [conf.start_state]
            value_stack = []
            span_stack = []
            checkpoints = []
            reduces = {}

        shift_values: Dict[int, Any] = {}

        def shifted_value(token: Token):
            key = id(token)
            try:
                return shift_values[key]
            except KeyError:
                pass
            if old_trace is not None:
                try:
                    value = old_trace.shift_values[key]
                except KeyError:
                    pass
                else:
                    shift_values[key] = value
                    return value
            value = token if token.type not in callbacks else callbacks[token.type](token)
            shift_values[key] = value
            return value

        if edit is None:
            def map_token(i: int) -> Optional[int]:
                return i

            def map_boundary(k: int) -> Optional[int]:
                return k
        else:
            e_start, e_end, e_new_len = edit
            delta = e_new_len - (e_end - e_start)

            def map_token(i: int) -> Optional[int]:
                if i < e_start:
                    return i
                if i >= e_start + e_new_len:
                    return i - delta
                return None

            def map_boundary(k: int) -> Optional[int]:
                if k <= e_start:
                    return k
                if k >= e_start + e_new_len:
                    return k - delta
                return None

        tokens_fed = resume_from
        trace = _Trace(tokens)

        def feed(token: Token, is_end: bool = False):
            nonlocal tokens_fed
            while True:
                state = state_stack[-1]
                try:
                    action, arg = states[state][token.type]
                except KeyError:
                    expected = {s for s in states[state] if s.isupper()}
                    raise UnexpectedToken(token, expected)

                if action is Shift:
                    assert not is_end
                    state_stack.append(arg)
                    value_stack.append(shifted_value(token))
                    old_i = map_token(tokens_fed)
                    span_stack.append((tokens_fed, tokens_fed + 1, old_i, None if old_i is None else old_i + 1))
                    tokens_fed += 1
                    stats.tokens_fed += 1
                    return None

                rule = arg
                size = len(rule.expansion)
                if size:
                    args = value_stack[-size:]
                    del state_stack[-size:]
                    del value_stack[-size:]
                    spans = span_stack[-size:]
                    del span_stack[-size:]
                    new_start, new_end = spans[0][0], spans[-1][1]
                    old_start, old_end = _combined_old_span(spans)
                else:
                    args = []
                    new_start = new_end = tokens_fed
                    old_start = old_end = map_boundary(tokens_fed)

                result = _MISSING
                if old_start is not None and old_trace is not None:
                    entry = old_trace.reduces.get((rule, old_start, old_end))
                    if entry is not None and _same_args(entry[0], args):
                        # Same rule, same input tokens (by identity): with pure
                        # callbacks the result is identical, so reuse it.
                        result = entry[1]
                        stats.rules_reused += 1
                if result is _MISSING:
                    result = callbacks[rule](args) if callbacks else args
                    stats.rules_executed += 1
                reduces[(rule, new_start, new_end)] = (tuple(args), result)

                value_stack.append(result)
                span_stack.append((new_start, new_end, old_start, old_end))
                _action, new_state = states[state_stack[-1]][rule.origin.name]
                assert _action is Shift
                state_stack.append(new_state)

                if is_end and state_stack[-1] == end_state:
                    return value_stack[-1]

        def record_checkpoint():
            checkpoints.append((
                tuple(state_stack),
                tuple(value_stack),
                tuple((s[0], s[1]) for s in span_stack),
            ))

        try:
            i = tokens_fed
            while i < len(tokens):
                record_checkpoint()
                feed(tokens[i])
                i = tokens_fed
            record_checkpoint()
            if tokens:
                end_token = Token.new_borrow_pos('$END', '', tokens[-1])
            else:
                end_token = Token('$END', '', 0, 1, 1)
            trace.result = feed(end_token, is_end=True)
        except UnexpectedInput as e:
            # Attach diagnostics, mirroring lark.parsers.lalr_parser._Parser.parse_from_state.
            # The interactive parser can answer choices()/accepts(), but cannot
            # resume (the session owns the token stream).
            parser_state = ParserState(conf, LexerThread(None, None), list(state_stack), list(value_stack))
            e.state = parser_state
            e.interactive_parser = InteractiveParser(self._parser, parser_state, parser_state.lexer)
            trace.error = e

        trace.checkpoints = checkpoints
        trace.reduces = reduces
        trace.shift_values = shift_values
        self.last_stats = stats
        return trace
