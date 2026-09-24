"""Incremental parsing sessions for the LALR parser.

This module implements an explicit incremental parsing session, aimed at
workloads such as editors and rule backends, where the input changes a few
tokens at a time but re-parsing the whole input each time is too expensive.

The session consumes a token sequence with *stable identities*: tokens that
did not change between two consecutive parses must be the *same objects*
('is'), so that the session can tell which parts of the input are
unaffected.  Edits are described as a single contiguous replacement range
over the previous token sequence, via IncrementalSession.parse_edit().
(Producing the tokens - incrementally or not - is the caller's
responsibility; incremental lexing is out of scope for this module.)

How it works:

- While parsing, the session saves resumable checkpoints (the parser's
  state and value stacks) at every token boundary.
- After an edit, parsing resumes from the latest checkpoint at or before
  the earliest affected token, instead of from the beginning of the input.
- Rule and terminal callbacks are memoized by the identity of their
  children, so reductions over unchanged input (including the whole
  suffix) are not re-run; their previous results are reused.
- As a fast path, when the parser reaches a position of the unchanged
  suffix where both the parse stacks and the upcoming token identities
  match the previous parse, the remaining suffix (and its result) is
  reused wholesale.

Because callbacks are not re-run for unchanged input, incremental parsing
is only sound when every callback (rule callbacks, terminal callbacks and
any embedded transformer) is a pure function of its children.  Use
Lark.incremental_session(), which requires the caller to explicitly
declare this, instead of instantiating this module directly.

Error handling: a failed parse raises the usual UnexpectedInput
diagnostics, and the session stays usable - a later parse_edit() that
fixes the input continues from the checkpoints of the failed parse.

Note: the session keeps the memoized reduction results (and therefore
references to the tokens and trees involved) for its whole lifetime.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .exceptions import UnexpectedInput
from .grammar import Rule
from .lexer import Token
from .tree import Tree
from .parsers.lalr_parser import LALR_Parser
from .parsers.lalr_parser_state import ParserState, ParseConf
from .parsers.lalr_interactive_parser import InteractiveParser


_NO_RESULT = object()


class _Checkpoint:
    """A resumable snapshot of the parser stacks at a token boundary."""
    __slots__ = ('state_stack', 'value_stack')

    def __init__(self, state: ParserState):
        self.state_stack: Tuple = tuple(state.state_stack)
        self.value_stack: List = list(state.value_stack)


def _values_equal(a: Any, b: Any) -> bool:
    """Equality for reuse decisions, short-circuiting on identity.

    Tokens (and any other non-Tree values) must be the *same objects*:
    tokens carry source positions, so value-equality is not enough to
    guarantee identical positions in the reused result.  Trees are
    compared structurally, but their leaves still have to be identical
    objects, which also guarantees equal propagated positions.
    """
    if a is b:
        return True
    if isinstance(a, Tree) and isinstance(b, Tree):
        return (a.data == b.data
                and len(a.children) == len(b.children)
                and all(_values_equal(x, y) for x, y in zip(a.children, b.children)))
    return False


def _value_stacks_equal(a: List, b: List) -> bool:
    return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))


def _memoize_rule_callback(rule: Rule, callback, memo: Dict, stats: Dict):
    def callback_with_memo(children):
        # The memo keeps strong references to the children, so their ids
        # cannot be recycled while an entry is alive; matching ids mean
        # the children really are the same objects.
        key = (id(rule), tuple(map(id, children)))
        entry = memo.get(key)
        if entry is None:
            stats['rule_callbacks_run'] += 1
            result = callback(children)
            memo[key] = (list(children), result)
            return result
        stats['rule_callbacks_reused'] += 1
        return entry[1]
    return callback_with_memo


def _memoize_terminal_callback(name, callback, memo: Dict, stats: Dict):
    def callback_with_memo(token):
        key = (name, id(token))
        entry = memo.get(key)
        if entry is None:
            stats['terminal_callbacks_run'] += 1
            result = callback(token)
            memo[key] = (token, result)
            return result
        stats['terminal_callbacks_reused'] += 1
        return entry[1]
    return callback_with_memo


def _memoized_callbacks(callbacks: Dict, memo: Dict, stats: Dict) -> Dict:
    wrapped = {}
    for key, callback in callbacks.items():
        if isinstance(key, Rule):
            wrapped[key] = _memoize_rule_callback(key, callback, memo, stats)
        else:
            wrapped[key] = _memoize_terminal_callback(key, callback, memo, stats)
    return wrapped


class IncrementalSession:
    """An incremental parsing session over a token sequence with stable identities.

    Don't instantiate directly; use Lark.incremental_session().

    Attributes:
        stats: A dict with counters describing the most recent parse
            ('tokens_fed', 'restored_from', 'suffix_reused',
            'rule_callbacks_run', 'rule_callbacks_reused',
            'terminal_callbacks_run', 'terminal_callbacks_reused').
    """

    stats: Dict[str, Any]

    def __init__(self, parser: LALR_Parser, start: str):
        self._parser = parser.parser  # lark.parsers.lalr_parser._Parser
        self._memo: Dict = {}
        self.stats = {}
        self._reset_stats()
        callbacks = _memoized_callbacks(self._parser.callbacks, self._memo, self.stats)
        self._parse_conf = ParseConf(self._parser.parse_table, callbacks, start)
        self._tokens: Optional[List[Token]] = None
        self._checkpoints: List[Optional[_Checkpoint]] = []
        self._result: Any = _NO_RESULT

    def _reset_stats(self) -> None:
        self.stats.clear()
        self.stats.update({
            'tokens_fed': 0,
            'restored_from': 0,
            'suffix_reused': False,
            'rule_callbacks_run': 0,
            'rule_callbacks_reused': 0,
            'terminal_callbacks_run': 0,
            'terminal_callbacks_reused': 0,
        })

    @property
    def tokens(self) -> Optional[List[Token]]:
        """The token sequence of the most recent parse attempt, or None."""
        return self._tokens

    @property
    def result(self) -> Any:
        """The result of the most recent parse, or None if it failed."""
        return None if self._result is _NO_RESULT else self._result

    def parse(self, tokens: Sequence[Token]) -> Any:
        """Parse a full token sequence, replacing any previous session state.

        Use this for the first parse of a session, or to force a full
        re-parse.  Later edits should go through parse_edit().

        Parameters:
            tokens: The tokens to parse. Their identities are remembered
                and compared (by 'is') against future edits.
        """
        tokens = list(tokens)
        self._reset_stats()
        self._tokens = tokens
        self._checkpoints = [None] * (len(tokens) + 1)
        self._result = _NO_RESULT
        state = ParserState(self._parse_conf, None)
        self._checkpoints[0] = _Checkpoint(state)
        result = self._feed(state, tokens, 0, None)
        self._result = result
        return result

    def parse_edit(self, start: int, end: int, replacement: Sequence[Token]) -> Any:
        """Replace tokens[start:end] of the previous sequence and re-parse.

        The new sequence is 'old[:start] + replacement + old[end:]'.
        Parsing resumes from the latest checkpoint at or before 'start';
        the unchanged prefix is not re-parsed and its callbacks are not
        re-run.  The unchanged suffix is reused wherever the parse stacks
        and token identities allow it.

        Works after a failed parse too: the checkpoints recorded before
        the error are reused, so fixing the input is cheap.

        Parameters:
            start: Index of the first replaced token.
            end: Index one past the last replaced token.
            replacement: The tokens to insert in place of the replaced range.
        """
        if self._tokens is None:
            raise ValueError("parse_edit() requires a previous parse; call parse() first")
        old_tokens = self._tokens
        old_checkpoints = self._checkpoints
        old_result = self._result
        if not (0 <= start <= end <= len(old_tokens)):
            raise ValueError("invalid replacement range (%r, %r) for %d tokens"
                             % (start, end, len(old_tokens)))
        replacement = list(replacement)
        tokens = old_tokens[:start] + replacement + old_tokens[end:]
        delta = len(tokens) - len(old_tokens)
        suffix_start = start + len(replacement)

        self._reset_stats()
        self._tokens = tokens
        self._result = _NO_RESULT

        # Resume from the latest checkpoint at or before the earliest
        # affected token.  Checkpoints below 'start' are still valid,
        # because the tokens before 'start' are unchanged.
        c = start
        while c > 0 and old_checkpoints[c] is None:
            c -= 1
        cp = old_checkpoints[c]
        if cp is None:
            state = ParserState(self._parse_conf, None)
            self._checkpoints = [None] * (len(tokens) + 1)
            self._checkpoints[0] = _Checkpoint(state)
        else:
            state = ParserState(self._parse_conf, None, list(cp.state_stack), list(cp.value_stack))
            self._checkpoints = old_checkpoints[:c + 1] + [None] * (len(tokens) - c)
        self.stats['restored_from'] = c

        result = self._feed(state, tokens, c,
                            _ReuseContext(old_tokens, old_checkpoints, old_result, delta, suffix_start))
        self._result = result
        return result

    def _feed(self, state: ParserState, tokens: List[Token], index: int,
              reuse: Optional['_ReuseContext']) -> Any:
        """Feed tokens[index:] (plus $END) to the parser state.

        Records a checkpoint at every token boundary.  When 'reuse' is
        given and the parser reaches the unchanged suffix with parse
        stacks and token identities matching the previous parse, the
        previous result is reused and the remaining tokens are skipped.
        """
        stats = self.stats
        checkpoints = self._checkpoints
        n = len(tokens)
        i = index
        try:
            while i < n:
                if reuse is not None and i >= reuse.suffix_start and reuse.old_result is not _NO_RESULT:
                    j = i - reuse.delta
                    # In the unchanged suffix, tokens[i:] are the same
                    # objects as old_tokens[j:] by construction.
                    ocp = reuse.old_checkpoints[j]
                    if (ocp is not None
                            and tokens[i] is reuse.old_tokens[j]
                            and tuple(state.state_stack) == ocp.state_stack
                            and _value_stacks_equal(state.value_stack, ocp.value_stack)):
                        # The rest of the parse is identical to the
                        # previous one: reuse its checkpoints and result.
                        checkpoints[i:] = reuse.old_checkpoints[j:]
                        stats['suffix_reused'] = True
                        return reuse.old_result
                token = tokens[i]
                state.feed_token(token)
                checkpoints[i + 1] = _Checkpoint(state)
                stats['tokens_fed'] += 1
                i += 1
            if tokens:
                end_token = Token.new_borrow_pos('$END', '', tokens[-1])
            else:
                end_token = Token('$END', '', 0, 1, 1)
            return state.feed_token(end_token, True)
        except UnexpectedInput as e:
            # Attach an interactive parser for consistency with a regular
            # parse.  Resuming, however, should go through the session.
            e.interactive_parser = InteractiveParser(self._parser, state, None)
            raise


class _ReuseContext:
    __slots__ = ('old_tokens', 'old_checkpoints', 'old_result', 'delta', 'suffix_start')

    def __init__(self, old_tokens, old_checkpoints, old_result, delta, suffix_start):
        self.old_tokens = old_tokens
        self.old_checkpoints = old_checkpoints
        self.old_result = old_result
        self.delta = delta
        self.suffix_start = suffix_start
