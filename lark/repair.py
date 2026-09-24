"""Budgeted token-level repair search for LALR interactive parsing.

When a LALR parse fails with an ``UnexpectedToken`` error, ``InteractiveParser.accepts()``
can list the tokens accepted at the error position, but it cannot tell you which edits
would let the *rest* of the input parse as well. This module provides ``find_repairs()``,
a budgeted uniform-cost (Dijkstra) search over token-level edits - insertions, deletions
and replacements - that returns the cheapest edit plans which repair the remaining input.

Each candidate plan is verified by continuing the parse through all of the remaining
input (up to and including ``$END``), never just the error position. Because the search
space is potentially infinite (e.g. zero-cost insertions), the search is bounded by a
budget of state expansions; if the budget runs to exhaustion, the result is flagged
with ``proven_optimal=False``.

The search runs on parser states with all parse callbacks disabled, so transformers
with side effects are never triggered while searching. Only when a chosen plan is
applied to the real interactive parser (via ``RepairResult.apply()``) do the callbacks
fire - exactly as many times as they would in a normal parse of the repaired input.

Example::

    from lark import Lark, UnexpectedToken
    from lark.repair import find_repairs

    parser = Lark(grammar, parser='lalr')
    try:
        parser.parse(text)
    except UnexpectedToken as e:
        result = find_repairs(e, insert_cost=1, delete_cost=1, replace_cost=1)
        if result.plans:
            tree = result.apply(result.plans[0])
"""

from copy import copy
from dataclasses import dataclass
from heapq import heappush, heappop
from itertools import count
from math import inf
from typing import Iterable, List, Optional, Tuple, Union

from .exceptions import UnexpectedToken
from .lexer import Token
from .parsers.lalr_interactive_parser import InteractiveParser


@dataclass(frozen=True)
class Insert:
    """Insert a token of type ``token_type`` before the token at ``position``."""
    position: int
    token_type: str


@dataclass(frozen=True)
class Delete:
    """Delete the token at ``position``."""
    position: int
    token: Token


@dataclass(frozen=True)
class Replace:
    """Replace the token at ``position`` with a token of type ``token_type``."""
    position: int
    token_type: str
    token: Token


Edit = Union[Insert, Delete, Replace]


@dataclass(frozen=True)
class RepairPlan:
    """A single repair: an ordered sequence of edits, its total cost, and the
    position in the remaining token stream at which unedited parsing resumes.

    ``resume_index`` is an index into ``RepairResult.tokens``: tokens before it were
    consumed, skipped or replaced by the plan's edits, tokens from it on are fed
    unchanged. It equals ``len(tokens)`` when the edits only append tokens before EOF
    (e.g. a missing closing delimiter).
    """
    edits: Tuple[Edit, ...]
    cost: float
    resume_index: int


class RepairResult:
    """The result of ``find_repairs()``.

    Attributes:
        plans: The cheapest repair plans found, sorted by cost (at most ``max_plans``).
        proven_optimal: False when the search budget ran out before the search space
            was exhausted, meaning cheaper plans may still exist.
        tokens: The remaining tokens from the error position onward. Edit positions
            and ``RepairPlan.resume_index`` refer to this list.
        expansions: How many search states were expanded (for budget diagnostics).
    """

    def __init__(self, plans: List[RepairPlan], proven_optimal: bool, tokens: List[Token],
                 expansions: int, interactive_parser: Optional[InteractiveParser]):
        self.plans = plans
        self.proven_optimal = proven_optimal
        self.tokens = tokens
        self.expansions = expansions
        self.interactive_parser = interactive_parser

    def __repr__(self):
        return '%s(plans=%r, proven_optimal=%r)' % (type(self).__name__, self.plans, self.proven_optimal)

    def apply(self, plan: RepairPlan, interactive_parser: Optional[InteractiveParser] = None):
        """Apply a chosen plan to an interactive parser and finish the parse.

        Feeds the edited token stream into ``interactive_parser`` (defaults to the one
        the search was started from), followed by ``$END``, and returns the parse result.

        This is the only place where parser callbacks (e.g. transformers) run; they
        fire exactly as they would when parsing the repaired input from scratch.

        Note: this modifies the interactive parser in place.
        """
        ip = interactive_parser if interactive_parser is not None else self.interactive_parser
        if ip is None:
            raise ValueError("apply() needs an interactive parser, but none is associated with this result")

        tokens = self.tokens
        inserts = {}
        overridden = {}   # position -> token_type to feed instead, or None to skip the token
        for edit in plan.edits:
            if isinstance(edit, Insert):
                inserts.setdefault(edit.position, []).append(edit.token_type)
            elif isinstance(edit, Delete):
                overridden[edit.position] = None
            elif isinstance(edit, Replace):
                overridden[edit.position] = edit.token_type
            else:
                raise TypeError("Unknown edit: %r" % (edit,))

        last_token = None
        for index, token in enumerate(tokens):
            for token_type in inserts.get(index, ()):
                ip.feed_token(Token.new_borrow_pos(token_type, '', token))
            if index in overridden:
                replacement = overridden[index]
                if replacement is not None:
                    ip.feed_token(Token.new_borrow_pos(replacement, '', token))
            else:
                ip.feed_token(token)
            last_token = token
        for token_type in inserts.get(len(tokens), ()):
            fabricated = Token.new_borrow_pos(token_type, '', last_token) if last_token is not None else Token(token_type, '')
            ip.feed_token(fabricated)
        return ip.feed_eof(last_token)


def _resume_index(edits: Tuple[Edit, ...]) -> int:
    index = 0
    for edit in edits:
        end = edit.position if isinstance(edit, Insert) else edit.position + 1
        index = max(index, end)
    return index


def find_repairs(source: Union[UnexpectedToken, InteractiveParser],
                 candidates: Optional[Iterable[str]] = None, *,
                 insert_cost: float = 1.0,
                 delete_cost: float = 1.0,
                 replace_cost: float = 1.0,
                 max_plans: int = 3,
                 budget: int = 10000) -> RepairResult:
    """Search for the cheapest token-level edits that repair the remaining input.

    Parameters:
        source: The ``UnexpectedToken`` exception raised by the failed parse (preferred,
            as it includes the offending token), or an ``InteractiveParser`` positioned
            at the error. In the latter case the offending token is assumed to be
            already consumed (e.g. an error at EOF).
        candidates: Terminal types to try for insertions and replacements.
            Defaults to ``accepts()`` at the error position.
        insert_cost: Cost of inserting a candidate token.
        delete_cost: Cost of deleting an input token.
        replace_cost: Cost of replacing an input token with a candidate token.
            Costs must be non-negative; ``math.inf`` disables the operation.
        max_plans: Maximum number of plans to return.
        budget: Maximum number of search states to expand. If the budget runs out
            before the search space is exhausted, ``result.proven_optimal`` is False.

    Returns:
        A ``RepairResult``. Every returned plan was verified by parsing the remaining
        input to its end, so applying it is guaranteed to parse (barring exceptions
        raised by user callbacks).

    Note:
        The remaining input is lexed once, up front, using the frozen parser state at
        the error position. A lexer error (``UnexpectedCharacters``) in the remaining
        input will propagate, as token-level repair cannot fix it.
    """
    if min(insert_cost, delete_cost, replace_cost) < 0:
        raise ValueError("Edit costs must be non-negative")
    if max_plans < 1:
        raise ValueError("max_plans must be at least 1")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    first_tokens: List[Token] = []
    if isinstance(source, UnexpectedToken):
        ip = source.interactive_parser
        if ip is None:
            raise ValueError("The UnexpectedToken exception is not associated with an interactive parser")
        if source.token.type != '$END':
            first_tokens.append(source.token)
    elif isinstance(source, InteractiveParser):
        ip = source
    else:
        raise TypeError("find_repairs() expects an UnexpectedToken or an InteractiveParser, got %r" % (source,))

    # Search on a private copy with all parse callbacks disabled, so that
    # transformers with side effects are never triggered by the search itself.
    base = ip.copy(deepcopy_values=False)
    conf_no_callbacks = copy(base.parser_state.parse_conf)
    conf_no_callbacks.callbacks = {}
    base.parser_state.parse_conf = conf_no_callbacks

    tokens = first_tokens + list(base.lexer_thread.lex(base.parser_state))

    if candidates is None:
        candidates = sorted(base.accepts())
    candidates = [c for c in candidates if c != '$END']

    def borrow_pos(token_type: str) -> Token:
        neighbor = tokens[index] if index < len(tokens) else (tokens[-1] if tokens else None)
        return Token.new_borrow_pos(token_type, '', neighbor) if neighbor is not None else Token(token_type, '')

    def state_key(cursor: InteractiveParser):
        return tuple(cursor.parser_state.state_stack)

    serial = count()
    heap = [(0.0, next(serial), 0, base, ())]     # (cost, serial, token_index, cursor, edits)
    best = {}                                     # (state_key, token_index) -> cheapest cost seen
    plans: List[RepairPlan] = []
    expansions = 0
    exhausted = False

    while heap:
        if expansions >= budget:
            exhausted = True
            break
        cost, _, index, cursor, edits = heappop(heap)
        key = (state_key(cursor), index)
        if best.get(key, inf) <= cost:
            continue    # already reached this state at an equal or lower cost
        best[key] = cost
        expansions += 1

        if index == len(tokens):
            # All input consumed; a plan is valid only if the parser also accepts EOF.
            end_cursor = cursor.copy(deepcopy_values=False)
            eof = Token.new_borrow_pos('$END', '', tokens[-1]) if tokens else Token('$END', '')
            try:
                end_cursor.feed_token(eof)
            except UnexpectedToken:
                pass
            else:
                plans.append(RepairPlan(edits, cost, _resume_index(edits)))
                if len(plans) >= max_plans:
                    return RepairResult(plans, True, tokens, expansions, ip)

        choices = cursor.choices()

        if index < len(tokens):
            token = tokens[index]
            # Match: feed the input token as-is
            if token.type in choices:
                match_cursor = cursor.copy(deepcopy_values=False)
                try:
                    match_cursor.feed_token(token)
                except UnexpectedToken:
                    pass
                else:
                    heappush(heap, (cost, next(serial), index + 1, match_cursor, edits))
            # Delete: skip the input token
            if delete_cost != inf:
                heappush(heap, (cost + delete_cost, next(serial), index + 1, cursor,
                                edits + (Delete(index, token),)))
            # Replace: feed a candidate token instead of the input token
            if replace_cost != inf:
                for token_type in candidates:
                    if token_type == token.type or token_type not in choices:
                        continue
                    replace_cursor = cursor.copy(deepcopy_values=False)
                    try:
                        replace_cursor.feed_token(borrow_pos(token_type))
                    except UnexpectedToken:
                        continue
                    heappush(heap, (cost + replace_cost, next(serial), index + 1, replace_cursor,
                                    edits + (Replace(index, token_type, token),)))
        # Insert: feed a candidate token without consuming input.
        # Also allowed at the end of input, to repair unclosed constructs before EOF.
        if insert_cost != inf:
            for token_type in candidates:
                if token_type not in choices:
                    continue
                insert_cursor = cursor.copy(deepcopy_values=False)
                try:
                    insert_cursor.feed_token(borrow_pos(token_type))
                except UnexpectedToken:
                    continue
                heappush(heap, (cost + insert_cost, next(serial), index, insert_cursor,
                                edits + (Insert(index, token_type),)))

    return RepairResult(plans, not exhausted, tokens, expansions, ip)
