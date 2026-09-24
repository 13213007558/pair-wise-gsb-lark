# This module provides a LALR interactive parser, which is used for debugging and error handling

from typing import Iterator, List
from copy import copy
import warnings

from lark.exceptions import UnexpectedToken
from lark.lexer import Token, LexerThread
from .lalr_parser_state import ParserState

###{standalone

import heapq

class RepairEdit:
    """A single token-level edit, as part of a RepairPlan.

    Attributes:
        kind: One of 'insert', 'delete' or 'replace'.
        index: Position in the remaining token stream (RepairResult.tokens)
            at which the edit applies.
        token_type: For 'insert' and 'replace', the terminal to feed instead.
            For 'delete', the type of the removed token.
        replaced_type: For 'replace', the type of the original token. Otherwise None.
    """
    __slots__ = ('kind', 'index', 'token_type', 'replaced_type')

    def __init__(self, kind, index, token_type, replaced_type=None):
        self.kind = kind
        self.index = index
        self.token_type = token_type
        self.replaced_type = replaced_type

    def __eq__(self, other):
        if not isinstance(other, RepairEdit):
            return NotImplemented
        return (self.kind, self.index, self.token_type, self.replaced_type) == \
               (other.kind, other.index, other.token_type, other.replaced_type)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.kind, self.index, self.token_type, self.replaced_type))

    def __repr__(self):
        if self.kind == 'replace':
            return 'RepairEdit(%r, %d, %r, %r)' % (self.kind, self.index, self.token_type, self.replaced_type)
        return 'RepairEdit(%r, %d, %r)' % (self.kind, self.index, self.token_type)


class RepairPlan:
    """A sequence of edits that lets the parse finish, found by InteractiveParser.repair_search().

    Attributes:
        edits: A tuple of RepairEdit, in order of application.
        cost: The total cost of the edits.
        resume_index: Position in RepairResult.tokens from which the rest of
            the input is parsed without edits (i.e. right past the last edit).
    """
    __slots__ = ('edits', 'cost', 'resume_index')

    def __init__(self, edits, cost, resume_index):
        self.edits = edits
        self.cost = cost
        self.resume_index = resume_index

    def __eq__(self, other):
        if not isinstance(other, RepairPlan):
            return NotImplemented
        return (self.edits, self.cost, self.resume_index) == (other.edits, other.cost, other.resume_index)

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return 'RepairPlan(cost=%r, resume_index=%d, edits=%r)' % (self.cost, self.resume_index, list(self.edits))


class RepairResult:
    """The result of InteractiveParser.repair_search().

    Attributes:
        plans: The cheapest repair plans found, ordered by cost (cheapest first).
        optimal: False if the search budget ran out before the search space was
            exhausted, in which case cheaper or additional plans may exist.
        tokens: The remaining token stream that the plan indices refer to.
        expanded: How many search nodes were expanded (for diagnostics).
    """
    __slots__ = ('plans', 'optimal', 'tokens', 'expanded')

    def __init__(self, plans, optimal, tokens, expanded):
        self.plans = plans
        self.optimal = optimal
        self.tokens = tokens
        self.expanded = expanded

    @property
    def best(self):
        """The cheapest plan found, or None if there isn't one."""
        return self.plans[0] if self.plans else None

    def __repr__(self):
        return 'RepairResult(plans=%r, optimal=%r)' % (self.plans, self.optimal)


class InteractiveParser:
    """InteractiveParser gives you advanced control over parsing and error handling when parsing with LALR.

    For a simpler interface, see the ``on_error`` argument to ``Lark.parse()``.
    """
    def __init__(self, parser, parser_state: ParserState, lexer_thread: LexerThread):
        self.parser = parser
        self.parser_state = parser_state
        self.lexer_thread = lexer_thread
        self.result = None

    @property
    def lexer_state(self) -> LexerThread:
        warnings.warn("lexer_state will be removed in subsequent releases. Use lexer_thread instead.", DeprecationWarning)
        return self.lexer_thread

    def feed_token(self, token: Token):
        """Feed the parser with a token, and advance it to the next state, as if it received it from the lexer.

        Note that ``token`` has to be an instance of ``Token``.
        """
        return self.parser_state.feed_token(token, token.type == '$END')

    def iter_parse(self) -> Iterator[Token]:
        """Step through the different stages of the parse, by reading tokens from the lexer
        and feeding them to the parser, one per iteration.

        Returns an iterator of the tokens it encounters.

        When the parse is over, the resulting tree can be found in ``InteractiveParser.result``.
        """
        for token in self.lexer_thread.lex(self.parser_state):
            yield token
            self.result = self.feed_token(token)

    def exhaust_lexer(self) -> List[Token]:
        """Try to feed the rest of the lexer state into the interactive parser.

        Note that this modifies the instance in place and does not feed an '$END' Token
        """
        return list(self.iter_parse())


    def feed_eof(self, last_token=None):
        """Feed a '$END' Token. Borrows from 'last_token' if given."""
        eof = Token.new_borrow_pos('$END', '', last_token) if last_token is not None else self.lexer_thread._Token('$END', '', 0, 1, 1)
        return self.feed_token(eof)


    def __copy__(self):
        """Create a new interactive parser with a separate state.

        Calls to feed_token() won't affect the old instance, and vice-versa.
        """
        return self.copy()

    def copy(self, deepcopy_values=True):
        parser_state = self.parser_state.copy(deepcopy_values=deepcopy_values)
        return type(self)(self.parser, parser_state, parser_state.lexer)

    def __eq__(self, other):
        if not isinstance(other, InteractiveParser):
            return False

        return self.parser_state == other.parser_state and self.lexer_thread == other.lexer_thread

    def as_immutable(self):
        """Convert to an ``ImmutableInteractiveParser``."""
        p = copy(self)
        return ImmutableInteractiveParser(p.parser, p.parser_state, p.lexer_thread)

    def pretty(self):
        """Print the output of ``choices()`` in a way that's easier to read."""
        out = ["Parser choices:"]
        for k, v in self.choices().items():
            out.append('\t- %s -> %r' % (k, v))
        out.append('stack size: %s' % len(self.parser_state.state_stack))
        return '\n'.join(out)

    def choices(self):
        """Returns a dictionary of token types, matched to their action in the parser.

        Only returns token types that are accepted by the current state.

        Updated by ``feed_token()``.
        """
        return self.parser_state.parse_conf.parse_table.states[self.parser_state.position]

    def accepts(self):
        """Returns the set of possible tokens that will advance the parser into a new valid state."""
        accepts = set()
        conf_no_callbacks = copy(self.parser_state.parse_conf)
        # We don't want to call callbacks here since those might have arbitrary side effects
        # and are unnecessarily slow.
        conf_no_callbacks.callbacks = {}
        for t in self.choices():
            if t.isupper(): # is terminal?
                new_cursor = self.copy(deepcopy_values=False)
                new_cursor.parser_state.parse_conf = conf_no_callbacks
                try:
                    new_cursor.feed_token(self.lexer_thread._Token(t, ''))
                except UnexpectedToken:
                    pass
                else:
                    accepts.add(t)
        return accepts

    def repair_search(self, candidates=None, insert_cost=1, delete_cost=1, replace_cost=1, k=1, budget=10000, first_token=None):
        """Search for the cheapest token-level edits that let the parse run to completion.

        Unlike accepts(), which only looks at the current position, each plan
        returned here is verified by parsing the rest of the input through to
        the end, so it is guaranteed to recover the parse (barring lexer errors).

        The search never executes parser callbacks (transformers), so it is safe
        to use with transformers that have side-effects. Once a plan is chosen,
        apply it with apply_repair(), which runs the callbacks exactly as a
        normal parse would.

        Parameters:
            candidates: Terminal names allowed for insert/replace edits.
                Defaults to all terminals known to the parse table.
            insert_cost: Cost of inserting a candidate terminal.
            delete_cost: Cost of deleting an input token.
            replace_cost: Cost of replacing an input token with a candidate terminal.
            k: How many of the cheapest plans to return.
            budget: Maximum number of search nodes to expand. If the budget runs
                out before the search space is exhausted, the result's 'optimal'
                attribute is False, meaning cheaper or additional plans may exist.
            first_token: The token the parser choked on (e.g. UnexpectedToken.token),
                which the lexer has already consumed. A '$END' token is ignored.

        Returns:
            A RepairResult with up to k plans, ordered by cost.

        Note:
            Edit costs must be non-negative. Zero-cost edits are allowed; cycles
            they may form are handled, but they can make the search rely on the
            budget to terminate.

            Edit sequences that converge to the same parser state are merged,
            keeping only the cheapest, so fewer than k plans may be returned
            even when more equally-cheap edit sequences exist.
        """
        if insert_cost < 0 or delete_cost < 0 or replace_cost < 0:
            raise ValueError("Edit costs must be non-negative")
        if k < 1:
            raise ValueError("k must be at least 1")
        if budget < 0:
            raise ValueError("budget must be non-negative")

        # Materialize the remaining tokens on a copy of the lexer state, so the
        # search can look ahead (and a chosen plan can be applied) without
        # disturbing self. A contextual lexer is replaced by its root lexer,
        # which accepts every terminal regardless of the parser position.
        tokens = []
        if first_token is not None and first_token.type != '$END':
            tokens.append(first_token)
        lexer = getattr(self.lexer_thread.lexer, 'root_lexer', self.lexer_thread.lexer)
        tokens.extend(lexer.lex(copy(self.lexer_thread.state), self.parser_state))

        if candidates is None:
            candidates = {
                token_type
                for state in self.parser_state.parse_conf.states.values()
                for token_type in state
                if token_type.isupper()
            }
        candidates = set(candidates)
        candidates.discard('$END')

        # The search must never run user callbacks (they may have side-effects)
        conf_no_callbacks = copy(self.parser_state.parse_conf)
        conf_no_callbacks.callbacks = {}

        _Token = self.lexer_thread._Token

        start_state = self.parser_state.copy(deepcopy_values=False)
        start_state.parse_conf = conf_no_callbacks

        order = 0
        heap = [(0, order, start_state, 0, ())]
        closed = {}
        plans = []
        expanded = 0

        def push(cost, state, index, edits):
            nonlocal order
            order += 1
            heapq.heappush(heap, (cost, order, state, index, edits))

        # Uniform-cost search over (parser state, token index). Because all edge
        # costs are non-negative, plans are popped in non-decreasing cost order,
        # so the first k plans found are provably the k cheapest.
        while heap and len(plans) < k and expanded < budget:
            cost, _, state, index, edits = heapq.heappop(heap)
            key = (tuple(state.state_stack), index)
            if closed.get(key, float('inf')) <= cost:
                continue
            closed[key] = cost
            expanded += 1

            if index == len(tokens):
                # All input consumed; the parse succeeds if it can reduce to the end state
                cursor = state.copy(deepcopy_values=False)
                try:
                    cursor.feed_token(_Token('$END', ''), True)
                except UnexpectedToken:
                    pass
                else:
                    resume_index = 0
                    for edit in edits:
                        edit_end = edit.index if edit.kind == 'insert' else edit.index + 1
                        resume_index = max(resume_index, edit_end)
                    plans.append(RepairPlan(edits, cost, resume_index))
                    continue

            choices = state.parse_conf.states[state.position]
            token = tokens[index] if index < len(tokens) else None

            # Shift the next input token, if the parser accepts it
            if token is not None and token.type in choices:
                cursor = state.copy(deepcopy_values=False)
                try:
                    cursor.feed_token(token)
                except UnexpectedToken:
                    pass
                else:
                    push(cost, cursor, index + 1, edits)

            # Delete the next input token
            if token is not None:
                push(cost + delete_cost, state, index + 1,
                     edits + (RepairEdit('delete', index, token.type),))

            # Insert or replace with a candidate terminal
            for cand in candidates:
                if cand not in choices:
                    continue
                cursor = state.copy(deepcopy_values=False)
                try:
                    cursor.feed_token(_Token(cand, ''))
                except UnexpectedToken:
                    continue
                push(cost + insert_cost, cursor, index,
                     edits + (RepairEdit('insert', index, cand),))
                if token is not None:
                    push(cost + replace_cost, cursor, index + 1,
                         edits + (RepairEdit('replace', index, cand, token.type),))

        optimal = len(plans) >= k or not heap
        return RepairResult(plans, optimal, tokens, expanded)

    def apply_repair(self, plan, tokens):
        """Apply a RepairPlan chosen from repair_search() and finish the parse.

        Feeds the edited token stream through this parser, executing the parser
        callbacks exactly as a normal parse of the corrected input would, and
        returns the parse result.

        Parameters:
            plan: A RepairPlan from the search result.
            tokens: The 'tokens' attribute of the same RepairResult.
        """
        _Token = self.lexer_thread._Token
        index = 0
        for edit in plan.edits:
            while index < edit.index:
                self.feed_token(tokens[index])
                index += 1
            if edit.kind == 'insert':
                self.feed_token(_Token(edit.token_type, ''))
            elif edit.kind == 'delete':
                index += 1
            elif edit.kind == 'replace':
                self.feed_token(_Token(edit.token_type, ''))
                index += 1
            else:
                raise ValueError("Unknown edit kind: %r" % (edit.kind,))
        while index < len(tokens):
            self.feed_token(tokens[index])
            index += 1
        return self.feed_eof(last_token=tokens[-1] if tokens else None)

    def resume_parse(self):
        """Resume automated parsing from the current state.
        """
        return self.parser.parse_from_state(self.parser_state, last_token=self.lexer_thread.state.last_token)



class ImmutableInteractiveParser(InteractiveParser):
    """Same as ``InteractiveParser``, but operations create a new instance instead
    of changing it in-place.
    """

    result = None

    def __hash__(self):
        return hash((self.parser_state, self.lexer_thread))

    def feed_token(self, token):
        c = copy(self)
        c.result = InteractiveParser.feed_token(c, token)
        return c

    def exhaust_lexer(self):
        """Try to feed the rest of the lexer state into the parser.

        Note that this returns a new ImmutableInteractiveParser and does not feed an '$END' Token"""
        cursor = self.as_mutable()
        cursor.exhaust_lexer()
        return cursor.as_immutable()

    def as_mutable(self):
        """Convert to an ``InteractiveParser``."""
        p = copy(self)
        return InteractiveParser(p.parser, p.parser_state, p.lexer_thread)

###}
