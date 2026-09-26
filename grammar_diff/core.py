"""Core engine for bounded grammar comparison.

Two Lark grammars are compiled to LALR parsers.  Instead of enumerating
token sequences and parsing each one from scratch, the search walks the
product of the two LALR automatons breadth-first over shared prefixes:

* each search node carries one immutable LALR state stack per grammar, so
  all sequences sharing a prefix reuse the same parser state;
* a node is pruned for a grammar as soon as that grammar's stack provably
  cannot reach an accept state within the remaining token budget, using an
  exact 'minimum tokens to acceptance' distance computed on the LALR
  automaton (which encodes the derivable lengths of the grammar);
* nodes are deduplicated on the pair of stacks, so equivalent suffix
  behaviour is only explored once, at the shortest prefix.

Left recursion, nullable rules and mutually recursive rules are all handled
natively, because everything is driven by the compiled LALR tables rather
than by a recursive-descent walk of the grammar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from lark import Lark, Token
from lark.parsers.lalr_analysis import Shift

END = "$END"

#: Sentinel returned by LalrMachine.step when feeding the token completes a
#: parse (only possible for the END token).
_ACCEPT = object()

#: Sentinel used internally when the min-completion search gives up.
_UNKNOWN = object()

#: A LALR configuration is a tuple of automaton state ids.
Stack = Tuple[int, ...]
TokenSeq = Tuple[str, ...]


class LalrMachine:
    """An immutable, re-startable view over one grammar's LALR automaton."""

    #: Safety valve for a single min-completion BFS.  When it trips, the
    #: result is treated as 'unknown' and never causes pruning.
    MIN_EXPLORE_LIMIT = 200000

    def __init__(self, grammar: str, start: Optional[str] = None):
        self.grammar = grammar
        options = {"parser": "lalr"}
        if start is not None:
            options["start"] = start
        self.lark = Lark(grammar, **options)
        interactive = self.lark.parse_interactive(start=start)
        conf = interactive.parser_state.parse_conf
        self.start_symbol = conf.start
        self.states = conf.states
        self.start_stack: Stack = (conf.start_state,)
        self.end_state = conf.end_state
        terminals = set()
        for transitions in self.states.values():
            for symbol in transitions:
                if symbol.isupper():
                    terminals.add(symbol)
        terminals.discard(END)
        self.terminals = terminals
        self._min_cache: Dict[Stack, Tuple[Optional[int], int]] = {}

    def step(self, stack: Stack, token_type: str):
        """Advance the automaton by one token.

        Returns the new stack, ACCEPT (only for the END token), or None when
        the token is not a viable continuation of the prefix.
        """
        stack = list(stack)
        seen = set()
        while True:
            marker = tuple(stack)
            if marker in seen:
                # Reduce cycle; impossible for a valid LALR automaton, but
                # stay conservative instead of looping forever.
                return None
            seen.add(marker)
            entry = self.states.get(stack[-1], {}).get(token_type)
            if entry is None:
                return None
            action, arg = entry
            if action is Shift:
                stack.append(arg)
                return tuple(stack)
            rule = arg
            size = len(rule.expansion)
            if size:
                del stack[-size:]
            goto = self.states[stack[-1]][rule.origin.name][1]
            stack.append(goto)
            if token_type == END and stack[-1] == self.end_state:
                return _ACCEPT

    def accepts_eof(self, stack: Stack) -> bool:
        """True when the stack accepts end-of-input (prefix is a full parse)."""
        return self.step(stack, END) is _ACCEPT

    def expected_tokens(self, stack: Stack) -> List[str]:
        """Terminal types the automaton could consume next from this stack."""
        return sorted(s for s in self.states.get(stack[-1], {}) if s.isupper())

    def min_completion(self, stack: Stack, cap: int) -> Optional[int]:
        """Minimum number of extra tokens needed to reach acceptance.

        Returns None when it is proven that more than 'cap' tokens are
        needed.  Results are memoized; a cached exact value answers any cap,
        a cached None answers any cap up to the one it was proven for.
        """
        if cap < 0:
            return None
        if self.accepts_eof(stack):
            return 0
        entry = self._min_cache.get(stack)
        if entry is not None:
            value, cached_cap = entry
            if value is not None:
                return value if value <= cap else None
            if cap <= cached_cap:
                return None
        value = self._bfs_min_completion(stack, cap)
        if value is _UNKNOWN:
            # Conservative lower bound: never used to prune.
            return 0
        self._min_cache[stack] = (value, cap)
        return value

    def _bfs_min_completion(self, stack: Stack, cap: int):
        """Exact BFS over automaton configurations, bounded by 'cap' tokens."""
        visited = {stack}
        queue = deque([(stack, 0)])
        explored = 0
        while queue:
            current, cost = queue.popleft()
            if cost >= cap:
                continue
            for token_type in self.terminals:
                nxt = self.step(current, token_type)
                if nxt is _ACCEPT or nxt is None or nxt in visited:
                    continue
                visited.add(nxt)
                if self.accepts_eof(nxt):
                    return cost + 1
                queue.append((nxt, cost + 1))
                explored += 1
                if explored > self.MIN_EXPLORE_LIMIT:
                    return _UNKNOWN
        return None


@dataclass
class SideReport:
    """Parse outcome of the distinguishing sequence on one grammar."""

    accepted: bool
    detail: str
    tree: Optional[str] = None


@dataclass
class CompareResult:
    """Outcome of a bounded grammar comparison.

    status is one of:
      * 'different'               - a distinguishing sequence was found;
      * 'equivalent-within-bound' - the bounded space was exhausted and no
                                    sequence is accepted by exactly one side;
      * 'inconclusive-budget'     - the state budget ran out first.
    """

    status: str
    distinguishing_tokens: Optional[TokenSeq]
    distinguishing_text: Optional[str]
    side_a: Optional[SideReport]
    side_b: Optional[SideReport]
    exhausted: bool
    nodes_expanded: int
    branches_pruned: int
    budget: int
    max_tokens: int
    warnings: List[str] = field(default_factory=list)

    @property
    def is_different(self) -> bool:
        return self.status == "different"

    def summary(self) -> str:
        lines = ["status: %s" % self.status]
        if self.distinguishing_tokens is not None:
            shown = " ".join(self.distinguishing_tokens) or "<empty>"
            lines.append("distinguishing tokens: %s" % shown)
            lines.append("distinguishing text: %r" % (self.distinguishing_text,))
            for name, side in (("A", self.side_a), ("B", self.side_b)):
                lines.append("side %s: %s" % (name, side.detail))
        lines.append(
            "explored: %d nodes expanded, %d branches pruned, budget %d, "
            "max_tokens %d, range %s"
            % (
                self.nodes_expanded,
                self.branches_pruned,
                self.budget,
                self.max_tokens,
                "exhausted" if self.exhausted else "NOT exhausted",
            )
        )
        for warning in self.warnings:
            lines.append("warning: %s" % warning)
        return "\n".join(lines)


def _validate_samples(machine_a, machine_b, token_samples):
    if not token_samples:
        raise ValueError("token sample table must not be empty")
    for token_type, lexemes in token_samples.items():
        if not lexemes:
            raise ValueError("token %r has no sample lexemes" % (token_type,))
        for label, machine in (("A", machine_a), ("B", machine_b)):
            if token_type not in machine.terminals:
                raise ValueError(
                    "token %r is not a terminal of grammar %s (terminals: %s)"
                    % (token_type, label, ", ".join(sorted(machine.terminals)))
                )


def _lexeme_warnings(machine, label, token_samples):
    warnings = []
    for token_type, lexemes in token_samples.items():
        for lexeme in lexemes:
            try:
                tokens = list(machine.lark.lex(lexeme))
            except Exception as exc:  # noqa: BLE001 - report, never fail
                warnings.append(
                    "grammar %s: sample %r for %s does not lex (%s)"
                    % (label, lexeme, token_type, exc)
                )
                continue
            if len(tokens) != 1 or tokens[0].type != token_type:
                got = ", ".join("%s(%r)" % (t.type, str(t)) for t in tokens) or "nothing"
                warnings.append(
                    "grammar %s: sample %r for %s lexes as %s; the reported "
                    "distinguishing text may not reproduce from raw text"
                    % (label, lexeme, token_type, got)
                )
    return warnings


def _parse_tree(machine, seq, lexemes):
    try:
        interactive = machine.lark.parse_interactive(start=machine.start_symbol)
        for token_type, lexeme in zip(seq, lexemes):
            interactive.feed_token(Token(token_type, lexeme))
        return interactive.feed_eof().pretty()
    except Exception:  # noqa: BLE001 - the tree is a best-effort extra
        return None


def _report_side(machine, seq, lexemes):
    """Feed the sequence from scratch for a pruning-independent verdict."""
    stack = machine.start_stack
    for index, token_type in enumerate(seq):
        nxt = machine.step(stack, token_type)
        if nxt is None or nxt is _ACCEPT:
            expected = ", ".join(machine.expected_tokens(stack)) or "-"
            return SideReport(
                False,
                "rejected at token %d (%s); expected one of: %s"
                % (index + 1, token_type, expected),
            )
        stack = nxt
    if machine.accepts_eof(stack):
        return SideReport(True, "accepted", _parse_tree(machine, seq, lexemes))
    expected = ", ".join(machine.expected_tokens(stack)) or "-"
    return SideReport(
        False, "rejected at end of input; expected one of: %s" % expected
    )


def compare_grammars(
    grammar_a: str,
    grammar_b: str,
    token_samples: Dict[str, Sequence[str]],
    max_tokens: int,
    *,
    start: Optional[str] = None,
    budget: int = 200000,
) -> CompareResult:
    """Compare two Lark grammars over a bounded token-sequence space.

    Parameters:
        grammar_a, grammar_b: Lark grammar texts (EBNF).
        token_samples: mapping of terminal name to non-empty sample lexemes.
            The search runs over token *types*; the first lexeme of each type
            is used to render the distinguishing text.
        max_tokens: maximum sequence length considered (inclusive).
        start: common start rule; defaults to each grammar's default.
        budget: maximum number of search nodes expanded.

    Returns a CompareResult.  When status is 'different',
    distinguishing_tokens is the shortest sequence accepted by exactly one
    grammar and side_a/side_b describe each parse outcome.
    """
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    machine_a = LalrMachine(grammar_a, start=start)
    machine_b = LalrMachine(grammar_b, start=start)
    token_samples = {k: list(v) for k, v in token_samples.items()}
    _validate_samples(machine_a, machine_b, token_samples)
    warnings = _lexeme_warnings(machine_a, "A", token_samples)
    warnings += _lexeme_warnings(machine_b, "B", token_samples)

    token_types = list(token_samples)
    machines = (machine_a, machine_b)

    def prune(side, stack, length):
        """None out a stack that cannot accept within the remaining budget."""
        if stack is None:
            return None
        remaining = max_tokens - length
        if machines[side].min_completion(stack, remaining) is None:
            return None
        return stack

    root = ((), prune(0, machine_a.start_stack, 0), prune(1, machine_b.start_stack, 0))
    if root[1] is None and root[2] is None:
        # Neither grammar can accept anything within the bound.
        return CompareResult(
            status="equivalent-within-bound",
            distinguishing_tokens=None,
            distinguishing_text=None,
            side_a=None,
            side_b=None,
            exhausted=True,
            nodes_expanded=0,
            branches_pruned=1,
            budget=budget,
            max_tokens=max_tokens,
            warnings=warnings,
        )

    queue = deque([root])
    seen = {(root[1], root[2])}
    expanded = 0
    pruned = 0
    while queue:
        seq, stack_a, stack_b = queue.popleft()
        if expanded >= budget:
            return CompareResult(
                status="inconclusive-budget",
                distinguishing_tokens=None,
                distinguishing_text=None,
                side_a=None,
                side_b=None,
                exhausted=False,
                nodes_expanded=expanded,
                branches_pruned=pruned,
                budget=budget,
                max_tokens=max_tokens,
                warnings=warnings,
            )
        expanded += 1
        accepts_a = stack_a is not None and machine_a.accepts_eof(stack_a)
        accepts_b = stack_b is not None and machine_b.accepts_eof(stack_b)
        if accepts_a != accepts_b:
            lexemes = [token_samples[t][0] for t in seq]
            return CompareResult(
                status="different",
                distinguishing_tokens=seq,
                distinguishing_text=" ".join(lexemes),
                side_a=_report_side(machine_a, seq, lexemes),
                side_b=_report_side(machine_b, seq, lexemes),
                exhausted=False,
                nodes_expanded=expanded,
                branches_pruned=pruned,
                budget=budget,
                max_tokens=max_tokens,
                warnings=warnings,
            )
        if len(seq) >= max_tokens:
            continue
        for token_type in token_types:
            child_a = stack_a if stack_a is None else machine_a.step(stack_a, token_type)
            child_b = stack_b if stack_b is None else machine_b.step(stack_b, token_type)
            child_a = prune(0, child_a, len(seq) + 1)
            child_b = prune(1, child_b, len(seq) + 1)
            if child_a is None and child_b is None:
                pruned += 1
                continue
            key = (child_a, child_b)
            if key in seen:
                continue
            seen.add(key)
            queue.append((seq + (token_type,), child_a, child_b))

    return CompareResult(
        status="equivalent-within-bound",
        distinguishing_tokens=None,
        distinguishing_text=None,
        side_a=None,
        side_b=None,
        exhausted=True,
        nodes_expanded=expanded,
        branches_pruned=pruned,
        budget=budget,
        max_tokens=max_tokens,
        warnings=warnings,
    )
