"""Witness search for LALR parse-table conflicts.

When the LALR construction reports a conflict (shift/reduce in strict mode,
or reduce/reduce), this module searches for a *witness*: a sequence of
terminals that drives the parser state machine into the conflicting state.

The witness is verified by replaying it through the parser state machine.
Each competing action is then extended towards a full parse. If two actions
can complete *the same input*, the grammar is proven ambiguous at that point
and both derivations are reported. Otherwise the report only claims that the
conflict state is reachable - which, for LALR, may be an artifact of merged
states rather than a true ambiguity.

The search is bounded by an input-length budget and a state budget; when a
budget is exceeded, the report says so explicitly.

Witnesses are sequences of *terminals*, never nonterminals. Nonterminal
transitions in the automaton are expanded to their minimal terminal
derivation (the empty string for nullable rules), so nonterminals never leak
into the reported input.
"""

from collections import deque
import heapq
import itertools

from ..grammar import Terminal

END = Terminal('$END')


class WitnessBudget:
    """Budgets bounding the conflict witness search."""

    def __init__(self, max_input_len=16, max_states=4000, max_prefixes=32, max_derivations=16):
        self.max_input_len = max_input_len      # max terminals in any candidate input
        self.max_states = max_states            # max parser configurations/itemsets visited (total)
        self.max_prefixes = max_prefixes        # max distinct conflict-reaching prefixes to try
        self.max_derivations = max_derivations  # max full parses collected per action

    def describe(self):
        return ('max_input_len=%d, max_states=%d, max_prefixes=%d, max_derivations=%d'
                % (self.max_input_len, self.max_states, self.max_prefixes, self.max_derivations))


DEFAULT_BUDGET = WitnessBudget()


class _BudgetExceeded(Exception):
    pass


def format_rule(rule):
    "Format a rule, keeping its (possibly import-mangled) origin name and alias."
    text = '<%s : %s>' % (rule.origin.name, ' '.join(sym.name for sym in rule.expansion))
    if rule.alias:
        text += ' (alias: %s)' % rule.alias
    return text


def _format_input(terminals):
    return ' '.join(sym.name for sym in terminals) or '(empty input)'


def _min_terminal_derivations(analyzer, max_len):
    """For each nonterminal, a shortest terminal string it can derive (capped at max_len).

    Nullable nonterminals map to the empty tuple. Nonterminals that cannot
    derive any terminal string are absent from the result.
    """
    derivs = {}
    changed = True
    while changed:
        changed = False
        for origin, rules in analyzer.lr0_rules_by_origin.items():
            best = derivs.get(origin)
            best_len = len(best) if best is not None else max_len + 1
            for rule in rules:
                total = 0
                parts = []
                for sym in rule.expansion:
                    if sym.is_term:
                        parts.append(sym)
                        total += 1
                    elif sym in derivs:
                        parts.extend(derivs[sym])
                        total += len(derivs[sym])
                    else:
                        break
                    if total >= best_len:
                        break
                else:
                    if total < best_len:
                        derivs[origin] = tuple(parts)
                        changed = True
    return derivs


def _end_itemsets(analyzer):
    "Itemsets in which a root rule ($root_*) is satisfied, i.e. accept states."
    ends = set()
    for itemset in analyzer.lr0_itemsets:
        for rp in itemset.kernel:
            if rp.is_satisfied and rp.rule.origin.name.startswith('$root_'):
                ends.add(itemset)
                break
    return ends


def _reduce(stack, rule):
    "Apply a reduce of 'rule' to a parser stack (tuple of itemsets). None if impossible."
    if rule.expansion:
        if len(stack) <= len(rule.expansion):
            return None
        stack = stack[:-len(rule.expansion)]
    goto = stack[-1].transitions.get(rule.origin)
    if goto is None:
        return None
    return stack + (goto,)


class _Search:
    "Shared state for the bounded searches below."

    def __init__(self, analyzer, budget):
        self.analyzer = analyzer
        self.budget = budget
        self.states_seen = 0

    def tick(self, what):
        self.states_seen += 1
        if self.states_seen > self.budget.max_states:
            raise _BudgetExceeded('state budget exceeded (%d configurations) during %s'
                                  % (self.budget.max_states, what))


def _walks_to_state(search, target):
    """Yield (start_name, terminals) that drive the LR(0) automaton from a start
    state into 'target', in increasing length order.

    Walks may contain cycles, so this is a potentially infinite generator;
    the caller bounds it via the budgets.
    """
    analyzer = search.analyzer
    derivs = _min_terminal_derivations(analyzer, search.budget.max_input_len)
    counter = itertools.count()
    queue = [(0, next(counter), name, state, ())
             for name, state in sorted(analyzer.lr0_start_states.items())]
    heapq.heapify(queue)
    while queue:
        length, _, start_name, state, prefix = heapq.heappop(queue)
        search.tick('prefix search')
        if state is target:
            yield start_name, prefix
            continue
        for sym, next_state in state.transitions.items():
            if sym.is_term:
                step = (sym,)
            else:
                step = derivs.get(sym)
                if step is None:
                    continue
            new_length = length + len(step)
            if new_length > search.budget.max_input_len - 1:
                continue
            heapq.heappush(queue, (new_length, next(counter), start_name, next_state, prefix + step))


def _replay_to_state(search, start_state, fixed_input, target):
    """Replay 'fixed_input' through the parser state machine (nondeterministically,
    to account for the conflict itself), and return every configuration whose top
    state is 'target' after consuming all of 'fixed_input' except the lookahead.

    Returns a list of (stack, actions).
    """
    target_pos = len(fixed_input) - 1
    results = []
    visited = set()
    queue = deque([((start_state,), 0, ())])
    while queue:
        stack, pos, actions = queue.popleft()
        if pos == target_pos and stack[-1] is target:
            results.append((stack, actions))
            continue
        key = (stack, pos)
        if key in visited:
            continue
        visited.add(key)
        search.tick('witness replay')
        la = fixed_input[pos]
        state = stack[-1]
        if pos < target_pos:
            next_state = state.transitions.get(la)
            if next_state is not None:
                queue.append((stack + (next_state,), pos + 1, actions + (('shift', la),)))
        for rule in sorted(state.lookaheads.get(la, ()), key=str):
            new_stack = _reduce(stack, rule)
            if new_stack is not None:
                queue.append((new_stack, pos, actions + (('reduce', rule),)))
    return results


def _find_completions(search, end_states, init_stack, init_input, init_pending, init_actions):
    """Extend a parser configuration towards full parses, generating input
    terminals on demand.

    Returns (accepted, truncated_reason). 'accepted' is a list of
    (input, actions); 'truncated_reason' is None unless a budget cut the
    search short. Each call gets its own state budget, so that one
    unlucky configuration cannot starve the other candidates.
    """
    budget = search.budget
    counter = itertools.count()
    queue = [(len(init_input), next(counter), init_stack, init_input, init_pending, init_actions)]
    visited = set()
    accepted = []
    accepted_inputs = set()
    states_seen = 0
    truncated = None
    while queue:
        _, _, stack, input_, pending, actions = heapq.heappop(queue)
        if pending == END and stack[-1] in end_states:
            if input_ not in accepted_inputs:
                accepted_inputs.add(input_)
                accepted.append((input_, actions))
                if len(accepted) >= budget.max_derivations:
                    return accepted, 'derivation budget exceeded (%d)' % budget.max_derivations
            continue
        key = (stack, pending)
        if key in visited:
            continue
        visited.add(key)
        states_seen += 1
        if states_seen > budget.max_states:
            return accepted, ('state budget exceeded (%d configurations) '
                              'while extending an action to a full parse' % budget.max_states)
        state = stack[-1]
        if pending is None:
            # choose the next lookahead terminal (or $END)
            candidates = set(state.lookaheads)
            candidates.update(sym for sym in state.transitions if sym.is_term)
            if stack[-1] in end_states:
                candidates.add(END)
            for la in sorted(candidates, key=lambda s: s.name):
                heapq.heappush(queue, (len(input_), next(counter), stack, input_, la, actions))
            continue
        if pending == END:
            # only reduces (and the accept above) are possible on $END
            for rule in sorted(state.lookaheads.get(END, ()), key=str):
                new_stack = _reduce(stack, rule)
                if new_stack is not None:
                    heapq.heappush(queue, (len(input_), next(counter), new_stack, input_, END,
                                           actions + (('reduce', rule),)))
            continue
        next_state = state.transitions.get(pending)
        if next_state is not None:
            if len(input_) + 1 > budget.max_input_len:
                truncated = 'input length budget exceeded (%d terminals)' % budget.max_input_len
            else:
                heapq.heappush(queue, (len(input_) + 1, next(counter), stack + (next_state,),
                                       input_ + (pending,), None, actions + (('shift', pending),)))
        for rule in sorted(state.lookaheads.get(pending, ()), key=str):
            new_stack = _reduce(stack, rule)
            if new_stack is not None:
                heapq.heappush(queue, (len(input_), next(counter), new_stack, input_, pending,
                                       actions + (('reduce', rule),)))
    return accepted, truncated


class ConflictWitness:
    """Result of analyzing a single parse-table conflict."""

    def __init__(self):
        self.prefix = None          # terminals reaching the conflict state
        self.verified = False       # prefix verified by replaying the state machine
        self.common_input = None    # input that two competing actions both parse completely
        self.derivations = None     # (actions1, actions2) for common_input
        self.action_labels = None   # labels of the two actions of the proven derivations
        self.exhaustive = False     # search space exhausted (no budget truncation)
        self.reason = None          # why no witness / no proof was found


def _action_label(action, la=None):
    kind, payload = action
    if kind == 'shift':
        return 'shift %s' % (payload if payload is not None else la).name
    return 'reduce %s' % format_rule(payload)


def analyze_conflict(analyzer, itemset, la, actions, budget=None):
    """Search for a witness for the conflict in 'itemset' on lookahead 'la'.

    'actions' is a list of the competing actions as ('shift', None) or
    ('reduce', rule) pairs. Returns a ConflictWitness.
    """
    budget = budget or DEFAULT_BUDGET
    witness = ConflictWitness()
    search = _Search(analyzer, budget)
    end_states = _end_itemsets(analyzer)
    truncated_reasons = []
    prefixes_tried = 0
    walks_exhausted = False
    seen_prefixes = set()
    try:
        for start_name, prefix in _walks_to_state(search, itemset):
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            if prefixes_tried >= budget.max_prefixes:
                truncated_reasons.append('prefix budget exceeded (%d prefixes)' % budget.max_prefixes)
                break
            prefixes_tried += 1
            if witness.prefix is None:
                witness.prefix = prefix
            configs = _replay_to_state(search, analyzer.lr0_start_states[start_name],
                                       prefix + (la,), itemset)
            for stack, replay_actions in configs:
                witness.verified = True
                completions = []
                for kind, rule in actions:
                    if kind == 'shift':
                        init = (stack + (itemset.transitions[la],), prefix + (la,), None,
                                replay_actions + (('shift', la),))
                    else:
                        new_stack = _reduce(stack, rule)
                        if new_stack is None:
                            completions.append(None)
                            continue
                        init = (new_stack, prefix, la,
                                replay_actions + (('reduce', rule),))
                    accepted, truncated_reason = _find_completions(search, end_states, *init)
                    if truncated_reason:
                        truncated_reasons.append(truncated_reason)
                    completions.append(accepted)
                # look for one input that two different actions both parse
                for i in range(len(actions)):
                    if not completions[i]:
                        continue
                    inputs_i = {inp: acts for inp, acts in completions[i]}
                    for j in range(i + 1, len(actions)):
                        if not completions[j]:
                            continue
                        for inp, acts_j in completions[j]:
                            if inp in inputs_i:
                                witness.prefix = prefix
                                witness.common_input = inp
                                witness.derivations = (inputs_i[inp], acts_j)
                                witness.action_labels = (_action_label(actions[i], la),
                                                         _action_label(actions[j], la))
                                return witness
        else:
            walks_exhausted = True
    except _BudgetExceeded as e:
        truncated_reasons.append(str(e))

    if witness.prefix is None:
        witness.reason = (truncated_reasons[0] if truncated_reasons else None) or \
            'no input reaching the conflict state found within budget'
    else:
        witness.exhaustive = walks_exhausted and not truncated_reasons
        witness.reason = truncated_reasons[0] if truncated_reasons else None
    return witness


def format_witness_report(analyzer, itemset, la, actions, budget=None):
    """Build the witness section of a conflict error message.

    'actions' is a list of ('shift', None) / ('reduce', rule) pairs.
    """
    budget = budget or DEFAULT_BUDGET
    witness = analyze_conflict(analyzer, itemset, la, actions, budget)
    lines = []
    lines.append('Witness search (budget: %s):' % budget.describe())
    if witness.prefix is None:
        lines.append('  Could not find an input reaching the conflict state: %s'
                     % witness.reason)
        return '\n'.join(lines)

    lines.append('  Conflict state is reachable, e.g. on input: %s'
                 % _format_input(witness.prefix))
    lines.append('  with lookahead (next token): %s' % la.name)
    if witness.verified:
        lines.append('  (reachability verified by replaying the parser state machine)')
    lines.append('  Competing actions at that point:')
    for i, action in enumerate(actions, 1):
        lines.append('    %d. %s' % (i, _action_label(action, la)))

    if witness.common_input is not None:
        lines.append('  AMBIGUITY PROVEN: the input "%s" has two different parse trees.'
                     % _format_input(witness.common_input))
        for n, (label, derivation) in enumerate(zip(witness.action_labels,
                                                    witness.derivations), 1):
            lines.append('    Derivation %d (%s first):' % (n, label))
            for kind, payload in derivation:
                lines.append('      %s' % _action_label((kind, payload)))
    else:
        lines.append('  NO two-tree witness found: the conflict state is reachable,')
        lines.append('  but no single input (up to %d terminals) is parsed to' % budget.max_input_len)
        lines.append('  completion by both actions.')
        if witness.exhaustive:
            lines.append('  The search was exhaustive within budget; the grammar may be')
            lines.append('  unambiguous (e.g. an LALR-only conflict from merged states,')
            lines.append('  or a conflict that is harmless at parse time).')
        else:
            lines.append('  The search was cut short (%s), so this is inconclusive:' % witness.reason)
            lines.append('  the grammar may be ambiguous beyond the search budget, or the')
            lines.append('  conflict may be an LALR-only artifact of merged states.')
    return '\n'.join(lines)
