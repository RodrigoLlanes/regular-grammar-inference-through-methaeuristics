from __future__ import annotations

from copy import deepcopy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, Union, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Generic, Set

from utils import IdentityDefaultdict, IdentityFrozenSet, IdentitySet 


epsilon = None
T = TypeVar('T')
U = TypeVar('U')


class Node(ABC, Generic[T]):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def add_transition(self, term: T, dest: Node) -> None:
        raise NotImplementedError()

    @abstractmethod
    def transitions(self) -> Iterable[Tuple[T, Node]]:
        raise NotImplementedError


class _ENFA_Node(Node[Optional[str]]):
    def __init__(self) -> None:
        self._transitions: Dict[Optional[str], IdentitySet[_ENFA_Node]] = defaultdict(IdentitySet)
    
    def add_transition(self, term: Optional[str], dest: _ENFA_Node) -> None:
        self._transitions[term].add(dest)

    def transitions(self) -> Iterable[Tuple[Optional[str], _ENFA_Node]]:
        for term, dests in self._transitions.items():
            for dest in dests:
                yield term, dest


class _NFA_Node(Node[str]):
    def __init__(self) -> None:
        self._transitions: Dict[str, IdentitySet[_NFA_Node]] = defaultdict(IdentitySet)
    
    def add_transition(self, term: str, dest: _NFA_Node) -> None:
        self._transitions[term].add(dest)

    def transitions(self) -> Iterable[Tuple[str, _NFA_Node]]:
        for term, dests in self._transitions.items():
            for dest in dests:
                yield term, dest


class _DFA_Node(Node[str]):
    def __init__(self) -> None:
        self._transitions: Dict[str, _DFA_Node] = {}
    
    def add_transition(self, term: str, dest: _DFA_Node) -> None:
        self._transitions[term] = dest

    def transitions(self) -> Iterable[Tuple[str, _DFA_Node]]:
        for term, dest in self._transitions.items():
            yield term, dest

    def neightbours(self) -> IdentitySet[_DFA_Node]:
        return IdentitySet(self._transitions.values())

    def __contains__(self, key: str) -> bool:
        return key in self._transitions

    def __getitem__(self, key: str) -> _DFA_Node:
        return self._transitions[key]


def node_namer() -> Callable[[Node], Tuple[str, bool]]:
    ni = 0
    mem = {}
    def node_name(node: Node) -> Tuple[str, bool]:
        nonlocal ni
        key = id(node)
        first = key not in mem
        if first:
            mem[key] = f'q{ni}'
            ni += 1
        
        return mem[key], first
    return node_name


RLGProductions = Optional[Union[Tuple[str, str], Tuple[str]]]
RLG = Dict[str, Set[RLGProductions]] 


class EpsilonNFA:
    def __init__(self, empty: bool = True) -> None:
        self.initial_state: _ENFA_Node = _ENFA_Node()
        self.final_states: IdentitySet = IdentitySet()
        if not empty:
            self.final_states.add(self.initial_state)
    
    def _symbols(self) -> set[str]:
        res = set()
        visited = IdentitySet([self.initial_state])
        stack = [self.initial_state]
        while len(stack):
            for nt, n in stack.pop().transitions():
                res.add(nt)
                if n not in visited:
                    visited.add(n)
                    stack.append(n)
        return res

    def _nodes(self) -> IdentitySet[_ENFA_Node]:
        visited = IdentitySet([self.initial_state])
        stack = [self.initial_state]
        while len(stack):
            for _, n in stack.pop().transitions():
                if n not in visited:
                    visited.add(n)
                    stack.append(n)
        return visited

    def asdict(self) -> Tuple[Dict[str, Dict[Optional[str], Set[str]]], str, Set[str]]:
        d: Dict[str, Dict[Optional[str], Set[str]]] = defaultdict(lambda: defaultdict(set))
        node_name = node_namer()
        stack = [self.initial_state]
        while len(stack):
            node = stack.pop(0)
            name, _ = node_name(node)
            for nt, n in node.transitions():
                n_name, first = node_name(n)
                d[name][nt].add(n_name)
                if first:
                    stack.append(n)
        return d, node_name(self.initial_state)[0], {node_name(node)[0] for node in self.final_states}

    def render_dot(self) -> str:
        nodes = ['node [shape = point]; qi']
        edges = ['qi -> q0;']
        
        edges_names = defaultdict(set)
        node_name = node_namer()
        stack = [self.initial_state]
        while len(stack):
            node = stack.pop(0)
            name, _ = node_name(node)
            nodes.append(f'node [shape = {"circle" if node not in self.final_states else "doublecircle"}, label = "{name}"]; {name}')
            for nt, n in node.transitions():
                n_name, first = node_name(n)
                nt = nt if nt is not None else 'ɛ'
                edges_names[name, n_name].add(nt)
                if first:
                    stack.append(n)
        for (q0, q1), symbs in edges_names.items():
            edges.append(f'{q0} -> {q1} [label = "{", ".join(symbs)}"];')
        return 'digraph graph_rendered{\n    ' + '\n    '.join(nodes + edges) + '\n}'

    @staticmethod
    def combine(a: EpsilonNFA, b: EpsilonNFA, copy: bool = True) -> EpsilonNFA:
        ac = deepcopy(a) if copy else a
        bc = deepcopy(b) if copy else b
        for state in ac.final_states:
            state.add_transition(epsilon, bc.initial_state)
        ac.final_states = bc.final_states
        return ac
    
    @staticmethod
    def from_RLG(grammar: RLG, s: str) -> EpsilonNFA:
        nodes = defaultdict(_ENFA_Node)
        final = 0
        final_states = IdentitySet([nodes[final]])
        states = {final, s}
        unknown = [s]
        
        while len(unknown):
            current = unknown.pop()
            for prod in grammar[current]:
                if prod is None:
                    final_states.add(nodes[current])
                elif len(prod) == 1:
                    symbol = prod[0]
                    nodes[current].add_transition(symbol, nodes[final])
                else:
                    state, symbol = prod
                    nodes[current].add_transition(symbol, nodes[state])
                    if state not in states:
                        states.add(state)
                        unknown.append(state)
        enfa = EpsilonNFA()
        enfa.initial_state = nodes[s]
        enfa.final_states = final_states
        return enfa

    @staticmethod
    def from_RegEx(regex: str) -> EpsilonNFA:
        if not len(regex):
            return EpsilonNFA(empty=False)
        elif len(regex) == 1:
            res = EpsilonNFA()
            final = _ENFA_Node()
            res.initial_state.add_transition(regex, final)
            res.final_states.add(final)
            return res
        elif regex[0] == '(':
            i = regex.find(')')
            left = EpsilonNFA.from_RegEx(regex[1:i])
            if regex[i+1] == '*':
                for final in left.final_states:
                    final.add_transition(epsilon, left.initial_state)
                left.final_states.add(left.initial_state)
                right = EpsilonNFA.from_RegEx(regex[i + 2:])
            elif regex[i+1] == '+':
                for final in left.final_states:
                    final.add_transition(epsilon, left.initial_state)
                right = EpsilonNFA.from_RegEx(regex[i + 2:])
            else:
                right = EpsilonNFA.from_RegEx(regex[i + 1:])
            return EpsilonNFA.combine(left, right)
        elif regex[1] == '*':
            left = EpsilonNFA(empty=False)
            left.initial_state.add_transition(regex[0], left.initial_state)
            right = EpsilonNFA.from_RegEx(regex[2:])
            return EpsilonNFA.combine(left, right)
        elif regex[1] == '+':
            left = EpsilonNFA()
            final = _ENFA_Node()
            left.initial_state.add_transition(regex[0], final)
            final.add_transition(regex[0], final)
            left.final_states.add(final)
            right = EpsilonNFA.from_RegEx(regex[2:])
            return EpsilonNFA.combine(left, right)
        else:
            left = EpsilonNFA()
            final = _ENFA_Node()
            left.initial_state.add_transition(regex[0], final)
            left.final_states.add(final)
            right = EpsilonNFA.from_RegEx(regex[1:])
            return EpsilonNFA.combine(left, right)


class NFA(EpsilonNFA):
    def __init__(self, empty: bool = True) -> None:
        self.initial_state: _NFA_Node = _NFA_Node()
        self.final_states: IdentitySet = IdentitySet()
        if not empty:
            self.final_states.add(self.initial_state)

    @staticmethod
    def from_EpsilonNFA(automaton: EpsilonNFA) -> NFA:
        nodes = automaton._nodes()
        symbols = automaton._symbols()

        e_closure = IdentityDefaultdict(IdentitySet)
        for q0 in nodes:
            e_closure[q0].add(q0)
            stack = [q0]
            while len(stack):
                node = stack.pop()
                for nt, q1 in node.transitions():
                    if nt is epsilon and q1 not in e_closure[q0]:
                        e_closure[q0].add(q1)
                        stack.append(q1)

        delta = IdentityDefaultdict(lambda: defaultdict(IdentitySet))
        for q0 in nodes:
            for s in symbols - {epsilon, }:
                for node in e_closure[q0]:
                    for nt, q1 in node.transitions():
                        if nt == s:
                            delta[q0][s].extend(e_closure[q1])  # TODO: con .add(q1), parece que tambien funciona y queda más bonito, pero se supone que el algoritmo es así

        new_nodes = IdentityDefaultdict(_NFA_Node)
        for q0, transitions in delta.items():
            for nt, states in transitions.items():
                for q1 in states:
                    new_nodes[q0].add_transition(nt, new_nodes[q1])
        
        res = NFA()
        res.initial_state = new_nodes[automaton.initial_state]
        res.final_states = IdentitySet([new_nodes[node] for node in nodes if len(automaton.final_states.intersection(e_closure[node]))])
        return res 


class DFA(NFA):
    def __init__(self, empty: bool = True) -> None:
        self.initial_state: _DFA_Node = _DFA_Node()
        self.final_states: IdentitySet = IdentitySet()
        if not empty:
            self.final_states.add(self.initial_state)
        self.consistent: bool = self.is_consistent()

    def is_consistent(self) -> bool:
        finals = IdentitySet()
        visited = IdentitySet()
        stack = [self.initial_state]
        
        while len(stack):
            state = stack.pop()
            neightbours = state.neightbours()
            future = neightbours - visited
            for neightbour in future:
                visited.add(neightbour)
                stack.append(neightbour)
            if state in self.final_states or len(neightbours.intersection(finals)) > 0:
                finals.add(state)
                continue
            if len(future) > 0:
                continue
            return False
        return True

    def words_iterator(self) -> Iterator[str]:
        nodes: List[Tuple[str, _DFA_Node]] = [('', self.initial_state)]
        while len(nodes):
            string, state = nodes.pop(0)
            if state in self.final_states:
                yield string
            for symbol, target in state.transitions():
                nodes.append((symbol + string, target))

    def evaluate(self, word: Union[List[str], str]) -> float:
        current = self.initial_state
        for i, symbol in enumerate(reversed(word)):
            if symbol not in current:
                return i / len(word)
            current = current[symbol]
        return 1 if current in self.final_states else 1 - 1 / ((len(word) + 1) * 2)

    def accepts(self, word: Union[List[str], str]) -> bool:
        return self.evaluate(word) == 1
    
    @staticmethod
    def from_NFA(nfa: NFA) -> DFA:
        states = set()
        final_states = set()
        new_nodes = defaultdict(_DFA_Node)
        unknown = [IdentityFrozenSet([nfa.initial_state])]
        while len(unknown):
            q = unknown.pop()
            states.add(q)
            if len(q.intersection(nfa.final_states)):
                final_states.add(q)
        
            transitions = defaultdict(IdentitySet)
            for state in q:
                for k, v in state.transitions():
                    transitions[k].add(v)
            
            transitions = {k: IdentityFrozenSet(v) for k, v in transitions.items()}
            for k, v in transitions.items():
                new_nodes[q].add_transition(k, new_nodes[v])
                if v not in states:
                    unknown.append(v)
        dfa = DFA()
        dfa.initial_state = new_nodes[IdentityFrozenSet([nfa.initial_state])]
        dfa.final_states = IdentitySet(new_nodes[n] for n in final_states)
        return dfa

