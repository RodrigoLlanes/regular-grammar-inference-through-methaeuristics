from typing import List, Tuple
from random import choices, randint, shuffle

from automatons import DFA, NFA, EpsilonNFA
from parameters import *


def generate(dfa: DFA, positives: int, negatives: int) -> List[Tuple[str, bool]]:
    generator = dfa.words_iterator()
    cases: List[Tuple[str, bool]] = [(next(generator), True) for _ in range(positives)]

    max_size = len(cases[-1])
    while len(cases) < negatives + positives:
        word = choices(list(sigma), k=randint(0, max_size))
        word = ''.join(word)
        if not dfa.accepts(word):
            cases.append((word, False))
    return cases


def get_samples() -> List[Tuple[str, bool]]:
    enfa = EpsilonNFA.from_RegEx(grammar)
    nfa = NFA.from_EpsilonNFA(enfa)
    dfa = DFA.from_NFA(nfa)
    return generate(dfa, positives, negatives)

