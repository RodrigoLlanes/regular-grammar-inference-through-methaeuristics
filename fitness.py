from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Set, FrozenSet, Union, Generic, TypeVar

from parameters import *
from automatons import EpsilonNFA, NFA, DFA, RLG

def evaluate(dfa: DFA, samples: List[Tuple[str, bool]]) -> float:
    return mean_evaluation(dfa, samples)


def weight_evaluation(dfa: DFA, samples: List[Tuple[str, bool]], min_w: float, max_w: float) -> float:
    positives = []
    negatives = []
    for sample, accept in samples:
        (positives if accept else negatives).append(abs(int(not accept) - dfa.evaluate(sample)))
    if len(positives) == 0:
        return sum(negatives)/len(negatives)
    if len(negatives) == 0:
        return sum(positives)/len(positives)
    positives_fitness = sum(positives)/len(positives)
    negatives_fitness = sum(negatives)/len(negatives)
    return min(positives_fitness, negatives_fitness) * min_w + max(positives_fitness, negatives_fitness) * max_w


def min_evaluation(dfa: DFA, samples: List[Tuple[str, bool]]) -> float:
    positives = []
    negatives = []
    for sample, accept in samples:
        (positives if accept else negatives).append(abs(int(not accept) - dfa.evaluate(sample)))
    positives_fitness = sum(positives)/len(positives) if len(positives) > 0 else 1
    negatives_fitness = sum(negatives)/len(negatives) if len(negatives) > 0 else 1
    return min(positives_fitness, negatives_fitness)


def mean_evaluation(dfa: DFA, samples: List[Tuple[str, bool]]) -> float:
    result = 0
    for sample, accept in samples:
        result += abs(int(not accept) - dfa.evaluate(sample))
    return result / len(samples)


def RLG_evaluate(rlg: RLG, s: str, samples: List[Tuple[str, bool]]) -> float:
    enfa = EpsilonNFA.from_RLG(rlg, s)
    nfa = NFA.from_EpsilonNFA(enfa)
    dfa = DFA.from_NFA(nfa)
    return evaluate(dfa, samples)

