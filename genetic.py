from collections import defaultdict
from random import choices, randint, sample
from typing import List, Tuple, Optional
from statistics import mean

from fitness import evaluate
from automatons import DFA, NFA, RLG, EpsilonNFA
import parameters as params

Genome = List[Optional[str]]


def initialize(length: int) -> Genome:
    genome = []
    for _ in range(length):
        option = randint(0, 1)
        if option:
            genome += [params.random_non_term(), params.random_terminal_or_epsilon()]
        else:
            genome += [params.random_non_term(), params.random_non_term(), params.random_terminal()]
    return genome


def mutate(genome: Genome, mutations: int = 3) -> None:
    for _ in range(mutations):
        index = randint(0, len(genome) - 1)
        if genome[index] is None:
            genome[index] = params.random_terminal_or_epsilon()
        elif genome[index] in params.sigma:
            if genome[index-1] in params.N and genome[index-2] in params.N:
                genome[index] = params.random_terminal()
            else:
                genome[index] = params.random_terminal_or_epsilon()
        else:
            genome[index] = params.random_non_term()


# Asumming len(a) == len(b) and len(a) > splits
def recombine(a: Genome, b: Genome, splits: int = 2) -> Genome:
    splited_a = _split_genome(a)
    splited_b = _split_genome(b)
    n = len(splited_a)
    split_indexes = [0] + list(sorted(sample(range(1, n), splits))) + [n]
    res = []
    for i in range(1, len(split_indexes)):
        genome = splited_a if i%2 == 0 else splited_b
        res += sum(genome[split_indexes[i-1]:split_indexes[i]], start=[])
    return res


def _split_genome(genome: Genome) -> List[List[Optional[str]]]:
    genome = genome.copy()
    
    result = []
    while len(genome):
        symbol, first = genome.pop(0), genome.pop(0)
        if first is params.epsilon:
            result += [[symbol], [None]]
        elif first in params.sigma:
            result += [[symbol], [first]]
        else:
            result += [[symbol], [first, genome.pop(0)]]
    return result


def decode(genome: Genome) -> DFA:
    genome = genome.copy()
    
    grammar: RLG = defaultdict(set)
    while len(genome):
        symbol, first = genome.pop(0), genome.pop(0)
        if first is params.epsilon:
            grammar[symbol].add(None)
        elif first in params.sigma:
            grammar[symbol].add((first,))
        else:
            grammar[symbol].add((first, genome.pop(0)))
    enfa = EpsilonNFA.from_RLG(grammar, params.S)
    nfa = NFA.from_EpsilonNFA(enfa)
    dfa = DFA.from_NFA(nfa)
    return dfa


class Individual:
    def __init__(self, genome: Optional[Genome] = None, genome_length: int = 20) -> None:
        self._genome: Genome = genome if genome is not None else initialize(genome_length)
        self._dfa: DFA = decode(self._genome)
        self._scores: List[float] = []

    @property
    def fitness(self) -> Tuple[bool, float]:
        return self._dfa.consistent, mean(self._scores)

    @property
    def last_fitness(self) -> float:
        return self._scores[-1]
    
    def calculate_fitness(self, samples: List[Tuple[str, bool]]) -> float:
        return evaluate(self._dfa, samples)

    def evaluate(self, samples: List[Tuple[str, bool]]) -> None:
        self._scores.append(self.calculate_fitness(samples))


def get_children(n: int, parents: List[Individual]) -> List[Individual]:
    children = []

    parents = choices(parents, weights=[sum(parent.fitness) for parent in parents], k=n*2)
    for i in range(0, n*2, 2):
        a, b = parents[i], parents[i+1]
        child = recombine(a._genome, b._genome, params.splits)
        mutate(child, params.mutations)
        children.append(Individual(genome=child))
    return children


def iteration(population: List[Individual], samples: List[Tuple[str, bool]]) -> List[Individual]:
    for ind in population:
        ind.evaluate(samples)

    population.sort(key=lambda x: x.fitness)
    n = int(len(population) * params.replacement_rate)
    return get_children(n, population) + population[n:]


def best_fitness(population:List[Individual], samples: List[Tuple[str, bool]]) -> float:
    return max(evaluate(decode(individual._genome), samples) for individual in population)

 
def best_individual(population: List[Individual], samples: List[Tuple[str, bool]]) -> Individual:
    return sorted(population, key= lambda x: evaluate(decode(x._genome), samples))[-1]


def train(train_data: List[Tuple[str, bool]], test_data: List[Tuple[str, bool]] = []) -> Tuple[List[float], List[float], Individual]:
    population = [Individual(genome_length=params.genome_size) for _ in range(params.population_size)]

    historical_test = []
    historical_train = []
    for _ in range(params.iterations):
        population = iteration(population, train_data)
        historical_train.append(population[-1].fitness[1])
        historical_test.append(population[-1].calculate_fitness(test_data))
    return historical_train, historical_test, population[-1]

