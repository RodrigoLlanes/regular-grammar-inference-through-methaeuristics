from random import randint, sample, shuffle, uniform
from typing import List, Tuple, Optional
from math import exp

from genetic import Genome, initialize, decode
from fitness import evaluate
from automatons import DFA
import parameters as param


class Individual:
    def __init__(self, genome: Optional[Genome] = None, genome_length: int = 20) -> None:
        self._genome: Genome = genome if genome is not None else initialize(genome_length)
        self._dfa: DFA = decode(self._genome)
        self._score: float = 0

    @property
    def fitness(self) -> float:
        return (self._dfa.consistent + self._score) / 2

    @property
    def last_fitness(self) -> float:
        return self._score
    
    def calculate_fitness(self, samples: List[Tuple[str, bool]]) -> float:
        return evaluate(self._dfa, samples)

    def evaluate(self, samples: List[Tuple[str, bool]]) -> None:
        self._score = self.calculate_fitness(samples)



def random_neightbour(solution: Individual) -> Individual:
    genome = solution._genome.copy()
    
    for _ in range(param.neightbour_distance):
        index = randint(0, len(genome) - 1)
        
        if genome[index] in param.N:
            if genome[index-1] not in param.N:
                genome[index] = param.random_non_term()
            else:
                new_symbol = param.random_symbol()
                genome[index] = new_symbol
                if new_symbol not in param.N:
                    genome = genome[:index+1] + genome[index+2:]
        elif genome[index-2] in param.N:
            new_symbol = param.random_symbol()
            if new_symbol in param.N:
                genome = genome[:index-1] + [new_symbol, param.random_terminal()] + genome[index+1:]
            elif new_symbol is param.epsilon:
                genome = genome[:index-1] + [new_symbol] + genome[index+1:]
            else:
                genome[index] = new_symbol
        else:
            new_symbol = param.random_symbol()
            if new_symbol in param.N:
                genome = genome[:index] + [new_symbol, param.random_terminal()] + genome[index+1:]
            else:
                genome[index] = new_symbol
    return Individual(genome)


def train(train_data: List[Tuple[str, bool]], test_data: List[Tuple[str, bool]] = []) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float]], Individual]:
    solution = Individual(genome_length=param.genome_size)
    solution.evaluate(train_data)
    best = (solution.fitness, solution)

    historical_test = []
    historical_train = []

    hist_train_best = []
    hist_test_best = []
    
    t = param.initial_temperature

    while t > param.final_temperature:
        candidate = random_neightbour(solution)
        
        candidate.evaluate(train_data)

        cost = solution.fitness - candidate.fitness
        cost *= param.cost_scale
        
        if cost < 0 or uniform(0, 1) < exp(-cost / t):
            if candidate.fitness > best[0]:
                best = (candidate.fitness, candidate)
            solution = candidate

        
        historical_test.append(solution.calculate_fitness(test_data))
        historical_train.append(solution.calculate_fitness(train_data))

        hist_test_best.append(best[1].calculate_fitness(test_data))
        hist_train_best.append(best[1].calculate_fitness(train_data))
        t -= param.alpha

    return (historical_train, historical_test), (hist_train_best, hist_test_best), best[1]

