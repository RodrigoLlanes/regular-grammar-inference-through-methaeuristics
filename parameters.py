from random import sample
from typing import Optional

######################################
##       Tweakable parameters       ##
######################################

# General grammar parameters
S = 'S'
N = {S, 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'}
sigma = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
target_grammar = 'a+b+'

# Train data parameters
positives = 500
negatives = 500
test_rate = 0.1

# Genetical algorithms parameters
splits = 4
mutations = 6
iterations = 100
genome_size = 20
population_size = 200
replacement_rate = 0.4

# Simulated annealing algorithm parameters
initial_temperature = 90
alpha = 0.01
final_temperature = 0.1
cost_scale = 10000
neightbour_distance = 10



######################################
##     Non-tweakable parameters     ##
######################################

epsilon = None
terminal = sigma.union({epsilon})
all_symbols = N.union(terminal)

def random_terminal() -> str:
    return sample(sigma, 1)[0]

def random_terminal_or_epsilon() -> Optional[str]:
    return sample(terminal, 1)[0]

def random_non_term() -> str:
    return sample(N, 1)[0]

def random_symbol() -> Optional[str]:
    return sample(all_symbols, 1)[0]


