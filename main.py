from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import argparse

import parameters as params
from samples_generator import get_samples
from genetic import train as gen_train
from annealing import train as ann_train
from automatons import DFA, EpsilonNFA, NFA


def regex_test(grammar) -> None:
    enfa = EpsilonNFA.from_RegEx(grammar)
    rendered = enfa.render_dot()
    s = graphviz.Source(rendered, filename="tmp/enfa.gv", format="png")
    s.view()
    
    nfa = NFA.from_EpsilonNFA(enfa)
    rendered = nfa.render_dot()
    s = graphviz.Source(rendered, filename="tmp/nfa.gv", format="png")
    s.view()
    
    dfa = DFA.from_NFA(nfa)
    rendered = dfa.render_dot()
    s = graphviz.Source(rendered, filename="tmp/dfa.gv", format="png")
    s.view()



def main() -> None:
    parser = argparse.ArgumentParser(description='Regular grammar inference through methaeuristic algorithms')
    parser.add_argument('mode', choices=['genetic', 'annealing'], help='Methaeuristic to use')
    parser.add_argument('-r', dest='repetitions', type=int, default=1, help='Times the experiment will be repeated')
    parser.add_argument('-o', dest='out', type=str, help='Out file path')
    parser.add_argument('-g', dest='grammar', type=str, help='Grammar to infer.')

    args = parser.parse_args()
    
    target = params.target_grammar
    if args.grammar is not None:
        target = args.grammar
    
    data = get_samples(target)
    test_index = int(len(data) * params.test_rate)
    test_data, train_data = data[:test_index], data[test_index:]
    if args.mode == 'annealing':
        out_data = [[''] + list(np.arange(params.initial_temperature, params.final_temperature, -params.alpha))]
        for i in range(args.repetitions):
            (current_hist, _), (best_hist, _), best = ann_train(train_data, test_data)
            print(f'Score: {best_hist[-1]}')
            plt.plot(list(map(lambda x: 1-x, current_hist)), label='Current solution' + f' {i}'if args.repetitions > 1 else '')
            plt.plot(list(map(lambda x: 1-x, best_hist)), label='Best solution' + f' {i}'if args.repetitions > 1 else '')
            out_data.append([i]+best_hist)
        if args.out is not None:
            with open(args.out, 'w') as file:
                for i in range(len(out_data[0])):
                    file.write(','.join(str(col[i]) for col in out_data) + '\n')
    elif args.mode == 'genetic':
        out_data = [[''] + list(range(params.iterations))]
        for i in range(args.repetitions):
            hist, _, best = gen_train(train_data, test_data)
            print(f'Score: {hist[-1]}')
            plt.plot(list(map(lambda x: 1-x, hist)), label='Best solution'  + f' {i}'if args.repetitions > 1 else '')
            out_data.append([i] + hist)
        if args.out is not None:
            with open(args.out, 'w') as file:
                for i in range(len(out_data[0])):
                    file.write(','.join(str(col[i]) for col in out_data) + '\n')
    
    if args.out is None:
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()

