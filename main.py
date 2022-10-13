from random import shuffle
import matplotlib.pyplot as plt
import networkx as nx
import graphviz

import parameters as params
from samples_generator import get_samples
from genetic import train as gen_train
from annealing import train as ann_train
from automatons import DFA, EpsilonNFA, NFA


def regex_test() -> None:
    enfa = EpsilonNFA.from_RegEx(params.grammar)
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
    data = get_samples()
    shuffle(data)
    test_index = int(len(data) * params.test_rate)
    test_data, train_data = data[:test_index], data[test_index:]

    (ann_hist_train, ann_hist_test), (ann_best_hist_train, ann_best_hist_test), best = ann_train(train_data, test_data)
    plt.plot(list(map(lambda x: 1-x, ann_hist_train)), label='Annealing train')
    #plt.plot(list(map(lambda x: 1-x, ann_hist_test)), label='Annealing test')
    plt.plot(list(map(lambda x: 1-x, ann_best_hist_train)), label='Annealing train (best)')
    #plt.plot(list(map(lambda x: 1-x, ann_best_hist_test)), label='Annealing test (best)')

    gen_historical_train, gen_historical_test, best = gen_train(train_data, test_data)
    plt.plot(list(map(lambda x: 1-x, gen_historical_train)), label='Genetical train')
    #plt.plot(list(map(lambda x: 1-x, gen_historical_test)), label='Genetical test')
    plt.legend()
    plt.show()

    rendered = best._dfa.render_dot()
    s = graphviz.Source(rendered, filename="tmp/result.gv", format="png")
    s.view()
   
    print('')
    print(f'Simulated annealing: {ann_best_hist_train[-1]}') 
    print(f'Genetical algorithm: {gen_historical_train[-1]}')


if __name__ == '__main__':
    main()
    #regex_test()

