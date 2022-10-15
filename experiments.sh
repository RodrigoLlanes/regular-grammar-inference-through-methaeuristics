#!/bin/bash

source ../env/bin/activate

methaeuristics=("genetic" "annealing")
grammars=("(a*b*)+c(d*e*)+cf*")
for eur in ${methaeuristics[@]}
do
    for g in ${grammars[@]}
    do
        /usr/bin/python3 ../main.py $eur -r 11 -g "$g" -o "./${eur}_${g}.csv" 
    done
done
