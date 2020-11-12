#!/bin/bash
for n in 100 200 400 800 1600;
do
for p in 0.95 0.98 1.02 1.05;
do
python3 Gen_graphs.py $n $p
done
done
