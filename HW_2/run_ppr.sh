#!/bin/bash

for a in 0.95 0.98;
do
for k in 5 10 15 20;
do
python3 ppr.py $a $k
done
done
