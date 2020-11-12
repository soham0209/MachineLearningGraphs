Please keep myGraph.py and cora folder in the same directory.

myGraph.py contains:
1. Graph data structure
2. BFS routine
3. BFS routine with visited nodes
4. Clustering coefficient of a node
5. Clustering coefficient of a graph

I have used python 3.8 to run the code. Please make sure that you can run python3 from the terminal, i.e. python3 opens a python3 interpreter. 


Problem 3.1:
To generate ER graph with given parameters run the script run.sh

bash ~: ./run.sh
Note you can generate an ER graph with specified n, p  100 times as following:
python3 Gen_graphs.py n p


Problem 3.2:
bash ~: python3 sim_cora.py p

As calculated p = 0.00148119 will generate an ER-graph with expected number of edges 5429.

Problem 3.3:
python3 watts_strogatz.py N K

N = number of nodes [2708 for HW1]
K = Number of neighbors connected [4 for HW1]

This also reports estimated rewiring probability q(within accuracy of 0.05) such that the Watts-Strogatz model has same clustering coefficient as that of the cora network.

Problem 4.1:
python3 spectral_clustering.py


Reports min N-cut(S), N-cut(S`), R-cut(S) and R-cut(S`)

Problem 4.2:
Same as above.

