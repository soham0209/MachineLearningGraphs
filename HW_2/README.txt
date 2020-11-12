Question 1.2: run
python3 gen_sbm.py

Default is n = 200, p = 0.16, q = 0.04. You can specify python3 gen_spm.py n p q also. 
This saves a graph 'sbm_graph.pkl'. It is used in the next questions.


Question 1.3:
python3 commute_time.py

inblock.png and accrossblock.png shows the two histograms of commute time within community and across community.

Question 1.4:
./run_ppr.sh
run_ppr.sh actually calls:
python3 ppr.py alpha k

You can run ppr.py with alpha and k with your choice.

Question 1.5: 
Not attempted

Question 2.5:
./run.sh

run.sh has hyperparameters mentioned.

Question 2.6:
You can run APPNP model by specifying --model_type=APPNP in python3 train.py --model_type=APPNP

Note if you need to change alpha, niter you need to change line 63 and 64.