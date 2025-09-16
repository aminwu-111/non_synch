# non_synch

the code for the paper Bayesian Inference for Non-Synchronously Observed Diffusions 

the code 'pf_comp.m' is the code used to compare the likelihood for the particle filter.

the code 'Drift_OU.m' is the mlpmmh for the drift coefficient of the OU pporcess using 'X_18.mat' data

the code "OU_SIG.m' is the mlpmmh for the diffusion coefficient of the OU pporcess using 'X_18.mat' data

the code 'zebra_beest.m' is the mlpmmh for the beest zebra data using the 'beest_zebra.xlsx' data

the code 'Greek_cell.m' is for the mlpmmh for the Greek cell phone company share data; the data is inside 

For the rates of the OU process, I run the code on ibex. code 'pmmhOU_11.m' uses the data 'X.txt', repeat the code for different levels(maybe 9-12) each with 64 runs and then calculate the mean over each level and then plots the rates. Similar for the 'mlpmmhOU_17'.

For the rates of the slv process, I run the code on ibex. code 'pmmhOU_l5.m' uses the data 'X-1.txt', repeat the code for different levels(maybe 8-11) each with 64 runs and then calculate the mean over each level and then plots the rates. Similar for the 'mlpmmhOU_17'.
