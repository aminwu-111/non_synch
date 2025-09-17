# non_synch

This repository contains MATLAB code for replicating the results in the paper "Bayesian Inference for Non-Synchronously Observed Diffusions."

    pf_comp.m: Compares the likelihood estimates produced by the particle filter.
    
    Drift_OU.m: Implements MLP-MMH for estimating the drift coefficient of the Ornstein-Uhlenbeck (OU) process using the X_18.mat dataset.
    
    OU_SIG.m: Implements MLP-MMH for estimating the diffusion coefficient of the OU process, also using X_18.mat.
    
    zebra_beest.m: Runs MLP-MMH for the beest-zebra dataset, using data from beest_zebra.xlsx.
    
    Greek_cell.m: Applies MLP-MMH to the Greek cell phone company share data (data included in the repository).
    
    ARcomparison.m: Generates Figure 4, comparing acceptance rates of 0.07 and 0.23.


the code'DB_PMCMC.m' and 'EM_PMCMC.m' are used to compare the naive Euler-driven pMCMC approach (with a relatively coarse discretisation) to the proposed diffusion bridge-driven pMCMC approach, both in terms of posterior accuracy and computational cost

For the rates of the OU process, I run the code on ibex. code 'pmmhOU_11.m' uses the data 'X.txt', repeat the code for different levels(maybe 9-12) each with 64 runs and then calculate the mean over each level and then plots the rates. Similar for the 'mlpmmhOU_17'.

For the rates of the slv process, I run the code on ibex. code 'pmmhOU_l5.m' uses the data 'X-1.txt', repeat the code for different levels(maybe 8-11) each with 64 runs and then calculate the mean over each level and then plots the rates. Similar for the 'mlpmmhOU_17'.
