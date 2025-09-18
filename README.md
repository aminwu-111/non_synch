# non_synch

This repository contains MATLAB code for replicating the results in the paper "Bayesian Inference for Non-Synchronously Observed Diffusions."

    pf_comp.m: Compares the variance of likelihood estimates produced by the Euler driven Particle Filter(PF) and diffusion bridge driven PF (Figure 1).
    
    Drift_OU.m: Implements MLPMMH for estimating the drift coefficient of the Ornstein-Uhlenbeck (OU) process using the X_18.mat dataset (Figure 2 left) .
    
    OU_SIG.m: Implements MLPMMH for estimating the diffusion coefficient of the OU process, requires loading the data file X_18.mat(Figure 2 right).
    
    zebra_beest.m: Runs MLPMMH for estimating the drift coefficient of the SLV model of the beest-zebra dataset, using data from beest_zebra.xlsx (Figure 3 left).
    
    Greek_cell.m: Applies MLPMMH for estimating the drift coefficient of the SLV model of to the Greek cell phone company share data (data included in the repository)(Figure 3 right).
    
    ARcomparison.m: Comparing acceptance rates of 0.07 (Sherlock et al) and 0.23 for PMCMC (Figure 4).

    DB_PMCMC.m and EM_pmcmc.m: These scripts compare two PMCMC approaches for parameter inference for the OU process.
        EM_pmcmc.m implements the naive Euler-driven PMCMC method with discretization level 4.
        DB_PMCMC.m implements the proposed diffusion bridge-driven PMCMC method with discretization level 4.
    Both scripts evaluate and compare the methods in terms of posterior accuracy and computational cost.

For the rates of the OU process, run the code 'mlpmmhOU_111' with data 'X.txt' on ibex(super computer in KAUST) for different levels (maybe 7-10) each with 64 runs and then calculate the mean at each level and then calculate the rates for mlpmmh. (Table 1)

For the rates of the SLV process, run the code 'pmmh_l5.m' with the dataset X-1.txt (named X.txt within the code, but distinguished here for clarity) on ibex for different levels(maybe 8-11) each with 64 runs and then calculate the mean at each level and then calculate the rates for pmmh. Similar using the 'mlpmmh_l5' for mlpmmh rates.(Table 2)
