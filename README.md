# BNPDensityRegression_AdaptiveTruncation
Adaptive truncation algorithm for posterior inference of a Bayesian nonparametric multivariate density regression model.

This matlab code provides posterior inference for the multivariate density regression model with normalised weights. 

The folder Code: contains the matlab code for setting hyperparameters, defining link functions, posterior inference, and tools for prediction. The main file AT_NWR.m is an adaptive truncation algorithm, which produces weighted particles, representing draws from the posterior. The link function assumes b age at event variables, and d-b binary variables. This link needs to be changed for other response types (along with initialisation of the latent variables in the empirical hyperparameters and mcmc initialisation, and the relevant predictive quantities).

The folder Simulated Study: provides demo for reproducing the results in the paper. SimulatedStudyRun.m is the main file demonstrating how to run the code on simulated data.
