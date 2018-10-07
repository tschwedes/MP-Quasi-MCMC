#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:29:19 2018

@author: Tobias Schwedes

Script to compute the effective sample size from a given sequence of
1-dimensional random variables

"""


import numpy as np


def AutoCorrelation(Samples):
    
    """
    Computes autocorrelation of a 1-dimensional sequence of random
    variables
    
    Inputs:
    ------
    x       - array_like
            sequence of MCMC random variates
    
    Outputs:
    -------
    AutoCor - array_like 
            sequence of autocorrelations     
    
    """
    
    n = len(Samples)
    Variance = Samples.var()
    
    # Normalise Samples
    Samples = Samples-Samples.mean()
    
    # Compute correlation
    Correlations = np.correlate(Samples, Samples, mode = 'full')[-n:]
    
    assert np.allclose(Correlations, \
                        np.array([(Samples[:n-k]*Samples[-(n-k):]).sum() \
                        for k in range(n)]))
    
    # Compute Autocorrelation
    AutoCor = Correlations/(Variance*(np.arange(n, 0, -1)))

    return AutoCor


def EffectiveSampleSize(Samples, AutoCor):
    
    """
    Computes effective sample size for a sequence of 1-dimensional random 
    variates
    
    Inputs:
    ------
    Samples         - array_like 
                    1-dimensional sequence of MCMC random variates
    AutoCor         - array_like 
                    1-dimensional sequence of autocorrelation of Samples
    
    Outputs:
    -------
    ESS             - float
                    effective sample size   
    
    """
    
    N = Samples.shape[0]
    
    # Initital positive sequence estimator (Geyer 1992, p.477)
    if N % 2 == 0:
        auto = AutoCor[2:][::2] + AutoCor[3:][::2]
    elif N % 2 == 1:
        auto = AutoCor[2:-1][::2] + AutoCor[3:][::2]
    
    if any(auto<0):
        n_neg = min(np.where(auto <0)[0])
    else:
        print ("Increase sample size to have sufficiently \
               many autocorrelations for estimation")
        n_neg = len(auto)

    # Initial monotone sequence estimator (Geyer 1992, p.477)
    diff = np.diff(auto)
    if any(diff>0):
        n_mon = np.min(np.where(diff>0))
    else:
        n_mon = n_neg

    # Take sequence length as min between positive + montone sequence estimate
    n = min(n_neg, n_mon) + 1
    K = 2*n+2
   
    # Compute effective sample size
    tau = 1 + 2* np.sum(AutoCor[1:K])
    ESS = N * tau**(-1)

    return ESS