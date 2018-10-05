#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:11:15 2018

@author: Tobias Schwedes
"""

import time
import numpy as np
from BayesianLinReg import BayesianLinReg

if __name__ == '__main__':

    #############################
    # Parameters for simulation #
    #############################
    
    d           = 3                 # Dimension of posterior
    alpha       = 0.5               # Standard deviation of observation noise
    x0          = np.zeros(d)       # Starting value
    N           = 1024              # Number of proposed states
    StepSize    = np.sqrt(2)        # Proposal step size
    CovScaling  = 1.                # Proposal covariance scaling
    PowerOfTwo  = 15                # generates size of seed = 2**PowerOfTwo-1
    Stream      = 'cud'             # choose between 'iid' or 'cud' seed
    
    # Define optional Burn-In
    BurnIn = 0
    
    ##################
    # Run simulation #
    ##################

    # Starting time of simulation
    StartTime = time.time()

    # Run simulation
    BLR = BayesianLinReg(d, alpha, x0, N, StepSize, CovScaling, \
                             PowerOfTwo, Stream='cud')

    # Stopping time
    EndTime = time.time()

    print ("CPU time needed =", EndTime - StartTime)


    ###################
    # Analyse results #
    ###################

    # Samples
    Samples = BLR.GetSamples(BurnIn)

    # Plot marginal PDF histogram in Index-th coordinate
    Index = 0
    BarNum = 100
    BLR.GetMarginalHistogram(Index=0, BarNum=100, BurnIn=0)
    

    # IS estimate for posterior mean as approximate posterior mean
    ApprPostMean = BLR.GetIS_MeanEstimate(N, BurnIn)
    print ("IS posterior mean estimate = ", ApprPostMean)

    # Compute average acceptance rate 
    AcceptRate = BLR.GetAcceptRate(BurnIn)
    print ("Acceptance rate = ", AcceptRate)
