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
    
    d           = 1                 # Dimension of posterior
    alpha       = 0.5               # Standard deviation of observation noise
    x0          = np.zeros(d)       # Starting value
    N           = 1024              # Number of proposed states
    StepSize    = 1.2 #np.sqrt(2)   # Proposal step size
    PowerOfTwo  = 15                # generates size of seed = 2**PowerOfTwo-1
    Stream      = 'cud'             # choose between 'iid' or 'cud' seed
    
    # Define optional Burn-In
    BurnIn = 0
    
    ##################
    # Run simulation #
    ##################

    # Starting time of simulation
    StartTime = time.time()
    
    InitMean = np.ones(d)
    InitCov = np.identity(d)

    # Run simulation
    BLR = BayesianLinReg(d, alpha, x0, N, StepSize, \
                             PowerOfTwo, InitMean, InitCov, Stream='cud')

    # Stopping time
    EndTime = time.time()

    print ("CPU time needed =", EndTime - StartTime)


    ###################
    # Analyse results #
    ###################

    # Samples
    Samples = BLR.getSamples(BurnIn)

    # Plot marginal PDF histogram in Index-th coordinate
    Index = 0
    BarNum = 100
    BLR.getMarginalHistogram(Index=0, BarNum=100, BurnIn=0)
    

    # IS estimate for posterior mean as approximate posterior mean
    ApprPostMean = BLR.getIS_MeanEstimate(N, BurnIn)
    print ("IS posterior mean estimate = ", ApprPostMean)

    # Compute average acceptance rate 
    AcceptRate = BLR.getAcceptRate(BurnIn)
    print ("Acceptance rate = ", AcceptRate)
