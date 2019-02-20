#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:11:15 2018

@author: Tobias Schwedes
"""

import time
import numpy as np
from BayesianLogisticRegression import BayesianLogReg

if __name__ == '__main__':

    #############################
    # Parameters for simulation #
    #############################

    Case = 'ripley'         # Data case
    N = 1024                  # Number of proposed states
    StepSize = 1.2          # Proposal step size
    PowerOfTwo = 13         # Generates size of seed = 2**PowerOfTwo-1
    Stream = 'cud'          # Choose between 'iid' or 'cud' seed 
    InitMean = np.loadtxt('./GaussApproxims/ApprMean_{}.txt'.format(Case))
    InitCov = np.loadtxt('./GaussApproxims/ApprCov_{}.txt'.format(Case))     
    df = 250.                 # Degree of freedom for student distribution
    alpha = 100.            # Scaling of the prior covariance
   
    # Define optional Burn-In
    BurnIn = 0
    
    ##################
    # Run simulation #
    ##################

    # Starting time of simulation
    StartTime = time.time()

    # Run simulation
    BLR = BayesianLogReg(N, StepSize, PowerOfTwo, \
                 InitMean, InitCov, df, Case, alpha, Stream)


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