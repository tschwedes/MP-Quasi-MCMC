#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:31:25 2018

@author: Tobias Schwedes

Script to implement Bayesian linear (just standard Gaussian) regression using importance sampling
for multiple proposal Quasi-MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Seed import SeedGen


class BayesianLinReg:
    
    def __init__(self, d, x0, N, StepSize, PowerOfTwo, \
                 InitMean, InitCov, Stream):
    
        """
        Implements the Bayesian Linear Regression based on 
        Data set "Data.txt" by using multiple proposal quasi MCMC with 
        Importance Sampling (IS-MP-QMCMC)
    
        Inputs:
        -------   
        d               - int 
                        dimension of posterior    
        alpha           - float
                        Standard deviation for Observation noise
        x0              - array_like
                        d-dimensional array; starting value
        N               - int 
                        number of proposals per iteration
        StepSize        - float 
                        step size for proposed jump in mean
        CovScaling      - float 
                        scaling of proposal covariance
        PowerOfTwo      - int
                        defines size S of seed by S=2**PowerOfTwo-1
        Stream          - string
                        either 'cud' or 'iid'; defining what seed is used
        """
    
        
        ##################################
        # Choose stream for Markoc Chain #
        ##################################
    
        xs = SeedGen(d+1, PowerOfTwo, Stream)
    
        ###########################################
        # Compute prior and likelihood quantities #
        ###########################################

     
        ##################
        # Initialisation #
        ##################
    
        # List of samples to be collected
        self.xVals = list()
        self.xVals.append(x0)
    
        # Iteration number
        NumOfIter = int(int((2**PowerOfTwo-1)/(d+1))*(d+1)/(N))
        print ('Total number of Iterations = ', NumOfIter)
    
        # set up acceptance rate array
        self.AcceptVals = list()
    
        # initialise
        xI = self.xVals[0]
        I = 0
        

        # Weighted Sum and Covariance Arrays
        self.WeightedSum = np.zeros((NumOfIter,d))
        self.WeightedCov = np.zeros((NumOfIter,d,d)) 
    

        # Approximate Posterior Mean and Covariance as initial estimates
        self.ApprPostMean = InitMean
        self.ApprPostCov = InitCov   
        

        # Cholesky decomposition of initial Approximate Posterior Covariance
        CholApprPostCov = np.linalg.cholesky(self.ApprPostCov)
        InvApprPostCov = np.linalg.inv(self.ApprPostCov)
        
        
        ####################
        # Start Simulation #
        ####################
    
        for n in range(NumOfIter):
            
            ######################
            # Generate proposals #
            ######################
              
            # Load stream of points in [0,1]^d
            U = xs[n*(N):(n+1)*(N),:]
            
            # Sample new proposed States according to multivariate t-distribution               
            y = self.ApprPostMean + np.dot(norm.ppf(U[:,:d], loc=np.zeros(d), \
                                                    scale=StepSize), CholApprPostCov)
            
            # Add current state xI to proposals    
            Proposals = np.insert(y, 0, xI, axis=0)
    
    
            ########################################################
            # Compute probability ratios = weights of IS-estimator #
            ########################################################
    
            # Compute Log-posterior probabilities
            LogPosteriors   = -0.5*np.dot(np.dot(Proposals-np.zeros(d), np.identity(d)), \
                                 (Proposals - np.zeros(d)).T).diagonal(0)
    
            # Compute Log of transition probabilities
            LogK_ni = -0.5*np.dot(np.dot(Proposals-self.ApprPostMean, InvApprPostCov/(StepSize**2)), \
                                 (Proposals - self.ApprPostMean).T).diagonal(0)
            LogKs = np.sum(LogK_ni) - LogK_ni # from any state to all others
            

            # Compute weights
            LogPstates = LogPosteriors + LogKs
            Sorted_LogPstates = np.sort(LogPstates)
            LogPstates = LogPstates - (Sorted_LogPstates[0] + \
                    np.log(1 + np.sum(np.exp(Sorted_LogPstates[1:] - Sorted_LogPstates[0]))))
            Pstates = np.exp(LogPstates)
    
    
            #######################
            # Compute IS-estimate #
            #######################
    
            # Compute weighted sum as posterior mean estimate
            WeightedStates = np.tile(Pstates, (d,1)) * Proposals.T
            self.WeightedSum[n,:] = np.sum(WeightedStates, axis=1).copy()


            ##################################
            # Sample according to IS-weights #
            ##################################
    
            # Sample N new states 
            PstatesSum = np.cumsum(Pstates)
            Is = np.searchsorted(PstatesSum, U[:,d:].flatten())
#            IS = rv_discrete(values=(range(N+1),Pstates)).rvs(size=N)
            xvals_new = Proposals[Is]
            self.xVals.append(xvals_new)
    
            # Compute approximate acceptance rate
            AcceptValsNew = 1. - Pstates[Is]
            self.AcceptVals.append(AcceptValsNew)
    
            # Update current state
            I = Is[-1]
            xI = Proposals[I,:]
    
    
    def GetSamples(self, BurnIn=0):
        
        """
        Compute samples from posterior from MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - int 
                Burn-In period
        
        Outputs:
        -------
        Samples - array_like
                (Number of samples) x d-dimensional arrayof Samples      
        """
        
        Samples = np.concatenate(self.xVals[1:], axis=0)[BurnIn:,:]
                
        return Samples
       
        
    def GetAcceptRate(self, BurnIn=0):
        
        """
        Compute acceptance rate of MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - int
                Burn-In period
        
        Outputs:
        -------
        AcceptRate - float
                    average acceptance rate of MP-QMCMC 
        """    
        
        AcceptVals = np.concatenate(self.AcceptVals)[BurnIn:]
        AcceptRate = np.mean(AcceptVals)
        
        return AcceptRate

     
    def GetIS_MeanEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling estimate
             
        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """            
        
        WeightedMean = np.mean(self.WeightedSum[int(BurnIn/N):,:], axis=0)
        
        return WeightedMean
    

    def GetIS_FunMeanEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling estimate
             
        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """            
        
        WeightedMean = np.mean(self.WeightedFunSum[int(BurnIn/N):,:], axis=0)
        
        return WeightedMean    
  

    def GetIS_CovEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling covariance estimate
        
        
        Outputs:
        -------
        WeightedCov - d-dimensional array
        """            
        
        WeightedCov = np.mean(self.WeightedCov[int(BurnIn/N):,:,:], axis=0)
        
        return WeightedCov    
    
      
    def GetMarginalHistogram(self, Index=0, BarNum=100, BurnIn=0):
        
        """
        Plot histogram of marginal distribution for posterior samples using 
        MP-QMCMC
        
        Inputs:
        ------
        Index   - int
                index of dimension for marginal distribution
        BurnIn  - int
                Burn-In period
        
        Outputs:
        -------
        Plot
        """         

        Fig = plt.figure()
        SubPlot = Fig.add_subplot(111)
        SubPlot.hist(self.GetSamples(BurnIn)[:,Index], BarNum, label = "PDF Histogram", density = True)
        
        return Fig


