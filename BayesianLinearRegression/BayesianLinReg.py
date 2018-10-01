#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:31:25 2018

@author: Tobias Schwedes

Script to implement Bayesian linear regression using importance sampling
for multiple proposal Quasi-MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rv_discrete
from Data import DataGen
from Seed import SeedGen


class BayesianLinReg:
    
    def __init__(self, d, alpha, x0, N, StepSize, CovScaling, PowerOfTwo, Stream='cud'):
    
        """
        Implements the Bayesian Linear Regression based on 
        Data set "Data.txt" by using multiple proposal quasi MCMC with 
        Importance Sampling (IS-MP-QMCMC)
    
        inputs:
        -------   
        d               - Integer; dimension of posterior    
        alpha           - Standard deviation for observation noise
        x0              - D-dimensional array; starting value
        N               - Integer; number of proposals per iteration
        StepSize        - Float; step size for proposed jump in mean
        CovScaling      - Float; scaling of proposal covariance
        PowerOfTwo      - Defines size S of seed by S=2**PowerOfTwo-1
        stream          - String; either 'cud' or 'iid'; defining what seed is used
    
        outputs:
        ------- 
        xVals        - Array of sample values
        AcceptVals   - Array of acceptance probabilities
        WeightedSum  - Array of sum of weighted samples from every iteration
        """
    
        #################
        # Generate Data #
        #################
        
        Data        = DataGen(alpha, d)
        X           = Data.DesignMatrix()
        obs         = Data.Observations()
        n_samples   = Data.NumOfSamples()
        
        ##################################
        # Choose stream for Markoc Chain #
        ##################################
    
        xs = SeedGen(d, PowerOfTwo, Stream)
    
        ###########################################
        # Compute prior and likelihood quantities #
        ###########################################
        
        # Compute covariance of g-prior
        g = 1./n_samples
        sigmaSq = 1./alpha
        G_prior = sigmaSq / g * np.linalg.inv(np.dot(X.T,X))
        InvG_prior = np.linalg.inv(G_prior)
        
        # Fisher Information as constant metric tensor
        FisherInfo = InvG_prior + alpha*np.dot(X.T,X)
        InvFisherInfo = np.linalg.inv(FisherInfo)  
        
         
        ##################
        # Initialisation #
        ##################
    
        # List of samples to be collected
        self.xVals = list()
        self.xVals.append(x0)
    
        # Iteration number
        NumOfIter = int(int((2**PowerOfTwo-1)/d)*d/(N+1))
        print ('Total number of Iterations = ', NumOfIter)
    
        # set up acceptance rate array
        self.AcceptVals = list()
    
        # initialise
        xI = self.xVals[0]
        I = 0
        
        # Weighted Sum and Covariance Arrays
        self.WeightedSum = np.zeros((NumOfIter,d))
        
    
        ####################
        # Start Simulation #
        ####################
    
        for n in range(NumOfIter):
            
            ######################
            # Generate proposals #
            ######################
              
            # Load stream of points in [0,1]^d
            U = xs[n*(N+1):(n+1)*(N+1),:]
    
            # Compute proposal mean according to Langevin
            GradLog_xI = -np.dot(InvG_prior,xI) + alpha * np.dot(X.T, (obs - np.dot(X, xI)))
            Mean_xI = xI + StepSize**2/2.*np.dot(InvFisherInfo,GradLog_xI)
                
            # Generate auxiliary proposal state according to MALA 
            # (facilitates computation of proposing probabilities)
            z = Mean_xI + np.dot(norm.ppf(U[0,:], loc=np.zeros(d), scale=1.), \
                               np.linalg.cholesky(CovScaling**2*InvFisherInfo).T)
    
            # Compute mean of auxiliary proposal state according to MALA
            GradLog_z = -np.dot(InvG_prior,z) + alpha * np.dot(X.T, (obs - np.dot(X, z)))
            Mean_z = z + StepSize**2/2.*np.dot(InvFisherInfo,GradLog_z)
    
            # Generate proposals via inverse CDF transformation
            y = Mean_z + np.dot(norm.ppf(U[1:,:], loc=np.zeros(d), scale=1.), \
                              np.linalg.cholesky(CovScaling**2*InvFisherInfo).T)
    
            # Add current state xI to proposals    
            Proposals = np.insert(y, I, xI, axis=0)
    
    
            ########################################################
            # Compute probability ratios = weights of IS-estimator #
            ########################################################
    
            # Compute Log-posterior probabilities
            LogPriors = -0.5*np.dot(np.dot(Proposals, InvG_prior), Proposals.T).diagonal(0) # Zellner's g-prior
            fs = np.dot(X,Proposals.T)
            LogLikelihoods  = -0.5*alpha*np.dot(obs-fs.T, (obs-fs.T).T).diagonal(0)
            LogPosteriors   = LogPriors + LogLikelihoods
    
            # Compute Log of transition probabilities
            GradLog_states = - np.dot(InvG_prior,Proposals.T) \
                             + alpha * np.dot(X.T, (obs - np.dot(X, Proposals.T).T).T)
            Mean_Proposals = Proposals + StepSize**2/2.*np.dot(InvFisherInfo,GradLog_states).T
            LogKiz = -0.5*np.dot(np.dot(Mean_Proposals-z, FisherInfo/(CovScaling**2)), \
                                 (Mean_Proposals - z).T).diagonal(0) # from any state to z
            LogKzi = -0.5*np.dot(np.dot(Proposals-Mean_z, FisherInfo/(CovScaling**2)), \
                                 (Proposals - Mean_z).T).diagonal(0) # from z to any state
            LogKs = LogKiz + np.sum(LogKzi) - LogKzi
    
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
            Is = rv_discrete(values=(range(N+1),Pstates)).rvs(size=N)
            xvals_new = Proposals[Is]
            self.xVals.append(xvals_new)
    
            # Compute approximate acceptance rate
            AcceptValsNew = 1. - Pstates[Is]
            self.AcceptVals.append(AcceptValsNew)
    
            # Update current state
            I = Is[-1] #rv_discrete(values=(range(N+1),Pstates)).rvs(size=1)
            xI = Proposals[I,:]
    
    
    def Samples(self, BurnIn=0):
        
        """
        Compute samples from posterior from MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - Integer; Burn-In period
        
        Outputs:
        -------
        Samples - (Number of samples) x d-dimensional array        
        """
        
        Samples = np.concatenate(self.xVals[1:], axis=0)[BurnIn:,:]
                
        return Samples
       
        
    def AcceptRate(self, BurnIn=0):
        
        """
        Compute acceptance rate of MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - Integer; Burn-In period
        
        Outputs:
        -------
        AcceptRate - Float; average acceptance rate of MP-QMCMC 
        """    
        
        AcceptVals = np.concatenate(self.AcceptVals)[BurnIn:]
        AcceptRate = np.mean(AcceptVals)
        
        return AcceptRate

     
    def IS_MeanEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling estimate
        
        
        Outputs:
        -------
        WeightedMean - d-dimensional array
        """            
        
        WeightedMean = np.mean(self.WeightedSum[int(BurnIn/N):,:], axis=0)
        
        return WeightedMean
  
      
    def MarginalHistogram(self, Index=0, BarNum=100, BurnIn=0):
        
        """
        Plot histogram of marginal distribution for posterior samples using 
        MP-QMCMC
        
        Inputs:
        ------
        Index   - Integer; index of dimension for marginal distribution
        BurnIn  - Integer; Burn-In period
        
        Outputs:
        -------
        Plot
        """         

        Fig = plt.figure()
        SubPlot = Fig.add_subplot(111)
        SubPlot.hist(self.Samples(BurnIn)[:,Index], BarNum, label = "PDF Histogram", density = True)
        
        return Fig


