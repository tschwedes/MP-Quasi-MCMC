#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:31:25 2018

@author: Tobias Schwedes

Script to implement Bayesian linear regression using Rao-Blackwellisation / 
importance sampling for multiple proposal Quasi-MCMC with a SmMALA proposal
kernel.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from Data import DataGen
from Seed import SeedGen
#from Seed_digShift import SeedGen


class BayesianLinReg:
    
    def __init__(self, d, alpha, x0, N, StepSize, CovScaling, PowerOfTwo, Stream='cud'):
    
        """
        Implements the Bayesian Linear Regression based on Data set "Data.txt" 
        by using importance sampling / Rao-Blackwellisation for multiple 
        proposal Quasi-MCMC with with a SmMALA proposal kernel.
    
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
    
        #################
        # Generate Data #
        #################
        
        Data            = DataGen(alpha, d)
        X               = Data.getDesignMatrix()
        Obs             = Data.getObservations()
        NumOfSamples    = Data.getNumOfSamples()
        
        ##################################
        # Choose stream for Markoc Chain #
        ##################################
    
        xs = SeedGen(d+1, PowerOfTwo, Stream)
    
        ###########################################
        # Compute prior and likelihood quantities #
        ###########################################
        
        # Compute covariance of g-prior
        g = 1./NumOfSamples
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
#        NumOfIter = int(int((2**PowerOfTwo-1)/d)*d/(N+1))
        NumOfIter = int(int((2**PowerOfTwo-1)/(d+1))*(d+1)/(N+1))
        
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
#            U = xs[n*(N+1):(n+1)*(N+1),:]
            U = xs[n*(N):(n+1)*(N),:]
    
            # Compute proposal mean according to Langevin
            GradLog_xI = -np.dot(InvG_prior,xI) + alpha * np.dot(X.T, (Obs - np.dot(X, xI)))
            Mean_xI = xI + StepSize**2/2.*np.dot(InvFisherInfo,GradLog_xI)
                
            # Generate auxiliary proposal state according to MALA 
            # (facilitates computation of proposing probabilities)
            z = Mean_xI + np.dot(norm.ppf(U[0,:d], loc=np.zeros(d), scale=1.), \
                               np.linalg.cholesky(CovScaling**2*InvFisherInfo).T)    
            
            # Compute mean of auxiliary proposal state according to MALA
            GradLog_z = -np.dot(InvG_prior,z) + alpha * np.dot(X.T, (Obs - np.dot(X, z)))
            Mean_z = z + StepSize**2/2.*np.dot(InvFisherInfo,GradLog_z)
    
            # Generate proposals via inverse CDF transformation
            y = Mean_z + np.dot(norm.ppf(U[1:,:d], loc=np.zeros(d), scale=1.), \
                              np.linalg.cholesky(CovScaling**2*InvFisherInfo).T)
   
            
    
            # Add current state xI to proposals    
            Proposals = np.insert(y, I, xI, axis=0)
    
    
            ########################################################
            # Compute probability ratios = weights of IS-estimator #
            ########################################################
    
            # Compute Log-posterior probabilities
            LogPriors = -0.5*np.dot(np.dot(Proposals, InvG_prior), Proposals.T).diagonal(0) # Zellner's g-prior
            fs = np.dot(X,Proposals.T)
            LogLikelihoods  = -0.5*alpha*np.dot(Obs-fs.T, (Obs-fs.T).T).diagonal(0)
            LogPosteriors   = LogPriors + LogLikelihoods
    
            # Compute Log of transition probabilities
            GradLog_states = - np.dot(InvG_prior,Proposals.T) \
                             + alpha * np.dot(X.T, (Obs - np.dot(X, Proposals.T).T).T)
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
            # Replace Is sampling with QMC sampling step
            PstatesSum = np.cumsum(Pstates)
            Is = np.searchsorted(PstatesSum, U[:,d:].flatten())
            
            xvals_new = Proposals[Is]
            self.xVals.append(xvals_new)
    
            # Compute approximate acceptance rate
            AcceptValsNew = 1. - Pstates[Is]
            self.AcceptVals.append(AcceptValsNew)
    
            # Update current state
            I = Is[-1] #rv_discrete(values=(range(N+1),Pstates)).rvs(size=1)
            xI = Proposals[I,:]
    
    
    def getSamples(self, BurnIn=0):
        
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
       
        
    def getAcceptRate(self, BurnIn=0):
        
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

     
    def getIS_MeanEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling estimate

        Inputs:
        -------   
        N               - int 
                        number of proposals per iteration      
        BurnIn          - int
                        Burn-In period          
             
        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """            
        
        WeightedMean = np.mean(self.WeightedSum[int(BurnIn/N):,:], axis=0)
        
        return WeightedMean
  

    def getWeighted_Sums(self, N, BurnIn=0):
        
        """
        Compute importance sampling estimate

        Inputs:
        -------   
        N               - int 
                        number of proposals per iteration      
        BurnIn          - int
                        Burn-In period       
                        
        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """            
        
        WeightedSums = self.WeightedSum[int(BurnIn/N):,:]
        
        return WeightedSums
    
      
    def getMarginalHistogram(self, Index=0, BarNum=100, BurnIn=0):
        
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
        SubPlot.hist(self.getSamples(BurnIn)[:,Index], BarNum, label = "PDF Histogram", density = True)
        
        return Fig


