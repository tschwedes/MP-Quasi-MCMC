#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:56:36 2018

@author: Tobias Scw#!/usr/bin/env python3

Script to implement Bayesian logistic regression using importance sampling
for multiple proposal Quasi-MCMC.
"""

import numpy as np
import matplotlib.pyplot as plt
from StudentT import multivariate_t_rvs_custom_seed, multivariate_t_LogPdf
from Data import DataLoad
from Seed import SeedGen
#from Seed_digShift import SeedGen


class BayesianLogReg:
    

    def __init__(self, N, StepSize, PowerOfTwo, \
                 InitMean, InitCov, df, Case, alpha=100., Stream='cud', WeightIn=0):
        
        """
        Implements the Bayesian Logistic Regression based on the
        Data sets in ./Data/ by using multiple proposal quasi MCMC with 
        Importance Sampling (IS-MP-QMCMC)
    
        Inputs:
        -------   
        N               - int
                        number of proposals per iteration
        StepSize        - float
                        step size for proposed jump in mean
        PowerOfTwo      - int
                        Defines size S of seed by S=2**PowerOfTwo-1
        InitMean        - array_like 
                        d-dimensional initial proposal mean
        InitCov         - array_like
                        dxd-dimensional initial proposal covariance
        df              - float >2 
                        degree of freedom for student distribution
        Case            - string
                        determines the data used
        alpha           - float
                        1./alpha scales prior covariance
        Stream          - string
                        either 'cud' or 'iid'; defining what seed is used       
        WeightIn        - float
                        if BurnIn-run existed, weight initial esitmates
                        by int(WeightIn/N)-times
        """
    
        #############
        # Load Data #
        #############
        
        Data        = DataLoad(Case)
        d           = Data.getDimension()
        XX          = Data.getDesignMatrix()
        t           = Data.getResponses()

        
        ##################################
        # Choose stream for Markoc Chain #
        ##################################
    
        xs = SeedGen(d+2, PowerOfTwo, Stream)
    
         
        ##################
        # Initialisation #
        ##################
    
        # List of samples to be collected
        self.xVals = list()
        self.xVals.append(InitMean)
    
        # Iteration number
        NumOfIter = int(int((2**PowerOfTwo-1)/(d+2))*(d+2)/N)
        print ('Total number of Iterations = ', NumOfIter)
    
        # Set up acceptance rate array
        self.AcceptVals = list()
    
        # Initialise
        xI = self.xVals[0]
        I = 0

        # Number of iterations used for initial approximated posterior mean
        M = int(WeightIn/N)+1
        
        # Weighted Sum and Covariance Arrays
        self.WeightedSum = np.zeros((NumOfIter+M,d))
        self.WeightedCov = np.zeros((NumOfIter+M,d,d)) 
        self.WeightedSum[0:M,:] = InitMean
        self.WeightedCov[0:M,:] = InitCov        
    
        # Approximate Posterior Mean and Covariance as initial estimates
        self.ApprPostMean = InitMean
        self.ApprPostCov  = InitCov
        
        # Cholesky decomposition of initial Approximate Posterior Covariance
        CholApprPostCov   = np.linalg.cholesky(self.ApprPostCov)

    
        ####################
        # Start Simulation #
        ####################
    
        for n in range(NumOfIter):

            ######################
            # Generate proposals #
            ######################
            
            # Load stream of points in [0,1]^(d+1)
            U = xs[n*N:(n+1)*N,:]
            
            # Sample new proposed States according to multivariate t-distribution    
            y = multivariate_t_rvs_custom_seed(U[:,:d+1], self.ApprPostMean, \
                    StepSize*CholApprPostCov*np.sqrt((df-2.)/df), df=df)  
            
            # Add current state xI to proposals    
            Proposals = np.insert(y, 0, xI, axis=0)
             

            ########################################################
            # Compute probability ratios = weights of IS-estimator #
            ########################################################

            # Compute Log-posterior probabilities    
            LogPriors       = -0.5*np.dot(np.dot(Proposals, np.identity(d)/alpha), \
                                          (Proposals).T).diagonal(0)
            fs              = np.dot(XX, Proposals.T)
            LogLikelihoods  = np.dot(t,fs) - np.sum(np.log(1.+np.exp(fs)), axis=0)
            LogPosteriors   = LogPriors + LogLikelihoods       
                        
            
            # Compute Log of transition probabilities
            LogK_ni = multivariate_t_LogPdf(Proposals, self.ApprPostMean, \
                            StepSize**2*self.ApprPostCov*(df-2.)/df, df=df)
            LogKs   = np.sum(LogK_ni) - LogK_ni # from any state to all others
            
            # Compute weights
            LogPstates          = LogPosteriors + LogKs
            Sorted_LogPstates   = np.sort(LogPstates)
            LogPstates          = LogPstates - (Sorted_LogPstates[-1] + np.log(1 + \
                                np.sum(np.exp(Sorted_LogPstates[:-1] - \
                                              Sorted_LogPstates[-1]))))
            Pstates             = np.exp(LogPstates)         
            

            ########################
            # Compute IS-estimates #
            ########################       
            
            # Compute weighted sum as posterior mean estimate
            WeightedStates = np.tile(Pstates, (d,1)) * Proposals.T
            self.WeightedSum[n+M,:] = np.sum(WeightedStates, axis=1).copy()
            
            # Update Approximate Posterior Mean
            self.ApprPostMean = np.mean(self.WeightedSum[:n+M+1,:], axis=0) 

            # Compute weighted sum as posterior covariance estimate
            B1 = (Proposals - self.ApprPostMean).reshape(N+1,d,1) 
            B2 = np.transpose(B1,(0,2,1)) 
            A = np.matmul(B1, B2)
            self.WeightedCov[n+M,:,:] = np.sum((np.tile(Pstates, (d,d,1)) * A.T).T, axis=0)
            
            # Update Approximate Posterior Covariance
            if n> 2*d/N: # makes sure NumOfSamples > d for covariance estimate
                self.ApprPostCov = np.mean(self.WeightedCov[:n+M+1,:,:], axis=0)
                CholApprPostCov = np.linalg.cholesky(self.ApprPostCov)

            ##################################
            # Sample according to IS-weights #
            ##################################
    
            # Sample N new states 
            PstatesSum = np.cumsum(Pstates)
            Is = np.searchsorted(PstatesSum, U[:N-1,d+1:].flatten())
            PstatesSubSamSum = np.cumsum(np.bincount(Is)/(N-1))
            I = np.searchsorted(PstatesSubSamSum, U[N-1,d+1:])
            
            # Add new samples to list
            xValsNew = Proposals[Is]
            self.xVals.append(xValsNew.copy())
    
            # Compute approximate acceptance rate
            AcceptValsNew = 1. - Pstates[Is]
            self.AcceptVals.append(AcceptValsNew)
    
            # Update current state
#            I = Is[-1]
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
                (Number of samples) x d-dimensional array of Samples    
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
        Compute importance sampling mean estimate
        
        
        Outputs:
        -------
        WeightedMean    - array_like
                        d-dimensional array
        """            
        
        WeightedMean = np.mean(self.WeightedSum[int(BurnIn/N):,:], axis=0)
        
        return WeightedMean


    def getIS_CovEstimate(self, N, BurnIn=0):
        
        """
        Compute importance sampling covariance estimate
        
        
        Outputs:
        -------
        WeightedCov - d-dimensional array
        """            
        
        WeightedCov = np.mean(self.WeightedCov[int(BurnIn/N):,:,:], axis=0)
        
        return WeightedCov


    def getWeightedSum(self, N, BurnIn=0):
        
        """
        Compute samples from posterior from MP-QMCMC
        
        Inputs:
        ------
        BurnIn  - int 
                Burn-In period
        
        Outputs:
        -------
        Samples - array_like
                (Number of samples) x d-dimensional array of Samples     
        """
        
        WeightedSum = self.WeightedSum[int(BurnIn/N):,:]
                
        return WeightedSum
    
      
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





