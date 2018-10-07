#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 14:29:19 2018

@author: Tobias Schwedes

Script to determine a good choice of step size for the MP-(Q)MCMC.
This is done by finding the step size which maximises the effective sample
size (ESS) of the pseudo-random version of the algorithm (for N=4 proposals) 
over a trial set of possible step sizes. Maximal ESS corresponds to minimal 
empirical sample variance.

"""


import time
import numpy as np
from BayesianLogisticRegression import BayesianLogReg
from Data import DataLoad
from scipy.optimize import root
from ESS import AutoCorrelation, EffectiveSampleSize


if __name__ == '__main__':

    #############################
    # Parameters for simulation #
    #############################
    
    Range = np.linspace(1.0, 1.3, 13) # Range of step sizes
    Cases = ['ripley' , 'pima', 'heart', 'australian', 'german']
    EssMeans = np.zeros((len(Range), len(Cases))) # Means of ESS 
        
    c=0      
    for Case in Cases:

        #################
        # Generate Data #
        #################
    
        Data        = DataLoad(Case)
        d           = Data.GetDimension()
        XX          = Data.GetDesignMatrix()
        t           = Data.GetResponses()
        m           = Data.GetNumOfSamples()
        alpha       = 100. # hyperparameter for prior
 
    
        ###############################
        # Initial Mean and Covariance #
        ###############################       
        
        # Compute initial mean via root of posterior derivative
        def PostDer(z):
            f = np.dot(XX,z)
            PostDer = np.dot(XX.T, t - (np.exp(f)/(1+np.exp(f)))) - 1./alpha*np.dot(np.identity(d),z)   
            return PostDer 
    
        RootRes = root(PostDer, np.zeros(d), tol=1e-12)
        InitMean = RootRes.x   

        # Compute initital covariance as Inverse Fisher Info
        f = np.dot(XX,InitMean)
        p = 1./(1+np.exp(-f))
        v = p*(np.ones(m)-p)
        v1 = np.multiply(v,XX.T)
        Ginit = np.dot(v1,XX) + np.identity(d)/alpha   
        InitCov = np.linalg.inv(Ginit)


        ###############################
        # Initial Mean and Covariance #
        ###############################  

        s=0
        for StepSize in Range:
    
            N           = 4              # Number of proposed states
            PowerOfTwo  = 14             # Generates size of seed = 2**PowerOfTwo-1
            Stream      = 'iid'          # Choose between 'iid' or 'cud' seed 
#            InitMean = np.loadtxt('./GaussApproxims/ApprMean_{}.txt'.format(Case))
#            InitCov = np.loadtxt('./GaussApproxims/ApprCov_{}.txt'.format(Case))     
            df          = 250.            # Degree of freedom for student distribution
            alpha       = 100.           # Scaling of the prior covariance
            d           = len(InitMean)  # Dimension
           
    
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

            # Define Burn-In
            BurnInPowerOfTwo = 12
            BurnIn = 2**BurnInPowerOfTwo        
        
            # Samples
            Samples = BLR.GetSamples(BurnIn)
            
            # Initialise autocorrelations and effective sample sizes
            AutoCor = np.zeros((len(Samples), d))
            ESS = np.zeros(d)    
            
            # Compute autocorrelations and effective samples sizes
            for i in range(d):    
                AutoCor[:,i] = AutoCorrelation(Samples[:,i])
                ESS[i] = EffectiveSampleSize(Samples[:,i], AutoCor[:,i])   
                print ('ESS =', ESS[i])
            
            EssMean = np.mean(ESS)   
            EssMeans[s,c] =  EssMean    
            print ("Mean ESS estimate =", EssMean)
            
            s+=1
            print ("s = ", s)
            
        c+=1
        print ("c =", c)
        
    print ("ESS Means = \n" , EssMeans)        
    
    
    ############################
    # Choose optimal step size #
    ############################
    
    StepSizes = np.zeros(len(Cases))
    for k in range(len(Cases)):
        Kmax = np.where(EssMeans[:,k]==EssMeans[:,k].max())
        StepSizes[k] = Range[Kmax]
    print ("StepSizes for {} = ".format(Cases), StepSizes)