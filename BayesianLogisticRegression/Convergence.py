#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:25:19 2018

@author: Tobias Schwedes


Script to analyse the convergence in Empirical Variance and Squared Bias
(as well as combined = MSE) for importance sampling MP-MCMC driven by a
IID seed VS. by a CUD seed. The underlying sampling method makes use of
an independent adaptive proposal sampler
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import root
from scipy.stats import linregress
from BayesianLogisticRegression import BayesianLogReg
from Data import DataLoad


if __name__ == '__main__':

    #############################
    # Parameters for simulation #
    #############################  
    
    # Number of simulations
    NumOfSim = 25
    # Define size of seed by powers of two
    PowerOfTwoArray = np.arange(11,20) #11 for N=4 and 18 for N=1024
    # Define number of proposed states
    N_Array = np.array([4,8,16,32,64,128,256,512,1024])
    # Degree of freedom of student proposal sampler (df>>1 close to gaussian)
    df = 250.
    # Scaling for prior covariance
    alpha = 100.
    # Specify data set
    Cases = ['ripley']# , 'pima', 'heart', 'australian', 'german'][1:]
   
    # Specify StepSizes for individual data set (according to StepSize.py)
    StepSizes = np.array([1.2, 1.05, 1.15, 1.1, 1.15])

    # Specify BurnIn for individual data set
    AllBurnInPowerOfTwo = [12, 12, 12, 12, 13]
    

    ##########################################################################

    ##############################
    # Run convergence experiment #
    ##############################


    # Create directory to save results in
    DirName = 'results'
    try:
        # Create target Directory
        os.mkdir(DirName)
        print("Directory " , DirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , DirName ,  " already exists")


    c=0
    for Case in Cases:
    
        print ("%%%%%%%%%%%%%%%%%%%%")
        print ("Case = ", Case)    
        print ("%%%%%%%%%%%%%%%%%%%%")
        
        # Specify directory under which Case results are saved
        DirName2 = DirName+'/'+Case
        try:
            # Create target Directory
            os.mkdir(DirName2)
            print("Directory " , DirName2 ,  " Created ") 
        except FileExistsError:
            print("Directory " , DirName2 ,  " already exists")  


        # Starting Time
        StartTimeCase = time.time()
        
        # Proposal step size
        StepSize = StepSizes[c]

    
        #################
        # Generate Data #
        #################
    
        Data        = DataLoad(Case)
        d           = Data.GetDimension()
        XX          = Data.GetDesignMatrix()
        t           = Data.GetResponses()
        m           = Data.GetNumOfSamples()
        
        
        ######################################################
        # Compute initial mean and covariance for BurnIn-run #
        ######################################################
            
        # Compute initial estimate of posterior mean via root of posterior derivative
        def PostDer(z):
            f = np.dot(XX,z)
            PostDer = np.dot(XX.T, t - (np.exp(f)/(1+np.exp(f)))) - 1./alpha*np.dot(np.identity(d),z)   
            return PostDer 
    
        RootRes = root(PostDer, np.zeros(d), tol=1e-15)
        InitMean = RootRes.x    
        
        # Calculate initial estimate of posterior covariance via FisherInfo
        f = np.dot(XX,InitMean)
        p = 1./(1+np.exp(-f))
        v = p*(np.ones(m)-p)
        v1 = np.multiply(v,XX.T)
        Ginit = np.dot(v1,XX) + np.identity(d)/alpha   
        InitCov = np.linalg.inv(Ginit)
    

        ##############################################
        # Posterior goldstandard mean and covariance #
        ##############################################

        # Approximated gold standard mean and covariance of posterior distribution
        GoldStandardPostCov = np.loadtxt('./GaussApproxims/ApprCov_{}.txt'.format(Case))
        GoldStandardApprPostMean = np.loadtxt('./GaussApproxims/ApprMean_{}.txt'.format(Case))    


        ###############
        # Burn-In Run #
        ###############
    
        BurnInPowerOfTwo = AllBurnInPowerOfTwo[c]
        BurnInStepSize=1.2
        BurnInN = 8
    
        BurnInQMC_BLR = BayesianLogReg(BurnInN, BurnInStepSize, BurnInPowerOfTwo, \
             InitMean, InitCov, df, Case, alpha, Stream='cud')            

        # Estimates from BurnIn-run
        QMC_Samples = BurnInQMC_BLR.GetSamples()
        QMC_IS_MeanEstimate = BurnInQMC_BLR.GetIS_MeanEstimate(BurnInN)
        QMC_IS_CovEstimate = BurnInQMC_BLR.GetIS_CovEstimate(BurnInN)
        
    
        ##################
        # Initialisation #
        ##################

        # Define initial mean and covariance as estimates from BurnIn-run
        InitMean = QMC_IS_MeanEstimate
        InitCov = QMC_IS_CovEstimate

        # Arrays to be filled with IS posterior estimates
        QMC_EstimArray = np.zeros((len(N_Array), NumOfSim, d))
        PSR_EstimArray = np.zeros((len(N_Array), NumOfSim, d))
    
        for p in range(N_Array.shape[0]):
            
            Counter = 0
            
            for j in range(NumOfSim):
    
                #########################################################
                # Parameters for individual simulation with N proposals #
                #########################################################
                
                N = int(N_Array[p])
                PowerOfTwo = PowerOfTwoArray[p]
                NumOfIter = int((2**PowerOfTwo-1.)/N)
                WeightIn = 2**BurnInPowerOfTwo-1
        
                print ('Number of proposals = ', N)
    
                # Starting Time
                StartTime = time.time()
                                   
                ##################
                # Run simulation #
                ##################
                
                QMC_BLR = BayesianLogReg(N, StepSize, PowerOfTwo, \
                     InitMean, InitCov, df, Case, alpha, Stream='cud', WeightIn=WeightIn)            
                PSR_BLR = BayesianLogReg(N, StepSize, PowerOfTwo, \
                     InitMean, InitCov, df, Case, alpha, Stream='iid', WeightIn=WeightIn)       
                      
                # Stopping time
                EndTime = time.time()
                print ("CPU time for single pair of simulations =", EndTime - StartTime)
                
                # Percent of simulations for proposal number N done
                Counter += 1
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print ("{} % of simulations for N={} done: ".format(Counter/NumOfSim*100, N))
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                
                # Acceptance rate of MP-(Q)MCMC
                QMC_AcceptRate = QMC_BLR.GetAcceptRate()
                print ("QMC Acceptance Rate = ", QMC_AcceptRate)
                PSR_AcceptRate = PSR_BLR.GetAcceptRate()
                print ("PSR Acceptance Rate = ", PSR_AcceptRate)
    
    
                ################## QMC #####################
    
                # Compute estimated IS mean
                QMC_EstimArray[p,j,:] = QMC_BLR.GetIS_MeanEstimate(N, WeightIn)
    
                ################## PSR #####################
    
                # Compute estimated IS mean
                PSR_EstimArray[p,j,:] = PSR_BLR.GetIS_MeanEstimate(N, WeightIn)
    
        
        ###############################
        # TRACE OF EMPIRICAL VARIANCE #
        ###############################
        
        # Compute average estimator variance
        QMC_EstimAverageVar = np.var(QMC_EstimArray, axis=1)
        PSR_EstimAverageVar = np.var(PSR_EstimArray, axis=1)
        
        # Compute variance trace
        QMC_EstimAverageVarTrace = np.sum(QMC_EstimAverageVar, axis=1)
        PSR_EstimAverageVarTrace = np.sum(PSR_EstimAverageVar, axis=1)
        
        
       ######################################
        # SUM OF COMPONENTS OF SQUARED BIAS #
       ######################################
        
        # Compute average estimator squared bias
        QMC_EstimAverageSquareBias = (np.mean(QMC_EstimArray-GoldStandardApprPostMean, axis=1))**2
        PSR_EstimAverageSquareBias = (np.mean(PSR_EstimArray, axis=1)-GoldStandardApprPostMean)**2
        
        # Compute sum of components of squared bias
        QMC_EstimAverageSquareBiasTrace = np.sum(QMC_EstimAverageSquareBias, axis=1)
        PSR_EstimAverageSquareBiasTrace = np.sum(PSR_EstimAverageSquareBias, axis=1)
        
    
        ################## QMC #####################
    
        ##################################
        # VARIANCE of Empirical variance #
        ##################################
    
        # Error bar estimate for empirical variance estimates
        BatchSize = 5.
        QMC_EstimBatches            = np.array(np.array_split(QMC_EstimArray, BatchSize, axis=1))
        QMC_EstimBatchVar           = np.var(QMC_EstimBatches, axis=2)
        QMC_EstimBatchVarTrace      = np.sum(QMC_EstimBatchVar, axis=2)
        QMC_VarEstimBatchVarTrace   = np.var(QMC_EstimBatchVarTrace, axis=0)/BatchSize
        
        
        ######################################
        # VARIANCE of Empirical squared bias #
        # (not consistent estimate due to    #
        # invalid approximation of square of #
        # var equals var of squares)         #
        ######################################
    
        # Error bar estimator for empirical variance estimates here
        QMC_BiasBatches = np.array(np.array_split(QMC_EstimArray-GoldStandardApprPostMean, BatchSize, axis=1))
        QMC_BiasBatchSquareMean = np.mean(QMC_BiasBatches, axis=2)**2
        QMC_BiasBatchSquareMeanTrace = np.sum(QMC_BiasBatchSquareMean, axis=2)
        QMC_BiasBatchSquareMeanTraceVar = np.var(QMC_BiasBatchSquareMeanTrace, axis=0) /BatchSize
    
        #######################
        ### MSE COMPUTATION ###
        #######################
    
        # Compute MSE
        QMC_MSE_Trace = QMC_EstimAverageVarTrace + QMC_EstimAverageSquareBiasTrace
        
        # Compute MSE variance
        # (not consistent estimate since variance of MSE is not variance
        # of empirical variance plus variance of bias square)
        QMC_BatchMSE_TraceVar = QMC_VarEstimBatchVarTrace + QMC_BiasBatchSquareMeanTraceVar
    
    
    
        ################## PSR #####################
    
        ##################################
        # VARIANCE of Empirical variance #
        ##################################
    
        # Error bar estimate for empirical variance estimates
        BatchSize = 5.
        PSR_EstimBatches            = np.array(np.array_split(PSR_EstimArray, BatchSize, axis=1))
        PSR_EstimBatchVar           = np.var(PSR_EstimBatches, axis=2)
        PSR_EstimBatchVarTrace      = np.sum(PSR_EstimBatchVar, axis=2)
        PSR_VarEstimBatchVarTrace   = np.var(PSR_EstimBatchVarTrace, axis=0)/BatchSize
    
        ######################################
        # VARIANCE of Empirical squared bias #
        # (not consistent estimate due to    #
        # invalid approximation of square of #
        # var equals var of squares)         #
        ######################################
    
        # Error bar estimator for empirical variance estimates here
        PSR_BiasBatches = np.array(np.array_split(PSR_EstimArray-GoldStandardApprPostMean, BatchSize, axis=1))
        PSR_BiasBatchSquareMean = np.mean(PSR_BiasBatches, axis=2)**2
        PSR_BiasBatchSquareMeanTrace = np.sum(PSR_BiasBatchSquareMean, axis=2)
        PSR_BiasBatchSquareMeanTraceVar = np.var(PSR_BiasBatchSquareMeanTrace, axis=0) /BatchSize
    
    
        #######################
        ### MSE COMPUTATION ###
        #######################
    
        # Compute MSE
        PSR_MSE_Trace = PSR_EstimAverageVarTrace + PSR_EstimAverageSquareBiasTrace
        
        # Compute MSE variance
        # (not consistent estimate since variance of MSE is not variance
        # of empirical variance plus variance of bias square)
        PSR_BatchMSE_TraceVar = PSR_VarEstimBatchVarTrace + PSR_BiasBatchSquareMeanTraceVar
    
    
        # Overall End Time
        EndTimeCase = time.time()
        TimeCase = EndTimeCase - StartTimeCase
        print ("Overall CPU time =", TimeCase)
    
        #########################################################################
    
     
        ###############################
        ### Empirica Variance PLOTS ###
        ###############################
        
        # Fancier plots
        fig, ax1 = plt.subplots()
        fig.tight_layout()
    
        ax1.errorbar(N_Array, QMC_EstimAverageVarTrace, \
                    yerr=3*np.sqrt(QMC_VarEstimBatchVarTrace), fmt='-o', \
                    markersize=3, label = 'QMC',elinewidth = 1, capsize = 3, \
                    color='darkblue')    
        ax1.errorbar(N_Array, PSR_EstimAverageVarTrace, \
                    yerr=3*np.sqrt(PSR_VarEstimBatchVarTrace), fmt='--o', \
                    markersize=3, label = 'PSR',elinewidth = 1, capsize = 3, \
                    color='darkred')  
    
    
        ax1.errorbar(N_Array, 1*1e0*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax1.errorbar(N_Array, 0.5*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        ax1.set_xlabel('Number of Proposals $N$ \n (Step Size = %1.3f)' %StepSize)
    
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'Variance', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        x1_ticks_labels = [5,10,25,50,100,250,500,1000]
        ax1.set_xticks(np.array([5,10,25,50,100,250,500,1000]))
        ax1.set_xticklabels(x1_ticks_labels, fontsize=11)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True,which="both")
        
        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.errorbar(N_Array*NumOfIter, 1.*1e0*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax2.errorbar(N_Array*NumOfIter, 0.5*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        x2_ticks_labels = np.array([5000, 25000, 100000, 500000])
        ax2.set_xticks(x2_ticks_labels)
        ax2.set_xticklabels(x2_ticks_labels, fontsize=11)
        ax2.set_xlabel('Total Number of Samples $n$ \n (%i Iterations)' %NumOfIter, color='k')
    
        fig.tight_layout()
    #    plt.show()
        plt.savefig('results/{}/emprVar_{}mcmc.eps'.format(Case, NumOfSim), format='eps')
    
    
    
    
        #############################################
        ### Empirica Variance & Bias Square PLOTS ###
        #############################################
    
        # Fancier plots
        fig, ax1 = plt.subplots()
        fig.tight_layout()
    
        ax1.errorbar(N_Array, QMC_EstimAverageVarTrace, \
                    yerr=3*np.sqrt(QMC_VarEstimBatchVarTrace), fmt='--o', \
                    markersize=3, label = 'Var (QMC)',elinewidth = 1, capsize = 3, \
                    color='darkblue')
        ax1.errorbar(N_Array, QMC_EstimAverageSquareBiasTrace, fmt=':', \
    #                yerr=1*np.sqrt(QMC_BiasBatchSquareMeanTraceVar),              
                    markersize=3, label = r'$Bias^2$ (QMC)',elinewidth = 1, capsize = 3, \
                    color='blue')     
        
        ax1.errorbar(N_Array, PSR_EstimAverageVarTrace, \
                    yerr=3*np.sqrt(PSR_VarEstimBatchVarTrace), fmt='-o', \
                    markersize=3, label = 'Var (PSR)',elinewidth = 1, capsize = 3, \
                    color='darkred')
        ax1.errorbar(N_Array, PSR_EstimAverageSquareBiasTrace, \
    #                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                    fmt='-.', \
                    markersize=3, label = r'$Bias^2$ (PSR)',elinewidth = 1, capsize = 3, \
                    color='red')   
        
    
        
        ax1.errorbar(N_Array, 0.6*1e-1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax1.errorbar(N_Array, 0.6*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        ax1.set_xlabel('Number of Proposals $N$ \n (Step Size = %1.3f)' %StepSize)
    
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        x1_ticks_labels = [5,10,25,50,100,250,500,1000]
        ax1.set_xticks(np.array([5,10,25,50,100,250,500,1000]))
        ax1.set_xticklabels(x1_ticks_labels, fontsize=11)
        ax1.legend(loc='best', fontsize=9.)
        ax1.grid(True,which="both")
        
        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.errorbar(N_Array*NumOfIter, 0.6*1e-1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax2.errorbar(N_Array*NumOfIter, 0.6*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        x2_ticks_labels = np.array([5000, 25000, 100000, 500000])
        ax2.set_xticks(x2_ticks_labels)
        ax2.set_xticklabels(x2_ticks_labels, fontsize=11)
        ax2.set_xlabel('Total Number of Samples $n$ \n (%i Iterations)' %NumOfIter, color='k')
    
        fig.tight_layout()
    #    plt.show()
        plt.savefig('results/{}/emprVar_BiasSquare_{}mcmc.eps'.format(Case, NumOfSim), format='eps')
    
    
    
        #########################
        ### Bias Square PLOTS ###
        #########################
    
        # Fancier plots
        fig, ax1 = plt.subplots()
        fig.tight_layout()
    
        ax1.errorbar(N_Array, QMC_EstimAverageSquareBiasTrace, \
    #                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                    fmt='-o', \
                    markersize=3, label = r'QMC', elinewidth = 1, capsize = 3, \
                    color='darkblue')  
    
        ax1.errorbar(N_Array, PSR_EstimAverageSquareBiasTrace, \
    #                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                    fmt='--o', \
                    markersize=3, label = r'PSR', elinewidth = 1, capsize = 3, \
                    color='darkred')   
    
        ax1.errorbar(N_Array, 0.25*1e-1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax1.errorbar(N_Array, 0.25*1e1*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        ax1.set_xlabel('Number of Proposals $N$ \n (Step Size = %1.3f)' %StepSize)
    
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'$Bias^2$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        x1_ticks_labels = [5,10,25,50,100,250,500,1000] #[5,10,20,50,100] 
        ax1.set_xticks(np.array([5,10,25,50,100,250,500,1000])) # #[5,10,20,50,100]
        ax1.set_xticklabels(x1_ticks_labels, fontsize=11)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True,which="both")
        
        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.errorbar(N_Array*NumOfIter, 0.25*1e-1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax2.errorbar(N_Array*NumOfIter, 0.25*1e1*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        x2_ticks_labels = np.array([5000, 25000, 100000, 500000])
        ax2.set_xticks(x2_ticks_labels)
        ax2.set_xticklabels(x2_ticks_labels, fontsize=11)
        ax2.set_xlabel('Total Number of Samples $n$ \n (%i Iterations)' %NumOfIter, color='k')
    
        fig.tight_layout()
    #    plt.show()
        plt.savefig('results/{}/BiasSquare_{}mcmc.eps'.format(Case, NumOfSim), format='eps')
    
    
    
        #################
        ### MSE PLOTS ###
        #################
    
        # Fancier plots
        fig, ax1 = plt.subplots()
        fig.tight_layout()
    
        ax1.errorbar(N_Array, QMC_MSE_Trace, fmt='-o',\
    #                yerr=3*np.sqrt(QMC_BatchMSE_TraceVar), fmt='-o', \
                    markersize=3, label = 'QMC',elinewidth = 1, capsize = 3, \
                    color='darkblue')  
        ax1.errorbar(N_Array, PSR_MSE_Trace, fmt='--o', \
    #                yerr=3*np.sqrt(PSR_BatchMSE_TraceVar), fmt='-o', \
                    markersize=3, label = 'PSR',elinewidth = 1, capsize = 3, \
                    color='darkred')  
    
        ax1.errorbar(N_Array, 0.3*1e0*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax1.errorbar(N_Array, 0.6*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        ax1.set_xlabel('Number of Proposals $N$ \n (Step Size = %1.3f)' %StepSize)
    
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(r'$MSE$', color='k')
        ax1.tick_params('y', colors='k')
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        x1_ticks_labels = [5,10,25,50,100,250,500,1000] #[5,10,20,50,100] 
        ax1.set_xticks(np.array([5,10,25,50,100,250,500,1000])) # #[5,10,20,50,100]
        ax1.set_xticklabels(x1_ticks_labels, fontsize=11)
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True,which="both")
    
        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.errorbar(N_Array*NumOfIter, 0.3*1e0*(N_Array*NumOfIter)**(-1.), fmt='--', \
                    label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
        ax2.errorbar(N_Array*NumOfIter, 0.6*1e2*(N_Array*NumOfIter)**(-2.), fmt=':', \
                    label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
        x2_ticks_labels = np.array([5000, 25000, 100000, 500000])
        ax2.set_xticks(x2_ticks_labels)
        ax2.set_xticklabels(x2_ticks_labels, fontsize=11)
        ax2.set_xlabel('Total Number of Samples $n$ \n (%i Iterations)' %NumOfIter, color='k')
    #    ax2.grid()
    
        fig.tight_layout()
    #    plt.show()
        plt.savefig('results/{}/MSE_{}mcmc.eps'.format(Case, NumOfSim), format='eps')
    
    
    
        ####################################################################
    
    
        #################################################################################
        ### Linear Regression on log-log grahps for determination of convergence rate ###
        #################################################################################
       
        x = np.log(N_Array)
    
        #################################################
        ### Compute empirical variance convergence rate #
        #################################################
           
        # QMC
        QMC_EmpiricalVar_y = np.log(QMC_EstimAverageVarTrace)
        QMC_EmpiricalVarSlope, PSR_Intercept, PSR_rValue, PSR_pValue, PSR_StdErr = linregress(x,QMC_EmpiricalVar_y)    
        
        # PSR
        PSR_EmpiricalVar_y = np.log(PSR_EstimAverageVarTrace)    
        PSR_EmpiricalVarSlope, QMC_Intercept, QMC_rValue, QMC_pValue, QMC_StdErr = linregress(x,PSR_EmpiricalVar_y)    
      
        ###########################################
        ### Compute sqaured bias convergence rate #
        ###########################################
       
        # QMC
        QMC_biassq_y = np.log(QMC_EstimAverageSquareBiasTrace)
        QMC_biassq_slope, intercept, r_value, p_value, std_err = linregress(x,QMC_biassq_y)    
        
        # PSR
        PSR_biassq_y = np.log(PSR_EstimAverageSquareBiasTrace)    
        PSR_biassq_slope, intercept, r_value, p_value, std_err = linregress(x,PSR_biassq_y)      
            
        ##################################
        ### Compute MSE convergence rate #
        ##################################
    
        # QMC
        QMC_mse_y = np.log(QMC_MSE_Trace)
        QMC_mse_slope, intercept, r_value, p_value, std_err = linregress(x,QMC_mse_y)    
        
        # PSR
        PSR_mse_y = np.log(PSR_MSE_Trace)    
        PSR_mse_slope, intercept, r_value, p_value, std_err = linregress(x,PSR_mse_y)
    
    
    
    
    
    
    
    
        ########################
        # Save arrays to files #
        ########################
        
        # Mean estimates
        np.save('results/{}/PSR_EstimArray'.format(Case), PSR_EstimArray)
        np.save('results/{}/QMC_EstimArray'.format(Case), QMC_EstimArray)
        
        # Empirical variance
        np.savetxt('results/{}/QMC_EstimAverageVarTrace.txt'.format(Case), QMC_EstimAverageVarTrace) 
        np.savetxt('results/{}/PSR_EstimAverageVarTrace.txt'.format(Case), PSR_EstimAverageVarTrace)
        np.savetxt('results/{}/QMC_VarEstimBatchVarTrace.txt'.format(Case), QMC_VarEstimBatchVarTrace) 
        np.savetxt('results/{}/PSR_VarEstimBatchVarTrace.txt'.format(Case), PSR_VarEstimBatchVarTrace)
        
        # Squared bias
        np.savetxt('results/{}/QMC_EstimAverageSquareBiasTrace.txt'.format(Case), QMC_EstimAverageSquareBiasTrace)
        np.savetxt('results/{}/PSR_EstimAverageSquareBiasTrace.txt'.format(Case), PSR_EstimAverageSquareBiasTrace)
        np.savetxt('results/{}/QMC_BiasBatchSquareMeanTraceVar.txt'.format(Case), QMC_BiasBatchSquareMeanTraceVar)
        np.savetxt('results/{}/PSR_BiasBatchSquareMeanTraceVar.txt'.format(Case), PSR_BiasBatchSquareMeanTraceVar)
        
        # Mean squared error 
        np.savetxt('results/{}/QMC_MSE_Trace.txt'.format(Case), QMC_MSE_Trace)    
        np.savetxt('results/{}/PSR_MSE_Trace.txt'.format(Case), PSR_MSE_Trace)
        np.savetxt('results/{}/QMC_BatchMSE_TraceVar .txt'.format(Case), QMC_BatchMSE_TraceVar)    
        np.savetxt('results/{}/PSR_BatchMSE_TraceVar .txt'.format(Case), PSR_BatchMSE_TraceVar)  
    
        # Miscellaneous  
        np.savetxt('results/{}/N_Array.txt'.format(Case), N_Array)
        np.savetxt('results/{}/cpu_time.txt'.format(Case), np.array([TimeCase]))
        np.savetxt('results/{}/NumOfIter.txt'.format(Case), np.array([NumOfIter]))
        np.savetxt('results/{}/StepSize.txt'.format(Case), np.array([StepSize]))
        np.savetxt('results/{}/BurnInPowerOfTwo.txt'.format(Case), np.array([BurnInPowerOfTwo]))
        np.savetxt('results/{}/DegreeOfFreedom.txt'.format(Case), np.array([df]))

        # Empirical variance and MSE reductions    
        np.savetxt('results/{}/VarianceReductions.txt'.format(Case), PSR_EstimAverageVarTrace\
                                                                   / QMC_EstimAverageVarTrace)
        np.savetxt('results/{}/MSE_Reductions.txt'.format(Case), PSR_MSE_Trace/ QMC_MSE_Trace)
        
        # Empirical variance slope
        np.savetxt('results/{}/QMC_EmpiricalVarSlope.txt'.format(Case), np.array([QMC_EmpiricalVarSlope]))
        np.savetxt('results/{}/PSR_EmpiricalVarSlope.txt'.format(Case),np.array([PSR_EmpiricalVarSlope]))
        
        # Squared bias slope 
        np.savetxt('results/{}/QMC_biassq_slope.txt'.format(Case), np.array([QMC_biassq_slope]))
        np.savetxt('results/{}/PSR_biassq_slope.txt'.format(Case), np.array([PSR_biassq_slope])) 
        
        # MSE slope
        np.savetxt('results/{}/QMC_mse_slope.txt'.format(Case), np.array([QMC_mse_slope]))
        np.savetxt('results/{}/PSR_mse_slope.txt'.format(Case), np.array([PSR_mse_slope]))    
        
        # Next case
        c+=1