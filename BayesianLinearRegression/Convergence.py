#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 15:25:19 2018

@author: Tobias Schwedes
"""

"""
Script to analyse the convergence in Empirical Variance and Squared Bias
(as well as combined = MSE) for importance sampling MP-MCMC driven by a
IID seed VS. by a CUD seed. The underlying sampling method makes use of
Riemannian manifold Metropolis-Adjunsted Langevin proposals.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import root
from scipy import stats

from BayesianLinReg import BayesianLinReg
from Data import DataGen


# Specify directory under which results are saved
DirName = 'results'

# Create directory to save results in
try:
    # Create target Directory
    os.mkdir(DirName)
    print("Directory " , DirName ,  " Created ") 
except FileExistsError:
    print("Directory " , DirName ,  " already exists")
    


if __name__ == '__main__':

    # Starting Time
    StartTimeAll = time.time()

    #################
    # Generate Data #
    #################

    alpha       = 0.5
    d           = 10
    Data        = DataGen(alpha, d)
    X           = Data.DesignMatrix()
    obs         = Data.Observations()
    n_samples   = Data.NumOfSamples()
    
    
    #############################
    # Parameters for simulation #
    #############################  
    
    # Number of simulations
    NumOfSim = 5
    # Define size of seed by powers of two
    PowerOfTwoArray = np.arange(11,20) #10 for N=3 and 19 for N=1023
    # Define number of proposed states
    N_Array = np.array([3,7])#,15,31])#,63,127,255,511,1023])  
    # Proposal step size
    StepSize = np.sqrt(2) 
    # Proposal covariance scaling
    CovScaling = 1.
    
    
    
    ###########################################
    # Compute prior and likelihood quantities #
    ###########################################
    
    # Compute covariance of g-prior
    g = 1./n_samples
    sigmaSq = 1./alpha
    G_prior = sigmaSq / g * np.linalg.inv(np.dot(X.T,X))
    InvG_prior = np.linalg.inv(G_prior)
    Lambda0 = sigmaSq * InvG_prior
    
    # Gaussian approximations for independent sampler
    PostMean = np.dot(np.linalg.pinv(np.dot(X.T,X) + Lambda0), np.dot(X.T,obs))
    PostCov = sigmaSq*np.linalg.pinv(np.dot(X.T,X) + Lambda0)      
    
    # Fisher Information as constant metric tensor
    FisherInfo = InvG_prior + alpha*np.dot(X.T,X)
    InvFisherInfo = np.linalg.inv(FisherInfo) 


    ##########################################################
    # Compute starting value as root of posterior derivative #
    ##########################################################

    # Define posterior derivative
    def PostDeriv(z):
        Val = -np.dot(InvG_prior,z) + alpha * np.dot(X.T, (obs - np.dot(X, z)))
        return Val
    
    # Compute root
    RootRes = root(PostDeriv, np.zeros(d), tol=1e-12)
    x0 = RootRes.x  # Starting value      
    


    ##################
    # Initialisation #
    ##################

    # Arrays to be filled with IS posterior estimates
    QMC_EstimArray = np.zeros((len(N_Array), NumOfSim, d))
    PSR_EstimArray = np.zeros((len(N_Array), NumOfSim, d))

    for p in range(N_Array.shape[0]):
        
        Counter = 0
        
        for j in range(NumOfSim):

            #############################
            # Parameters for simulation #
            ############################# 
            
            N = int(N_Array[p])
            PowerOfTwo = PowerOfTwoArray[p]
            NumOfIter = int((2**PowerOfTwo-1.)/(N+1))
            BurnIn = 0

            print ('Number of proposals = ', N)

            # Starting Time
            StartTime = time.time()
               
            
            ##################
            # Run simulation #
            ##################
            
            QMC_BLR = BayesianLinReg(d, alpha, x0, N, StepSize, CovScaling, \
                             PowerOfTwo, Stream='cud')            
            PSR_BLR = BayesianLinReg(d, alpha, x0, N, StepSize, CovScaling, \
                             PowerOfTwo, Stream='iid')     
                  
            # Stopping time
            EndTime = time.time()
            print ("CPU time for single pair of simulations =", EndTime - StartTime)
            
            # Percent of simulations for proposal number N done
            Counter += 1
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print ("{} % of simulations for N={} done: ".format(Counter/NumOfSim*100, N))
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            
            # Acceptance rate of MP-(Q)MCMC
            QMC_AcceptRate = QMC_BLR.AcceptRate(BurnIn)
            print ("QMC Acceptance Rate = ", QMC_AcceptRate)
            PSR_AcceptRate = PSR_BLR.AcceptRate(BurnIn)
            print ("PSR Acceptance Rate = ", PSR_AcceptRate)


            ################## QMC #####################

            # Compute estimated IS mean
            QMC_EstimArray[p,j,:] = QMC_BLR.IS_MeanEstimate(N, BurnIn)

            ################## PSR #####################

            # Compute estimated IS mean
            PSR_EstimArray[p,j,:] = PSR_BLR.IS_MeanEstimate(N, BurnIn)


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
    QMC_EstimAverageSquareBias = (np.mean(QMC_EstimArray-PostMean, axis=1))**2
    PSR_EstimAverageSquareBias = (np.mean(PSR_EstimArray, axis=1)-PostMean)**2
    
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
    QMC_BiasBatches = np.array(np.array_split(QMC_EstimArray-PostMean, BatchSize, axis=1))
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
    PSR_BiasBatches = np.array(np.array_split(PSR_EstimArray-PostMean, BatchSize, axis=1))
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
    EndTimeAll = time.time()
    print ("Overall CPU time =", EndTimeAll - StartTimeAll)

    #########################################################################

 
    ###############################
    ### Empirica Variance PLOTS ###
    ###############################
    
    # Fancier plots
    fig, ax1 = plt.subplots()
    fig.tight_layout()
    
    ax1.errorbar(N_Array, PSR_EstimAverageVarTrace, \
                yerr=3*np.sqrt(PSR_VarEstimBatchVarTrace), fmt='--o', \
                markersize=3, label = 'PSR',elinewidth = 1, capsize = 3, \
                )
    ax1.errorbar(N_Array, QMC_EstimAverageVarTrace, \
                yerr=3*np.sqrt(QMC_VarEstimBatchVarTrace), fmt='-o', \
                markersize=3, label = 'QMC',elinewidth = 1, capsize = 3, \
                )

    ax1.errorbar(N_Array, .1*1e1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
    ax1.errorbar(N_Array, 0.3*1e3*(N_Array*NumOfIter)**(-2.), fmt=':', \
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

    ax2 = ax1.twiny()
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.errorbar(N_Array*NumOfIter, 0.1*1e1*(N_Array*NumOfIter)**(-1.), fmt='--', \
                label = r'$\sim n^{-1}$', elinewidth = 1, color='0.5')
    ax2.errorbar(N_Array*NumOfIter, 0.3*1e3*(N_Array*NumOfIter)**(-2.), fmt=':', \
                label = r'$\sim n^{-2}$', elinewidth = 1, color='0.5')
    x2_ticks_labels = np.array([5000, 25000, 100000, 500000])
    ax2.set_xticks(x2_ticks_labels)
    ax2.set_xticklabels(x2_ticks_labels, fontsize=11)
    ax2.set_xlabel('Total Number of Samples $n$ \n (%i Iterations)' %NumOfIter, color='k')

    fig.tight_layout()
#    plt.show()
    plt.savefig('{}/emprVar_{}mcmc.eps'.format(DirName, NumOfSim), format='eps')




    #############################################
    ### Empirica Variance & Bias Square PLOTS ###
    #############################################

    # Fancier plots
    fig, ax1 = plt.subplots()
    fig.tight_layout()

    ax1.errorbar(N_Array, PSR_EstimAverageVarTrace, \
                yerr=3*np.sqrt(PSR_VarEstimBatchVarTrace), fmt='-o', \
                markersize=3, label = 'Var (PSR)',elinewidth = 1, capsize = 3, \
                )
    ax1.errorbar(N_Array, PSR_EstimAverageSquareBiasTrace, \
#                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                fmt='-.', \
                markersize=3, label = r'$Bias^2$ (PSR)',elinewidth = 1, capsize = 3, \
                )   
    
    
    ax1.errorbar(N_Array, QMC_EstimAverageVarTrace, \
                yerr=3*np.sqrt(QMC_VarEstimBatchVarTrace), fmt='--o', \
                markersize=3, label = 'Var (QMC)',elinewidth = 1, capsize = 3, \
                )
    ax1.errorbar(N_Array, QMC_EstimAverageSquareBiasTrace, fmt=':', \
#                yerr=1*np.sqrt(QMC_BiasBatchSquareMeanTraceVar),              
                markersize=3, label = r'$Bias^2$ (QMC)',elinewidth = 1, capsize = 3, \
                )    
    
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
    plt.savefig('{}/emprVar_BiasSquare_{}mcmc.eps'.format(DirName, NumOfSim), format='eps')



    #########################
    ### Bias Square PLOTS ###
    #########################

    # Fancier plots
    fig, ax1 = plt.subplots()
    fig.tight_layout()

    ax1.errorbar(N_Array, PSR_EstimAverageSquareBiasTrace, \
#                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                fmt='--o', \
                markersize=3, label = r'PSR', elinewidth = 1, capsize = 3, \
                )   
    
    ax1.errorbar(N_Array, QMC_EstimAverageSquareBiasTrace, \
#                yerr=1*np.sqrt(PSR_BiasBatchSquareMeanTraceVar), 
                fmt='-o', \
                markersize=3, label = r'QMC', elinewidth = 1, capsize = 3, \
                )   


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
    plt.savefig('{}/BiasSquare_{}mcmc.eps'.format(DirName, NumOfSim), format='eps')



    #################
    ### MSE PLOTS ###
    #################

    # Fancier plots
    fig, ax1 = plt.subplots()
    fig.tight_layout()

    ax1.errorbar(N_Array, QMC_MSE_Trace, fmt='-o',\
#                yerr=3*np.sqrt(QMC_BatchMSE_TraceVar), fmt='-o', \
                markersize=3, label = 'QMC',elinewidth = 1, capsize = 3, \
                )#color = '0.25')
    ax1.errorbar(N_Array, PSR_MSE_Trace, fmt='--o', \
#                yerr=3*np.sqrt(PSR_BatchMSE_TraceVar), fmt='-o', \
                markersize=3, label = 'PSR',elinewidth = 1, capsize = 3, \
                )#color = '0.25')

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

    fig.tight_layout()
#    plt.show()
    plt.savefig('{}/MSE_{}mcmc.eps'.format(DirName, NumOfSim), format='eps')



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
    QMC_EmpiricalVarSlope, PSR_Intercept, PSR_rValue, PSR_pValue, PSR_StdErr = stats.linregress(x,QMC_EmpiricalVar_y)    
    
    # PSR
    PSR_EmpiricalVar_y = np.log(PSR_EstimAverageVarTrace)    
    PSR_EmpiricalVarSlope, QMC_Intercept, QMC_rValue, QMC_pValue, QMC_StdErr = stats.linregress(x,PSR_EmpiricalVar_y)    
  
    ###########################################
    ### Compute sqaured bias convergence rate #
    ###########################################
   
    # QMC
    QMC_biassq_y = np.log(QMC_EstimAverageSquareBiasTrace)
    QMC_biassq_slope, intercept, r_value, p_value, std_err = stats.linregress(x,QMC_biassq_y)    
    
    # PSR
    PSR_biassq_y = np.log(PSR_EstimAverageSquareBiasTrace)    
    PSR_biassq_slope, intercept, r_value, p_value, std_err = stats.linregress(x,PSR_biassq_y)      
        
    ##################################
    ### Compute MSE convergence rate #
    ##################################

    # QMC
    QMC_mse_y = np.log(QMC_MSE_Trace)
    QMC_mse_slope, intercept, r_value, p_value, std_err = stats.linregress(x,QMC_mse_y)    
    
    # PSR
    PSR_mse_y = np.log(PSR_MSE_Trace)    
    PSR_mse_slope, intercept, r_value, p_value, std_err = stats.linregress(x,PSR_mse_y)








    ########################
    # Save arrays to files #
    ########################
    
    # Mean estimates
    np.save('{}/PSR_EstimArray'.format(DirName), PSR_EstimArray)
    np.save('{}/QMC_EstimArray'.format(DirName), QMC_EstimArray)
    
    # Empirical variance
    np.savetxt('{}/QMC_EstimAverageVarTrace.txt'.format(DirName), QMC_EstimAverageVarTrace) 
    np.savetxt('{}/PSR_EstimAverageVarTrace.txt'.format(DirName), PSR_EstimAverageVarTrace)
    np.savetxt('{}/QMC_VarEstimBatchVarTrace.txt'.format(DirName), QMC_VarEstimBatchVarTrace) 
    np.savetxt('{}/PSR_VarEstimBatchVarTrace.txt'.format(DirName), PSR_VarEstimBatchVarTrace)
    
    # Squared bias
    np.savetxt('{}/QMC_EstimAverageSquareBiasTrace.txt'.format(DirName), QMC_EstimAverageSquareBiasTrace)
    np.savetxt('{}/PSR_EstimAverageSquareBiasTrace.txt'.format(DirName), PSR_EstimAverageSquareBiasTrace)
    np.savetxt('{}/QMC_BiasBatchSquareMeanTraceVar.txt'.format(DirName), QMC_BiasBatchSquareMeanTraceVar)
    np.savetxt('{}/PSR_BiasBatchSquareMeanTraceVar.txt'.format(DirName), PSR_BiasBatchSquareMeanTraceVar)
    
    # Mean squared error 
    np.savetxt('{}/QMC_MSE_Trace.txt'.format(DirName), QMC_MSE_Trace)    
    np.savetxt('{}/PSR_MSE_Trace.txt'.format(DirName), PSR_MSE_Trace)
    np.savetxt('{}/QMC_BatchMSE_TraceVar .txt'.format(DirName), QMC_BatchMSE_TraceVar)    
    np.savetxt('{}/PSR_BatchMSE_TraceVar .txt'.format(DirName), PSR_BatchMSE_TraceVar)  

    # Miscellaneous  
    np.savetxt('{}/N_Array.txt'.format(DirName), N_Array)
    np.savetxt('{}/cpu_time.txt'.format(DirName), np.array([EndTimeAll - StartTimeAll]))
    np.savetxt('{}/NumOfIter.txt'.format(DirName), np.array([NumOfIter]))
    np.savetxt('{}/StepSize.txt'.format(DirName), np.array([StepSize]))
    np.savetxt('{}/CovScaling.txt'.format(DirName), np.array([CovScaling]))

    # Empirical variance and MSE reductions    
    np.savetxt('{}/VarianceReductions.txt'.format(DirName), PSR_EstimAverageVarTrace\
                                                               / QMC_EstimAverageVarTrace)
    np.savetxt('{}/MSE_Reductions.txt'.format(DirName), PSR_MSE_Trace/ QMC_MSE_Trace)
    
    # Empirical variance slope
    np.savetxt('{}/QMC_EmpiricalVarSlope.txt'.format(DirName), np.array([QMC_EmpiricalVarSlope]))
    np.savetxt('{}/PSR_EmpiricalVarSlope.txt'.format(DirName),np.array([PSR_EmpiricalVarSlope]))
    
    # Squared bias slope 
    np.savetxt('{}/QMC_biassq_slope.txt'.format(DirName), np.array([QMC_biassq_slope]))
    np.savetxt('{}/PSR_biassq_slope.txt'.format(DirName), np.array([PSR_biassq_slope])) 
    
    # MSE slope
    np.savetxt('{}/QMC_mse_slope.txt'.format(DirName), np.array([QMC_mse_slope]))
    np.savetxt('{}/PSR_mse_slope.txt'.format(DirName), np.array([PSR_mse_slope]))    