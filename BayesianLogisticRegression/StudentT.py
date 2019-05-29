#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 13:27:50 2018

@author: Tobias Schwedes

Multivariate student's t continuous random variables

"""

import numpy as np
from scipy.stats import gamma, norm, multivariate_normal
from scipy.special import gamma as GammaFun
import matplotlib.pylab as plt


def multivariate_t_rvs(Mean, Sigma, df=np.inf, n=1):
    
    """
    Generate random variables of multivariate t distribution
    
    Inputs:
    -------
    Mean    - array_like
            mean of random variable, length determines dimension of random variable
    Sigma   - array_like
            square array of covariance  matrix
    df      - int or float
            degrees of freedom
    n       - int
            number of observations, return random array will be (n, len(Mean))
    
    Outputs:
    -------
    rvs : ndarray, (n, len(Mean)); each row is a draw of a multivariate t distributed
            random variable
    """
    
    d = Mean.shape[0]
    
    if df == np.inf:
        U = 1.
    else:
        U = np.random.gamma(df/2., 2./df, size=(n,1))     
              
    X = np.random.multivariate_normal(np.zeros(d),Sigma,(n,))
    rvs = Mean + X/np.tile(np.sqrt(U), [1,d])
     
    return rvs




def multivariate_t_rvs_custom_seed(Seed, Mean, CholSigma, df=np.inf):
    
    """
    Generate samples of multivariate t distribution from a custom seed
    
    Inputs:
    -------
    Seed       - array_like
            seed used to generate samples, shape of (n, len(Mean))
    Mean       - array_like
            mean of random variable, length determines dimension of random variable
    CholSigma   - array_like
            square array of cholesky decomposition of covariance  matrix
    df          - int or float
            degrees of freedom

    Outputs:
    --------
    rvs         - ndarray, (n, len(m))
            each row is an independent draw of a multivariate t distributed
            random variable
    """
    
    d = Mean.shape[0]
    
    if df == np.inf:
        U = 1.
    else:
        U = gamma.ppf([Seed[:,0]], df/2., scale=2./df).T
       
        
    X = np.dot(CholSigma, norm.ppf(Seed[:,1:], loc=0., scale=1).T).T   
    rvs = Mean + X/np.tile(np.sqrt(U), [1,d])
     
    return rvs





def multivariate_t_pdf(X, Mean, Sigma, df=np.inf):

    """
    Computes multivariate t-student density
    
    Inputs:
    ------    
    X           - array_like
            d-dimensional evaluation point
    mu          - array_like
            d-dimensional mean value
    Sigma       - array_like
            square array (dxd) scale matrix
    df          - int or float
            degrees of freedom
    d           - int
            dimension        
        
    Outputs:
    --------    
    the density of the given evaluation point

    """   
    
    d = Mean.shape[0]
    
    if df == np.inf:
        vals = multivariate_normal.pdf(X, Mean, Sigma)
    else: 
        Num1 = GammaFun((df+d)/2.)
        Denom = GammaFun(df/2.)*df**(d/2.)*np.pi**(d/2.)*np.linalg.det(Sigma)**(1./2)
        if len(X.shape)>1:
            Num2 = (1. + 1./df * np.dot( np.dot( (X-Mean), np.linalg.inv(Sigma)), \
                                        (X-Mean).T).diagonal(0))**(-(df+d)/2.)     
        else:    
            Num2 = (1. + 1./df * np.dot( np.dot( (X-Mean), np.linalg.inv(Sigma)), \
                                        (X-Mean)))**(-(df+d)/2.)   
            
        vals = Num1 * Num2 / Denom 

    return vals


def multivariate_t_LogPdf(X, Mean, Sigma, df=np.inf):
    
    """
    Computes Logarithm of multivariate t-student density
    
    Inputs:
    ------    
    X           - array_like
            d-dimensional evaluation point
    mu          - array_like
            d-dimensional mean value
    Sigma       - array_like
            square array (dxd) scale matrix
    df          - int or float
            degrees of freedom
    d           - int
            dimension        
        
    Outputs:
    --------    
    the log-density of the given evaluation point

    """ 
    
    d = Mean.shape[0]
    
    if df == np.inf:
        vals = np.log(multivariate_normal.pdf(X, Mean, Sigma))
        
    else: 
        Term1 = np.log(GammaFun((df+d)/2.)) 
        Term2 = - np.log(GammaFun(df/2.)*df**(d/2.)*np.pi**(d/2.)*np.linalg.det(Sigma)**(1./2))
        if len(X.shape)>1:
            Term3 = -(df+d)/2. * np.log((1. + 1./df * np.dot( np.dot( (X-Mean), np.linalg.inv(Sigma)), \
                                        (X-Mean).T).diagonal(0)))
        else:
            Term3 = -(df+d)/2. * np.log((1. + 1./df * np.dot( np.dot( (X-Mean), np.linalg.inv(Sigma)), \
                                        (X-Mean))))
            
        vals = Term1+Term2+Term3

    return vals



if __name__ == '__main__':

    # Parameters
    n = int(1e4)
    Mean = np.array([1,2])
    Sigma = np.array([[1.,0.3],[0.3,3.]])
    df=20
    
    # Generate random variables
    X = multivariate_t_rvs(Mean, Sigma, df=df, n=n)
    print ('Mean = ', np.mean(X, axis=0))
    print ('Covariance = ', np.cov(X.T))
    
    # Compare to variates generated by custom seed
    CholSigma = np.linalg.cholesky(Sigma)
    #np.random.seed(0)    
    Seed = np.random.uniform(0,1,size=(n, len(Mean)+1))
    Y = multivariate_t_rvs_custom_seed(Seed, Mean, CholSigma, df=df)
    print ('Mean = ', np.mean(Y, axis=0))
    print ('Covariance = ', np.cov(Y.T))   
    
    # Plots histgrams of both
    plt.hist(X[:,0], bins=30, density=True, lw=2, alpha = 0.5, color= 'b')
    plt.hist(Y[:,0], bins=30, density=True, lw=2, alpha = 0.5, color= 'r')
    plt.rcParams["patch.force_edgecolor"] = True    
    plt.show()
