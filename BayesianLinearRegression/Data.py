#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:48:36 2018

@author: Tobias Schwedes


Script to generate data for a linear Bayesian regression problem.
"""

import numpy as np


class DataGen:
    
    def __init__(self, alpha, d):
        

        """    
        Inputs:
        ------
        alpha           - float 
                        Obervation noise scaling   
        d               - int 
                        dimension of posterior    
        """                
        
        np.random.seed(0)
        
        # Create design matrix
        self.n_samples, n_features   = int(d**(1./2)*100), d
        self.X                       = np.random.randn(self.n_samples, \
                                                       n_features) # Create Gaussian data        
        
        # Weights
        self.w = np.ones(d)
        
        # Create noise with a precision alpha
        self.noise = np.random.normal(loc=0, scale=1. / np.sqrt(alpha), \
                                      size=self.n_samples)
        
        # Create observations
        self.obs = np.dot(self.X, self.w) + self.noise
        
        np.random.seed()  

    def getNumOfSamples(self):
        return self.n_samples
    
    def getWeights(self):
        return self.w

    def getNoise(self):
        return self.noise

    def getObservations(self):
        return self.obs

    def getDesignMatrix(self):
        return self.X
