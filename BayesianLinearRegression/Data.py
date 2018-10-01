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

    def NumOfSamples(self):
        return self.n_samples
    
    def Weights(self):
        return self.w

    def Noise(self):
        return self.noise

    def Observations(self):
        return self.obs

    def DesignMatrix(self):
        return self.X
