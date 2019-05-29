#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 20 15:11:48 2019

@author: Tobias Schwedes

Script to generate the seed used for (Quasi-)MCMC with multiple
proposals and importance sampling applied to Bayesian linear regression.
"""

import numpy as np


def SeedGen(d, PowerOfTwo, Stream):

    """
    Function to generate the seed used to run (Quasi-)MCMC

    inputs:
    -------   
    d               - int
                    dimension of posterior    
    PowerOfTwo      - int in [10,23]
                    defines size S of seed by S=2**PowerOfTwo-1
    Stream          - string
                    either 'cud' or 'iid'; defining what seed is used

    outputs:
    ------- 
    xs              - array_like
                    (2**PowerOfTwo) x d-Array of seed
    """
    
    if Stream == 'iid':
        # Generate iid random uniformly distributed RV in [0,1]
        xs = np.random.uniform(0,1,(int((2**PowerOfTwo-1)/d)*d,d))

    elif Stream == 'cud':
        # Load cud point sequence
        cuds = np.load('../CUDs/ChenEtAl/CudsChen_{}.npy'.format(PowerOfTwo))
        
        # Create d-dimensional sequence by shifted cud sequence
        UsedLength = int(cuds.shape[0]/d)*d
        TrimmedCuds = cuds[:UsedLength]
        xs = TrimmedCuds
        
        for i in range(d)[1:]:
            xs = np.append(xs, np.roll(TrimmedCuds,-i))        
        
        xs = xs.reshape(UsedLength,d)
        xs = np.append(np.zeros(d)+1e-9,xs).reshape(UsedLength+1,d)
        u_rand = np.random.uniform(0,1,d)
        xs_sh = xs + u_rand
        xs = xs_sh - np.floor(xs_sh) 

    else:
        raise ValueError('Stream must be chose either as "iid" or as "cud"')

    return xs
