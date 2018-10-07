#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:22:20 2018

@author: Tobias Schwedes
"""

import numpy as np


def tribble_construction(d, N):
       
    """
    
    Constructs a d-dimensional CUD-sequence of length N according to the
    linear congruential generator introduced in
      
        "Construction of weakly CUD sequences for MCMC sampling"
        
    by
    
        Tribble and Owen (2008)
        
    
    Input:
    ------        
    d 	- int
        	dimension
    N 	- int
        	length of CUD sequence
    
    Output:
    -------
    cuds 	- array_like
        		CUD sequence, also saved as .npy file        
    
    """    
    
    # Set up array of CUD numbers
    cuds = np.zeros((N-1)*d)
    print ("Constructing a {}-dimensional CUD sequence of length =".format(d), N)
     
               
    # Differentiate cases of constructions depending on length N
    if N==251:
        a=55
    elif N==509:
        a=35
    elif N==1021:
        a=65   
    elif N==2039:
        a=393        
    elif N==4093:
        a=235
    elif N==8191:
        a=884       
    elif N==16381:
        a=665         
    elif N==32749:
        a=219              
    elif N==65521:
        a=2469           
    elif N==131071:
        a=29223           
    elif N==262139:
        a=21876                     
    elif N==524287:
        a=37698           
    elif N==1048573:
        a=22202        
    elif N==2097143:
        a=360889          
    elif N==4194301:
        a=914334   
    else:
        raise Exception('Oops! That was no valid CUD length. Choose number '\
                        'among: N= 251, 509, 1021, 2039, 4093, 8191, 16381, '\
                        '32749, 65521, 131071, 262139, 524287, 1048573, '\
                        '2097143, 4194301.')
 
    # Construct CUD sequence
    for i in range((N-1)*d):
        cuds[i] = np.mod(a**i,N)
    cuds = np.append(np.zeros(d), cuds) / N
    cuds = cuds.reshape(N,d)

    # Save CUD sequence as .npy      
    np.save('CudsTribble_dim{}_{}'.format(d, N), cuds)
    
    return cuds
            
if __name__ == '__main__':
    
    # Construct CUD sequence
    d=2
    N=1021
    Cuds = tribble_construction(d,N)
