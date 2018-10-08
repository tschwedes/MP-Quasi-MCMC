#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:22:20 2018

@author: Tobias Schwedes
"""

import numpy as np



def chen_construction(m):
    
    """
    
    Constructs a 1-dimensional CUD-sequence of length 2^m-1 with 10 <= m <= 32    
    according to the construction introduced in       
      
        "New inputs and methods for Markov chain quasi-Monte Carlo"
        
    by
    
        Chen, Matsumoto, Nishimura and Owen (2012)
        
    and coefficient lists of primite polynomials provided in
    
        "Primite polynomials over finite fields"
        
    by

        Hansen and Mullen (1992)
    
    
    Input: 
    ------       
    m 	- int
        	length parameter
    
    Output:
    ------
    cuds 	- array_like
        		CUD sequence, also saved as .npy file        
    
    """

    # Make sure valid CUD length parameter is chosen
    if m<10 or m>32:
        raise Exception('Oops! That was no valid CUD length parameter. '\
                        'Choose integer among: 10 <= m <= 32.') 
        
    # LFSR parameter tupels [m,s=s(m)] such that 2^m-1 and f^s(m) are coprime 
    PrimList = [[10,115], [11,291], [12,172], [13,267], [14,332], [15,388],\
            [16,283], [17,514], [18,698], [19,706], [20,1304], [21,920],\
            [22,1336], [23,1236], [24,1511], [25,1445], [26,1906],\
            [27,1875], [28,2573], [29,2633], [30,2423], [31,3573], [32,3632]] 
    PrimArray = np.array(PrimList)    
    
    # Get parameter s(m) from tupel array
    n = np.where(PrimArray==m)[0][0]
    s = PrimArray[n,1] 
    
    # Define fractions of form 2^-i for binary representation
    bins = 2.**(-np.arange(1,m+1))
    
    # Set up array of CUD numbers
    cuds = np.zeros(2**m-1)
    print ("Constructing a CUD sequence of length =", 2**m-1)
    
    # Starting coefficients
    s0 = np.ones(m)
    so = s0
    
    # Differentiate cases of constructions depending on m and s(m)
    if m==10:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)
    elif m==11:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[2])%2)
            cuds[i] = np.sum(so*bins)
    elif m==12:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[4]+so[6])%2)
            cuds[i] = np.sum(so*bins)
    elif m==13:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[3]+so[4])%2)
            cuds[i] = np.sum(so*bins) 
    elif m==14:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[3]+so[5])%2)
            cuds[i] = np.sum(so*bins)         
    elif m==15:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1])%2)
            cuds[i] = np.sum(so*bins)            
    elif m==16:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[2]+so[3]+so[5])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==17:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==18:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[7])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==19:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[2]+so[5])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==20:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==21:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[2])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==22:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==23:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[5])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==24:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[3]+so[4])%2)
            cuds[i] = np.sum(so*bins)          
    elif m==25:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==26:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[2]+so[6])%2)
            cuds[i] = np.sum(so*bins)            
    elif m==27:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[2]+so[5])%2)
            cuds[i] = np.sum(so*bins)         
    elif m==28:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)               
    elif m==29:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[2])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==30:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[1]+so[4]+so[6])%2)
            cuds[i] = np.sum(so*bins)           
    elif m==31:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[3])%2)
            cuds[i] = np.sum(so*bins)                      
    elif m==32:
        for i in range(2**m-1):
            for j in range(s):
                so = np.append(so[1:], (so[0]+so[2]+so[6]+so[7])%2)
            cuds[i] = np.sum(so*bins)      

    # Save CUD sequence as .npy      
    np.save('CudsChen_{}'.format(m), cuds)
        
    return cuds


if __name__ == '__main__':
    
    # Construct CUD sequence
    Cuds = chen_construction(10)      
       