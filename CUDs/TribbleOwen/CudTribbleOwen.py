#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:22:20 2018

@author: Tobias Schwedes
"""

import numpy as np
from math import gcd


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
    
           
    # Differentiate cases of constructions depending on length N
    if N==251:
        if d<12:
            a=33
        else:    
            a=55
    elif N==509:
        a=35
    elif N==1021:
        if d<12 or d>24:
            a=65
        else:    
            a=331   
    elif N==2039:
        if d<12:
            a=995
        elif 12<=d<=24:    
            a=328      
        else:
            a=393 
    elif N==4093:
        if d<12:
            a=209
        else:
            a=219 
    elif N==8191:
        if d<12:
            a=884
        else:
            a=1716
    elif N==16381:
        if d<12:
            a=572
        else:
            a=665
    elif N==32749:
        if d<12:
            a=219
        else:
            a=9515            
    elif N==65521:
        if d<12:
            a=17364
        else:
            a=2469          
    elif N==131071:
        if d<12:
            a=43165
        elif d<=12<=24:
            a=29223
        else:
            a=29803         
    elif N==262139:
        if d<12:
            a=92717
        else:
            a=21876                  
    elif N==524287:
        if d<12:
            a=283741
        elif d<=12<=24:
            a=37698
        else:
            a=155411     
    elif N==1048573:
        if d<12:
            a=380985
        elif d<=12<=24:
            a=100768
        else:
            a=22202       
    elif N==2097143:
        if d<12:
            a=360889
        elif d<=12<=24:
            a=1043187
        else:
            a=1939807        
    elif N==4194301:
        if d<12:
            a=914334
        else:
            a=1731287
            
    else:
        raise Exception('Oops! That was no valid CUD length. Choose number '\
                        'among: N= 251, 509, 1021, 2039, 4093, 8191, 16381, '\
                        '32749, 65521, 131071, 262139, 524287, 1048573, '\
                        '2097143, 4194301.')
 
    # Set up array of CUD numbers
    cuds = np.zeros((N-1)*d)
    print ("Constructing a CUD sequence of length =", N)      
    
    # Construct CUD sequence
    A=1
    for i in range((N-1)*d):
        cuds[i]=A
        A=(A*a)%N   
    cuds = np.append(np.zeros(d), cuds) 
    cuds = cuds.reshape(N,d)        
    
    # Multiply k-th identical block with a^(k-1) (mod N) if GCD(N-1,d)>1 
    g = gcd(N-1,d)
    if g==1:
        cuds=cuds/N
    else:
        b=int((N-1)/g)
        A=1
        for j in range(b):
            cuds[1+j*b:1+(j+1)*b]=np.mod(cuds[1+j*b:1+(j+1)*b]*A,N)/N
            A=(A*a)%N
            
    # Save CUD sequence as .npy      
    np.save('CudsTribble_dim{}_{}'.format(d, N), cuds)
    
    return cuds
          

if __name__ == '__main__':
  
    for N in [509,1021,2039,4093,8191,16381,32749,65521,131071,262139,524287]:
        import time
        start=time.time()
        A = tribble_construction(26,N)
        end=time.time()
        print("CPU time = ", end-start)
    
#    import matplotlib.pylab as plt
#    plt.plot(A[:,1],A[:,3], '.')  
#    plt.show()    
