#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:58:44 2018

@author: Tobias Schwedes

Script to load data for logistic Bayesian regression problems.
"""

import numpy as np

class DataLoad:

    def __init__(self, case):
    
        if case == 'ripley':        
    
            # Two hyperparameters of model
            Polynomial_Order = 1;
    
            # Load and prepare Train and Test Data
            X = np.loadtxt('./Data/ripley.txt')
            self.t = X[:,-1]
            X = X[:,:-1]
            self.m = X.shape[0]
    
            # Standardise Data
            X = (X - np.tile(np.mean(X, axis=0),(self.m,1))) / \
                np.tile(np.std(X, axis=0, ddof=1),(self.m,1))
    
            #Create Polynomial Basis
            self.XX = np.ones((self.m,1))
            for i in range(Polynomial_Order+1)[1:]:
                self.XX = np.concatenate((self.XX, X**i), axis=1)
    
            [self.m,self.d] = self.XX.shape
    

        elif case == 'pima':
            
    
            # Two hyperparameters of model
            Polynomial_Order = 1;
    
            # Load and prepare Train and Test Data
            X = np.loadtxt('./Data/pima.txt')
            self.t = X[:,-1]
            X = X[:,:-1]
            self.m = X.shape[0]
    
            # Standardise Data
            X = (X - np.tile(np.mean(X, axis=0),(self.m,1))) / \
                np.tile(np.std(X, axis=0, ddof=1),(self.m,1))
    
            #Create Polynomial Basis
            self.XX = np.ones((self.m,1))
            for i in range(Polynomial_Order+1)[1:]:
                self.XX = np.concatenate((self.XX, X**i), axis=1)
    
            [self.m,self.d] = self.XX.shape
    
    
        elif case == 'german':
    
            # Two hyperparameters of model
            Polynomial_Order = 1;
    
            # Load and prepare Train and Test Data
            X = np.loadtxt('./Data/german.txt')
            self.t = X[:,-1]
            X = X[:,:-1]
            self.m = X.shape[0]
    
            # Replace all 1s in t with 0s
            self.t[self.t==1] = 0
            # Replace all 2s in t with 1s
            self.t[self.t==2] = 1
    
            # Standardise Data
            X = (X - np.tile(np.mean(X, axis=0),(self.m,1))) / \
                np.tile(np.std(X, axis=0, ddof=1),(self.m,1))
    
            #Create Polynomial Basis
            self.XX = np.ones((self.m,1))
            for i in range(Polynomial_Order+1)[1:]:
                self.XX = np.concatenate((self.XX, X**i), axis=1)
    
            [self.m,self.d] = self.XX.shape
    

        elif case == 'heart':
    
            # Two hyperparameters of model
            Polynomial_Order = 1;
    
            # Load and prepare Train and Test Data
            X = np.loadtxt('./Data/heart.txt')
            self.t = X[:,-1]
            X = X[:,:-1]
            self.m = X.shape[0]
    
            # Replace all 1s in t with 0s
            self.t[self.t==1] = 0
            # Replace all 2s in t with 1s
            self.t[self.t==2] = 1
    
            # Standardise Data
            X = (X - np.tile(np.mean(X, axis=0),(self.m,1))) / \
                np.tile(np.std(X, axis=0, ddof=1),(self.m,1))
    
            #Create Polynomial Basis
            self.XX = np.ones((self.m,1))
            for i in range(Polynomial_Order+1)[1:]:
                self.XX = np.concatenate((self.XX, X**i), axis=1)
    
            [self.m,self.d] = self.XX.shape
   
    
        elif case == 'australian':
    
    
            # Two hyperparameters of model
            Polynomial_Order = 1;
    
            # Load and prepare Train and Test Data
            X = np.loadtxt('./Data/australian.txt')
            self.t = X[:,-1]
            X = X[:,:-1]
            self.m = X.shape[0]
    
            # Standardise Data
            X = (X - np.tile(np.mean(X, axis=0),(self.m,1))) / \
                np.tile(np.std(X, axis=0, ddof=1),(self.m,1))
    
            #Create Polynomial Basis
            self.XX = np.ones((self.m,1))
            for i in range(Polynomial_Order+1)[1:]:
                self.XX = np.concatenate((self.XX, X**i), axis=1)
    
            [self.m,self.d] = self.XX.shape
            
        else:
            raise ValueError("case must be chosen from one of the following: 'ripley',\
                             'pima', 'heart', 'australian', 'german'")
          
    def GetDimension(self):
        return self.d
        
    def GetDesignMatrix(self):
        return self.XX
    
    def GetNumOfSamples(self):
        return self.m        

    def GetResponses(self):
        return self.t       