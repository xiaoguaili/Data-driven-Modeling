#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:16:34 2018

@author: xiaoguai
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from PDEsolver_tf import NeuralNetwork
#from models_pytorch2 import NeuralNetwork

if __name__ == "__main__": 
    
    N_start = 3 # number of training points starts from N_start. 
    deltaN = 3 # deltaN denotes increment.
    N_num = 10 # N_num different numbers are studied.
    Nlist = np.array([N_start + i*deltaN for i in range(0, N_num)])   
    N_u = 2 # number of boundary points
    error = np.array([]) #record error for each number of training points
    for i in range(0, N_num):
        N_f = Nlist[i]
        layers = np.array([1,50,50,1])
        # Generate data
        X_u = np.array([[-1], [1]])
        Y_u = np.array([[0], [0]])
        X_f = -1 + 2*lhs(1, N_f)
        Y_f = -(np.pi**2+1)*np.sin(np.pi*X_f)
    
        # Create model
        model = NeuralNetwork(X_u, Y_u, X_f, Y_f, layers)
    
        # Training
        model.train(nIter = 40000, batch_size = N_u)
    
        # Generate Test Data
        N_star = 1000
        X_star = -1 + 2*np.linspace(0,1,N_star)[:,None]
        U_star = np.sin(np.pi*X_star)
    
        # Prediction
        U_pred = model.predict(X_star)
        error = np.append(error,[sum((U_pred - U_star)**2) / sum(U_star**2)]) 
        print(error)
    #%%
        # Plotting
        plt.figure(i)
        plt.plot(X_star, U_star, 'b-', linewidth=2)
        plt.plot(X_star, U_pred, 'r--', linewidth=2)
        plt.xlabel('$x$')
        plt.ylabel('$u(x)$')
        plt.legend(['$u(x)$', 'prediction'], loc='lower right')
        name = "%dTrainingPoints.png" % N_f
        plt.savefig(name, format='png')
#%%        
    # Plotting error
    plt.figure(N_num+1)
    plt.plot(Nlist, error, linewidth=2)
    plt.xlabel('Number of training points')
    plt.ylabel('error')
    plt.savefig('error.png')
    