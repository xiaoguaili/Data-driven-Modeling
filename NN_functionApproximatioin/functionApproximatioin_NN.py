#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:34:55 2018

@author: xiaoguai
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

from models_numpy import NeuralNetwork

if __name__ == "__main__": 
    
    N_start = 500 # number of training points starts from N_start. 
    deltaN = 500 # deltaN denotes increment.
    N_num = 10 # N_num different numbers are studied.
    Nlist = np.array([N_start + i*deltaN for i in range(0, N_num)])
    error = np.array([]) #record error for each number of training points
#%%
    for i in range(0, N_num):
        N = Nlist[i]
        
        X_dim = 2
        Y_dim = 1
        layers = np.array([X_dim,50,50,Y_dim])
        noise = 0.2
    
        # Generate Training Data   
        def f(x):
            return np.reshape(np.cos(np.pi*x[:,0])*np.cos(np.pi*x[:,1]),(x.shape[0],-1))
    
        # Specify input domain bounds
        lb = 50*np.ones((1,X_dim))
        ub = 54*np.ones((1,X_dim)) 
    
        # Generate data
        X = lb + (ub-lb)*lhs(X_dim, N)
        Y = f(X) + noise*np.random.randn(N,Y_dim)

        # Generate Test Data
        N_star = 1000
        X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
        Y_star = f(X_star)
            
        # Create model
        model = NeuralNetwork(X, Y, layers)
        
        # Training
        model.train(nIter = 40000, batch_size = 50)
  
        # Prediction
        Y_pred = model.predict(X_star)  
        error = np.append(error,[sum((Y_pred - Y_star)**2) / sum(Y_star**2)]) 
        print(error)
#%%
        # Plotting
        plt.figure(i)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax = plt.axes(projection='3d')
        ax.plot3D(X_star[:,0], X_star[:,1], Y_star.flatten(), 'b-', linewidth=2)
        ax.plot3D(X_star[:,0], X_star[:,1], Y_pred.flatten(), 'r--', linewidth=2)
        ax.scatter3D(X[:,0], X[:,1], Y)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$f(x)$');
        plt.legend(['$f(x)$', 'prediction', '%d training data' % N], loc='upper right')
        name = "%dTrainingData.png" % N
        plt.savefig(name, format='png')
#%%        
    # Plotting error
    plt.figure(1)
    plt.plot(Nlist, error, linewidth=2)
    plt.xlabel('Number of training data')
    plt.ylabel('error')
    plt.savefig('error.png')
