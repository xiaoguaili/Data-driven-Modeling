#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:57:04 2018

@author: xiaoguai
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

from models import BayesianLinearRegression
from basisFunctions import Basis

if __name__ == "__main__": 
    
    # N is the number of training points.
    N = 200
    noise_var = np.sqrt(0.1)
    alpha = 10.0
    beta = 0.05
    
    # Create random input and output data
    X = lhs(1, N)
    y = np.sin(2.0*np.pi*X) + noise_var*np.random.randn(N,1)
    
    # Define basis functions
    basisN = 3 #highest order of basis functions
    p = Basis(X, basisN)
    
    # choice 1: use monomials as basis functions
#    phi = p.monomials()
    # choice 2: use Fourier as basis functions
#    phi = p.Fourier()
    # choice 3: use Legendre polynomials as basis functions
    phi = p.Legendre()
    
    # Define model
    m = BayesianLinearRegression(phi, y, alpha, beta)
      
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()
    
    # Predict at a set of test points
    Num_X_star = 200
    X_star = np.linspace(0,1,Num_X_star)[:,None]
    p_star = Basis(X_star, basisN)
    
    # phi_star should be consistent with the basis funcitons that was used above
#    phi_star = p_star.monomials() 
#    phi_star = p_star.Fourier()
    phi_star = p_star.Legendre()
    
    y_pred_MLE = np.matmul(phi_star, w_MLE)
    y_pred_MAP = np.matmul(phi_star, w_MAP)
    
    # Draw sampes from the predictive posterior
    num_samples = 1000
    mean_star, var_star = m.predictive_distribution(phi_star)
    mean_star = np.array(mean_star)
    mean_star = np.reshape(mean_star, Num_X_star)
    samples = np.random.multivariate_normal(mean_star, var_star, num_samples)
    
    # Plot
    plt.figure(1, figsize=(8,6))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i,:], 'k', linewidth=0.05)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.axis('tight')


    # Plot distribution of w
    # With different number of basis funcitons, there are different number of w entries.
    # The range of x need to change according to basis functions and their number.
    for n in range(0, np.shape(w_MAP)[0]):
        plt.subplot(1,2,2)
        x_axis = np.linspace(0.5, 3, 1000)[:,None]
        plt.plot(x_axis, norm.pdf(x_axis,w_MAP[n],Lambda_inv[n,n]), label = 'p(w'+ str(n)+'|D)')
        plt.legend()
        plt.xlabel('$w$')
        plt.ylabel('$p(w|D)$')
        plt.axis('tight')
