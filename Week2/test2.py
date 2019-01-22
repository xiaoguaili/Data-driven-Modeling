#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 01:16:14 2018

@author: xiaoguai
"""

import numpy as np

if __name__ == "__main__":
    N = 10
    X = np.linspace(1/4,N/4,N)[:,None]
#    print(np.shape(X)[0])
    basisN = 3
    
    #Legendre
    phi = np.matrix([np.polynomial.legendre.Legendre.basis(i, domain = [0,1])(X[0]) for i in range(1, basisN+1)]).T
    for n in range(1, np.shape(X)[0]):
        phi = np.vstack((phi, np.matrix([np.polynomial.legendre.Legendre.basis(i, domain = [0,1])(X[n]) for i in range(1, basisN+1)]).T))
    print(np.shape(phi))

    #Fourier
#    phi1 = np.matrix([np.sin(2.0*np.pi*i*X[0]) for i in range(1, basisN+1)]).T
#    phi2 = np.matrix([np.cos(2.0*np.pi*i*X[0]) for i in range(0, basisN+1)]).T
#    phi = np.hstack((phi1,phi2))
     
#    for n in range(1, np.shape(X)[0]):
#        phi1 = np.matrix([np.sin(2.0*np.pi*i*X[n]) for i in range(1, basisN+1)]).T
#        phi2 = np.matrix([np.cos(2.0*np.pi*i*X[n]) for i in range(0, basisN+1)]).T
#        phi = np.vstack((phi, np.hstack((phi1,phi2)))) 
#    print(np.shape(phi))
#    for j in range(0, N):
#        print(phi[j])

    # monomials
  #  phi = np.matrix([X[0]**i for i in range(0, basisN)]).T
  #  for n in range(1, N):
  #      phi = np.vstack((phi, np.matrix([X[n]**i for i in range(0, basisN)]).T))
  #  print(np.shape(phi))
  #  y = np.matmul(X.T, phi)
  #  print(np.shape(y))
  #  for n in range(0, N):
  #      print(phi[n])
        
  #  phi.shape
  
  