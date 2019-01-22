#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 00:11:44 2018

@author: xiaoguai
"""

import numpy as np
class Basis:
  """
    Linear regression model: y = (w.T)*phi, where phi maps x (d dimensions) to phi(x) (m dimensions)
  """
  def __init__(self, X, basisN):
      
      self.X = X
      self.basisN = basisN
      
  def monomials(self): 
      phi = np.matrix([self.X[0]**i for i in range(0, self.basisN+1)]).T 
      # basisN is defined as highest order of basis functions such that there are basisN+1 functions
      for n in range(1, np.shape(self.X)[0]):
          phi = np.vstack((phi, np.matrix([self.X[n]**i for i in range(0, self.basisN+1)]).T))
      return phi
  
  def Fourier(self):
      phi1 = np.matrix([np.sin(2.0*np.pi*i*self.X[0]) for i in range(1, self.basisN+1)]).T
      phi2 = np.matrix([np.cos(2.0*np.pi*i*self.X[0]) for i in range(0, self.basisN+1)]).T
      phi = np.hstack((phi1,phi2))
      # the 0th order basis function is a constant. It's computed in phi2.
      for n in range(1, np.shape(self.X)[0]):
          phi1 = np.matrix([np.sin(2.0*np.pi*i*self.X[n]) for i in range(1, self.basisN+1)]).T
          phi2 = np.matrix([np.cos(2.0*np.pi*i*self.X[n]) for i in range(0, self.basisN+1)]).T
          phi = np.vstack((phi, np.hstack((phi1,phi2))))    
      return phi
  
  def Legendre(self): 
      phi = np.matrix([np.polynomial.legendre.Legendre.basis(i, domain = [0,1])(self.X[0]) for i in range(1, self.basisN+1)]).T
      for n in range(1, np.shape(self.X)[0]):
          phi = np.vstack((phi, np.matrix([np.polynomial.legendre.Legendre.basis(i, domain = [0,1])(self.X[n]) for i in range(1, self.basisN+1)]).T))
      return phi