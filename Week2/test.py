#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 23:41:49 2018

@author: xiaoguai
"""
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":   
    mean = [0, 0]
    cov = [[1, 0], [0, 100]]  # diagonal covariance
    print(np.shape(mean))
    x= np.random.multivariate_normal(mean, cov, 5000)
    x.shape
    print(x.shape)
    #plt.plot(x, y, 'x')
    #plt.axis('equal')
    #plt.show()