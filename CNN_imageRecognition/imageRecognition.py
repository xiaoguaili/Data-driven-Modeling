#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:01:46 2018

@author: xiaoguai
"""

import numpy as np
#import cPickle, gzip
import gzip
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
from models_pytorch_CNN import ConvNet_CIFAR10
#from models_pytorch import ConvNet


if __name__ == "__main__": 
    
    def plot_random_sample(images, labels):
        X = images.transpose(0,2,3,1).astype("uint8")
        idx = np.random.choice(range(len(X)))
        plt.figure(figsize=(1,1))
        plt.imshow(X[idx:idx+1][0])
        print('This is a %d' % train_labels[idx])
        plt.show()
        del(X)
    
    # Load the dataset
    def load_batch(file):
        path = 'cifar-10-batches-py/'
        file = 'data_batch_1'
        with open(path+file, 'rb') as fo:
            u = pickle._Unpickler(fo)
            u.encoding = 'latin1'
            mydict = u.load()
        
        images = mydict['data']
        labels = mydict['labels']
        imagearray = np.array(images)   #   (10000, 3072)
        labelarray = np.array(labels)   #   (10000,)   
        return imagearray, labelarray
    
    # Training data
    train_images, train_labels = load_batch('data_batch_1')
    for i in range(2,6):
        file = 'data_batch_%d' % i
        images, labels = load_batch(file)
        train_images = np.vstack((train_images,images))
        train_labels = np.hstack((train_labels,labels))
        del(images, labels, file, i)
    N_train = train_images.shape[0]
    train_images = train_images.reshape([N_train, 3, 32, 32])
    
    # Test data
    test_images, test_labels = load_batch('test_batch')
    N_test = test_images.shape[0]
    test_images = test_images.reshape([N_test, 3, 32, 32])
       
    # Check a few samples to make sure the data was loaded correctly    
    # plot_random_sample(train_images, train_labels)

    # Define model
    model = ConvNet_CIFAR10(train_images, train_labels)

    # Train
    epochdata, lossdata = model.train()
    
    # Evaluate test performance
    model.test(test_images, test_labels)
    
    # Predict
    predicted_labels = np.argmax(model.predict(test_images),1)
    print(confusion_matrix(test_labels, predicted_labels))
    
    #plot loss as a function of the training iterations    
    plt.figure(1)
    plt.plot(epochdata[1:], lossdata[1:], 'b-', linewidth = 2)
    plt.axis('tight')
    plt.xlabel('$epoch$')
    plt.ylabel('$loss$')
    plt.savefig('loss', format='png')
    
    # Plot a random prediction
    idx = 3452
    plt.figure(1)
    img = test_images[idx,0,:,:]
    plt.imshow(img, cmap=plt.get_cmap('gray_r'))
    print('Correct label: %d, Predicted label: %d' % (test_labels[idx], predicted_labels[idx]))
    plt.show()