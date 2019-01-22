#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 20:02:20 2018

@author: xiaoguai
"""

import torch
import torch.utils.data
from torch.autograd import Variable, grad
import timeit
import numpy as np

# Define CNN architecture and forward pass
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)) 
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)) 
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(20, 20, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2))
        self.fc = torch.nn.Linear(4*4*20, 10)
        
    def forward_pass(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
 #       out=out.view(-1, 8*8*32)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class ConvNet_CIFAR10:
    # Initialize the class
    def __init__(self, X, Y):  
        
        # Check if there is a GPU available
        if torch.cuda.is_available() == True:
            self.dtype_double = torch.cuda.FloatTensor
            self.dtype_int = torch.cuda.LongTensor
        else:
            self.dtype_double = torch.FloatTensor
            self.dtype_int = torch.LongTensor
        
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        self.train_data = torch.utils.data.TensorDataset(X, Y)
        
        # Define architecture and initialize
        self.net = CNN()
        
        # Define the loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        
           
    # Trains the model by minimizing the Cross Entropy loss
    def train(self, num_epochs = 2, batch_size = 128):
        
        # Create a PyTorch data loader object
        self.trainloader = torch.utils.data.DataLoader(self.train_data, 
                                                  batch_size=batch_size, 
                                                  shuffle=True)
       
        start_time = timeit.default_timer()
        lossdata = np.array([0.0])
        epochdata = np.array([0.0])
        for epoch in range(num_epochs):
            for it, (images, labels) in enumerate(self.trainloader):
                images = Variable(images)
                labels = Variable(labels)
        
                # Reset gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.net.forward_pass(images)
                
                # Compute loss
                loss = self.loss_fn(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                self.optimizer.step()
        
                if (it+1) % 100 == 0:
                    elapsed = timeit.default_timer() - start_time
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Time: %2fs' 
                           %(epoch+1, num_epochs, it+1, len(self.train_data)//batch_size, loss.data[0], elapsed))
                    start_time = timeit.default_timer()
                    
                    # collect loss data for a given epoch
                    lossdata = np.vstack((lossdata, np.array([loss.data[0]]))) 
                    epochdata = np.vstack((epochdata, np.array([epoch+1+(it+1)/(len(self.train_data)//batch_size)])))
        return epochdata, lossdata
                    
    def test(self, X, Y):
        # Define PyTorch dataset
        X = torch.from_numpy(X).type(self.dtype_double) # num_images x num_pixels_x x num_pixels_y
        Y = torch.from_numpy(Y).type(self.dtype_int) # num_images x 1
        test_data = torch.utils.data.TensorDataset(X, Y)
       
        # Create a PyTorch data loader object
        test_loader = torch.utils.data.DataLoader(test_data, 
                                                  batch_size=128, 
                                                  shuffle=True)
        
        # Test prediction accuracy
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = Variable(images)
            outputs = self.net.forward_pass(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Test Accuracy of the model on the %d test images: %d %%' % (len(test_data), 100 * correct / total))
        
    
    # Evaluates predictions at test points    
    def predict(self, X_star):
        X_star = torch.from_numpy(X_star).type(self.dtype_double) 
        X_star = Variable(X_star, requires_grad=False)
        y_star = self.net.forward_pass(X_star)
        y_star = y_star.data.numpy()
        return y_star