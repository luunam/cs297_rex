import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class BaselineNet(nn.Module):
  def __init__(self, state_dim, output_dim, epochs=5000, debug=True):
    super(BaselineNet, self).__init__()

    hidden_dim = 256
    self.linear1 = nn.Linear(state_dim, hidden_dim)
    self.linear2 = nn.Linear(hidden_dim, hidden_dim)
    self.linear3 = nn.Linear(hidden_dim, output_dim)
    self.epochs = epochs
    self.debug = debug
    
  def forward(self, x):
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = torch.tanh(self.linear3(x))

    return x

  def feed(self, X, y):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
    cnt = 0

    for epoch in range(self.epochs):
      outputs = self.forward(X.float())
      labels = torch.max(y, 1)[1]
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      cnt+=1
      if self.debug and cnt % 100 == 0:
        correct = 0
        total = y.shape[0]
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum()
        accuracy = 100 * correct/total
        print("Iteration: {}. Loss: {}. Correct: {}. Accuracy: {}.".format(cnt, loss.item(), correct, accuracy))

    


class AE(nn.Module):
  def __init__(self, dim, epochs=5000):
    super(AE, self).__init__()
    self.enc1 = nn.Linear(dim, 128)
    self.enc2 = nn.Linear(128, 64)
    self.enc3 = nn.Linear(64, 32)
    self.enc4 = nn.Linear(32, 16)
    
    self.dec1 = nn.Linear(16, 32)
    self.dec2 = nn.Linear(32, 64)
    self.dec3 = nn.Linear(64, 128)
    self.dec4 = nn.Linear(128, dim)
    
    self.epochs = epochs
    
  def forward(self, x):
    x = self.encode(x)
    x = self.decode(x)
    
    return x
  
  def feed(self, X):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    for epoch in range(self.epochs):
      X = X.float()
      
      optimizer.zero_grad()
      outputs = self.forward(X)
      loss = criterion(outputs, X)
      
      loss.backward()
      optimizer.step()
      
      if (epoch % 50 == 0):
        print('Epoch: {}, loss: {}'.format(epoch, loss))
      
  def encode(self, x):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
    
    return x
  
  def decode(self, x):
    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    
    return x