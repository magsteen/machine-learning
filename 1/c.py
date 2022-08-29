from pickletools import optimize
from statistics import mode
from turtle import color
from xml.etree.ElementTree import tostring
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

from csv_loader import load_csv

tensor = load_csv("./c.csv")

x_train, y_train = tensor[:, :1], tensor[:, 1:]

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def f(self, x):
        z = x @ self.W + self.b
        return 20 * (1 / (1 + torch.exp(-z))) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(1_000_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10000 == 0: print("epoch: ", epoch)
    
# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('day')
plt.ylabel('head circumference')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x_train, model.f(x_train).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
