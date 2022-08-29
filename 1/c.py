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
        return 20 * torch.nn.Sigmoid()(x @ self.W + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.W, model.b], 0.000_000_1)
for epoch in range(1_000_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10000 == 0: print("epoch: ", epoch)
    
# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

steps = 2000
x = torch.linspace(torch.min(x_train), torch.max(x_train), steps).reshape(-1, 1)
y = model.f(x).detach()

print(x);print(y)

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('day')
plt.ylabel('head circumference')
plt.plot(x, y, label='$\\hat y = f(x) = xW+b$')
plt.savefig("c.png")
