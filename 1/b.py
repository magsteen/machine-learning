from pickletools import optimize
from statistics import mode
from turtle import color
from xml.etree.ElementTree import tostring
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv

from csv_loader import load_csv

tensor = load_csv("./b.csv")

length_data, weight_data, day_data = tensor[:, 1:2], tensor[:, 2:], tensor[:, :1]

x_train, y_train = tensor[:, 1:], tensor[:, :1]

class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0, 0.0], [0.0, 0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def f(self, x):
        return x @ self.W + self.b

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
ax = plt.axes(projection='3d')
ax.scatter3D(length_data, weight_data, day_data)

a = np.linspace(torch.min(length_data), torch.max(length_data), 2)
b = np.linspace(torch.min(weight_data), torch.max(weight_data), 2)
x, y = np.meshgrid(a, b)
z = torch.tensor([
    [torch.min(length_data), torch.min(weight_data)], 
    [torch.max(length_data), torch.max(weight_data)]
])

ax.plot_surface(x, y, model.f(z).detach(), color="orange")
ax.set_xlabel('length')
ax.set_ylabel('weight')
ax.set_zlabel('day')

plt.show()
