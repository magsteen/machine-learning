from curses import termname
from pickletools import optimize
from statistics import mode
from xml.etree.ElementTree import tostring
import torch
import numpy as np
import matplotlib.pyplot as plt
import random as rand

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

class CrossEntropyXORModel:
    def __init__(self):
        # Model variables
        self.W_1 = torch.tensor([
            [rand.uniform(0.0, 1.0), rand.uniform(-1.0, 0.0)],
            [rand.uniform(0.0, 1.0), rand.uniform(-1.0, 0.0)]
        ], requires_grad=True)
        self.W_2 = torch.tensor([
            [rand.uniform(0.0, 1.0)], [rand.uniform(0.0, 1.0)],
        ], requires_grad=True)
        self.b_1 = torch.tensor([
            [rand.uniform(-1.0, 0.0), rand.uniform(0.0, 1.0)]
        ], requires_grad=True)
        self.b_2 = torch.tensor([
            [rand.uniform(-1.0, 0.0)]
        ], requires_grad=True)
        
    def logits(self, x):
        return x @ self.W + self.b
        
    def f_1(self, x):
        return torch.nn.Sigmoid()(x @ self.W_1 + self.b_1)
    
    def f_2(self, h):
        return torch.nn.Sigmoid()(h @ self.W_2 + self.b_2)
    
    def f(self, x):
        return self.f_2(self.f_1(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


model = CrossEntropyXORModel()
optimizer = torch.optim.SGD([model.W_1, model.W_2, model.b_1, model.b_2], 0.1)
for epoch in range(100_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10000 == 0: print("epoch: ", epoch)
    
#Print model variables and loss
print("W_1 = %s, W_2 = %s, b_1 = %s, b_2 = %s, loss = %s" 
      % (model.W_1, model.W_2, model.b_1, model.b_2, model.loss(x_train, y_train))
)

print("0.0 + 0.0 should give 0.0 => %s" % model.f(x_train[0:1, :]))
print("0.0 + 1.0 should give 1.0 => %s" % model.f(x_train[1:2, :]))
print("1.0 + 0.0 should give 1.0 => %s" % model.f(x_train[2:3, :]))
print("1.0 + 1.0 should give 0.0 => %s" % model.f(x_train[3:4, :]))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('input')
plt.ylabel('output')
plt.plot(x_train, model.f(x_train).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.savefig("./2/c.png")
