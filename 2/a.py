from curses import termname
from pickletools import optimize
from statistics import mode
from xml.etree.ElementTree import tostring
import torch
import numpy as np
import matplotlib.pyplot as plt

x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])

class CrossEntropyNOTModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def logits(self, x):
        return x @ self.W + self.b
        
    def f(self, x):
        return torch.nn.Sigmoid()(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = CrossEntropyNOTModel()
optimizer = torch.optim.SGD([model.W, model.b], 10.0)
for epoch in range(1_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10000 == 0: print("epoch: ", epoch)
    
#Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

print("Give 0.0 and expect 1.0 => %s" % model.f(x_train[0:1, :]))
print("Give 1.0 and expect 0.0 => %s" % model.f(x_train[1:2, :]))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('input')
plt.ylabel('output')
plt.plot(x_train, model.f(x_train).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.savefig("./2/a.png")
