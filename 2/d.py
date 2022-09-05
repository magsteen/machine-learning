from curses import termname
from pickletools import optimize
from statistics import mode
from unicodedata import decimal
from xml.etree.ElementTree import tostring
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import random as rand

# Load observations from the mnist dataset. The observations are divided into a training set and a test set

mnist_train = torchvision.datasets.MNIST('./2/data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./2/data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
x_train /= 255
x_test /= 255

class SoftMaxNumbers:
    def __init__(self):
        # Model variables
        self.W = torch.zeros([784, 10], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        
    def logits(self, x):
        return x @ self.W + self.b
        
    def f(self, x):
        return torch.nn.Softmax()(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = SoftMaxNumbers()
optimizer = torch.optim.SGD([model.W, model.b], 1.0)
for epoch in tqdm(range(10000)):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

#Print model variables and loss
print("W = %s\nb = %s\ntrain-loss = %s\ntest-loss = %s" % 
    (model.W, model.b, model.loss(x_train, y_train), model.loss(x_test, y_test))
)

print(model.accuracy(x_test, y_test))

for i in range(0,10):
    file_path = "./2/d_pics/w_%d.png" % (i)
    plt.imsave(file_path, model.W[:, i:i+1].reshape(28, 28).detach().numpy())
