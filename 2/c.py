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
optimizer = torch.optim.SGD([model.W_1, model.W_2, model.b_1, model.b_2], 1.0)
for epoch in range(100_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 10000 == 0: print("epoch: ", epoch)
    
#Print model variables and loss
print("W_1 = %s\nW_2 = %s\nb_1 = %s\nb_2 = %s\nloss = %s" 
      % (model.W_1, model.W_2, model.b_1, model.b_2, model.loss(x_train, y_train))
)

print("0.0 + 0.0 should give 0.0 => %s" % model.f(x_train[0:1, :]))
print("0.0 + 1.0 should give 1.0 => %s" % model.f(x_train[1:2, :]))
print("1.0 + 0.0 should give 1.0 => %s" % model.f(x_train[2:3, :]))
print("1.0 + 1.0 should give 0.0 => %s" % model.f(x_train[3:4, :]))

fig = plt.figure("Logistic regression: the logical XOR operator")

plot1 = fig.add_subplot(131, projection='3d')

x1_grid, x2_grid = torch.meshgrid(torch.linspace(-0.25, 1.25, 10), torch.linspace(-0.25, 1.25, 10))
h1_grid = torch.empty([10, 10])
h2_grid = torch.empty([10, 10])

for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        h = model.f_1(torch.stack([x1_grid[i, j], x2_grid[i, j]]))
        h1_grid[i, j] = h[0, 1]
        h2_grid[i, j] = h[0, 0]

plot1_h1 = plot1.plot_wireframe(
    x1_grid.detach().numpy(),
    x2_grid.detach().numpy(),
    h1_grid.detach().numpy(), 
    color="magenta")
plot1_h2 = plot1.plot_wireframe(
    x1_grid.detach().numpy(), 
    x2_grid.detach().numpy(), 
    h2_grid.detach().numpy(), 
    color="cyan")


plot1_info = fig.text(0.15, 0.9, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$h_1,h_2$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

plt.show()
