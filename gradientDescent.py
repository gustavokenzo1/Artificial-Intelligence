from pickletools import optimize
from tkinter import X
from typing import final
from sklearn.model_selection import learning_curve
import torch
import time
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Get random data
x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

# Convert numpy data to pytorch
x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

plt.plot(x_numpy, y_numpy, 'ro')

# Define model
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# Define cost function and optimizer
learning_rate = 0.01
criterion = nn.MSELoss() # Mean Squared Error
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # Stochastic Gradient Descent

# Training Loop
num_epochs = 1000
cost_counter = []

for epoch in range(num_epochs):
    y_hat = model(x)
    loss = criterion(y_hat, y) # (y^ - y)²
    cost_counter.append(loss)

    # Backward pass (reduce derivative)
    loss.backward()

    # Update weights
    optimizer.step()

    if (epoch + 1)%10 == 0:
        print('Epoch:', epoch)
        print('Cost: {:.20f}'.format(loss.item()))
        print('Coeficients:')
        print('m: {:.20f}'.format(model.weight.data.detach().item()))
        print('m (gradient): {:.20f}'.format(model.weight.grad.detach().item()))
        print('b: {:.20f}'.format(model.bias.data.detach().item()))
        print('b (gradiente): {:.20f}'.format(model.bias.grad.detach().item()))

        final_prediction = y_hat.detach().numpy()
        plt.plot(x_numpy, y_numpy, 'ro')
        plt.plot(x_numpy, final_prediction, 'b')
        plt.show()
    
    optimizer.zero_grad()

print('Cost Function Graph')
plt.plot(cost_counter, 'b')
plt.show()

