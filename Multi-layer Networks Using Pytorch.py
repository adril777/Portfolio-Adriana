import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
import numpy as np

# variation of the hyperparameters which are:

# batch sizes (batch_size): 256, 128, 64
# number of epochs (num_epochs): 10, 20, 30
# learning rate (lr): 0.1, 0.01, 0.2

def generate_synthetic_data (w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter (batch_size, features, labels):
    num_examples = features.shape[0]
    indices = np.random.permutation (num_examples)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - y)**2 / 2

def stochastic_gradient_descent (params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#--------------------EJECUTION----------------------------------

true_w = torch.tensor([2, -3.4, 5.6])
true_b = 4.2
features, labels = generate_synthetic_data(true_w, true_b, 1000)

# Split the data into training and test set
train_features, test_features = features[:800], features[800:]
train_labels, test_labels = labels[:800], labels[800:]

# Hyperparameters to test
batch_sizes = [256, 128, 64]
num_epochs_list = [10, 20, 30]
learning_rates = [0.1, 0.01, 0.2]

results = []

for batch_size in batch_sizes:
    for num_epochs in num_epochs_list:
        for lr in learning_rates:
            # Initialize parameters
            w = torch.normal(0, 0.01, size=(3, 1), requires_grad=True)
            b = torch.zeros(1, requires_grad=True)

            loss_values = []
            for epoch in range(num_epochs):
                for X, y in data_iter(batch_size, train_features, train_labels):
                    l = squared_loss(linreg(X, w, b), y)
                    l.sum().backward()
                    stochastic_gradient_descent([w, b], lr, batch_size)

                with torch.no_grad():
                    train_l = squared_loss(linreg(train_features, w, b), train_labels)
                    test_l = squared_loss(linreg(test_features, w, b), test_labels)
                    loss_values.append(float(train_l.mean()))

            with torch.no_grad():
                train_acc = 1 - torch.mean(torch.abs((linreg(train_features, w, b) - train_labels) / train_labels))
                test_acc = 1 - torch.mean(torch.abs((linreg(test_features, w, b) - test_labels) / test_labels))

            results.append((batch_size, num_epochs, lr, float(train_l.mean()), float(test_l.mean()), float(train_acc), float(test_acc)))

# Show results
for result in results:
    print(f'batch_size: {result[0]}, num_epochs: {result[1]}, lr: {result[2]}, '
          f'train_loss: {result[3]:.6f}, test_loss: {result[4]:.6f}, train_acc: {result[5]:.6f}, test_acc: {result[6]:.6f}')

