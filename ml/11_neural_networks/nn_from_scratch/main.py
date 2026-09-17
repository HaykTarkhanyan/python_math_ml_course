import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# fix seed for reproducibility
np.random.seed(509)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

INPUT_SIZE = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1

def he_initialization(size):
    return np.random.randn(size) * np.sqrt(2. / size)

def xaavier_initialization(size):
    return np.random.randn(size) * np.sqrt(1. / size)

def initialize_weights(input_size, hidden_size, output_size):
    weights = {
        'W1': he_initialization((input_size, hidden_size)),
        'b1': np.zeros((1, hidden_size)),
        'W2': he_initialization((hidden_size, output_size)),
        'b2': np.zeros((1, output_size))
    }
    return weights

def l2_loss(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

def l2_loss_derivative(Y_true, Y_pred):
    return 2 * (Y_pred - Y_true) / Y_true.size

def cross_entropy_loss(Y_true, Y_pred):
    epsilon = 1e-12
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    return -np.mean(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred))

def cross_entropy_loss_derivative(Y_true, Y_pred):
    epsilon = 1e-12
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    return (Y_pred - Y_true) / (Y_pred * (1 - Y_pred) * Y_true.size)

def forward_propagation(X, weights):
    Z1 = np.dot(X, weights['W1']) + weights['b1']
    A1 = relu(Z1)
    Z2 = np.dot(A1, weights['W2']) + weights['b2']
    A2 = sigmoid(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def backward_propagation(X, Y, weights, cache):
    m = X.shape[0]
    Z1, A1, Z2, A2 = cache

    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, weights['W2'].T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    return gradients

def update_weights(weights, gradients, learning_rate):
    weights['W1'] -= learning_rate * gradients['dW1']
    weights['b1'] -= learning_rate * gradients['db1']
    weights['W2'] -= learning_rate * gradients['dW2']
    weights['b2'] -= learning_rate * gradients['db2']
    return weights

def train(X, Y, input_size, hidden_size, output_size, epochs, learning_rate):
    weights = initialize_weights(input_size, hidden_size, output_size)
    loss_history = []

    for epoch in tqdm(range(epochs)):
        A2, cache = forward_propagation(X, weights)
        loss = l2_loss(Y, A2)
        loss_history.append(loss)

        gradients = backward_propagation(X, Y, weights, cache)
        weights = update_weights(weights, gradients, learning_rate)

        if epoch % 200 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return weights, loss_history

from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=1000, noise=0.1, random_state=509)

X_train, X_test = X[:800], X[800:]
Y_train, Y_test = Y[:800].reshape(-1, 1), Y[800:].reshape(-1, 1)

def predict(X, weights, threshold=0.5):
    A2, _ = forward_propagation(X, weights)
    return (A2 > threshold).astype(int)

def plot_data(X, Y):
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', marker='o')
    plt.show()
    


def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', marker='o')
    plt.show()
    
plot_data(X_train, Y_train)

LR = 0.01
EPOCHS = 10000


weights, loss_history = train(X_train, Y_train, 
                              INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 
                              EPOCHS, LR)   

print(f'Final Loss: {loss_history[-1]}')
