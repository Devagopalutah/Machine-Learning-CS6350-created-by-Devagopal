import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

class BanknoteClassifier:
    def __init__(self, hidden_width: int, random_init: bool = True):
        self.layer_sizes = [4, hidden_width, hidden_width, 1]
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            if random_init:
                w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1])
                b = np.zeros((1, self.layer_sizes[i+1]))
            else:
                w = np.zeros((self.layer_sizes[i], self.layer_sizes[i+1]))
                b = np.zeros((1, self.layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        sx = self.sigmoid(x)
        return sx * (1 - sx)

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        z_values = []

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        return activations, z_values

    def backward(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations, z_values = self.forward(x)

        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # Output layer error
        delta = (activations[-1] - y) * self.sigmoid_derivative(z_values[-1])

        # Backward pass
        for l in range(len(self.weights) - 1, -1, -1):
            weight_gradients[l] = np.dot(activations[l].T, delta)
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True)

            if l > 0:
                delta = np.dot(delta, self.weights[l].T) * self.sigmoid_derivative(z_values[l-1])

        return weight_gradients, bias_gradients

    def train(self, X_train: np.ndarray, y_train: np.ndarray, learning_rate_init: float,
              decay: float, epochs: int) -> Tuple[List[float], List[float]]:
        n_samples = X_train.shape[0]
        losses = []
        errors = []
        global_step = 0

        for epoch in range(epochs):
           
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            for i in range(n_samples):
               
                lr = learning_rate_init / (1 + decay * global_step)
                global_step += 1

                x = X_shuffled[i:i+1]
                y = y_shuffled[i:i+1]

                weight_gradients, bias_gradients = self.backward(x, y)
                for j in range(len(self.weights)):
                    self.weights[j] -= lr * weight_gradients[j]
                    self.biases[j] -= lr * bias_gradients[j]

                
                pred = self.forward(x)[0][-1]
                epoch_loss += float(np.mean((pred - y)**2))

            avg_loss = epoch_loss / n_samples
            current_error = self.evaluate(X_train, y_train)
            losses.append(float(avg_loss))
            errors.append(float(current_error))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {float(avg_loss):.4f}, Error: {float(current_error):.4f}")

        return losses, errors

    def predict(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return (activations[-1] >= 0.5).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return float(np.mean(predictions != y))

def verify_gradients():
    print("Gradient Verification:")
    X = np.array([[1.0, 1.0]])
    y = np.array([[1.0]])

    model = BanknoteClassifier(hidden_width=2)
    model.weights = [
        np.array([[0.1, 0.2], [0.3, 0.4]]),
        np.array([[0.5], [0.6]])
    ]
    model.biases = [
        np.array([[0.1, 0.1]]),
        np.array([[0.1]])
    ]

    weight_grads, bias_grads = model.backward(X, y)
    print("\nComputed gradients:")
    for i, w_grad in enumerate(weight_grads):
        print(f"Layer {i+1} weights:\n{w_grad}")
    for i, b_grad in enumerate(bias_grads):
        print(f"Layer {i+1} biases:\n{b_grad}")

def run_experiments(X_train, y_train, X_test, y_test):
    widths = [5, 10, 25, 50, 100]
    learning_rate = 0.1
    decay = 0.001
    epochs = 100

    results = {'random_init': [], 'zero_init': []}

    
    print("\nRunning experiments with random initialization:")
    for width in widths:
        print(f"\nWidth: {width}")
        model = BanknoteClassifier(width, random_init=True)
        losses, errors = model.train(X_train, y_train, learning_rate, decay, epochs)

        train_error = model.evaluate(X_train, y_train)
        test_error = model.evaluate(X_test, y_test)

        results['random_init'].append({
            'width': width,
            'train_error': train_error,
            'test_error': test_error,
            'losses': losses
        })

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title(f'Training Loss (width={width})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1, 2, 2)
        plt.plot(errors)
        plt.title(f'Training Error (width={width})')
        plt.xlabel('Epoch')
        plt.ylabel('Error Rate')
        plt.show()

    
    print("\nRunning experiments with zero initialization:")
    for width in widths:
        print(f"\nWidth: {width}")
        model = BanknoteClassifier(width, random_init=False)
        losses, errors = model.train(X_train, y_train, learning_rate, decay, epochs)

        train_error = model.evaluate(X_train, y_train)
        test_error = model.evaluate(X_test, y_test)

        results['zero_init'].append({
            'width': width,
            'train_error': train_error,
            'test_error': test_error,
            'losses': losses
        })

    return results

if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'bank', 'train.csv')
    test_file = os.path.join(base_dir, 'bank', 'test.csv')
    data_txt = os.path.join(base_dir, 'bank', 'data-desc.txt')

    
    column_names = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    train_df = pd.read_csv(train_file, names=column_names)
    test_df = pd.read_csv(test_file, names=column_names)

    
    X_train = train_df.iloc[:, :4].values
    y_train = train_df.iloc[:, 4].values.reshape(-1, 1)
    X_test = test_df.iloc[:, :4].values
    y_test = test_df.iloc[:, 4].values.reshape(-1, 1)

    
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    
    verify_gradients()

    results = run_experiments(X_train, y_train, X_test, y_test)

    
    print("\nSummary of Results:")
    print("\nRandom Initialization:")
    for result in results['random_init']:
        print(f"Width {result['width']}: Train Error = {result['train_error']:.4f}, "
              f"Test Error = {result['test_error']:.4f}")

    print("\nZero Initialization:")
    for result in results['zero_init']:
        print(f"Width {result['width']}: Train Error = {result['train_error']:.4f}, "
              f"Test Error = {result['test_error']:.4f}")