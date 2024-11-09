import numpy as np
import pandas as pd
import os
from collections import Counter

# Set up data paths
base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, 'bank-note', 'train.csv')
data_txt = os.path.join(base_dir, 'bank-note', 'data-desc.txt')
test_file = os.path.join(base_dir, 'bank-note', 'test.csv')

def read_data_description(desc_path):
    
    with open(desc_path, 'r') as f:
        data_desc = f.read()
    print("Data Description:")
    print(data_desc)
    print("\nFeatures will be loaded according to this description.")

def load_and_preprocess_data():

    read_data_description(data_txt)

    
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1

    print(f"\nDataset Statistics:")
    print(f"Training set: {len(X_train)} examples")
    print(f"Test set: {len(X_test)} examples")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, y_train, X_test, y_test

class Perceptron:
    """Base class for all Perceptron variants"""
    def __init__(self, max_epochs=10):
        self.max_epochs = max_epochs

    def predict(self, X):
        raise NotImplementedError

class StandardPerceptron(Perceptron):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_epochs):
            for i in range(n_samples):
                prediction = np.sign(np.dot(X[i], self.weights) + self.bias)
                if prediction != y[i]:
                    self.weights += y[i] * X[i]
                    self.bias += y[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

class VotedPerceptron(Perceptron):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights_list = []
        self.bias_list = []
        self.counts_list = []

        current_weights = np.zeros(n_features)
        current_bias = 0
        current_count = 0

        for _ in range(self.max_epochs):
            for i in range(n_samples):
                prediction = np.sign(np.dot(X[i], current_weights) + current_bias)

                if prediction == y[i]:
                    current_count += 1
                else:
                    if current_count > 0:
                        self.weights_list.append(current_weights.copy())
                        self.bias_list.append(current_bias)
                        self.counts_list.append(current_count)

                    current_weights = current_weights + y[i] * X[i]
                    current_bias = current_bias + y[i]
                    current_count = 1

        if current_count > 0:
            self.weights_list.append(current_weights.copy())
            self.bias_list.append(current_bias)
            self.counts_list.append(current_count)

    def predict(self, X):
        predictions = np.zeros(len(X))
        for i in range(len(X)):
            vote = 0
            for weights, bias, count in zip(self.weights_list, self.bias_list, self.counts_list):
                vote += count * np.sign(np.dot(X[i], weights) + bias)
            predictions[i] = np.sign(vote)
        return predictions

class AveragePerceptron(Perceptron):
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.avg_weights = np.zeros(n_features)
        self.avg_bias = 0

        update_count = 1

        for _ in range(self.max_epochs):
            for i in range(n_samples):
                prediction = np.sign(np.dot(X[i], self.weights) + self.bias)

                if prediction != y[i]:
                    self.weights += y[i] * X[i]
                    self.bias += y[i]

                self.avg_weights += self.weights
                self.avg_bias += self.bias
                update_count += 1

        self.avg_weights = self.avg_weights / update_count
        self.avg_bias = self.avg_bias / update_count

    def predict(self, X):
        return np.sign(np.dot(X, self.avg_weights) + self.avg_bias)

def calculate_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

if __name__ == "__main__":
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    
    perceptrons = {
        "Standard": StandardPerceptron(max_epochs=10),
        "Voted": VotedPerceptron(max_epochs=10),
        "Average": AveragePerceptron(max_epochs=10)
    }

    for name, perceptron in perceptrons.items():
        print(f"\nTesting {name} Perceptron:")
        perceptron.fit(X_train, y_train)
        test_predictions = perceptron.predict(X_test)
        error = calculate_error(y_test, test_predictions)
        print(f"Average prediction error on test set: {error:.6f}")

        if name == "Standard":
            print("\nLearned weight vector:")
            for i, w in enumerate(perceptron.weights):
                print(f"w{i+1}: {w:.6f}")
            print(f"bias: {perceptron.bias:.6f}")
        elif name == "Voted":
            print("\nWeight vectors and their counts:")
            for i, (weights, bias, count) in enumerate(zip(
                perceptron.weights_list,
                perceptron.bias_list,
                perceptron.counts_list)):
                print(f"\nWeight vector {i+1}:")
                for j, w in enumerate(weights):
                    print(f"w{j+1}: {w:.6f}")
                print(f"bias: {bias:.6f}")
                print(f"count: {count}")
            print(f"\nNumber of weight vectors: {len(perceptron.weights_list)}")
        else:  # Average
            print("\nLearned average weight vector:")
            for i, w in enumerate(perceptron.avg_weights):
                print(f"w{i+1}: {w:.6f}")
            print(f"bias: {perceptron.avg_bias:.6f}")