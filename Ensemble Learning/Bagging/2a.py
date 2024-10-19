import pandas as pd
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt

class Node:
    def __init__(self, attribute=None, threshold=None, label=None, branches=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.branches = branches or {}

def weighted_entropy(y, weights):
    classes = np.unique(y)
    probabilities = np.array([np.sum(weights[y == c]) for c in classes])
    probabilities /= np.sum(probabilities)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def weighted_information_gain(X, y, attribute, weights):
    parent_entropy = weighted_entropy(y, weights)
    
    weighted_child_entropy = 0
    for value in X[attribute].unique():
        mask = X[attribute] == value
        child_y = y[mask]
        child_weights = weights[mask]
        weight = np.sum(child_weights) / np.sum(weights)
        weighted_child_entropy += weight * weighted_entropy(child_y, child_weights)
    
    return parent_entropy - weighted_child_entropy

def decision_stump(X, y, weights):
    best_gain = -float('inf')
    best_attribute = None
    
    for attribute in X.columns:
        gain = weighted_information_gain(X, y, attribute, weights)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute
    
    node = Node(attribute=best_attribute)
    
    for value in X[best_attribute].unique():
        mask = X[best_attribute] == value
        subset_y = y[mask]
        subset_weights = weights[mask]
        
        if len(subset_y) == 0:
            node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
        else:
            classes, _ = np.unique(subset_y, return_counts=True)
            weighted_counts = np.array([np.sum(subset_weights[subset_y == c]) for c in classes])
            majority_class = classes[np.argmax(weighted_counts)]
            node.branches[value] = Node(label=majority_class)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    
    value = instance.get(node.attribute)
    if value not in node.branches:
        return max(node.branches.values(), key=lambda x: x.label if x.label is not None else "").label
    
    return predict(node.branches[value], instance)

def adaboost(X, y, T):
    n_samples, n_features = X.shape
    weights = np.ones(n_samples) / n_samples
    
    classifiers = []
    alphas = []
    
    for _ in range(T):
        stump = decision_stump(X, y, weights)
        predictions = np.array([predict(stump, X.iloc[i]) for i in range(n_samples)])
        
        error = np.sum(weights * (predictions != y)) / np.sum(weights)
        alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))
        
        weights *= np.exp(-alpha * y * predictions)
        weights /= np.sum(weights)
        
        classifiers.append(stump)
        alphas.append(alpha)
    
    return classifiers, alphas

def predict_adaboost(classifiers, alphas, instance):
    predictions = np.array([predict(clf, instance) for clf in classifiers])
    return np.sign(np.sum(alphas * predictions))

def evaluate_adaboost(classifiers, alphas, X, y):
    predictions = np.array([predict_adaboost(classifiers, alphas, X.iloc[i]) for i in range(len(X))])
    return np.mean(predictions == y)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, 'bank', 'train.csv')
data_txt = os.path.join(base_dir, 'bank', 'data-desc.txt')
test_file = os.path.join(base_dir, 'bank', 'test.csv')

train_data = pd.read_csv(train_file, header=None)
test_data = pd.read_csv(test_file, header=None)




print(f"Number of columns in train_data: {len(train_data.columns)}")
print(f"Number of columns in test_data: {len(test_data.columns)}")

with open(data_desc,'r') as f:
    attribute_names = [line.strip().split(':')[0] for line in f if ':' in line]

print(f"Number of attribute names read: {len(attribute_names)}")
print("Attribute names:", attribute_names)

if len(attribute_names) != len(train_data.columns):
    print("Mismatch between number of attributes and number of columns.")
    print("Using default column names.")
    attribute_names = [f'attr_{i}' for i in range(len(train_data.columns) - 1)] + ['y']

train_data.columns = attribute_names
test_data.columns = attribute_names

print("Assigned column names:", train_data.columns.tolist())

def impute_unknown(train_df, test_df):
    for column in train_df.columns:
        if train_df[column].dtype == 'object':  # Check if column is categorical
            majority_value = train_df[train_df[column] != 'unknown'][column].mode().iloc[0]
            
            train_df[column] = train_df[column].replace('unknown', majority_value)
            test_df[column] = test_df[column].replace('unknown', majority_value)
    
    return train_df, test_df

X_train, X_test = impute_unknown(train_data.drop('y', axis=1), test_data.drop('y', axis=1))
y_train, y_test = train_data['y'], test_data['y']

numerical_columns = X_train.select_dtypes(include=[np.number]).columns
print("Numerical columns:", numerical_columns.tolist())

numerical_medians = X_train[numerical_columns].median()

for col in numerical_columns:
    threshold = numerical_medians[col]
    X_train[col] = (X_train[col] > threshold).astype(int)
    X_test[col] = (X_test[col] > threshold).astype(int)

print("\nUnique values in each column after preprocessing:")
for col in X_train.columns:
    print(f"{col}: {X_train[col].unique()}")

print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)


y_train = y_train.map({'yes': 1, 'no': -1})
y_test = y_test.map({'yes': 1, 'no': -1})


T_values = range(1, 501, 10)  # 1 to 500, step 10 for faster execution
train_errors = []
test_errors = []

for T in T_values:
    classifiers, alphas = adaboost(X_train, y_train, T)
    
    train_accuracy = evaluate_adaboost(classifiers, alphas, X_train, y_train)
    test_accuracy = evaluate_adaboost(classifiers, alphas, X_test, y_test)
    
    train_errors.append(1 - train_accuracy)
    test_errors.append(1 - test_accuracy)
    
    print(f"Iterations: {T}, Train Error: {1-train_accuracy:.4f}, Test Error: {1-test_accuracy:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(T_values, train_errors, label='Training Error')
plt.plot(T_values, test_errors, label='Test Error')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error Rate')
plt.title('AdaBoost Error Rates')
plt.legend()
plt.show()


print(f"Final AdaBoost Training Error: {train_errors[-1]:.4f}")
print(f"Final AdaBoost Test Error: {test_errors[-1]:.4f}")