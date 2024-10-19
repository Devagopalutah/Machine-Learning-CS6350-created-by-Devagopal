import pandas as pd
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

class Node:
    def __init__(self, attribute=None, threshold=None, label=None, branches=None):
        self.attribute = attribute
        self.threshold = threshold
        self.label = label
        self.branches = branches or {}

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def information_gain(X, y, attribute):
    parent_entropy = entropy(y)
    
    weighted_child_entropy = 0
    for value in X[attribute].unique():
        child_y = y[X[attribute] == value]
        weight = len(child_y) / len(y)
        weighted_child_entropy += weight * entropy(child_y)
    
    return parent_entropy - weighted_child_entropy

def id3(X, y, attributes):
    if len(np.unique(y)) == 1:
        return Node(label=y.iloc[0])
    
    if len(attributes) == 0:
        return Node(label=Counter(y).most_common(1)[0][0])
    
    gains = [information_gain(X, y, attr) for attr in attributes]
    best_attribute = attributes[np.argmax(gains)]
    
    node = Node(attribute=best_attribute)
    
    for value in X[best_attribute].unique():
        X_subset = X[X[best_attribute] == value].drop(best_attribute, axis=1)
        y_subset = y[X[best_attribute] == value]
        
        if len(X_subset) == 0:
            node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = id3(X_subset, y_subset, new_attributes)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    
    value = instance.get(node.attribute)
    if value not in node.branches:
        return max(node.branches.values(), key=lambda x: x.label if x.label is not None else "").label
    
    return predict(node.branches[value], instance)

def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices]

def build_tree(X, y, attributes):
    X_sample, y_sample = bootstrap_sample(X, y)
    return id3(X_sample, y_sample, attributes)

def bagging(X, y, n_trees):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(build_tree, X, y, list(X.columns)) for _ in range(n_trees)]
        trees = [future.result() for future in as_completed(futures)]
    return trees

def predict_bagging(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    return Counter(predictions).most_common(1)[0][0]

def evaluate(model, X, y, is_bagging=False):
    if is_bagging:
        predictions = [predict_bagging(model, X.iloc[i]) for i in range(len(X))]
    else:
        predictions = [predict(model, X.iloc[i]) for i in range(len(X))]
    return np.mean(predictions == y)



base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, 'bank', 'train.csv')
data_txt = os.path.join(base_dir, 'bank', 'data-desc.txt')
test_file = os.path.join(base_dir, 'bank', 'test.csv')

train_data = pd.read_csv(train_file, header=None)
test_data = pd.read_csv(test_file, header=None)


print(f"Number of columns in train_data: {len(train_data.columns)}")
print(f"Number of columns in test_data: {len(test_data.columns)}")



with open(data_txt,'r') as f:
    attribute_names = [line.strip().split(':')[0] for line in f if ':' in line]

if len(attribute_names) != len(train_data.columns):
    print("Mismatch between number of attributes and number of columns.")
    print("Using default column names.")
    attribute_names = [f'attr_{i}' for i in range(len(train_data.columns) - 1)] + ['y']

train_data.columns = attribute_names
test_data.columns = attribute_names

def impute_unknown(train_df, test_df):
    for column in train_df.columns:
        if train_df[column].dtype == 'object':
            majority_value = train_df[train_df[column] != 'unknown'][column].mode().iloc[0]
            train_df[column] = train_df[column].replace('unknown', majority_value)
            test_df[column] = test_df[column].replace('unknown', majority_value)
    return train_df, test_df

X_train, X_test = impute_unknown(train_data.drop('y', axis=1), test_data.drop('y', axis=1))
y_train, y_test = train_data['y'], test_data['y']

numerical_columns = X_train.select_dtypes(include=[np.number]).columns
numerical_medians = X_train[numerical_columns].median()

for col in numerical_columns:
    threshold = numerical_medians[col]
    X_train[col] = (X_train[col] > threshold).astype(int)
    X_test[col] = (X_test[col] > threshold).astype(int)


n_trees_values = [1, 5, 10, 50, 100, 200, 300, 400, 500]
bagged_train_errors = []
bagged_test_errors = []

for n_trees in n_trees_values:
    trees = bagging(X_train, y_train, n_trees)
    
    train_accuracy = evaluate(trees, X_train, y_train, is_bagging=True)
    test_accuracy = evaluate(trees, X_test, y_test, is_bagging=True)
    
    bagged_train_errors.append(1 - train_accuracy)
    bagged_test_errors.append(1 - test_accuracy)
    
    print(f"Number of trees: {n_trees}, Train Error: {1-train_accuracy:.4f}, Test Error: {1-test_accuracy:.4f}")


single_tree = id3(X_train, y_train, list(X_train.columns))
single_train_error = 1 - evaluate(single_tree, X_train, y_train)
single_test_error = 1 - evaluate(single_tree, X_test, y_test)

print(f"Single Tree - Train Error: {single_train_error:.4f}, Test Error: {single_test_error:.4f}")


plt.figure(figsize=(10, 5))
plt.plot(n_trees_values, bagged_train_errors, label='Bagged Trees Training Error')
plt.plot(n_trees_values, bagged_test_errors, label='Bagged Trees Test Error')
plt.axhline(y=single_train_error, color='r', linestyle='--', label='Single Tree Training Error')
plt.axhline(y=single_test_error, color='g', linestyle='--', label='Single Tree Test Error')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.title('Bagged Trees vs Single Tree Error Rates')
plt.legend()
plt.xscale('log')
plt.show()


print(f"Final Bagged Trees Training Error: {bagged_train_errors[-1]:.4f}")
print(f"Final Bagged Trees Test Error: {bagged_test_errors[-1]:.4f}")
print(f"Single Tree Training Error: {single_train_error:.4f}")
print(f"Single Tree Test Error: {single_test_error:.4f}")