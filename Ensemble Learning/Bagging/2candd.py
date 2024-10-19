import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time
import os 

class Node:
    def __init__(self, attribute=None, label=None, branches=None):
        self.attribute = attribute
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

def id3(X, y, attributes, max_depth=5):
    if len(np.unique(y)) == 1 or max_depth == 0 or len(attributes) == 0:
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
            node.branches[value] = id3(X_subset, y_subset, new_attributes, max_depth-1)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    
    value = instance.get(node.attribute)
    if value not in node.branches:
        return most_common_label(node)
    
    return predict(node.branches[value], instance)

def most_common_label(node):
    if node.label is not None:
        return node.label
    labels = []
    for branch in node.branches.values():
        label = most_common_label(branch)
        if label is not None:
            labels.append(label)
    return Counter(labels).most_common(1)[0][0] if labels else None

def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices]

def build_tree(args):
    X, y, attributes, max_depth = args
    X_sample, y_sample = bootstrap_sample(X, y)
    return id3(X_sample, y_sample, attributes, max_depth)

def bagging(X, y, n_trees, max_depth):
    attributes = list(X.columns)
    with ProcessPoolExecutor() as executor:
        trees = list(executor.map(build_tree, [(X, y, attributes, max_depth)] * n_trees))
    return trees

def predict_bagging(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    return np.mean(predictions)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, 'bank', 'train.csv')
data_txt = os.path.join(base_dir, 'bank', 'data-desc.txt')
test_file = os.path.join(base_dir, 'bank', 'test.csv')

train_data = pd.read_csv(train_file, header=None)
test_data = pd.read_csv(test_file, header=None)

print(f"Number of columns in train_data: {len(train_data.columns)}")
print(f"Number of columns in test_data: {len(test_data.columns)}")


with open(data_txt, 'r') as f:
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


y_train = y_train.map({'yes': 1, 'no': 0})
y_test = y_test.map({'yes': 1, 'no': 0})

def run_experiment(X_train, y_train, X_test, y_test):
    n_repeats = 100  
    n_samples = 1000
    n_trees = 500  
    
    
    single_tree_predictions = []
    bagged_tree_predictions = []
    
    for _ in range(n_repeats):
        sample_indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train.iloc[sample_indices]
        y_sample = y_train.iloc[sample_indices]
        
        single_tree = id3(X_sample, y_sample, list(X_sample.columns))
        bagged_trees = bagging(X_sample, y_sample, n_trees)
        
        single_tree_pred = [predict(single_tree, X_test.iloc[i]) for i in range(len(X_test))]
        bagged_tree_pred = [predict_bagging(bagged_trees, X_test.iloc[i]) for i in range(len(X_test))]
        
        single_tree_predictions.append(single_tree_pred)
        bagged_tree_predictions.append(bagged_tree_pred)
    
    return np.array(single_tree_predictions), np.array(bagged_tree_predictions)

start_time = time.time()
single_tree_predictions, bagged_tree_predictions = run_experiment(X_train, y_train, X_test, y_test)
end_time = time.time()
print(f"Experiment runtime: {end_time - start_time:.2f} seconds")

def compute_bias_variance(predictions, y_true):
    biases = []
    variances = []
    for i in range(len(y_true)):
        mean_prediction = np.mean(predictions[:, i])
        bias = (mean_prediction - y_true[i]) ** 2
        variance = np.var(predictions[:, i])
        biases.append(bias)
        variances.append(variance)
    
    return np.mean(biases), np.mean(variances)

single_tree_bias, single_tree_variance = compute_bias_variance(single_tree_predictions, y_test)
bagged_tree_bias, bagged_tree_variance = compute_bias_variance(bagged_tree_predictions, y_test)

print("Single Tree:")
print(f"Bias: {single_tree_bias:.4f}")
print(f"Variance: {single_tree_variance:.4f}")
print(f"General Squared Error: {single_tree_bias + single_tree_variance:.4f}")

print("\nBagged Trees:")
print(f"Bias: {bagged_tree_bias:.4f}")
print(f"Variance: {bagged_tree_variance:.4f}")
print(f"General Squared Error: {bagged_tree_bias + bagged_tree_variance:.4f}")


labels = ['Single Tree', 'Bagged Trees']
bias = [single_tree_bias, bagged_tree_bias]
variance = [single_tree_variance, bagged_tree_variance]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, bias, width, label='Bias')
rects2 = ax.bar(x + width/2, variance, width, label='Variance')

ax.set_ylabel('Error')
ax.set_title('Bias-Variance Decomposition')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.tight_layout()
plt.show()


####Part D

def random_forest_tree(X, y, attributes, feature_subset_size):
    if len(np.unique(y)) == 1:
        return Node(label=y.iloc[0])
    
    if len(attributes) == 0:
        return Node(label=Counter(y).most_common(1)[0][0])
    
    feature_subset = np.random.choice(attributes, min(feature_subset_size, len(attributes)), replace=False)
    gains = [information_gain(X, y, attr) for attr in feature_subset]
    best_attribute = feature_subset[np.argmax(gains)]
    
    node = Node(attribute=best_attribute)
    
    for value in X[best_attribute].unique():
        X_subset = X[X[best_attribute] == value].drop(best_attribute, axis=1)
        y_subset = y[X[best_attribute] == value]
        
        if len(X_subset) == 0:
            node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = random_forest_tree(X_subset, y_subset, new_attributes, feature_subset_size)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    
    value = instance[node.attribute]
    if value not in node.branches:
        return max(node.branches.values(), key=lambda x: x.label if x.label is not None else float('-inf')).label
    
    return predict(node.branches[value], instance)

def bootstrap_sample(X, y):
    n_samples = len(X)
    indices = np.random.choice(n_samples, n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices]

def build_rf_tree(args):
    X, y, attributes, feature_subset_size = args
    X_sample, y_sample = bootstrap_sample(X, y)
    return random_forest_tree(X_sample, y_sample, attributes, feature_subset_size)

def random_forest(X, y, n_trees, feature_subset_size):
    attributes = list(X.columns)
    with ProcessPoolExecutor() as executor:
        trees = list(executor.map(build_rf_tree, [(X, y, attributes, feature_subset_size)] * n_trees))
    return trees

def predict_forest(trees, instance):
    predictions = [predict(tree, instance) for tree in trees]
    return Counter(predictions).most_common(1)[0][0]



def evaluate_forest(X_train, y_train, X_test, y_test, n_estimators, feature_subset_size):
    train_errors = []
    test_errors = []
    
    for i in range(1, n_estimators + 1):
        forest = random_forest(X_train, y_train, i, feature_subset_size)
        
        train_preds = [predict_forest(forest, X_train.iloc[j]) for j in range(len(X_train))]
        test_preds = [predict_forest(forest, X_test.iloc[j]) for j in range(len(X_test))]
        
        train_error = 1 - np.mean(train_preds == y_train)
        test_error = 1 - np.mean(test_preds == y_test)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        if i % 50 == 0:
            print(f"Trees: {i}, Feature subset size: {feature_subset_size}, "
                  f"Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")
    
    return train_errors, test_errors


n_estimators = 500
feature_subset_sizes = [2, 4, 6]
results = {}

for feature_subset_size in feature_subset_sizes:
    start_time = time.time()
    train_errors, test_errors = evaluate_forest(X_train, y_train, X_test, y_test, n_estimators, feature_subset_size)
    end_time = time.time()
    print(f"Time for feature subset size {feature_subset_size}: {end_time - start_time:.2f} seconds")
    results[feature_subset_size] = (train_errors, test_errors)


plt.figure(figsize=(15, 10))
for i, feature_subset_size in enumerate(feature_subset_sizes):
    plt.subplot(2, 2, i+1)
    train_errors, test_errors = results[feature_subset_size]
    plt.plot(range(1, n_estimators + 1), train_errors, label='Training Error')
    plt.plot(range(1, n_estimators + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title(f'Random Forest - Feature Subset Size: {feature_subset_size}')
    plt.legend()
    plt.xscale('log')

plt.tight_layout()
plt.show()

for feature_subset_size in feature_subset_sizes:
    train_errors, test_errors = results[feature_subset_size]
    print(f"\nFeature Subset Size: {feature_subset_size}")
    print(f"Final Training Error: {train_errors[-1]:.4f}")
    print(f"Final Test Error: {test_errors[-1]:.4f}")