import pandas as pd
import numpy as np
from collections import Counter
import os

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

def majority_error(y):
    if len(y) == 0:
        return 0
    majority = Counter(y).most_common(1)[0][1]
    return (len(y) - majority) / len(y)

def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities**2)

def information_gain(X, y, attribute, impurity_func):
    parent_impurity = impurity_func(y)
    
    weighted_child_impurity = 0
    for value in X[attribute].unique():
        child_y = y[X[attribute] == value]
        weight = len(child_y) / len(y)
        weighted_child_impurity += weight * impurity_func(child_y)
    
    return parent_impurity - weighted_child_impurity

def id3(X, y, attributes, max_depth, impurity_func):
    if len(np.unique(y)) == 1:
        return Node(label=y.iloc[0])
    
    if len(attributes) == 0 or max_depth == 0:
        return Node(label=Counter(y).most_common(1)[0][0])
    
    gains = [information_gain(X, y, attr, impurity_func) for attr in attributes]
    best_attribute = attributes[np.argmax(gains)]
    
    node = Node(attribute=best_attribute)
    
    for value in X[best_attribute].unique():
        X_subset = X[X[best_attribute] == value].drop(best_attribute, axis=1)
        y_subset = y[X[best_attribute] == value]
        
        if len(X_subset) == 0:
            node.branches[value] = Node(label=Counter(y).most_common(1)[0][0])
        else:
            new_attributes = [attr for attr in attributes if attr != best_attribute]
            node.branches[value] = id3(X_subset, y_subset, new_attributes, max_depth-1, impurity_func)
    
    return node

def predict(node, instance):
    if node.label is not None:
        return node.label
    
    if node.attribute not in instance:
        return max(node.branches.values(), key=lambda x: x.label if x.label else "").label
    
    value = instance[node.attribute]
    if value not in node.branches:
        return max(node.branches.values(), key=lambda x: x.label if x.label else "").label
    
    return predict(node.branches[value], instance)

def evaluate(tree, X, y):
    y_pred = []
    for i in range(len(X)):
        try:
            y_pred.append(predict(tree, X.iloc[i]))
        except Exception as e:
            print(f"Error predicting for instance {i}: {e}")
            y_pred.append(None)
    
    y_pred = [pred for pred in y_pred if pred is not None]
    y = y[:len(y_pred)]
    
    accuracy = sum(np.array(y_pred) == y) / len(y)
    return accuracy

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

impurity_functions = {
    'Information Gain': entropy,
    'Majority Error': majority_error,
    'Gini Index': gini_index
}

max_depths = range(1, 17)  # 1 to 16
results = []
for impurity_measure, impurity_func in impurity_functions.items():
    for max_depth in max_depths:
        tree = id3(X_train, y_train, list(X_train.columns), max_depth, impurity_func)
        train_accuracy = evaluate(tree, X_train, y_train)
        test_accuracy = evaluate(tree, X_test, y_test)
        results.append({
            'Impurity Measure': impurity_measure,
            'Max Depth': max_depth,
            'Train Error': 1 - train_accuracy,
            'Test Error': 1 - test_accuracy
        })

results_df = pd.DataFrame(results)
table = pd.pivot_table(results_df, 
                       values=['Train Error', 'Test Error'], 
                       index=['Impurity Measure', 'Max Depth'], 
                       aggfunc='first')
print("\nAverage Prediction Errors:")
print(table.to_string(float_format="{:.4f}".format))

# Calculate and print average errors across all depths for each impurity measure
average_errors = results_df.groupby('Impurity Measure')[['Train Error', 'Test Error']].mean()
print("\nAverage Errors Across All Depths:")
print(average_errors.to_string(float_format="{:.4f}".format))