import pandas as pd
import numpy as np
from collections import Counter
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
train_file = os.path.join(base_dir, 'car', 'train.csv')
test_file = os.path.join(base_dir, 'car', 'test.csv')

train_data = pd.read_csv(train_file, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])
test_data = pd.read_csv(test_file, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

attributes = list(X_train.columns)

impurity_functions = {
    'entropy': entropy,
    'majority_error': majority_error,
    'gini_index': gini_index
}

max_depths = [1, 2, 3, 4, 5, 6]
impurity_measures = ['entropy', 'majority_error', 'gini_index']

for impurity_measure in impurity_measures:
    print(f"\nImpurity measure: {impurity_measure}")
    for max_depth in max_depths:
        tree = id3(X_train, y_train, attributes, max_depth, impurity_functions[impurity_measure])
        train_accuracy = evaluate(tree, X_train, y_train)
        test_accuracy = evaluate(tree, X_test, y_test)
        print(f"Max depth: {max_depth}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")


max_depths = range(1, 7)  # 1 to 6
results = []

for impurity_measure, impurity_func in impurity_functions.items():
    for max_depth in max_depths:
        tree = id3(X_train, y_train, attributes, max_depth, impurity_func)
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

print("Average Prediction Errors:")
print(table.to_string(float_format="{:.4f}".format))

average_errors = results_df.groupby('Impurity Measure')[['Train Error', 'Test Error']].mean()
print("\nAverage Errors Across All Depths:")
print(average_errors.to_string(float_format="{:.4f}".format))
