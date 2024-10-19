import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt


base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, 'default of credit card clients.xls')
data = pd.read_excel(data_file, header=1)
X = data.drop('default payment next month', axis=1)
y = data['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=24000, test_size=6000, random_state=42)

continuous_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 
                       'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
scaler = StandardScaler()
X_train[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test[continuous_features] = scaler.transform(X_test[continuous_features])

def calculate_error(y_true, y_pred):
    return 1 - np.mean(y_true == y_pred)


dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_train_error = calculate_error(y_train, dt.predict(X_train))
dt_test_error = calculate_error(y_test, dt.predict(X_test))


def evaluate_ensemble(model, X_train, y_train, X_test, y_test, n_estimators):
    train_errors = []
    test_errors = []
    for i in [1, 10, 50, 100, 200, 500]:  # Reduced number of evaluation points
        if i > n_estimators:
            break
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        train_errors.append(calculate_error(y_train, model.predict(X_train)))
        test_errors.append(calculate_error(y_test, model.predict(X_test)))
    return train_errors, test_errors


bagging = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42, max_depth=None), n_estimators=500, random_state=42, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
adaboost = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), n_estimators=500, random_state=42)
bagging_train_errors, bagging_test_errors = evaluate_ensemble(bagging, X_train, y_train, X_test, y_test, 500)
rf_train_errors, rf_test_errors = evaluate_ensemble(rf, X_train, y_train, X_test, y_test, 500)
adaboost_train_errors, adaboost_test_errors = evaluate_ensemble(adaboost, X_train, y_train, X_test, y_test, 500)


plt.figure(figsize=(12, 8))
iterations = [1, 10, 50, 100, 200, 500]

plt.plot(iterations[:len(bagging_train_errors)], bagging_train_errors, label='Bagging (Train)', alpha=0.7)
plt.plot(iterations[:len(bagging_test_errors)], bagging_test_errors, label='Bagging (Test)', alpha=0.7)
plt.plot(iterations[:len(rf_train_errors)], rf_train_errors, label='Random Forest (Train)', alpha=0.7)
plt.plot(iterations[:len(rf_test_errors)], rf_test_errors, label='Random Forest (Test)', alpha=0.7)
plt.plot(iterations[:len(adaboost_train_errors)], adaboost_train_errors, label='AdaBoost (Train)', alpha=0.7)
plt.plot(iterations[:len(adaboost_test_errors)], adaboost_test_errors, label='AdaBoost (Test)', alpha=0.7)

plt.axhline(y=dt_train_error, color='r', linestyle='--', label='Single DT (Train)')
plt.axhline(y=dt_test_error, color='g', linestyle='--', label='Single DT (Test)')

plt.xlabel('Number of Estimators')
plt.ylabel('Error Rate')
plt.title('Comparison of Ensemble Methods on Credit Default Dataset')
plt.legend()
plt.grid(True)
plt.show()


print(f"Single Decision Tree - Train Error: {dt_train_error:.4f}, Test Error: {dt_test_error:.4f}")
print(f"Bagged Trees - Train Error: {bagging_train_errors[-1]:.4f}, Test Error: {bagging_test_errors[-1]:.4f}")
print(f"Random Forest - Train Error: {rf_train_errors[-1]:.4f}, Test Error: {rf_test_errors[-1]:.4f}")
print(f"AdaBoost - Train Error: {adaboost_train_errors[-1]:.4f}, Test Error: {adaboost_test_errors[-1]:.4f}")