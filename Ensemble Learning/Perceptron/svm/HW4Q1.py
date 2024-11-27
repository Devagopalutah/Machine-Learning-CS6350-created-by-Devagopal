import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from itertools import product
import os

= 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'bank-note')
train_file = os.path.join(data_dir, 'train.csv')
data_txt = os.path.join(data_dir, 'data-desc.txt')
test_file = os.path.join(data_dir, 'test.csv')

class PrimalSVM:
   
    def __init__(self, C, max_epochs=100, gamma0=0.1, a=1, schedule_type='a'):
       
        self.C = C
        self.max_epochs = max_epochs
        self.gamma0 = gamma0
        self.a = a
        self.schedule_type = schedule_type
        self.w = None
        self.b = None
        self.objective_values = []
        
    def _learning_rate(self, t):
        if self.schedule_type == 'a':
            return self.gamma0 / (1 + (self.gamma0/self.a) * t)
        else:
            return self.gamma0 / (1 + t)
    
    def _calculate_objective(self, X, y):
        margin = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.maximum(0, 1 - margin)
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(hinge_loss)
    
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        t = 0
        
        for epoch in range(self.max_epochs):
            X_shuffled, y_shuffled = shuffle(X, y)
            epoch_obj = 0
            
            for i in range(n_samples):
                t += 1
                gamma_t = self._learning_rate(t)
                
                margin = y_shuffled[i] * (np.dot(X_shuffled[i], self.w) + self.b)
                
                if margin < 1:
                    grad_w = self.w - self.C * y_shuffled[i] * X_shuffled[i]
                    grad_b = -self.C * y_shuffled[i]
                else:
                    grad_w = self.w
                    grad_b = 0
                
                self.w -= gamma_t * grad_w
                self.b -= gamma_t * grad_b
            
            obj_value = self._calculate_objective(X, y)
            self.objective_values.append(obj_value)
    
    def predict(self, X):
        
        return np.sign(np.dot(X, self.w) + self.b)
    
    def score(self, X, y):
        
        predictions = self.predict(X)
        return np.mean(predictions == y)

def load_and_preprocess_data():
    
    columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    
    
    train_data = pd.read_csv(train_file, header=None, names=columns)
    test_data = pd.read_csv(test_file, header=None, names=columns)
    
    
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    
    
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    
    print("Dataset information:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print("\nFeature statistics:")
    print(pd.DataFrame(X_train, columns=columns[:-1]).describe())
    
    return X_train, y_train, X_test, y_test

def tune_parameters(X_train, y_train, X_test, y_test, C):
    
    gamma0_values = [0.001, 0.01, 0.1, 1.0]
    a_values = [1, 5, 10, 50]
    
    best_params = {'gamma0': None, 'a': None}
    best_obj = float('inf')
    
    results = []
    
    for gamma0, a in product(gamma0_values, a_values):
        print(f"\nTrying gamma0={gamma0}, a={a}")
        svm = PrimalSVM(C=C, gamma0=gamma0, a=a, schedule_type='a')
        svm.fit(X_train, y_train)
        
        final_obj = svm.objective_values[-1]
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        results.append({
            'gamma0': gamma0,
            'a': a,
            'final_obj': final_obj,
            'train_score': train_score,
            'test_score': test_score,
            'convergence': svm.objective_values
        })
        
        if final_obj < best_obj:
            best_obj = final_obj
            best_params['gamma0'] = gamma0
            best_params['a'] = a
    
    
    plt.figure(figsize=(15, 10))
    for result in results:
        plt.plot(result['convergence'], 
                label=f"γ0={result['gamma0']}, a={result['a']}")
    plt.xlabel('Epoch')
    plt.ylabel('Objective Value')
    plt.title(f'Parameter Tuning: Convergence Plot (C={C})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return best_params, results

def run_complete_experiment():
    
    print("Loading banknote authentication dataset...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    C_values = [100/873, 500/873, 700/873]
    final_results = []
    
    for C in C_values:
        print(f"\n{'='*50}")
        print(f"Testing C = {C:.4f}")
        print(f"{'='*50}")
        
        
        print("\nTuning parameters for schedule A...")
        best_params, tuning_results = tune_parameters(X_train, y_train, X_test, y_test, C)
        gamma0_a = best_params['gamma0']
        a_value = best_params['a']
        
        print(f"\nBest parameters found:")
        print(f"γ0 = {gamma0_a}")
        print(f"a = {a_value}")
        
        
        print("\nTraining with schedule A (best parameters)...")
        svm_a = PrimalSVM(C=C, gamma0=gamma0_a, a=a_value, schedule_type='a')
        svm_a.fit(X_train, y_train)
        train_score_a = svm_a.score(X_train, y_train)
        test_score_a = svm_a.score(X_test, y_test)
        
        
        print("\nTraining with schedule B...")
        gamma0_b = gamma0_a
        svm_b = PrimalSVM(C=C, gamma0=gamma0_b, schedule_type='b')
        svm_b.fit(X_train, y_train)
        train_score_b = svm_b.score(X_train, y_train)
        test_score_b = svm_b.score(X_test, y_test)
        
        
        w_diff = np.linalg.norm(svm_a.w - svm_b.w)
        b_diff = abs(svm_a.b - svm_b.b)
        train_diff = abs(train_score_a - train_score_b)
        test_diff = abs(test_score_a - test_score_b)
        
        
        result = {
            'C': C,
            'schedule_a': {
                'gamma0': gamma0_a,
                'a': a_value,
                'train_score': train_score_a,
                'test_score': test_score_a,
                'final_obj': svm_a.objective_values[-1],
                'feature_weights': svm_a.w
            },
            'schedule_b': {
                'gamma0': gamma0_b,
                'train_score': train_score_b,
                'test_score': test_score_b,
                'final_obj': svm_b.objective_values[-1],
                'feature_weights': svm_b.w
            },
            'differences': {
                'w_diff': w_diff,
                'b_diff': b_diff,
                'train_diff': train_diff,
                'test_diff': test_diff
            }
        }
        final_results.append(result)
        
        
        print("\nResults Summary:")
        print("-" * 40)
        print("\nSchedule A (γt = γ0/(1 + (γ0/a)t)):")
        print(f"Training accuracy: {train_score_a:.4f}")
        print(f"Test accuracy: {test_score_a:.4f}")
        print("Feature weights:")
        for feature, weight in zip(['Variance', 'Skewness', 'Curtosis', 'Entropy'], svm_a.w):
            print(f"  {feature}: {weight:.4f}")
        print(f"Bias term: {svm_a.b:.4f}")
        
        print("\nSchedule B (γt = γ0/(1 + t)):")
        print(f"Training accuracy: {train_score_b:.4f}")
        print(f"Test accuracy: {test_score_b:.4f}")
        print("Feature weights:")
        for feature, weight in zip(['Variance', 'Skewness', 'Curtosis', 'Entropy'], svm_b.w):
            print(f"  {feature}: {weight:.4f}")
        print(f"Bias term: {svm_b.b:.4f}")
        
        print("\nDifferences between schedules:")
        print(f"Weight difference (L2 norm): {w_diff:.6f}")
        print(f"Bias difference: {b_diff:.6f}")
        print(f"Training accuracy difference: {train_diff:.6f}")
        print(f"Test accuracy difference: {test_diff:.6f}")
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(svm_a.objective_values, label='Schedule A')
        plt.plot(svm_b.objective_values, label='Schedule B')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.title(f'Schedule Comparison (C={C:.4f})')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    return final_results

if __name__ == "__main__":
    results = run_complete_experiment()