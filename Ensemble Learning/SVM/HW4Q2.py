import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import os

 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, 'bank-note')
train_file = os.path.join(data_dir, 'train.csv')
test_file = os.path.join(data_dir, 'test.csv')

class DualSVM:
    def __init__(self, C, kernel='linear', gamma=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
        self.w = None  # For linear kernel only
        
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'gaussian':
            return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.gamma))
    
    def _compute_kernel_matrix(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i,j] = self._kernel_function(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        K = self._compute_kernel_matrix(X)
        
        
        def objective(alpha):
            return 0.5 * np.sum((alpha.reshape(-1,1) * alpha.reshape(1,-1)) * 
                               (y.reshape(-1,1) * y.reshape(1,-1)) * K) - np.sum(alpha)
        
        
        def gradient(alpha):
            return np.sum((alpha.reshape(-1,1) * y.reshape(-1,1)) * 
                         (y.reshape(1,-1) * K), axis=1) - 1
        
        
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.dot(x, y)},  # Sum(alpha_i * y_i) = 0
        ]
        
    
        bounds = [(0, self.C) for _ in range(n_samples)]
        
        
        alpha0 = np.zeros(n_samples)
        
        
        result = minimize(objective, alpha0, method='SLSQP', jac=gradient,
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 1000})
        
        self.alpha = result.x
        
        
        sv_threshold = 1e-5
        sv_indices = np.where(self.alpha > sv_threshold)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_indices = sv_indices
        
        
        if self.kernel == 'linear':
            self.w = np.sum((self.alpha[sv_indices].reshape(-1,1) * 
                            self.support_vector_labels.reshape(-1,1) * 
                            self.support_vectors), axis=0)
            margins = np.dot(self.support_vectors, self.w)
            self.b = np.mean(self.support_vector_labels - margins)
        else:
            K_sv = self._compute_kernel_matrix(self.support_vectors, self.support_vectors)
            margins = np.sum((self.alpha[sv_indices].reshape(-1,1) * 
                            self.support_vector_labels.reshape(-1,1) * K_sv), axis=0)
            self.b = np.mean(self.support_vector_labels - margins)
    
    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            K = self._compute_kernel_matrix(X, self.support_vectors)
            y_pred = np.sum((self.alpha[self.support_vector_indices].reshape(1,-1) * 
                            self.support_vector_labels.reshape(1,-1) * K), axis=1) + self.b
            return np.sign(y_pred)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

class KernelPerceptron:
    def __init__(self, kernel='gaussian', gamma=1.0, max_epochs=100):
        self.kernel = kernel
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None
        
    def _kernel_function(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * self.gamma))
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        
        for epoch in range(self.max_epochs):
            mistakes = 0
            for i in range(n_samples):
                prediction = 0
                for j in range(n_samples):
                    prediction += self.alpha[j] * y[j] * self._kernel_function(X[j], X[i])
                prediction = np.sign(prediction)
                
                if prediction != y[i]:
                    self.alpha[i] += 1
                    mistakes += 1
            
            if mistakes == 0:
                break
        
        support_indices = np.where(self.alpha > 0)[0]
        self.support_vectors = X[support_indices]
        self.support_vector_labels = y[support_indices]
        self.alpha = self.alpha[support_indices]
    
    def predict(self, X):
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            pred = 0
            for sv, label, alpha in zip(self.support_vectors, 
                                      self.support_vector_labels, 
                                      self.alpha):
                pred += alpha * label * self._kernel_function(sv, x)
            predictions[i] = np.sign(pred)
        return predictions
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

def load_data():
    
    train_data = np.loadtxt(train_file, delimiter=',')
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]
    y_train = 2 * y_train - 1  
    
   
    test_data = np.loadtxt(test_file, delimiter=',')
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]
    y_test = 2 * y_test - 1  
    
    print("Dataset information:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test

def run_complete_experiment():
    X_train, y_train, X_test, y_test = load_data()
    C_values = [100/873, 500/873, 700/873]
    gamma_values = [0.1, 0.5, 1, 5, 100]
    
    
    print("\nPart A: Linear Dual SVM")
    print("=" * 50)
    linear_results = []
    
    for C in C_values:
        print(f"\nC = {C}")
        svm = DualSVM(C=C, kernel='linear')
        svm.fit(X_train, y_train)
        
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        result = {
            'C': C,
            'train_error': 1 - train_score,
            'test_error': 1 - test_score,
            'w': svm.w,
            'b': svm.b,
            'n_sv': len(svm.support_vectors)
        }
        linear_results.append(result)
        
        print(f"Training error: {1-train_score:.4f}")
        print(f"Test error: {1-test_score:.4f}")
        print(f"Number of support vectors: {len(svm.support_vectors)}")
        print("Weights:", svm.w)
        print("Bias:", svm.b)
    
    
    print("\nPart B: Gaussian Kernel SVM")
    print("=" * 50)
    gaussian_results = []
    
    for C in C_values:
        for gamma in gamma_values:
            print(f"\nC = {C}, gamma = {gamma}")
            svm = DualSVM(C=C, kernel='gaussian', gamma=gamma)
            svm.fit(X_train, y_train)
            
            train_score = svm.score(X_train, y_train)
            test_score = svm.score(X_test, y_test)
            
            result = {
                'C': C,
                'gamma': gamma,
                'train_error': 1 - train_score,
                'test_error': 1 - test_score,
                'n_sv': len(svm.support_vectors),
                'sv_indices': svm.support_vector_indices
            }
            gaussian_results.append(result)
            
            print(f"Training error: {1-train_score:.4f}")
            print(f"Test error: {1-test_score:.4f}")
            print(f"Number of support vectors: {len(svm.support_vectors)}")
    
    
    print("\nPart C: Support Vector Analysis")
    print("=" * 50)
    C_fixed = 500/873
    
    for i in range(len(gamma_values)-1):
        gamma1 = gamma_values[i]
        gamma2 = gamma_values[i+1]
        
        sv1 = next(r['sv_indices'] for r in gaussian_results 
                  if r['C'] == C_fixed and r['gamma'] == gamma1)
        sv2 = next(r['sv_indices'] for r in gaussian_results 
                  if r['C'] == C_fixed and r['gamma'] == gamma2)
        
        overlap = len(np.intersect1d(sv1, sv2))
        print(f"Overlap between gamma={gamma1} and gamma={gamma2}: {overlap} support vectors")
    
   
    print("\nPart D: Kernel Perceptron")
    print("=" * 50)
    perceptron_results = []
    
    for gamma in gamma_values:
        print(f"\nGamma = {gamma}")
        perceptron = KernelPerceptron(gamma=gamma)
        perceptron.fit(X_train, y_train)
        
        train_score = perceptron.score(X_train, y_train)
        test_score = perceptron.score(X_test, y_test)
        
        result = {
            'gamma': gamma,
            'train_error': 1 - train_score,
            'test_error': 1 - test_score,
            'n_sv': len(perceptron.support_vectors)
        }
        perceptron_results.append(result)
        
        print(f"Training error: {1-train_score:.4f}")
        print(f"Test error: {1-test_score:.4f}")
        print(f"Number of support vectors: {len(perceptron.support_vectors)}")
    
    return linear_results, gaussian_results, perceptron_results

if __name__ == "__main__":
    linear_results, gaussian_results, perceptron_results = run_complete_experiment()
