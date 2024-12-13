import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os

class LogisticRegression:
    def __init__(self, prior_variance: float = None, learning_rate: float = 0.1,
                 d: float = 1.0, epochs: int = 100):
        self.prior_variance = prior_variance  
        self.learning_rate = learning_rate
        self.d = d
        self.epochs = epochs
        self.weights = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        z = np.dot(X, self.weights)
        likelihood = -np.mean(y * np.log(self._sigmoid(z) + 1e-10) +
                            (1 - y) * np.log(1 - self._sigmoid(z) + 1e-10))

        
        if self.prior_variance is not None:
            prior_term = np.sum(self.weights**2) / (2 * self.prior_variance)
            return likelihood + prior_term
        return likelihood

    def _get_learning_rate(self, t: int) -> float:
        return self.learning_rate / (1 + (self.learning_rate * t / self.d))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                xi = X_shuffled[i]
                yi = y_shuffled[i]

                
                z = np.dot(xi, self.weights)
                pred = self._sigmoid(z)
                gradient = (pred - yi) * xi

                
                if self.prior_variance is not None:
                    gradient += self.weights / self.prior_variance

                
                learning_rate = self._get_learning_rate(epoch * n_samples + i)
                self.weights -= learning_rate * gradient

            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)

def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'bank', 'train.csv')
    test_file = os.path.join(base_dir, 'bank', 'test.csv')
    
   
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

    
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train, y_train, X_test, y_test

def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray,
                        variance: float = None) -> Tuple[float, float]:
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    d_values = [1, 10, 100]
    best_loss = float('inf')
    best_params = None

    for lr in learning_rates:
        for d in d_values:
            model = LogisticRegression(
                prior_variance=variance,
                learning_rate=lr,
                d=d,
                epochs=100
            )
            model.fit(X_train, y_train)

            
            last_losses = model.loss_history[-10:]
            loss_std = np.std(last_losses)
            mean_loss = np.mean(last_losses)

            if loss_std < 0.01 and mean_loss < best_loss:
                best_loss = mean_loss
                best_params = (lr, d)

            
            plt.figure(figsize=(10, 5))
            plt.plot(model.loss_history)
            plt.title(f'Convergence Plot (lr={lr}, d={d})')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()

    return best_params

def comprehensive_evaluation():
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    
    variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    map_results = []

    print("Part (a): MAP Estimation")
    print("Tuning parameters for each variance value...")

    for v in variances:
        print(f"\nTuning for variance = {v}")
        best_lr, best_d = tune_hyperparameters(X_train, y_train, variance=v)
        print(f"Best parameters: learning_rate = {best_lr}, d = {best_d}")

        
        model = LogisticRegression(prior_variance=v, learning_rate=best_lr, d=best_d)
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        map_results.append({
            'variance': v,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'best_lr': best_lr,
            'best_d': best_d
        })

    
    print("\nPart (b): ML Estimation")
    print("Tuning parameters for ML estimation...")

    best_lr, best_d = tune_hyperparameters(X_train, y_train)
    print(f"Best parameters: learning_rate = {best_lr}, d = {best_d}")

    ml_model = LogisticRegression(prior_variance=None, learning_rate=best_lr, d=best_d)
    ml_model.fit(X_train, y_train)
    ml_train_score = ml_model.score(X_train, y_train)
    ml_test_score = ml_model.score(X_test, y_test)

    
    print("\nMAP Estimation Results:")
    print("Variance | Train Acc | Test Acc  | Best lr | Best d")
    print("-" * 55)
    for result in map_results:
        print(f"{result['variance']:8.2f} | {result['train_accuracy']:8.4f} | "
              f"{result['test_accuracy']:8.4f} | {result['best_lr']:7.4f} | {result['best_d']:6.0f}")

    print("\nML Estimation Results:")
    print(f"Train Accuracy: {ml_train_score:.4f}")
    print(f"Test Accuracy: {ml_test_score:.4f}")
    print(f"Best learning rate: {best_lr}")
    print(f"Best d: {best_d}")

    
    print("\nPart (c): Analysis")
    print("\nComparison of MAP vs ML estimation:")

    
    best_map = max(map_results, key=lambda x: x['test_accuracy'])

    print("\n1. Performance Comparison:")
    print(f"Best MAP Test Accuracy: {best_map['test_accuracy']:.4f} (v={best_map['variance']})")
    print(f"ML Test Accuracy: {ml_test_score:.4f}")

    print("\n2. Conclusions:")
    if best_map['test_accuracy'] > ml_test_score:
        print("- MAP estimation provides better generalization than ML")
        print(f"- Optimal prior variance (v={best_map['variance']}) suggests appropriate "
              "level of regularization")
    else:
        print("- ML estimation performs comparably to MAP in this case")
        print("- Data might have low noise or sufficient samples for good ML estimation")

    print("\n3. Comparison of v (MAP) vs C (SVM):")
    print("- v in MAP controls the strength of the Gaussian prior")
    print("- Small v (like large C in SVM) enforces stronger regularization")
    print("- Large v (like small C in SVM) allows weights to take larger values")
    print("- Both hyperparameters balance between data fit and model complexity")

    return map_results, (ml_train_score, ml_test_score)

if __name__ == "__main__":
    
    map_results, ml_results = comprehensive_evaluation()