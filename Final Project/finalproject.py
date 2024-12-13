import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

class CustomPerceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000, random_state=42):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.errors_ = []
        
    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()
        
        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                prediction = self._step_function(np.dot(xi, self.weights) + self.bias)
                error = target - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * xi
                    self.bias += self.learning_rate * error
                    errors += 1
            self.errors_.append(errors)
            if errors == 0:
                break
        return self
    
    def predict(self, X):
        return self._step_function(np.dot(X, self.weights) + self.bias)

class CustomSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, max_iter=1000, random_state=42):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _hinge_loss(self, y, y_pred):
        return np.maximum(0, 1 - y * y_pred)
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        
        y = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()
        
        for _ in range(self.max_iter):
            for idx, xi in enumerate(X):
                condition = y[idx] * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(xi, y[idx]))
                    self.bias -= self.learning_rate * y[idx]
            
            # Calculate loss for monitoring
            output = np.dot(X, self.weights) + self.bias
            loss = np.mean(self._hinge_loss(y, output))
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        output = np.dot(X, self.weights) + self.bias
        return np.where(output <= 0, 0, 1)

class CustomNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, max_iter=1000, random_state=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.losses = []
        
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        sx = self._sigmoid(x)
        return sx * (1 - sx)
    
    def _initialize_parameters(self):
        np.random.seed(self.random_state)
        for i in range(len(self.layer_sizes) - 1):
            self.weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 0.01)
            self.biases.append(np.random.randn(1, self.layer_sizes[i+1]) * 0.01)
    
    def _forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            net = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activations.append(self._sigmoid(net))
        return activations
    
    def _backward_propagation(self, X, y, activations):
        m = X.shape[0]
        deltas = []
        
        # Output layer error
        delta = activations[-1] - y.reshape(-1, 1)
        deltas.append(delta)
        
        # Hidden layers error
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self._sigmoid_derivative(activations[i])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def fit(self, X, y):
        self._initialize_parameters()
        y = y.reshape(-1, 1)
        
        for _ in range(self.max_iter):
            # Forward propagation
            activations = self._forward_propagation(X)
            
            # Backward propagation
            self._backward_propagation(X, y, activations)
            
            # Calculate loss
            loss = np.mean(np.square(activations[-1] - y))
            self.losses.append(loss)
        
        return self
    
    def predict(self, X):
        activations = self._forward_propagation(X)
        return (activations[-1] >= 0.5).astype(int)

def load_and_validate_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
        print("\nColumns in dataset:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None

def prepare_initial_data(df):
    available_columns = []
    if 'RSRP' in df.columns:
        available_columns.append('RSRP')
    if 'squal' in df.columns:
        available_columns.append('squal')
    elif 'QUAL' in df.columns:
        available_columns.append('QUAL')

    if not available_columns:
        raise ValueError("Neither RSRP nor signal quality measurements found in the dataset")

    df[available_columns] = df[available_columns].apply(pd.to_numeric, errors='coerce')
    
    id_columns = [col for col in ['MCC', 'MNC', 'PCI'] if col in df.columns]
    if id_columns:
        df['station_id'] = df[id_columns].astype(str).agg('_'.join, axis=1)
    else:
        df['station_id'] = df.index.astype(str)

    return df, available_columns

def create_idle_labels(df, available_columns, rsrp_threshold=-75, squal_threshold=22):
    conditions = []
    if 'RSRP' in available_columns:
        conditions.append(df['RSRP'].notna() & (df['RSRP'] < rsrp_threshold))
    if 'squal' in available_columns or 'QUAL' in available_columns:
        qual_column = 'squal' if 'squal' in available_columns else 'QUAL'
        conditions.append(df[qual_column].notna() & (df[qual_column] < squal_threshold))

    if not conditions:
        return pd.Series(0, index=df.index)

    return (pd.concat(conditions, axis=1).all(axis=1)).astype(int)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, feature_names, model_name):
    print(f"\nTraining {model_name}...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    if hasattr(model, 'errors_'):
        plt.plot(model.errors_, label='Errors')
        plt.ylabel('Number of Misclassifications')
    elif hasattr(model, 'losses'):
        plt.plot(model.losses, label='Loss')
        plt.ylabel('Loss')
    plt.title(f'Training Progress - {model_name}')
    plt.xlabel('Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot feature importance (for Perceptron and SVM)
    if hasattr(model, 'weights') and len(model.weights.shape) == 1:
        plt.figure(figsize=(8, 4))
        importance = np.abs(model.weights)
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        plt.show()
    
    return model

def predict_idle_stations(model, df, feature_names, imputer, scaler):
    X = df[feature_names].copy()
    X = scaler.transform(imputer.transform(X))
    predictions = model.predict(X)
    
    return df[predictions == 1]['station_id'].tolist()

def main():
    # Set up file path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base_dir, 'data.csv')
    
    df = load_and_validate_csv(data_file)
    
    if df is not None:
        try:
            df, available_columns = prepare_initial_data(df)
            
            X = df[available_columns].copy()
            y = create_idle_labels(df, available_columns)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            
            X_train = scaler.fit_transform(imputer.fit_transform(X_train))
            X_test = scaler.transform(imputer.transform(X_test))
            
            # Initialize models
            perceptron = CustomPerceptron(learning_rate=0.01, max_iter=1000)
            svm = CustomSVM(learning_rate=0.001, lambda_param=0.01, max_iter=1000)
            nn = CustomNeuralNetwork(layer_sizes=[X_train.shape[1], 10, 5, 1], learning_rate=0.01, max_iter=1000)
            
            models = [
                (perceptron, "Perceptron"),
                (svm, "SVM"),
                (nn, "Neural Network")
            ]
            
            trained_models = {}
            
            # Train and evaluate each model
            for model, name in models:
                trained_model = train_and_evaluate_model(
                    model, X_train, X_test, y_train, y_test, available_columns, name
                )
                trained_models[name] = trained_model
            
            # Make predictions using each model
            print("\nTime-based Predictions:")
            for period in ['Morning', 'Afternoon', 'Evening', 'Night']:
                print(f"\n{period}:")
                sample = df.sample(n=min(len(df), 10000), replace=False)
                
                for model_name, model in trained_models.items():
                    print(f"\n{model_name} predictions:")
                    idle_stations = predict_idle_stations(
                        model,
                        sample,
                        available_columns,
                        imputer,
                        scaler
                    )
                    print(f"Number of idle stations: {len(idle_stations)}")
                    for station in idle_stations[:5]:
                        print(f"- {station}")
                    if len(idle_stations) > 5:
                        print(f"... and {len(idle_stations) - 5} more stations")
                    
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            print("Please check your data format and column names")

if __name__ == "__main__":
    main()