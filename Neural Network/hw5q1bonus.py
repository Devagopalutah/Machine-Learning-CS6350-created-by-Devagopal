import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

class BanknoteDataset(Dataset):
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class BanknoteNet(nn.Module):
    
    def __init__(self, input_size: int, hidden_width: int, depth: int, activation: str = 'tanh'):
        super().__init__()
        layers = []

        
        if activation == 'tanh':
            
            layers.append(nn.Linear(input_size, hidden_width))
            nn.init.xavier_uniform_(layers[-1].weight)
        else:  
            
            layers.append(nn.Linear(input_size, hidden_width))
            nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')

        
        for _ in range(depth - 2):
            if activation == 'tanh':
                layers.append(nn.Tanh())
                layers.append(nn.Linear(hidden_width, hidden_width))
                nn.init.xavier_uniform_(layers[-1].weight)
            else:
                layers.append(nn.ReLU())
                layers.append(nn.Linear(hidden_width, hidden_width))
                nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity='relu')

        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train_model(model: nn.Module,
                train_loader: DataLoader,
                test_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                epochs: int) -> dict:
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    history = {
        'train_loss': [],
        'train_error': [],
        'test_error': []
    }

    for epoch in range(epochs):
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        train_loss = train_loss / len(train_loader)
        train_error = 1 - (correct / total)

        
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        test_error = 1 - (correct / total)

        
        history['train_loss'].append(train_loss)
        history['train_error'].append(train_error)
        history['test_error'].append(test_error)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, '
                  f'Train Error = {train_error:.4f}, Test Error = {test_error:.4f}')

    return history

if __name__ == "__main__":
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(base_dir, 'bank', 'train.csv')
    test_file = os.path.join(base_dir, 'bank', 'test.csv')
    data_txt = os.path.join(base_dir, 'bank', 'data-desc.txt')

    
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

    
    train_dataset = BanknoteDataset(X_train, y_train)
    test_dataset = BanknoteDataset(X_test, y_test)

    
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    activations = ['tanh', 'relu']

    results = []

    
    for activation in activations:
        for depth in depths:
            for width in widths:
                print(f"\nTraining model with {activation}, depth={depth}, width={width}")

               
                model = BanknoteNet(input_size=4, hidden_width=width, depth=depth, activation=activation)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                
                history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

                
                results.append({
                    'activation': activation,
                    'depth': depth,
                    'width': width,
                    'final_train_error': history['train_error'][-1],
                    'final_test_error': history['test_error'][-1]
                })

                
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'])
                plt.title(f'Training Loss ({activation}, d={depth}, w={width})')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')

                plt.subplot(1, 2, 2)
                plt.plot(history['train_error'], label='Train')
                plt.plot(history['test_error'], label='Test')
                plt.title('Error Rates')
                plt.xlabel('Epoch')
                plt.ylabel('Error Rate')
                plt.legend()
                plt.tight_layout()
                plt.show()

    
    print("\nSummary of Results:")
    for result in results:
        print(f"\nActivation: {result['activation']}")
        print(f"Depth: {result['depth']}, Width: {result['width']}")
        print(f"Train Error: {result['final_train_error']:.4f}")
        print(f"Test Error: {result['final_test_error']:.4f}")