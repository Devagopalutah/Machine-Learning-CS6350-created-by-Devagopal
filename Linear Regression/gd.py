import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]


def calculate_cost(X, y, w):
    m = len(y)
    predictions = X.dot(w)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost


def batch_gradient_descent(X, y, learning_rate, tolerance=1e-6, max_iterations=10000):
    m, n = X.shape
    w = np.zeros(n)
    costs = []
    weight_diffs = []

    for i in range(max_iterations):
        predictions = X.dot(w)
        gradient = (1 / m) * X.T.dot(predictions - y)
        w_new = w - learning_rate * gradient

        weight_diff = np.linalg.norm(w_new - w)
        weight_diffs.append(weight_diff)

        if weight_diff < tolerance:
            break

        w = w_new
        cost = calculate_cost(X, y, w)
        costs.append(cost)

    return w, costs, weight_diffs


def stochastic_gradient_descent(X, y, learning_rate, max_updates=50000, convergence_window=1000):
    m, n = X.shape
    w = np.zeros(n)
    costs = []

    for update in range(max_updates):
        random_index = np.random.randint(m)
        xi = X[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        prediction = xi.dot(w)
        gradient = xi.T.dot(prediction - yi)
        w = w - learning_rate * gradient

        if update % 100 == 0:  # Calculate cost every 100 updates to save time
            cost = calculate_cost(X, y, w)
            costs.append(cost)

        # Check for convergence
        if len(costs) > convergence_window and np.std(costs[-convergence_window:]) < 1e-6:
            break

    return w, costs


def analytical_solution(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(base_dir, 'concrete', 'train.csv')
test_file = os.path.join(base_dir, 'concrete', 'test.csv')


X_train, y_train = load_data(train_file)
X_test, y_test = load_data(test_file)

X_train_bias = add_bias(X_train)
X_test_bias = add_bias(X_test)


learning_rates = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
for lr in learning_rates:
    w_bgd, costs_bgd, weight_diffs_bgd = batch_gradient_descent(X_train_bias, y_train, lr)
    if len(costs_bgd) < 10000:  # Converged
        print(f"Batch Gradient Descent converged with learning rate: {lr}")
        break

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(costs_bgd)
plt.title('BGD: Cost vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.subplot(1, 2, 2)
plt.plot(weight_diffs_bgd)
plt.title('BGD: Weight Difference Norm')
plt.xlabel('Iterations')
plt.ylabel('Weight Difference Norm')
plt.yscale('log')
plt.tight_layout()
plt.show()

print("Learned weight vector (BGD):", w_bgd)
print("Final training cost (BGD):", costs_bgd[-1])
print("Test cost (BGD):", calculate_cost(X_test_bias, y_test, w_bgd))


sgd_learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
sgd_results = []

for lr in sgd_learning_rates:
    w_sgd, costs_sgd = stochastic_gradient_descent(X_train_bias, y_train, lr)
    test_cost = calculate_cost(X_test_bias, y_test, w_sgd)
    sgd_results.append((lr, w_sgd, costs_sgd, test_cost))

    plt.figure(figsize=(10, 5))
    plt.plot(costs_sgd)
    plt.title(f'SGD: Cost vs Updates (lr={lr})')
    plt.xlabel('Updates (x100)')
    plt.ylabel('Cost')
    plt.show()


for lr, w, costs, test_cost in sgd_results:
    print(f"\nSGD Learning rate: {lr}")
    print(f"Final training cost: {costs[-1]:.6f}")
    print(f"Test cost: {test_cost:.6f}")


best_lr, best_w_sgd, _, best_test_cost = min(sgd_results, key=lambda x: x[3])
print(f"\nBest SGD learning rate: {best_lr}")
print(f"Best SGD test cost: {best_test_cost:.6f}")


w_analytical = analytical_solution(X_train_bias, y_train)

print("\nOptimal weight vector (Analytical):", w_analytical)
print("Training cost (Analytical):", calculate_cost(X_train_bias, y_train, w_analytical))
print("Test cost (Analytical):", calculate_cost(X_test_bias, y_test, w_analytical))


print("\nWeight vector comparison:")
print("BGD:", w_bgd)
print("SGD (best):", best_w_sgd)
print("Analytical:", w_analytical)

print("\nTest cost comparison:")
print("BGD:", calculate_cost(X_test_bias, y_test, w_bgd))
print("SGD (best):", best_test_cost)
print("Analytical:", calculate_cost(X_test_bias, y_test, w_analytical))


weight_vectors = {
    'BGD': w_bgd,
    'SGD (best)': best_w_sgd,
    'Analytical': w_analytical
}

plt.figure(figsize=(12, 6))
bar_width = 0.2
indices = np.arange(len(weight_vectors['BGD']))


for i, (label, weights) in enumerate(weight_vectors.items()):
    plt.bar(indices + i * bar_width, weights, width=bar_width, label=label)

plt.title('Comparison of Weight Vectors')
plt.xlabel('Feature Index')
plt.ylabel('Weight Value')
plt.xticks(indices + bar_width, range(len(weight_vectors['BGD'])))
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(12, 6))
plt.plot(costs_bgd, label='BGD')
plt.plot(np.arange(0, len(costs_sgd)*100, 100), costs_sgd, label=f'SGD (lr={best_lr})')
plt.axhline(y=calculate_cost(X_train_bias, y_train, w_analytical), color='r', linestyle='--', label='Analytical')
plt.title('Cost Comparison: BGD vs SGD vs Analytical')
plt.xlabel('Iterations / Updates')
plt.ylabel('Cost')
plt.legend()
plt.show()
