import numpy as np

def relu(x: np.array) -> np.array:
    a = np.copy(x)
    a[a < 0] = 0
    return a

def relu_derivative(h: np.array) -> np.array:
    return np.diag((h > 0).astype(float))

def softmax(x: np.array) -> np.array:
    sum = np.sum(np.e ** x)
    x = np.e ** x / sum
    return x