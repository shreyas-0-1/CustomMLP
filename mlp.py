import numpy as np
import pandas as pd
from activations import relu, relu_derivative

class MLP:
    def __init__(self, features: int, width: int, depth: int, output = 1):
        self.width = width
        self.depth = depth
        self.W, self.b = [np.random.randn(width, features) * np.sqrt(2/features)], [np.zeros(width)]
        for _ in range(depth - 1):
            self.W.append(np.random.randn(width, width) * np.sqrt(2/width))
            self.b.append(np.zeros(width))

        self.W.append(np.random.randn(output, width) * np.sqrt(2/width))
        self.b.append(np.zeros(output))
        self.h, self.f = [], []
        self.dLdh, self.dLdf, self.dLdW, self.dLdb = [], [], [], []
        
    def forward_pass(self, x: pd.Series):
        self.h = [x]
        self.f = [self.b[0] + self.W[0] @ x]
        for i in range(1, self.depth + 1):
            self.h.append(relu(self.f[i-1])) # dim R^D_i x 1
            self.f.append(self.b[i] + self.W[i] @ self.h[i]) # dim R^D_i+1 x 1

        return self.f[-1]

    def backward_pass(self, x: pd.Series, y: pd.Series):
        self.dLdf = [self.forward_pass(x) - y]
        self.dLdh = [self.W[-1].T @ self.dLdf[-1]]
        self.dLdW = [np.outer(self.dLdf[-1], self.h[-1])]
        self.dLdb = [self.dLdf[-1]]
        for i in range(self.depth - 1, -1, -1):
            self.dLdf.append(relu_derivative(self.h[i+1]) @ self.dLdh[-1])
            self.dLdh.append(self.W[i].T @ self.dLdf[-1])
            self.dLdW.append(np.outer(self.dLdf[-1], self.h[i]))      
            self.dLdb.append(self.dLdf[-1])
        
        self.dLdW.reverse() # [dLdW0, ..., dLdWd]
        self.dLdb.reverse() # [dLdb0, ..., dLdbd]

        return self.dLdW, self.dLdb 

    def grad_descent():
        pass

    def fit():
        pass

    def predict():
        pass

if __name__ == "__main__":
    net = MLP(2, 10, 10, 1)
    print(net.backward_pass(np.array([2, 4]), [2]))