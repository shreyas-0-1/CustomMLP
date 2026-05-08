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

        return self.f[-1] # model's prediction for regression

    def backward_pass(self, x: np.array, y: pd.Series):
        self.dLdf = [self.forward_pass(x) - y]
        self.dLdh = [self.W[-1].T @ self.dLdf[-1]]
        self.dLdW = [np.outer(self.dLdf[-1], self.h[-1])]
        self.dLdb = [self.dLdf[-1]]
        for i in range(self.depth - 1, -1, -1):
            self.dLdf.append(relu_derivative(self.h[i+1]) @ self.dLdh[-1])
            self.dLdh.append(self.W[i].T @ self.dLdf[-1])
            self.dLdW.append(np.outer(self.dLdf[-1], self.h[i]))      
            self.dLdb.append(self.dLdf[-1])
        
        self.dLdW.reverse() # [dL/dW0, ..., dL/dWd]
        self.dLdb.reverse() # [dL/db0, ..., dL/dbd]

        return self.dLdW, self.dLdb 

    # vanilla GD
    def fit(self, X: pd.DataFrame, y: pd.Series, epochs: int, learning: float): 
        for i in range(epochs): 
            print(f"Epoch {i+1}...")
            avg_dLdW, avgdLdB = self.backward_pass(X.iloc[0, :], y[0])
            batch_size = len(X)
            for i in range(1, batch_size):
                wrtW, wrtb = self.backward_pass(X.iloc[i, :], y[i])
                for j in range(0, len(avg_dLdW)):
                    avg_dLdW[j] += wrtW[j]
                    avgdLdB[j] += wrtb[j]
            
            for i in range(0, len(avg_dLdW)):
                avg_dLdW[i] /= batch_size
                avgdLdB[i] /= batch_size

            for i in range(0, len(avg_dLdW)):
                self.W[i] -= learning * avg_dLdW[i]
                self.b[i] -= learning * avgdLdB[i]

    def predict(X: pd.DataFrame):
        pass

if __name__ == "__main__":
    X = pd.read_csv("data.csv")[["x1", "x2"]]
    y = pd.read_csv("data.csv")["y"]
    net = MLP(2, 10, 10, 1)
    net.fit(X, y, 10, 0.01)