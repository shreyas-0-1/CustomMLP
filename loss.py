import numpy as np

def l2_loss(obs: np.array, pred: np.array):
    return 0.5 * np.sum((pred - obs)**2) / len(pred)