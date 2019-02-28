import numpy as np

def min_max_scaler(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))