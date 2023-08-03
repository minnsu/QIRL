import numpy as np
import pickle
import torch

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def store_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def softmax(x):
    tmp = torch.exp(x - torch.max(x))
    return tmp / tmp.sum()

def min_max_scaler(arr):
    if len(arr) == 0:
        return []
    minimum = min(arr)
    maximum = max(arr)
    return (arr - minimum) / (maximum - minimum)

def preprocess(raw, training=True):
    new_data = []
    raw = raw.swapaxes(0, 1)
    raw = raw[:, :, 3]
    if training:
        for idx in [0,1,4,5,7,8]:
            raw[:, idx] = min_max_scaler(raw[:, idx])
    return torch.tensor(raw, dtype=torch.float32)