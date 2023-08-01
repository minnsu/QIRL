import numpy as np
import pickle

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def store_data(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def softmax(x):
    tmp = np.exp(x - np.max(x))
    return tmp / tmp.sum()