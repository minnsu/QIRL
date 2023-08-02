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

def min_max_scaler(arr):
    if len(arr) == 0:
        return []
    minimum = min(arr)
    maximum = max(arr)
    return (arr - minimum) / (maximum - minimum)

def preprocess(src_path, dst_path, training=True):
    tmp = load_data(src_path)
    new_data = []
    for stock in tmp:
        tmp = stock.swapaxes(0, 2)
        tmp = tmp[3]
        if training:
            for idx in [0,1,4,5,7,8]:
                tmp[:, idx] = min_max_scaler(tmp[:, idx])
        new_data.append(tmp)
    store_data(dst_path, new_data)