import numpy as np
from sklearn.metrics import f1_score

def CCC_score(y_pred, y_true):
    y_true_mean = np.mean(y_true, axis=0, keepdims=True)
    y_pred_mean = np.mean(y_pred, axis=0, keepdims=True)

    y_true_var = np.var(y_true, axis=0, keepdims=True)
    y_pred_var = np.var(y_pred, axis=0, keepdims=True)

    cov = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=0, keepdims=True)

    ccc = 2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
    ccc = np.squeeze(ccc)
    return ccc

def VA_metric(x, y):
    return CCC_score(x, y)

def EXPR_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    f1 = f1_score(x, y, average= 'macro')
    return f1

def averaged_f1_score(input, target):
    N, label_size = input.shape
    
    f1s = []
    for i in range(label_size):
        f1 = f1_score(input[:, i], target[:, i])
        f1s.append(f1)
    return np.mean(f1s)

def AU_metric(x, y):
    x = np.around(x)
    f1_au = averaged_f1_score(x, y)
    return f1_au