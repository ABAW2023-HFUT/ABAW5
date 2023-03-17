import torch
import numpy as np
import random
import sys
from eval import VA_metric, EXPR_metric, AU_metric
import torch.nn as nn

class WeightedAsymmetricLoss(nn.Module):
    def __init__(self, device, eps=1e-8, disable_torch_grad=True, weight=None):
        super(WeightedAsymmetricLoss, self).__init__()
        self.disable_torch_grad = disable_torch_grad
        self.eps = eps
        self.weight = weight
        self.device = device

    def forward(self, x, y):
        
        xs_pos = x
        xs_neg = 1 - x

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        # Asymmetric Focusing
        if self.disable_torch_grad:
            torch.set_grad_enabled(False)
        neg_weight = 1 - xs_neg
        if self.disable_torch_grad:
            torch.set_grad_enabled(True)
        loss = los_pos + neg_weight * los_neg
        loss = loss.to(self.device)
        # pdb.set_trace()
        # print('loss',str(loss.data))
        # print('weight',str(self.weight.data))
        if self.weight is not None:

            loss = loss * self.weight.view(1,-1)

        # loss = loss.to(self.device)
        loss = loss.mean(dim=-1)

        # print(loss.shape)
        return -loss.mean()
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

def get_annotations(task):
    annotations = {
        'VA':'VA_Estimation_Challenge',
        'EXPR':'EXPR_Classification_Challenge',
        'AU':'AU_Detection_Challenge'
    }
    return annotations[task]

def get_disregard(task):
    disregard = {
        'VA':-5,
        'EXPR':-1,
        'AU':-1
    }
    return disregard[task]

def get_activation(task):
    if task == 'VA':
        return nn.Tanh()
    elif task == 'EXPR':
        return nn.Sigmoid()
    elif task == 'AU':
        return nn.Sigmoid()

def get_num_output(task):
    num_output = {
        'VA':1,
        'EXPR':8,
        'AU':1
    }
    return num_output[task]

def get_num_linear(task):
    num_linear = {
        'VA':2,
        'EXPR':1,
        'AU':12
    }
    return num_linear[task]

def get_loss_fn(task, device):
    if task == 'VA':
        return nn.MSELoss()
    elif task == 'EXPR':
        return nn.CrossEntropyLoss()
    elif task == 'AU':
        train_weight = torch.from_numpy(np.loadtxt('./AU_train_weight.txt'))
        train_weight = train_weight.to(device)
        return WeightedAsymmetricLoss(device, weight=train_weight)
        # return nn.BCEWithLogitsLoss()

def get_eval_fn(task):
    if task == 'VA':
        return VA_metric
    elif task == 'EXPR':
        return EXPR_metric
    elif task == 'AU':
        return AU_metric

def flatten_pred_label(lst):
    return np.concatenate([np.array(l) for l in lst])

def get_num_label(task):
    num_label = {
        'VA':2,
        'EXPR':1,
        'AU':12
    }
    return num_label[task]