import torch
import torch.nn as nn

def initialize_weights(model):
    model.apply(weights_init)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=10)    
    