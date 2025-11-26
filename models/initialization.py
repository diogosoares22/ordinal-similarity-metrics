import torch
import torch.nn as nn

def initialize_weights(model, gain=1):
    model.apply(lambda m: weights_init(m, gain))

def weights_init(m, gain=1):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=gain)    
    