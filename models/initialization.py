import torch
import torch.nn as nn
import math

def initialize_weights(model, gain=1):
    model.apply(lambda m: weights_init(m, gain))

def weights_init(m, gain=1):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, a=gain*math.sqrt(5))    
    