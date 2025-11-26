import torch
import torch.nn as nn

def initialize_weights(model, gain=sqrt(5)):
    model.apply(lambda m: weights_init(m, gain))

def weights_init(m, gain=sqrt(5)):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data, a=gain)    
    