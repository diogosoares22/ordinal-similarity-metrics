import numpy as np
from typing import Callable
from dataclasses import dataclass

@dataclass
class RepresentationPair:
    X: np.ndarray
    Y: np.ndarray
    d_x: Callable
    d_y: Callable
    
    