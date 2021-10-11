import numpy as np
import torch
import random
from config import LABEL2CHAR,CHAR2LABEL
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def decode_target(sequence):
    return ''.join([LABEL2CHAR[x] for x in sequence]).replace(' ', '')
