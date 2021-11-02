import os

import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
import config
from config import LABEL2CHAR,CHAR2LABEL,configs
from models import crnn,crnn_v2,resnet_rnn,resnet_gru
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def decode_target(sequence):
    return ''.join([LABEL2CHAR[x] for x in sequence]).replace(' ', '')


models = {
    'crnn': crnn.CRNN,
    'crnn_v2':crnn_v2.CRNN_V2,
    'resnet_rnn':resnet_rnn.ResNetRNN,
    'resnet_gru':resnet_gru.ResNetRRU
}

def make_model(model_name):
    model = models[model_name]
    assert model
    model_params = configs[model_name]
    crnn = model((config.channel,config.height,config.width),
                 config.num_class,model_params['map_to_seq_hidden'],
                 model_params['rnn_hidden'],model_params['leaky_relu'])
    return crnn

def dataset_check(path):
    labels = pd.read_csv(os.path.join(path,'train_label.csv'))
    keys = list(labels['ID'].values)
    vals = list(labels['label'].values)
    for key,val in tqdm(zip(keys,vals)):
        try:
            val = val.lower()
            label = [CHAR2LABEL[ch] for ch in val]
        except Exception as e:
            print(val,key,e)



if __name__ == '__main__':
    dataset_check('~/Documents/docs/dataset/label_data/captcha_generator')