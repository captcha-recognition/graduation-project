import torch
import torch.nn as nn
import torch.nn.functional as F
from models.crnn import CRNN
import config
from config import crc_train_config
from logger import logger

def load_model(checkpoint_path):
    """
    加载模型
    :param checkpoint_path: 模型路径
    :return:
    """
    crnn = CRNN(config.channel, config.height, config.weight, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Train from the least model {checkpoint_path}")


def predict(crnn,img):
    pass