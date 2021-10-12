import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import dataset
from models.crnn import CRNN
import config
from config import crc_train_config
from logger import logger
from tqdm import tqdm
from ctc import ctc_decode
from dataset import CaptchaDataset
def load_model(checkpoint_path):
    """
    加载模型
    :param checkpoint_path: 模型路径
    :return:
    """
    crnn = CRNN(config.channel, config.height, config.width, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Train from the least model {checkpoint_path}")


def predict(crnn,test_loader,label2char,device,decode_method,beam_size):
    crnn.eval()
    pbar = tqdm(total=len(test_loader), desc="Predict")
    all_preds = []
    test_data = []
    with torch.no_grad():
        for image_paths ,data in test_loader:
            test_data += image_paths
            images = data.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                               label2char=label2char)
            all_preds += preds
            pbar.update(1)
        pbar.close()

    return all_preds,test_data

def show_result(paths, preds):
    print('\n===== result =====')
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        print(f'{path} > {text}')

def main(test_path,checkpoint_path,decode_method = "beam_search",beam_size = 10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    predict_loader = dataset.test_loader(test_path)
    crnn = CRNN(config.channel, config.height, config.width, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    crnn.to(device)

    preds,images = predict(crnn, predict_loader,config.LABEL2CHAR, device,
                           decode_method = decode_method,beam_size=beam_size)

    show_result(images, preds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict captcha model')
    parser.add_argument('--test_path', type=str, required=False, default=config.test_data_path,
                        help='The path of test dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, default= crc_train_config['reload_checkpoint'],
                        help='The path of test dataset')
    args = parser.parse_args()
    main(args.test_path, args.checkpoint_path)