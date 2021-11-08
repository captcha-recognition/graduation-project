import pandas as pd
import torch
import torch.nn.functional as F
import dataset
import util
import os
from models.crnn import CRNN
import config
from config import crc_train_config
from logger import logger
from tqdm import tqdm
from ctc import ctc_decode
from logger import init_log

def load_model(checkpoint_path, device,model_name):
    """
    加载模型
    :param checkpoint_path: 模型路径
    :return:
    """
    crnn = util.make_model(model_name)
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    logger.info(f"Predict from the least model {checkpoint_path}, model {crnn.name()}")
    return crnn


def predict(crnn,test_loader,label2char,device,decode_method,beam_size):
    crnn.eval()
    pbar = tqdm(total=len(test_loader), desc="Predict")
    tot_correct = 0
    tot_count = 0
    wrong_cases = []
    with torch.no_grad():
        for images, targets, target_lengths  in test_loader:
            images = images.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            target_lengths = target_lengths.cpu().numpy().tolist()
            reals = targets.cpu().numpy().tolist()
            target_length_counter = 0
            tot_count += len(target_lengths)
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if real == pred:
                    tot_correct += 1
                else:
                    wrong_cases.append([util.decode_target(real),util.decode_target(pred)])
            pbar.update(1)
        pbar.close()
    
    print(f"acc: {tot_correct}/{tot_count} {tot_correct*1.0/tot_count}")
    print(wrong_cases)
    return wrong_cases





def main(test_path,checkpoint_path,model_name,multi,has_label = False,decode_method = "beam_search",beam_size = 10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, model_name {model_name} has_label {has_label},multi {multi}')
    predict_loader = dataset.test_loader(test_path,multi = multi)
    crnn = load_model(checkpoint_path,device,model_name)
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    crnn.to(device)

    wrong_cases = predict(crnn, predict_loader,config.LABEL2CHAR, device,
                           decode_method = decode_method,beam_size=beam_size)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict captcha model')
    parser.add_argument('--test_path', type=str, required=False, default=config.test_data_path,
                        help='The path of test dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,help='The path of test dataset')
    parser.add_argument('--model', type=str, required=True, default='crnn',
                        help='The mode of predict')
    parser.add_argument('--multi', type=bool, required=False, default=False,
                        help='multi or not')
    parser.add_argument('--has_label', type=bool, required=False, default=False,
                        help='The mode of predict')
    args = parser.parse_args()
    print(args)
    #init_log(args.model)
    main(args.test_path, args.checkpoint_path,args.model,args.multi,args.has_label)