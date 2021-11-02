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
    count = len(paths)
    right = 0
    errors = []
    for path, pred in zip(paths, preds):
        text = ''.join(pred)
        real = path.split('/')[-1].split('.')[0]
        print(f'{path} > {text}')
        if real.lower() == text:
            right += 1
        else:
            errors.append([real,text])
    print(f'{right}/{count} {right/count}')


    

def show_label_result(paths,preds,labels):

    print('\n===== result =====')
    total = len(paths)
    acc = 0
    errors = []
    reals = []
    res = []
    for path, pred in zip(paths, preds):
        try:
            text = ''.join(pred)
            img_path = path.split('/')[-1]
            real = str(labels[img_path])
            print(f'{path}: {real}> {text}')
            reals.append(real)
            res.append(preds)
            if real == text or real.lower() == text.lower():
                acc += 1
            else:
                errors.append((path,real.lower(),text.lower()))
            print(f"acc: {acc}/{total} {acc*1.0/total}")
        except Exception as e:
            print(e)
    #print(errors)
    ans = pd.DataFrame({'real':reals,'pred':res})
    ans.to_json('ans.json')
    



def main(test_path,checkpoint_path,model_name,has_label = False,decode_method = "beam_search",beam_size = 10):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, model_name {model_name} has_label {has_label}')
    predict_loader = dataset.test_loader(test_path)
    crnn = load_model(checkpoint_path,device,model_name)
    crnn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    crnn.to(device)

    preds,images = predict(crnn, predict_loader,config.LABEL2CHAR, device,
                           decode_method = decode_method,beam_size=beam_size)
    if has_label:
        data = pd.read_json(os.path.join(test_path,'train_label.json'))
        keys = list(data['ID'].values)
        values = list(data['label'].values)
        labels = {k:v for k,v in zip(keys,values)}
        show_label_result(images, preds,labels)
    else:
        show_result(images, preds)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict captcha model')
    parser.add_argument('--test_path', type=str, required=False, default=config.test_data_path,
                        help='The path of test dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True,help='The path of test dataset')
    parser.add_argument('--model', type=str, required=True, default='crnn',
                        help='The mode of predict')
    parser.add_argument('--has_label', type=bool, required=False, default=False,
                        help='The mode of predict')
    args = parser.parse_args()
    init_log(args.model)
    main(args.test_path, args.checkpoint_path,args.model,args.has_label)