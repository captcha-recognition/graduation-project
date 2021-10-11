# training model
import os
from util import  setup_seed
from  logger import  logger
import torch
import  torch.nn.functional as F
import torch.optim as optim
from torch.nn import CTCLoss
from preprocessed import train_loader,test_loader
from models.crnn import CRNN
from config import crc_train_config
import config
from ctc import  ctc_decode
from  tqdm import  tqdm
import random

def evaluate(crnn, dataloader, criterion,
             max_iter=None, decode_method='beam_search', beam_size=10):
    crnn.eval()
    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    pbar_total = max_iter if max_iter else len(dataloader)
    pbar = tqdm(total=pbar_total, desc="Evaluate")

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if max_iter and i >= max_iter:
                break
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            images, targets, target_lengths = [d.to(device) for d in data]
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)
            batch_size = images.size(0)
            input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
            reals = targets.cpu().numpy().tolist()
            target_lengths = target_lengths.cpu().numpy().tolist()
            tot_count += batch_size
            tot_loss += loss.item()
            target_length_counter = 0
            for pred, target_length in zip(preds, target_lengths):
                real = reals[target_length_counter:target_length_counter + target_length]
                target_length_counter += target_length
                if pred == real:
                    tot_correct += 1
                else:
                    wrong_cases.append((real, pred))

            pbar.update(1)
        pbar.close()

    evaluation = {
        'loss': tot_loss / tot_count,
        'acc': tot_correct / tot_count,
        'wrong_cases': wrong_cases
    }
    logger.debug(wrong_cases)
    return evaluation


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]
    logits = crnn(images)
    log_probs = F.log_softmax(logits, dim=2)
    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main(train_data_path,goto_train):
    epochs = crc_train_config['epochs']
    model_name = crc_train_config['name']
    lr = crc_train_config['lr']
    momentum = crc_train_config['momentum']
    show_interval = crc_train_config['show_interval']
    valid_interval = crc_train_config['valid_interval']
    early_stop = crc_train_config['early_stop']
    checkpoints_dir = crc_train_config['checkpoints_dir']
    reload_checkpoint = crc_train_config['reload_checkpoint']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'当前运行环境为device: {device}')

    t_loader,valid_loader = train_loader(train_data_path)
    crnn = CRNN(config.channel, config.height, config.weight, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    if goto_train:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
        logger.info(f"Train from the least model {reload_checkpoint}")
    crnn.to(device)
    optimizer = optim.RMSprop(crnn.parameters(), lr=lr,momentum=momentum)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)
    early_num = 0
    val_loss = (1 << 10)
    for epoch in tqdm(range(1, epochs + 1)):
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in t_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            tot_train_loss += loss
            tot_train_count += train_size

        if epoch % show_interval == 0:
            logger.info(f'Train epoch :{epoch}, train_loss: {tot_train_loss / tot_train_count}')
        if epoch % valid_interval == 0:
            evaluation = evaluate(crnn, valid_loader, criterion,
                                  decode_method=crc_train_config['decode_method'],
                                  beam_size=crc_train_config['beam_size'])
            valid_loss = evaluation['loss']
            valid_acc = evaluation['acc']
            logger.info(f'Valid epoch:{epoch}, loss:{valid_loss}, acc:{valid_acc}')
            if val_loss > valid_loss:
                val_loss = valid_loss
                early_num = 0
                save_model_path = os.path.join(checkpoints_dir,
                                               f'{model_name}_{epoch}_loss{valid_loss}.pt')
                torch.save(crnn.state_dict(), save_model_path)
                logger.info('save model at ', save_model_path)
            else:
                early_num += 1
            if early_num > early_stop:
                logger.info(f"Early Stop in epoch:{epoch}")
                break


if __name__ == '__main__':
    import argparse
    seed = 100
    setup_seed(seed)
    parser = argparse.ArgumentParser(description='Train captcha model')
    parser.add_argument('--train_path', type=str,required= False,default= config.train_data_path, help='The path of train dataset')
    parser.add_argument('--goto_train',type=bool,required= False,default= False,help="Train from checkpoint or not")
    args = parser.parse_args()
    main(args.train_path,args.goto_train)