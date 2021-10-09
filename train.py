# training model

import os
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
    return evaluation


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    #print(data[0].shape)
    images, targets, target_lengths = [d.to(device) for d in data]
    #print(images.shape,targets.shape,target_lengths.shape)
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


def main():
    epochs = crc_train_config['epochs']
    lr = crc_train_config['lr']
    show_interval = crc_train_config['show_interval']
    valid_interval = crc_train_config['valid_interval']
    save_interval = crc_train_config['save_interval']
    reload_checkpoint = crc_train_config['reload_checkpoint']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'当前运行环境为device: {device}')

    t_loader,valid_loader = train_loader(config.train_data_path)
    crnn = CRNN(config.channel, config.height, config.weight, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    # if reload_checkpoint:
    #     crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in tqdm(range(1, epochs + 1)):
        tot_train_loss = 0.
        tot_train_count = 0
        for train_data in t_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)
            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                logger.info(f' epoch:{epoch}, iter:{i}, train_batch_loss [{loss/train_size}]')
            if i % valid_interval == 0:
                evaluation = evaluate(crnn, valid_loader, criterion,
                                      decode_method=config['decode_method'],
                                      beam_size=config['beam_size'])
                print('valid_evaluation: loss={loss}, acc={acc}'.format(**evaluation))

                if i % save_interval == 0:
                    prefix = 'crc'
                    loss = evaluation['loss']
                    save_model_path = os.path.join(config['checkpoints_dir'],
                                                   f'{prefix}_{i:06}_loss{loss}.pt')
                    torch.save(crnn.state_dict(), save_model_path)
                    print('save model at ', save_model_path)

            i += 1

        logger.info(f' epoch :{epoch}, train_loss: {tot_train_loss / tot_train_count}')


if __name__ == '__main__':
    main()