# training model

import os
from  logger import  logger
import torch
import  torch.nn as nn
import  torch.nn.functional as F
import torch.optim as optim
from torch.nn import CTCLoss
from preprocessed import train_loader,test_loader
from models.crnn import CRNN
from config import crc_train_config
import config

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


def main():
    epochs = crc_train_config['epochs']
    lr = crc_train_config['lr']
    show_interval = crc_train_config['show_interval']
    valid_interval = crc_train_config['valid_interval']
    save_interval = crc_train_config['save_interval']
    reload_checkpoint = crc_train_config['reload_checkpoint']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'当前运行环境为device: {device}')

    crnn = CRNN(1, config.x, config.y, config.num_class,
                map_to_seq_hidden=crc_train_config['map_to_seq_hidden'],
                rnn_hidden=crc_train_config['rnn_hidden'],
                leaky_relu=crc_train_config['leaky_relu'])
    if reload_checkpoint:
        crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=lr)
    criterion = CTCLoss(reduction='sum')
    criterion.to(device)

    assert save_interval % valid_interval == 0
    i = 1
    for epoch in range(1, epochs + 1):
        print(f'epoch: {epoch}')
        tot_train_loss = 0.
        tot_train_count = 0
        for images, labels in train_loader:
            loss = train_batch(crnn, item, label, optimizer, criterion, device)
            train_size = len(label)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % show_interval == 0:
                logger.info(f'{i} epoch, train_batch_loss[{loss / train_size}]')

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

        print('train_loss: ', tot_train_loss / tot_train_count)


if __name__ == '__main__':
    main()