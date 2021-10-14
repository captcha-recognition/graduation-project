# 数据预处理
import logging
import os

import pandas as pd
import torch
import torchvision.utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import dataset,dataloader
import config
from logger import  logger


class CaptchaDataset(dataset.Dataset):
    """
    ## 加载数据，数据格式为
    # train: label.png
    # test: index.png
    """

    def __init__(self, root, transformer = None,train = True):
        """
        captcha dataset
        :param root: the path of dataset, 数据类型为 root/label.png ...
        :param transformer: transformer for image
        :param train: train of not
        """
        super(CaptchaDataset, self).__init__()
        assert root
        self.root = root
        self.train = train
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = transforms.ToTensor()
        self.labels = None
        if self.train:
            data = pd.read_csv(os.path.join(self.root,'train_label.csv'),skip_blank_lines=True)
            self.labels = list(data['label'].values)
            self.image_paths = list(data['ID'].values)
            assert len(self.labels) == len(self.image_paths)
        else:
            paths = os.listdir(self.root)
            self.image_paths = []
            for path in paths:
                if path.endswith(".png") or path.endswith(".jpg") or path.endswith("jpeg"):
                    self.image_paths.append(path)
        assert self.image_paths


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        #print(self.image_paths[idx])
        image_path = os.path.join(self.root,self.image_paths[idx])
        img = Image.open(image_path)
        # print(self.labels[idx])
        img = img.convert("RGB")
        if self.train:
            label = self.labels[idx]
            label = label.lower()
            #print(label)
            target = [config.CHAR2LABEL[c] for c in label]
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            img = self.transformer(img)
            return img, target, target_length
        else:
            return image_path, self.transformer(img)

def captcha_collate_fn(batch):

    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

def train_loader(train_path,train_rate = config.train_rate,batch_size = config.batch_size,
                 height = config.height, width = config.width,collate_fn = captcha_collate_fn,
                 transformer = None):
    """
    
    :param train_path:  the path of training data
    :param batch_size: 
    :param height resize height
    :param width: resize width
    :return: 
    """""
    if transformer is None:
        transformer = transforms.Compose(
            [
              # transforms.RandomAffine((0.9,1.1)),
              # transforms.RandomRotation(8),
              transforms.Resize((height, width)),
              transforms.ToTensor(),
              transforms.Normalize(mean=config.mean,std= config.std)
             ]
        )
    train_set = CaptchaDataset(train_path, transformer=transformer)
    train_len = int(len(train_set)*train_rate)
    train_data, val_data = torch.utils.data.random_split(train_set,[train_len,len(train_set)-train_len])
    return dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn= collate_fn),\
           dataloader.DataLoader(val_data, batch_size=batch_size, shuffle=True,collate_fn= collate_fn)


def test_loader(test_path,batch_size = config.test_batch_size, height = config.height,
                width = config.width,transformer = None):
    """

    :param test_path:
    :param batch_size:
    :param x: resize
    :param y:
    :return:
    """
    if transformer is None:
        transformer = transforms.Compose(
        [transforms.Resize((height, width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=config.mean, std=config.std)
         ]
    )
    test_set = CaptchaDataset(test_path,train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False)



if __name__ == '__main__':
     train_loader,val_loader = train_loader(config.train_data_path)
     #print(train_set.image_paths[1089])
     # imgs, targets, target_lens  = next(iter(train_loader))
     # grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     # print(grid_img.shape)
     # print(targets)
     # plt.imshow(grid_img.permute(1, 2, 0))
     # plt.imsave(f"pres/preprocessed_{height}_{weight}.jpg",grid_img.permute(1, 2, 0).numpy())
     num = 0
     for imgs, targets, target_lens  in train_loader:
         num += len(imgs)
         logger.info(f"imgs:{imgs.shape}, {num}")



