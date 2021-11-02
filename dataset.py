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
from logger import  logger,init_log


class CaptchaDataset(dataset.Dataset):
    """
    ## 加载数据，数据格式为
    # train: label.png
    # test: index.png
    """

    def __init__(self, root,multi = False, transformer = None,train = True):
        """
        captcha dataset
        :param root: the paths of dataset, 数据类型为 root/label.png ...
        :param transformer: transformer for image
        :param train: train of not
        """
        super(CaptchaDataset, self).__init__()
        assert root
        self.root = root
        self.train = train
        self.transformer = transformer
        self.labels = None
        if multi:
            paths = [os.path.join(self.root,path) for path in os.listdir(self.root)]
        else:
            paths = [self.root]
        self._extract_images(paths)

    def __extract_images(self, paths):

        image_paths = []
        labels = []
        logger.info(f'read data from {paths}')
        for item_path in paths:
            logger.info(f'read data from {item_path}')
            info = pd.read_json(os.path.join(item_path,'train_label.json',),dtype=str)
            item_image_paths = [os.path.join(item_path,path) for path in list(info['ID'].values)]
            item_labels = list(info['label'].values)
            image_paths += item_image_paths
            labels += item_labels
        self.image_paths = image_paths
        self.labels = labels

        assert len(self.image_paths) == len(self.labels)
    
    def _extract_images(self,paths):
        self.image_paths = []
        self.labels = []
        for path in paths:
            logger.info(path)
            if not os.path.isdir(path):
                continue
            file_paths = os.listdir(path)
            for file_path in file_paths:
                if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith("jpeg"):
                    self.image_paths.append(os.path.join(path,file_path))
                    if self.train:
                        self.labels.append(file_path.split('_')[1].split('.')[0])
                    else:
                        self.labels.append(file_path.split('.')[0])

        assert len(self.image_paths) == len(self.labels) 
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        fail = False
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                r,g,b,a = img.split()
                img.load() # required for png.split()
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=a) # 3 is the alpha channel
                img  =  background
        except Exception as e:
            logger.error(e)
            img = Image.open('2-mc5m.png')
            img = img.convert("RGB")
            fail = True
        if self.train:
            if fail:
                label = 'mc5m'
            else:
                label = str(self.labels[idx])
                label = label.lower()
            if len(label) != 4:
                logger.info(f'{label} length error')
            #logger.info(label)
            target = [config.CHAR2LABEL[c] for c in label]
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            if self.transformer:
                img = self.transformer(img)
            return img, target, target_length
        else:
            return image_path, self.transformer(img)

def resizeNormalize(image,imgH, imgW):
    """
    resize and normalize image
    """
    transformer = transforms.Compose(
        [transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
        #  transforms.Normalize(mean=config.mean, std=config.std)
         ]
    )
    return transformer(image)

def captcha_collate_fn(batch,imgH = 32, imgW = 100, keep_ratio = True):
    images, targets, target_lengths = zip(*batch)
    if keep_ratio:
        max_ratio = 0.0
        for image in images:
            w,h = image.size
            max_ratio = max(max_ratio,w/float(h))
        imgW = max(int(max_ratio*imgH),imgW)
    images = [resizeNormalize(image,imgH,imgW) for image in images]
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths

def train_loader(train_path,multi = False,train_rate = config.train_rate,batch_size = config.batch_size,
                 height = config.height, width = config.width,collate_fn = captcha_collate_fn,
                 transformer = None):
    """
    
    :param train_path:  the path of training data
    :param batch_size: 
    :param height resize height
    :param width: resize width
    :return: 
    """""
    # if transformer is None:
    #     transformer = transforms.Compose(
    #         [
    #           #transforms.RandomAffine((0.9,1.1)),
    #           #transforms.RandomRotation(8),
    #           transforms.Resize((height, width)),
    #           transforms.ToTensor(),
    #           transforms.Normalize(mean=config.mean,std= config.std)
    #          ]
    #     )
    train_set = CaptchaDataset(train_path,multi = multi, transformer=transformer)
    train_len = int(len(train_set)*train_rate)
    train_data, val_data = torch.utils.data.random_split(train_set,[train_len,len(train_set)-train_len])
    return dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn= collate_fn),\
           dataloader.DataLoader(val_data, batch_size=batch_size, shuffle=True,collate_fn= collate_fn)


def test_loader(test_path,batch_size = config.test_batch_size, height = config.height,
                width = config.width,collate_fn = captcha_collate_fn,transformer = None):
    """

    :param test_path:
    :param batch_size:
    :param x: resize
    :param y:
    :return:
    """
    # if transformer is None:
    #     transformer = transforms.Compose(
    #     [transforms.Resize((height, width)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=config.mean, std=config.std)
    #      ]
    # )
    test_set = CaptchaDataset(test_path,train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False,collate_fn = collate_fn)



if __name__ == '__main__':
     init_log('test')
     height,width = 32,100
    #  transformer = transforms.Compose(
    #     [
    #         #transforms.RandomAffine((0.9, 1.1)),
    #         #transforms.RandomRotation(8),
    #         transforms.Resize((32, int(width/(height/3)))),
    #         transforms.ToTensor(),
    #     ]
    #  )
     path = '/Users/sjhuang/Documents/docs/dataset/train'
     train_loade,val_loader = train_loader(path,multi = True,transformer = None)
     imgs, targets, target_lens  = next(iter(train_loade))
     grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     plt.imshow(grid_img.permute(1, 2, 0))
     plt.imsave(f"pres/preprocessed_{height}_{width}.jpg",grid_img.permute(1, 2, 0).numpy())
     # num = 0
     # for imgs, targets, target_lens  in train_loader:
     #     num += len(imgs)
     #     logger.info(f"imgs:{imgs.shape}, {num}")



