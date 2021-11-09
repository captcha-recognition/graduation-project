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
from util import init

class CaptchaDataset(dataset.Dataset):
    """
    ## 加载数据，数据格式为
    # train: label.png
    # test: index.png
    """

    def __init__(self, root:str, char2labels:dict,logger:logging, multi = False, transformer = None,train = True):
        """
        captcha dataset
        :param root: the paths of dataset, 数据类型为 root/label.png ...
        :param transformer: transformer for image
        :param train: train of not
        """
        super(CaptchaDataset, self).__init__()
        assert root and char2labels
        self.char2labels = char2labels
        self.root = root
        self.train = train
        self.transformer = transformer
        self.labels = None
        self.logger = logger
        if multi:
            paths = [os.path.join(self.root,path) for path in os.listdir(self.root)]
        else:
            paths = [self.root]
        self._extract_images(paths)
        if self.train:
            self._check_images()


    def _extract_images(self,paths):
        self.image_paths = []
        self.labels = []
        for path in paths:
            self.logger.info(path)
            if not os.path.isdir(path):
                continue
            file_paths = os.listdir(path)
            for file_path in file_paths:
                if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith("jpeg"):
                    self.image_paths.append(os.path.join(path,file_path))
                    name = file_path.split('.')[0]
                    if name.find('-') != -1:
                        label = name.split('-')[1]
                    elif name.find('_') != -1:
                        label = name.split('_')[1]
                    else:
                        label = name
                    self.labels.append(label)
        assert len(self.image_paths) == len(self.labels) 
    
    def _check_images(self):
        err_labels = []
        for img_path,label in zip(self.image_paths,self.labels):
            for c in label:
                if not ((c >= '0' and c <= '9') or (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')):
                    err_labels.append(img_path)
                    break
        if len(err_labels) > 0 and self.train:
            self.logger.error(f'Please check {err_labels}')

        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            img.load() # required for png.split()
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=a) # 3 is the alpha channel
            img  =  background
        #print(img.size)
        label = str(self.labels[idx])
        label = label.lower()
        if len(label) != 4 and self.train:
            self.logger.info(f'{label} length not 4')
        target = [self.char2labels[c] for c in label]
        target_length = [len(target)]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)
        if self.transformer:
            img = self.transformer(img)
        return img, target, target_length
 

def resizeNormalize(image,imgH, imgW,mean,std,train = False):
    """
    resize and normalize image
    """
    if train:
        transformer = transforms.Compose(
        [
         transforms.RandomAffine((0.9,1.1)),
         transforms.RandomRotation(6),
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    else:
        transformer = transforms.Compose(
        [
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
         ]
    )
    return transformer(image)


class CaptchaCollateFn(object):

    def __init__(self,imgH=32, imgW=100, keep_ratio=False,train = False, mean = (0.485, 0.456, 0.406), std =  (0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.train = train
        self.mean = mean
        self.std = std
    
    def __call__(self, batch):
        images, targets, target_lengths = zip(*batch)
        if self.keep_ratio:
            max_ratio = 0.0
            for image in images:
                w,h = image.size
                max_ratio = max(max_ratio,w/float(h))
            self.imgW = max(int(max_ratio*self.imgH),self.imgW)
        # print(images)=
        images = [resizeNormalize(image,self.imgH,self.imgW,self.mean,self.std,self.train) for image in images]
        # print(images[0].shape)
        images = torch.stack(images, 0)
        targets = torch.cat(targets, 0)
        target_lengths = torch.cat(target_lengths, 0)
        return images, targets, target_lengths
        
    

def train_loader(config,logger,transformer = None):
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
    train_set = CaptchaDataset(config['train']['input_path'],char2labels = config['base']['char2labels'],logger = logger,
                multi = config['train']['multi'], transformer=transformer)
    train_len = int(len(train_set)*config['base']['train_rate'])
    train_data, val_data = torch.utils.data.random_split(train_set,[train_len,len(train_set)-train_len])
    return dataloader.DataLoader(train_data, batch_size=config['train']['batch_size'], shuffle=True,
           collate_fn= CaptchaCollateFn(config['base']['height'],config['base']['width'],config['train']['keep_ratio'],True,config['base']['mean'],config['base']['std'])),\
           dataloader.DataLoader(val_data, batch_size=config['train']['batch_size'], shuffle=True,
           collate_fn= CaptchaCollateFn(config['base']['height'],config['base']['width'],config['train']['keep_ratio'],False,config['base']['mean'],config['base']['std']))


def test_loader(config:dict,logger:logging,transformer = None):
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
    test_set = CaptchaDataset(config['test']['input_path'],char2labels = config['base']['char2labels'],logger = logger,
                            multi = config['test']['multi'],train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=config['test']['batch_size'], shuffle=False,collate_fn = 
               CaptchaCollateFn(config['base']['height'],config['base']['width'],config['test']['keep_ratio'],False,
               config['base']['mean'],config['base']['std']))



if __name__ == '__main__':
     height,width = 32,100
    #  transformer = transforms.Compose(
    #     [
    #         #transforms.RandomAffine((0.9, 1.1)),
    #         #transforms.RandomRotation(8),
    #         transforms.Resize((32, int(width/(height/3)))),
    #         transforms.ToTensor(),
    #     ]
    #  )
     config, logger = init(100,'configs/local/resnet_rnn_ctc.yaml',save_log=False)
     train_loade,val_loader = train_loader(config,logger,transformer = None)
     imgs, targets, target_lens  = next(iter(train_loade))
     grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     plt.imshow(grid_img.permute(1, 2, 0))
     plt.imsave(f"pres/preprocessed_{height}_{width}.jpg",grid_img.permute(1, 2, 0).numpy())
     # num = 0
     # for imgs, targets, target_lens  in train_loader:
     #     num += len(imgs)
     #     logger.info(f"imgs:{imgs.shape}, {num}")



