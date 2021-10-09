# 数据预处理
import logging
import os
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
        self.image_paths = os.listdir(self.root)
        assert self.image_paths
        self.labels = None
        if self.train:
            self.labels = [image_path.split('.')[0] for image_path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root,self.image_paths[idx])
        img = Image.open(image_path)
        # print(self.labels[idx])
        img = img.convert("RGB")
        if self.train:
            label = self.labels[idx]
            target = [config.CHAR2LABEL[c] for c in label]
            target_length = [len(target)]
            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            img = self.transformer(img)
            #print(img.shape)
            return img, target, target_length
        else:
            return self.transformer(img)

def train_loader(train_path,batch_size = config.batch_size, height = config.height, weight = config.weight):
    """
    
    :param train_path:  the path of training data
    :param batch_size: 
    :param x: resize
    :param y: 
    :return: 
    """""
    transformer = transforms.Compose(
        [transforms.Resize((height, weight)),
         transforms.ToTensor(),
         ]
    )
    train_set = CaptchaDataset(train_path, transformer=transformer)
    return dataloader.DataLoader(train_set, batch_size=batch_size, shuffle=True)


def test_loader(test_path,batch_size = config.batch_size, height = config.height, weight = config.weight):
    """

    :param test_path:
    :param batch_size:
    :param x: resize
    :param y:
    :return:
    """
    transformer = transforms.Compose(
        [transforms.Resize((height, weight)),
         transforms.ToTensor(),
         ]
    )
    test_set = CaptchaDataset(test_path,train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False)



if __name__ == '__main__':

     height = 32
     weight = 100
     transformer = transforms.Compose(
         [transforms.Resize((height, weight)),
          transforms.ToTensor(),
          ]
     )
     train_set = CaptchaDataset(config.train_data_path,transformer=transformer)
     train_loader = dataloader.DataLoader(train_set,batch_size= 64,shuffle=False,drop_last=True)
     print(train_set.image_paths[1089])
     # imgs, targets, target_lens  = next(iter(train_loader))
     # grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     # print(grid_img.shape)
     # print(targets)
     # plt.imshow(grid_img.permute(1, 2, 0))
     # plt.imsave(f"pres/preprocessed_{height}_{weight}.jpg",grid_img.permute(1, 2, 0).numpy())
     num = 0
     for imgs, targets, target_lens  in train_loader:
         num += len(targets)
         logger.info(f"imgs:{imgs.shape}, {num}")



