# 图片预处理
import os
from tqdm import  tqdm
import json
import pandas as pd

def handle_dataset(path,begin = 1):
    """
    将 label.jpg 形式转化为 train.json 形式
    :param path:
    :param begin:
    :return:
    """
    imgs = os.listdir(path)
    idx = begin
    keys =[]
    vals =[]
    for img in tqdm(imgs):
        label, ty = img.split(".")
        if ty == 'png' or ty == 'jpg' or ty == 'jpeg':
            img_path = os.path.join(path, img)
            new_path = os.path.join(path, f"{idx}.{ty}")
            os.rename(img_path, new_path)
            keys.append(f"{idx}.{ty}")
            vals.append(label)
            idx += 1
    data = pd.DataFrame({'ID':keys,'label':vals})
    data.to_csv(os.path.join(path, "train_label.csv"), index=False)

def fix_idx(path,begin = 1):
    idx = begin
    data = pd.read_csv(os.path.join(path, "train_label.csv"))
    keys = list(data['ID'].values)
    new_keys = []
    vals = list(data['label'].values)
    for k, v in tqdm(zip(keys,vals)):
        _, ty = k.split(".")
        if ty == 'png' or ty == 'jpg' or ty == 'jpeg':
            new_k = f"{idx}.{ty}"
            new_keys.append(new_k)
            img_path = os.path.join(path, k)
            new_path = os.path.join(path, new_k)
            os.rename(img_path, new_path)
            idx += 1

    data = pd.DataFrame({'ID': new_keys, 'label': vals})
    data.to_csv(os.path.join(path, "train_label.csv"), index=False)

def merge_dataset(p1,p2,p):
    d1 = pd.read_csv(os.path.join(p1, "train_label.csv"))
    d2 = pd.read_csv(os.path.join(p2, "train_label.csv"))
    data = pd.concat([d1,d2])
    print(data.head())
    data.to_csv(os.path.join(p, "train_label.csv"), index = False)

def fix_label(path):
    data = pd.read_csv(path,skip_blank_lines=True)
    pass

if __name__ == '__main__':
   pass



