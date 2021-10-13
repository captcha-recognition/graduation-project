import string

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import  os
import pandas as pd

def generator(width,height,num,label_length,characters,path):
    """

    :param width:
    :param height:
    :param num:
    :param label_length:
    :param characters:
    :param path:
    :return:
    """
    generator = ImageCaptcha(width,height,font_sizes=(21, 25, 28))
    labels = []
    imgs = []
    for idx in tqdm(range(1,num+1)):
        label = ''.join([random.choice(characters[1:]) for j in range(label_length)])
        img_src = generator.generate_image(label)
        img_name = f"{idx}.jpg"
        img_path = os.path.join(path,img_name)
        img_src.save(img_path)
        labels.append(label)
        imgs.append(img_name)
    data = pd.DataFrame({"ID":imgs,"label":labels})
    data.to_csv(os.path.join(path,"train_label.csv"),index=False)


if __name__ == '__main__':
    path = "/Users/sjhuang/Documents/docs/dataset/unlabel_data/unlabel_captcha_generator"
    width = 100
    height = 32
    num = 1000
    label_length = 4
    characters = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase
    generator(width,height,num,label_length,characters,path)

