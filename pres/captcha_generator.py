import string

from captcha.image import ImageCaptcha
from tqdm import tqdm
import random
import  os
import pandas as pd

def generator(width,height,num,label_length,characters,path):
    generator = ImageCaptcha(width=width, height=height)
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
    path = "/Users/sjhuang/Documents/docs/dataset/captcha/captcha_generator"
    width = 100
    height = 32
    num = 200000
    label_length = 4
    characters = '-' + string.digits + string.ascii_uppercase + string.ascii_lowercase
    generator(width,height,num,label_length,characters,path)

