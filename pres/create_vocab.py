import os 
import random
from langconv import Converter
from tqdm import tqdm
word_list = []
datas = []

converter = Converter('zh-hans')

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def is_number(uchar):
    """判断一个unicode是否是半角数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
        return True
    else:
        return False

def is_english(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u0061' and uchar<=u'\u007a':
        return True
    else:
        return False

def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

# 读取标注文件
with open('../dataset/train.list', 'r', encoding='UTF-8') as f:
    for line in tqdm(f):
        name, label = line[:-1].split('\t')[-2:]
        label = label.replace('　','')
        label = converter.convert(label)
        label.lower()
        new_label = []
        for word in label:
            word = Q2B(word)
            if is_chinese(word) or is_number(word) or is_english(word):
                new_label.append(word)
                if word not in word_list:
                    word_list.append(word)
        if new_label!=[]:
            datas.append('%s\t%s\n' % (os.path.join('dataset/train_images',name), ''.join(new_label)))

word_list.sort()

# 生成词表
with open('../dataset/vocab.txt', 'w', encoding='UTF-8') as f:
    for word in word_list:
        f.write(word+'\n')

# 分割数据为训练和验证集
with open('../dataset/dataset.txt', 'w', encoding='UTF-8') as f:
    for line in datas:
        f.write(line)
