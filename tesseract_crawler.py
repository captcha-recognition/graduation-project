# 验证码采集
#!/usr/bin/env python
# encoding=utf-8
import time
import requests
from tqdm import  tqdm
import os
from logger import  logger

def split_name(url):
    words = url.split('/')
    return words[2]


def doGetApi(path,name,idx,base_url,params,headers):
    url = base_url + "?1=1"
    for k, v in params:
        url += f"&{k}={v}"
    req = requests.get(url=url, headers=headers)
    if (req.status_code == 200):
        text = req.text
        print(text)


def doGet(path,name,idx, base_url, params,headers):
    url = base_url + "?1=1"
    for k,v in params:
        url += f"&{k}={v}"
    req = requests.get(url = url,headers = headers)
    if(req.status_code == 200):
        text = req.content
        with open(f'{path}/{name}_{idx}_{time.time()}.png','wb') as f:
            f.write(text)


if __name__ == '__main__':
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    url = "https://fuwu.most.gov.cn/authentication/servlet/validateCodeServlet?width=100&height=32&1634122558325"
    name = split_name(url)
    nums = 100
    path = f"/Users/sjhuang/Documents/docs/dataset/unlabel_data/{name}"
    if not os.path.exists(path):
        logger.info(f"create {path} successful!\n")
        os.mkdir(path)
    for idx in tqdm(range(nums)):
        doGet(path,name,idx,url,{},headers)



