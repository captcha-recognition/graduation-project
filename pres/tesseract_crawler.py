# 验证码采集
#!/usr/bin/env python
# encoding=utf-8
import time
import requests
from tqdm import  tqdm
import os
import json

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
            #print(text)
            f.write(text)


def handle_html(path,name,idx, base_url, params,headers):
    url = base_url
    req = requests.get(url = url,headers = headers)
    if(req.status_code == 200):
        text = req.content
        js = json.loads(text)
        base_url = url + '/' + js['results']['captchaId']
        resp = requests.get(url = base_url,headers = headers)
        if(resp.status_code == 200):
            text = resp.content
            with open(f'{path}/{name}_{idx}_{time.time()}.png','wb') as f:
                # print(text)
                f.write(text)


if __name__ == '__main__':
    headers = {
        'Connection': 'keep-alive',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
        'Accept-Encoding': 'gzip, deflate,br',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    url = 'https://anhui.12388.gov.cn/xinfang/servlet/randimg?0.7848386119499069'
    name = split_name(url)
    nums = 200
    path = f"/Users/sjhuang/Documents/docs/dataset/crawlers/{name}"
    if not os.path.exists(path):
        os.mkdir(path)
    for idx in tqdm(range(nums)):
        # handle_html(path,name,idx,url,{},headers)
        # time.sleep(2)
        doGet(path,name,idx,url,{},headers)



