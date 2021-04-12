import urllib.request, urllib.parse, urllib.error
import json
import hashlib
import pandas as pd
import os
import re
import pandas as pd
import jieba
import re
from sklearn import *
from numpy import *
import numpy as np
import jieba
MyAK = 'KXLsvqI2US66a1X9Nq3wt51paYW6L8zL'
MySK = 'jlLAa3YhCLH9uL4PVhpUdzeq5W1gjPFN'

#处理得到url
def get_url(name):
    #GET请求 http://api.map.baidu.com/geocoding/v3/?address=北京市海淀区上地十街10号&output=json&ak=您的ak&callback=showLocation
    queryStr = '/geocoding/v3/?address={}&output=json&ak={}'.format(name,MyAK)
    # 对queryStr进行转码，safe内的保留字符不转换
    encodedStr = urllib.parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
    # 在最后追加sk
    rawStr = encodedStr + MySK
    # 计算sn
    sn = (hashlib.md5(urllib.parse.quote_plus(rawStr).encode("utf8")).hexdigest())
    #由于URL里面含有中文，所以需要用parse.quote进行处理，然后返回最终可调用的url
    url = urllib.parse.quote("http://api.map.baidu.com" + queryStr + "&sn=" + sn, safe="/:=&?#+!$,;'@()*[]")
    return url

#得到json数据
def get_json(url):
    # 从API读取数据
    req = urllib.request.urlopen(url)
    res = req.read().decode()

    # 解析数据
    try:
        # 将 JSON 对象转换为 Python 字典
        json_data = json.loads(res)
    except:
        json_data = None
    if not json_data or 'status' not in json_data or json_data['status'] != 0:
        print('json数据获取失败')
    else:
        #输出Json数据
        json.dumps(json_data, indent=4, ensure_ascii=False)
    return json_data

# 获取经纬度坐标
def get_lat(json_data):
    lat = json_data["result"]["location"]["lat"]
    return lat


def get_lng(json_data):
    lng = json_data["result"]["location"]["lng"]
    return lng




if __name__ == '__main__':
    #得到经纬度

    df = pd.read_excel(r'E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\Prediction\Shanghai_location_cleaned.xlsx')
    latitude = []
    longitude = []
    for text in df['Text']:
        print(text)
        url = get_url(text)
        json_data = get_json(url)
        latitude.append(get_lat(json_data))
        longitude.append(get_lng(json_data))
    df['latitude'] = latitude
    df['longitude'] = longitude
    print(df)
    df.to_excel(r'E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\Prediction\Shanghai_final.xlsx')
