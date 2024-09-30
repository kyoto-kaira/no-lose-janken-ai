import pickle
import numpy as np
import pandas as pd
import glob
import time
import torch

def point_complement(p1, p2, num):
    return np.concatenate([np.linspace(p2[0], p1[0], num, endpoint=False)[::-1].reshape(-1,1), np.linspace(p2[1], p1[1], num, endpoint=False)[::-1].reshape(-1,1)], axis=1)

def stroke_complement(stroke, num):
    num_point = len(stroke)-1
    if num_point == 0:
        return np.array(stroke.tolist()*(num+1))
    #stroke = np.array(stroke)
    a, b = divmod(num, num_point)
    tmp = np.array([stroke[0]])
    for i in range(num_point):
        tmp = np.concatenate([tmp, point_complement(stroke[i], stroke[i+1], a+2 if i<b else a+1)])
    return tmp

def pic_allstroke_complement(pic, num):
    data = np.array([[]])
    for i in range(len(pic)):
        num_point = num - sum([len(x) for x in pic[:i+1]])
        a, b = divmod(num_point, i+1)
        tmp = np.array([[]])
        for j in range(i+1):
            if j==0:
                tmp = stroke_complement(pic[j], a+1 if j<b else a)
            else:
                tmp = np.concatenate([tmp, stroke_complement(pic[j], a+1 if j<b else a)])
        if i==0:
            data = tmp.reshape(1, -1, 2)
        else:
            data = np.concatenate([data, tmp.reshape(1, -1, 2)], axis=0)
    return data

def pic_complement(pic, num):
    num_point = num - sum([len(x) for x in pic]) - len(pic) - 1
    a, b = divmod(num_point, len(pic))
    tmp = np.array([[-1, -1]])
    for j in range(len(pic)):
        tmp = np.concatenate([tmp, stroke_complement(pic[j], a+1 if j<b else a), [[-1,-1]]])
    data = tmp.reshape(-1, 2)
    return data

def craete_data(name, num_points):
    files = ["D:\Python/NF_2024\quick_draw\data\exp\simplified/ndjson/full_simplified_"+name+".ndjson"]
    all_data = []
    for file in files:
        data = pd.read_json(file, lines=True)
        print(len(data))
        data = data["drawing"][data["recognized"]]
        print(len(data))
        for i in data:
            tmp = []
            if len(i)==1 or len(i)>20:
                continue
            for j in i:
                tmp.append(np.array(j, dtype=float).T)
            if sum([len(x) for x in tmp])<=num_points:
                all_data.append(pic_allstroke_complement(tmp, num_points))

    print(len(all_data))

    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/512point/512point_'+name+'.pickle', mode='wb') as f:
        pickle.dump(all_data, f)

def convert_alldata():
    data_X = []
    files = glob.glob("D:\Python/NF_2024\quick_draw\data\exp\simplified/512point/*.pickle")
    print(files)
    for file in files:
        with open(file, mode='br') as f:
            tmp = pickle.load(f)
            tmp = tmp[:int(len(tmp)/4)]
            for l in tmp:
                data_X.append(l)
        print("a")
    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/all/512point_quarter_X.pickle', mode='wb') as f:
        pickle.dump(data_X, f)

def convert_data(name):
    data_X = []
    file = "D:\Python/NF_2024\quick_draw\data\exp\simplified/512point/512point_"+name+".pickle"
    with open(file, mode='br') as f:
        tmp = pickle.load(f)
        tmp = tmp[:int(len(tmp)/4)]
        for l in tmp:
            data_X.append(l)
    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_quarter/512point_quarter_'+name+'.pickle', mode='wb') as f:
        pickle.dump(data_X, f)

def create_transformer_data(name):
    all_data = []
    file = "D:\Python/NF_2024\quick_draw\data\exp\simplified/ndjson/full_simplified_"+name+".ndjson"
    data = pd.read_json(file, lines=True)
    data = data["drawing"][data["recognized"]]
    for i in data:
        tmp = []
        if len(i)==1 or len(i)>20:
            continue
        for j in i:
            tmp.append(np.array(j, dtype=float).T)
        if sum([len(x) for x in tmp])<=(511-len(tmp)):
                all_data.append(pic_complement(tmp, 512))
    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_trans/512point_'+name+'.pickle', mode='wb') as f:
        pickle.dump(all_data, f)

def create_diffdata(name, num_point):
    data_X = []
    data_y = []
    file = "D:\Python/NF_2024\quick_draw\data\exp\simplified/ndjson/full_simplified_"+name+".ndjson"
    data = pd.read_json(file, lines=True)
    print(len(data))
    data = data["drawing"][data["recognized"]]
    print(len(data))
    for i in data[:int(len(data)/4)]:
        tmp = []
        if len(i)==1 or len(i)>20:
            continue
        for j in i:
            tmp.append(np.array(j, dtype=float).T)
        if sum([len(x) for x in tmp])<=num_point:
            data_X.append(pic_allstroke_complement(tmp, num_point)[:-1])
            data_y.append(pic_allstroke_complement(tmp[::-1], num_point)[-2::-1])
    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_diff_quarter/X/512point_'+name+'.pickle', mode='wb') as f:
        pickle.dump(data_X, f)
    with open('D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_diff_quarter/y/512point_'+name+'.pickle', mode='wb') as f:
        pickle.dump(data_y, f)

for name in ["airplane", "ambulance", "backpack", "banana", "bicycle", "birthday cake", "book", "bus", "camera", "cat", "cloud", "dog", "helicopter", "house", "octopus", "sailboat", "The Eiffel Tower", "The Great Wall of China", "tree", "violin"]:
    print(name)
    create_diffdata(name, 512)

"""
512
airplane 151623 -> 125152 306
ambulance 148004 -> 133318 650
backpack 125801 -> 117159 327
banana 307936 -> 170060 273
bicycle 126527 -> 121821 793
birthday cake 144982 -> 136354 582
book 119364 -> 109695 424
bus 166208 -> 143817 783
camera 128772 -> 124425 237
cat 123202 -> 102229 664
cloud 120265 -> 25026 40
dog 152159 -> 132964 690
helicopter 159938 -> 137006 731
house 135420 -> 117843 176
octopus 150152 -> 94620 286
sailboat 136506 -> 130882 149
The Eiffel Tower 134801 -> 119046 236
The Great Wall of China 193015 -> 135787 453
tree 144721 -> 124263 249
violin 217260 -> 172919 782
"""
