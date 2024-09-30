import torch
from models.pointnet_ex import * 
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.point_draw import *
from utils.loss_functions import *

torch.set_printoptions(edgeitems=10000)

model = PointGenNet1(512, 512, 512)
model.load_state_dict(torch.load("model_weights\pointgennet1_1024_512_6_pointbatch512_scheduler.pth"))
model.to("cuda")
#model.eval()

model_dict = model.state_dict()
print(model_dict.keys())
print(model_dict["main.main.0.main.0.BatchNorm.running_mean"])
print(model_dict["main.main.0.main.0.BatchNorm.running_var"])
print("-"*50)
print(model_dict["main.main.12.BatchNorm.running_mean"])
print(model_dict["main.main.12.BatchNorm.running_var"])

X_train = []
for i, name in enumerate(["airplane", "ambulance", "backpack", "banana", "bicycle", "birthday cake", "book", "bus", "camera", "cat", "cloud", "dog", "helicopter", "house", "octopus", "sailboat", "The Eiffel Tower", "The Great Wall of China", "tree", "violin"]):
    with open("D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_quarter/512point_quarter_"+name+".pickle", mode='br') as f:
        tmp = pickle.load(f)
        X_train += tmp

"""
for pic in X_train:
    for i in pic:
        points_show(i)
    print("finish")
"""
class Point_Dataset0(Dataset):
    def __init__(self, X):
        self.X = []
        for pic in X:
            for i in range(len(pic)):
                self.X.append(torch.tensor(pic[i], dtype=torch.float32))
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

dataset = Point_Dataset0(X_train)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

N = int(len(train_dataloader)/100)
print(N)
per = 0
count = 0
loss_fn = ChamferDistance()
#loss_fn = torch.nn.MSELoss()

for i, x in enumerate(train_dataloader):
    a = torch.isnan(model(x.to("cuda"))).sum()
    if a>0:
        points_show(x)
        print(a)
        tmp = [p[1]-p[0] for p in x[0]]
        print(max(tmp))
        print(min(tmp))
        count+=1
        print("count:", count)
    else:
        pic = model(x.to("cuda"))
        for j in range(1):
            points_show(pic[j].to("cpu").detach().numpy())
        
    if i%N==0:
        print(per)
        per+=1