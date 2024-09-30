import pickle
from utils.datasets import *
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from models.pointnet import *
from utils.loss_functions import *
from sklearn.model_selection import train_test_split
import time
import wandb

X_train = []
cla = []
for i, name in enumerate(["airplane", "ambulance", "backpack", "banana", "bicycle", "birthday cake", "book", "bus", "camera", "cat", "cloud", "dog", "helicopter", "house", "octopus", "sailboat", "The Eiffel Tower", "The Great Wall of China", "tree", "violin"]):
    with open("data\exp\simplified/512point_quarter/512point_quarter_"+name+".pickle", mode='br') as f:
        tmp = pickle.load(f)
        X_train += tmp
        cla += [i]*len(tmp)

X_train, X_test, cla_train, cla_test = train_test_split(X_train, cla, test_size=0.2, stratify=cla)

batch_size = 512
num_points = 512

train_dataset = Point_Dataset1(X_train, cla)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Point_Dataset1(X_test, cla)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

model = PointGenClaNet(num_points, num_points, 20)

optimizer = optim.Adam(model.parameters(), lr=0.05)
genloss_fn = ChamferDistance()
#genloss_fn = nn.MSELoss()
claloss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda")
model.to(device)
model.train()
num_epoch = 7

run = wandb.init(project="QD_pointgencla512")

#s = time.time()
for epoch in range(num_epoch):
    train_loss = 0
    eval_loss = 0
    model.train()
    for X, y_gen, y_cla in train_dataloader:
        optimizer.zero_grad()
        X, y_gen, y_cla = X.to(device), y_gen.to(device), y_cla.to(device)  # データをGPUにおく
        pred_gen, pred_cla = model(X)  # 推論（forward）
        loss_gen = genloss_fn(pred_gen, y_gen)
        loss_cla = claloss_fn(pred_cla, y_cla)
        loss = loss_gen/300+loss_cla
        loss.backward()  # 逆伝搬
        optimizer.step()  # 重みの更新
        train_loss += loss.item()
        print("gen", loss_gen.item())
        print("cla", loss_cla.item())
        run.log({"loss":loss})
        run.log({"loss_gen":loss_gen})
        run.log({"loss_cla":loss_cla})
    print(f"Epoch: {epoch}, Train Loss: {train_loss / len(train_dataloader)}")
    
    # 検証用データを使って、定期的に精度の検証もする
    model.eval()
    with torch.no_grad(): 
        for X, y_gen, y_cla in test_dataloader:
            X, y_gen, y_cla = X.to(device), y_gen.to(device), y_cla.to(device)  # データをGPUにおく
            pred_gen, pred_cla = model(X)  # 推論（forward）
            loss_gen = genloss_fn(pred_gen, y_gen)
            loss_cla = claloss_fn(pred_cla, y_cla)
            loss = loss_gen/300+loss_cla
            eval_loss += loss.item()
            print("gen", loss_gen.item())
            print("cla", loss_cla.item())
            run.log({"eval_loss":loss})
            run.log({"eval_loss_gen":loss_gen})
            run.log({"eval_loss_cla":loss_cla})
    print(f"Evaluation Loss: {eval_loss / len(test_dataloader)}")
torch.save(model.state_dict(), "model_weights/pointgenclanet512_512_7.pth")
#e = time.time()
#print(e-s)

#batch512で1epoch 4350秒 72分
#1epochでも、loss300切るかどうか　Epoch: 0, Train Loss: 487.80872536252326　
#Epoch: 2, Train Loss: 279.6558717872763
#pointgennet512_512_4.pthは torch.save(model) PointGenNet(num_points, num_points, 256)
#pointgennet512_512_5.pthは torch.save(model.state_dict()) PointGenNet(num_points, num_points, 256)

#mseのevalはこんな感じ
"""
0.049074359238147736
0.05057292431592941
0.04901440441608429
53.487510681152344
42491101184.0
0.04893666133284569
0.050290998071432114
0.05017959326505661
0.051275063306093216
0.049564629793167114
0.0509619265794754
0.050299376249313354
0.049811333417892456
4308311.0
0.052149806171655655
0.04931291192770004
0.050634756684303284
0.051047615706920624
0.05032689869403839
0.05165691301226616
0.050018008798360825
0.04997618496417999
0.05008633807301521
0.04958861321210861
0.05237536132335663
0.05019895359873772
0.04917418956756592
0.04969214275479317
0.04801969975233078
0.04978609085083008
5918142976.0
128307483901952.0
0.04965618997812271
0.05044250935316086
0.04957311972975731
0.048251792788505554
0.050120942294597626
0.04896903410553932
1991.287841796875
0.04983747377991676
0.05051996558904648
0.04976337403059006
0.04956623911857605
0.04940562695264816
0.0499597042798996
0.05032818019390106
0.050531934946775436
0.050748296082019806
0.04895215481519699
0.04955060034990311
0.05102672800421715
0.050215259194374084
0.04964403063058853
562939148894208.0
0.04877884313464165
Evaluation Loss: 1.2506238605408774e+16
"""