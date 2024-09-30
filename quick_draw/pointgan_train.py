import pickle
from utils.datasets import *
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from models.pointgan import *
from utils.loss_functions import *
from sklearn.model_selection import train_test_split
import time
import wandb

X_train = []
cla = []
for i, name in enumerate(["airplane", "ambulance", "backpack", "banana", "bicycle", "birthday cake", "book", "bus", "camera", "cat", "cloud", "dog", "helicopter", "house", "octopus", "sailboat", "The Eiffel Tower", "The Great Wall of China", "tree", "violin"]):
    with open("D:\Python/NF_2024\quick_draw\data\exp\simplified/512point_quarter/512point_quarter_"+name+".pickle", mode='br') as f:
        tmp = pickle.load(f)
        X_train += tmp
        cla += [i]*len(tmp)
X_train, X_test = train_test_split(X_train, test_size=0.1, stratify=cla)

batch_size = 128
num_points = 512
#noise_dim = 64
num_gen_train = 64

train_dataset = Point_Dataset0(X_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = Point_Dataset0(X_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model_G = PointGANGenerator1(num_points, num_points)#, noise_dim)
model_G.load_state_dict(torch.load("model_weights\pointgan_G_pre3_2.pth"))
model_D = PointGANDiscriminator1(num_points, num_points)

optimizer_G = optim.Adam(model_G.parameters(), lr=0.005)
optimizer_D = optim.Adam(model_D.parameters(), lr=0.001)
loss_fn = ChamferDistance()
criterion = nn.BCELoss()

device = torch.device("cuda")
model_D.to(device)
model_G.to(device)
model_D.train()
model_G.train()
num_epoch = 10

run = wandb.init(project="QD_pointgan1")

#s = time.time()
for epoch in range(num_epoch):
    eval_loss = 0
    model_G.train()
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        optimizer_D.zero_grad()
        real_labels = torch.full((batch_size, 1), 0.9).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        # ノイズベクトルを生成
        #z = torch.randn(batch_size, noise_dim).to(device)

        if i%num_gen_train==0:
            real_loss = criterion(model_D(y, X), real_labels)
            fake_data = model_G(X)#, z)
            fake_loss = criterion(model_D(fake_data, X), fake_labels)
            D_loss = real_loss + fake_loss
            D_loss.backward()
            optimizer_D.step()

        #for i in range(num_gen_train):
        optimizer_G.zero_grad()
        fake_data = model_G(X)#, z)
        G_loss = criterion(model_D(fake_data, X), real_labels)  # ジェネレータはディスクリミネータを騙す必要がある
        G_loss.backward()
        optimizer_G.step()

        print("D_loss", D_loss.item())
        print("G_loss", G_loss.item())
        run.log({"D_loss":D_loss, "G_loss":G_loss})

    torch.save(model_G.state_dict(), "model_weights/pointgan_G2_"+str(epoch+1)+".pth")
    torch.save(model_D.state_dict(), "model_weights/pointgan_D2_"+str(epoch+1)+".pth")
    # 検証用データを使って、定期的に精度の検証もする
    model_G.eval()
    with torch.no_grad(): 
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            #z = torch.zeros(batch_size, noise_dim).to(device)
            pred = model_G(X)#, z)  # 推論（forward）
            loss = loss_fn(pred, y)
            eval_loss += loss.item()
            print(loss.item())
            run.log({"eval_loss":loss})
    print(f"Evaluation Loss: {eval_loss / len(test_dataloader)}")
#torch.save(model.state_dict(), "model_weights/pointgennet1_1024_512_10_pointbatch512_scheduler.pth")
#e = time.time()
#print(e-s)

#batch512で1epoch 4350秒 72分
#1epochでも、loss300切るかどうか　Epoch: 0, Train Loss: 487.80872536252326　
#Epoch: 2, Train Loss: 279.6558717872763
#pointgennet512_512_4.pthは torch.save(model) PointGenNet(num_points, num_points, 256)
#pointgennet512_512_5.pthは torch.save(model.state_dict()) PointGenNet(num_points, num_points, 256)

#BatchNromがバグる原因？
#動画入力ネットワーク向けに使用すると，隣接フレーム同士でデータの相関(もとい類似性)が高いことから，平均・分散の推定が安定せず，学習がなかなか進行しない．
#BatchNormを除くと、trainでも早々に発散してしまった。

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