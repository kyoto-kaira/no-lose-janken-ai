import numpy as np
import pandas as pd
import glob
import pickle
import utils.datasets
import torch
import torch.nn as nn
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.lstm import *
from models.pointnet import *
from sklearn.model_selection import train_test_split

data_X = []
files = glob.glob("data\exp\simplified\list\*.pickle")
for file in files:
    with open(file, mode='br') as f:
        tmp = pickle.load(f)
        for l in tmp:
            data_X.append(torch.tensor(l, dtype=torch.float32))

X_train, X_test = train_test_split(data_X, test_size=0.2)
batch_size = 1024

def collate_fn(batch):
    bx, by = list(zip(*batch))
    bx = pad_sequence(list(bx), batch_first=True, padding_value=-3)
    by = pad_sequence(list(by), batch_first=True, padding_value=-3)
    return bx, by

train_dataset = utils.datasets.Dataset0(X_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataset = utils.datasets.Dataset0(X_test)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = LSTM_Net0(2, 100)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index = -3)

device = torch.device("cuda")
model.to(device)
model.train()
num_epoch = 5

for epoch in range(num_epoch):
    train_loss = 0
    eval_loss = 0
    for X, y in train_dataloader:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)  # データをGPUにおく
        x1_pred, x2_pred = model(X)  # 推論（forward）
        loss = loss_fn(torch.cat((x1_pred, x2_pred), dim=1).reshape(-1, 258), torch.permute(y, (0,2,1)).reshape(-1))
        loss.backward()  # 逆伝搬
        optimizer.step()  # 重みの更新
        train_loss += loss.item()
        print(loss)
    print(f"Epoch: {epoch}, Train Loss: {train_loss / len(train_dataloader)}")
    
    # 検証用データを使って、定期的に精度の検証もする
    correct=0
    num=0
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():  # 勾配計算しなくて良い
            x1_pred, x2_pred = model(X)  # 推論（forward）
            loss = loss_fn(torch.cat((x1_pred, x2_pred), dim=1).reshape(-1, 258), torch.permute(y, (0,2,1)).reshape(-1))
            eval_loss += loss.item()
    print(f"Evaluation Loss: {eval_loss / len(test_dataloader)}")
