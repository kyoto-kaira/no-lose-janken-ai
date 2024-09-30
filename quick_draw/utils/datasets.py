import torch
from torch.utils.data import Dataset

class Dataset0(Dataset):
    def __init__(self, X):
        self.X = X
        tmp = []
        for l in X:
            tmp.append(torch.cat(((l[1:]+2).int(), torch.tensor([[0,0]]))))
        self.y = tmp
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Point_Dataset0(Dataset):
    def __init__(self, X):
        self.X = []
        self.y = []
        for pic in X:
            for i in range(len(pic)-1):
                self.X.append(torch.tensor(pic[i], dtype=torch.float32))
                self.y.append(torch.tensor(pic[-1], dtype=torch.float32))
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class Point_Dataset1(Dataset):
    def __init__(self, X, cla):
        self.X = []
        self.y_gen = []
        self.y_cla = []
        for pic, y in zip(X, cla):
            for i in range(len(pic)-1):
                self.X.append(torch.tensor(pic[i], dtype=torch.float32))
                self.y_gen.append(torch.tensor(pic[-1], dtype=torch.float32))
                self.y_cla.append(y)
        self.y_cla = torch.tensor(self.y_cla)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_gen[idx], self.y_cla[idx]

class Point_Dataset2(Dataset):
    def __init__(self, X, y):
        self.X = []
        self.y = []
        for i in range(len(X)):
            for j in range(len(X[i])):
                self.X.append(torch.tensor(X[i][j], dtype=torch.float32))
                self.y.append(torch.tensor(y[i][j], dtype=torch.float32))
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class Trans_Dataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        y = torch.cat([self.X[idx][1:], torch.tensor([[-1.0, -1.0]])])
        return self.X[idx], y, (y==-1.0).T[0].float()