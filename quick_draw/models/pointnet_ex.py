import torch
import torch.nn as nn
import torch.optim as optim

class PointNorm(nn.Module):
    def __init__(self, num_point, dim):
        super(PointNorm, self).__init__()
        self.num_point = num_point
        self.dim = dim
    
    def forward(self, input):
        input = input.reshape(-1, self.num_point, self.dim)
        mean = input.mean(dim=1).unsqueeze(dim=1).expand(-1, self.num_point, self.dim)
        var = input.var(dim=1).unsqueeze(dim=1).expand(-1, self.num_point, self.dim)
        return ((input-mean)/torch.sqrt(var+1e-4)).reshape(-1, self.dim)

class NonLinear(nn.Module):
    def __init__(self, input_channels, output_channels, num_point=None):
        super(NonLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_point = output_channels if num_point==None else num_point*output_channels

        self.main = nn.Sequential(
            nn.Linear(self.input_channels, self.output_channels),
            nn.ReLU(inplace=True))
        self.BatchNorm = nn.BatchNorm1d(self.num_point)

    def forward(self, input_data):
        input_data = self.main(input_data).reshape(-1, self.num_point)
        return self.BatchNorm(input_data).reshape(-1, self.output_channels)

class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, input_data):
        out = input_data.view(-1, self.num_channels, self.num_points)
        out = self.main(out)
        out = out.view(-1, self.num_channels)
        return out

class InputTNet(nn.Module):
    def __init__(self, num_points):
        super(InputTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(2, 64, num_points),
            NonLinear(64, 128, num_points),
            NonLinear(128, 1024, num_points),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 2, 2)
        out = torch.matmul(input_data.view(-1, self.num_points, 2), matrix)
        out = out.view(-1, 2)
        return out

class FeatureTNet(nn.Module):
    def __init__(self, num_points):
        super(FeatureTNet, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            NonLinear(64, 64, num_points),
            NonLinear(64, 128, num_points),
            NonLinear(128, 1024, num_points),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            NonLinear(512, 256),
            nn.Linear(256, 4096)
        )

    # shape of input_data is (batchsize x num_points, channel)
    def forward(self, input_data):
        matrix = self.main(input_data).view(-1, 64, 64)
        out = torch.matmul(input_data.view(-1, self.num_points, 64), matrix)
        out = out.view(-1, 64)
        return out

class PointNet(nn.Module):
    def __init__(self, num_points, num_gen_point):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_gen_point = num_gen_point

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonLinear(2, 64, num_points),
            NonLinear(64, 64, num_points),
            FeatureTNet(self.num_points),
            NonLinear(64, 64, num_points),
            NonLinear(64, 128, num_points),
            NonLinear(128, 1024, num_points),
            MaxPool(1024, self.num_points),
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, self.num_gen_point),
            )

    def forward(self, input_data):
        return self.main(input_data)
    
class PointGenNet(nn.Module):
    def __init__(self, num_input_point, num_gen_point, dim_last_hidden):
        super(PointGenNet, self).__init__()
        self.num_gen_point = num_gen_point
        self.main = PointNet(num_input_point, dim_last_hidden)
        self.generater = nn.Linear(dim_last_hidden, self.num_gen_point*2)
    
    def forward(self, input_data):
        input_data = self.main(input_data.reshape(-1,2))
        return self.generater(input_data).reshape(-1, self.num_gen_point, 2)
    
class PointGenNet1(nn.Module):
    def __init__(self, num_input_point, num_gen_point, dim_last_hidden):
        super(PointGenNet1, self).__init__()
        self.num_gen_point = num_gen_point
        self.main = PointNet(num_input_point, dim_last_hidden)
        self.generater = nn.Linear(dim_last_hidden, self.num_gen_point*2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_data):
        input_data = self.main(input_data.reshape(-1,2))
        return self.sigmoid(self.generater(input_data)).reshape(-1, self.num_gen_point, 2) *255
    
class PointBase(nn.Module):
    def __init__(self, num_points):
        super(PointBase, self).__init__()
        self.num_points = num_points

        self.main = nn.Sequential(
            InputTNet(self.num_points),
            NonLinear(2, 64),
            NonLinear(64, 64),
            FeatureTNet(self.num_points),
            NonLinear(64, 64),
            NonLinear(64, 128),
            NonLinear(128, 1024),
            MaxPool(1024, self.num_points)
            )

    def forward(self, input_data):
        return self.main(input_data)

class PointGenClaNet(nn.Module):
    def __init__(self, num_input_point, num_gen_point, num_class):
        super(PointGenClaNet, self).__init__()
        self.num_gen_point = num_gen_point

        self.pointbase = PointBase(num_input_point)
        self.classify = nn.Sequential(
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, num_class)
        )
        self.generater = nn.Sequential(
            NonLinear(1024, 512),
            nn.Dropout(p = 0.3),
            NonLinear(512, 256),
            nn.Dropout(p = 0.3),
            NonLinear(256, 256),
            nn.Linear(256, self.num_gen_point*2)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):
        input_data = self.pointbase(input_data.reshape(-1,2))
        return self.sigmoid(self.generater(input_data)).reshape(-1, self.num_gen_point, 2)*255, self.classify(input_data)
