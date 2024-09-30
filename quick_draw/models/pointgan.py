import torch
import torch.nn as nn

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
    def __init__(self, num_points):
        super(PointNet, self).__init__()
        self.num_points = num_points

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
            nn.Dropout(p = 0.3)
            )

    def forward(self, input_data):
        return self.main(input_data)
    
class PointGANGenerator1(nn.Module):
    def __init__(self, num_input_point, num_gen_point, dim_noise=0):
        super(PointGANGenerator1, self).__init__()
        self.num_gen_point = num_gen_point
        self.encoder = PointNet(num_input_point)
        self.generater = nn.Sequential(
            NonLinear(256+dim_noise, 512),
            nn.Dropout(p=0.3),
            NonLinear(512, num_gen_point),
            nn.Dropout(p=0.3),
            nn.Linear(num_gen_point, num_gen_point*2)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_data):#, z):
        input_data = self.encoder(input_data.reshape(-1,2))
        #input_data = torch.cat([input_data, z], dim=1)
        return self.sigmoid(self.generater(input_data)).reshape(-1, self.num_gen_point, 2) *255
    
class PointGANDiscriminator1(nn.Module):
    def __init__(self, num_input_point, num_gen_point):
        super(PointGANDiscriminator1, self).__init__()
        self.encoder1 = PointNet(num_gen_point)
        self.encoder2 = PointNet(num_input_point)
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, full, partial):
        full = self.encoder1(full)
        partial = self.encoder2(partial)
        return self.classifier(torch.cat([full, partial], dim=1))