import torch
import torch.nn as nn

class ChamferDistance(nn.Module):
    def __init__(self):
        super(ChamferDistance, self).__init__()

    def forward(self, points1, points2):
        dist_matrix = self.pairwise_dist(points1, points2)

        min_dist1, _ = torch.min(dist_matrix, dim=2)  # (batch_size, num_points1)
        min_dist2, _ = torch.min(dist_matrix, dim=1)  # (batch_size, num_points2)

        chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)
        return chamfer_dist

    def pairwise_dist(self, points1, points2):
        points1_expanded = points1.unsqueeze(2)  # (batch_size, num_points1, 1, 2)
        points2_expanded = points2.unsqueeze(1)  # (batch_size, 1, num_points2, 2)

        dist_matrix = torch.sum((points1_expanded - points2_expanded) ** 2, dim=-1)  # (batch_size, num_points1, num_points2)
        return dist_matrix