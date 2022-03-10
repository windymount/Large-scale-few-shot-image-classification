from turtle import forward
import torch.nn as nn
import torch


class baseline_advanced(nn.Module):
    def __init__(self, output_feature, backbone_weights) -> None:
        super().__init__()
        self.weight = nn.parameter.Parameter(nn.functional.normalize(backbone_weights))
        self.bias = nn.parameter.Parameter(torch.zeros((output_feature, )))
    
    def forward(self, x):
        x = nn.functional.normalize(x)
        norm_weight = nn.functional.normalize(self.weight)
        return nn.functional.linear(x, norm_weight, self.bias)
