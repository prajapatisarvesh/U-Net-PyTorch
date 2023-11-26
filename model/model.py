import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

class DoubleConvolution(BaseModel):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_convolution = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    

    def forward(self, x):
        return self.double_convolution(x)



class UNet(BaseModel):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        '''
        To-Do
        '''
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConvolution(in_channels, feature))
            in_channels = feature

        for feature in features[::-1]:
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConvolution(feature*2, feature))
        self.bottom = DoubleConvolution(features[-1], features[-1] * 2)
        self.last = nn.Conv2d(features[0], out_channels, kernel_size=1)
        

    '''
    Forward function for tensors
    '''
    def forward(self, x):
        '''
        To-Do
        '''
        raise NotImplementedError