import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import torchvision.transforms.functional as functional
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
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottom(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.last(x)
