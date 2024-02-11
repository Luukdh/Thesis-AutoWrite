import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.nn.functional import relu


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        #input: 1x256x3
        self.e11 = nn.Conv1d(3, 64, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x256x64
        self.e12 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x256x64
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x128x64

        # input: 1x128x64
        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x128x128
        self.e22 = nn.Conv1d(128, 128, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x128x128
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x64x128

        # input: 1x64x128
        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x64x256
        self.e32 = nn.Conv1d(256, 256, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x64x256
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x32x256

        # input: 1x32x256
        self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x32x512
        self.e42 = nn.Conv1d(512, 512, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x32x512
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x16x512

        # input: 1x16x512
        self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x16x1024
        self.e52 = nn.Conv1d(1024, 1024, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x16x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1, padding_mode='replicate')
        self.d12 = nn.Conv1d(512, 512, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1, padding_mode='replicate')
        self.d22 = nn.Conv1d(256, 256, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1, padding_mode='replicate')
        self.d32 = nn.Conv1d(128, 128, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.d42 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')

        # Output layer
        self.outconv = nn.Conv1d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xe13 = relu(self.e12(xe12))
        xe14 = relu(self.e12(xe13))
        xp1 = self.pool1(xe14)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xe23 = relu(self.e22(xe22))
        xe24 = relu(self.e22(xe23))
        xp2 = self.pool2(xe24)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xe33 = relu(self.e32(xe32))
        xe34 = relu(self.e32(xe33))
        xp3 = self.pool3(xe34)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xe43 = relu(self.e42(xe42))
        xe44 = relu(self.e42(xe43))
        xp4 = self.pool4(xe44)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        xe53 = relu(self.e52(xe52))
        xe54 = relu(self.e52(xe53))
        
        # Decoder
        xu1 = self.upconv1(xe54)
        xu11 = torch.cat([xu1, xe44], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))
        xd13 = relu(self.d12(xd12))
        xd14 = relu(self.d12(xd13))

        xu2 = self.upconv2(xd14)
        xu22 = torch.cat([xu2, xe34], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))
        xd23 = relu(self.d22(xd22))
        xd24 = relu(self.d22(xd23))

        xu3 = self.upconv3(xd24)
        xu33 = torch.cat([xu3, xe24], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))
        xd33 = relu(self.d32(xd32))
        xd34 = relu(self.d32(xd33))

        xu4 = self.upconv4(xd34)
        xu44 = torch.cat([xu4, xe14], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))
        xd43 = relu(self.d42(xd42))
        xd44 = relu(self.d42(xd43))

        # Output layer
        out = self.outconv(xd44)

        return out
    

if __name__ == "__main__":
    input = torch.Tensor(1, 1, 3, 256)
    model = UNet(3)
    output = model(input[0])
    print(output)