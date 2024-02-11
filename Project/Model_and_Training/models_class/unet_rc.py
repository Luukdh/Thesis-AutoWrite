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
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x128x64

        # input: 1x128x64
        self.e21 = nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x128x128
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x64x128

        # input: 1x64x128
        self.e31 = nn.Conv1d(128, 256, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x64x256
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x32x256

        # input: 1x32x256
        self.e41 = nn.Conv1d(256, 512, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x32x512
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x16x512

        # input: 1x16x512
        self.e51 = nn.Conv1d(512, 1024, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x16x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(1024, 512, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(512, 256, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(256, 128, kernel_size=3, padding=1, padding_mode='replicate')

        self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(128, 64, kernel_size=3, padding=1, padding_mode='replicate')

        # Output layer
        self.outconv = nn.Conv1d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        # print("xe11: ", xe11.shape)
        xp1 = self.pool1(xe11)
        # print("xp1: ", xp1.shape)

        xe21 = relu(self.e21(xp1))
        # print("xe21: ", xe21.shape)
        xp2 = self.pool2(xe21)
        # print("xp2: ", xp2.shape)

        xe31 = relu(self.e31(xp2))
        # print("xe31: ", xe31.shape)
        xp3 = self.pool3(xe31)
        # print("xp3: ", xp3.shape)

        xe41 = relu(self.e41(xp3))
        # print("xe41: ", xe41.shape)
        xp4 = self.pool4(xe41)
        # print("xp4: ", xp4.shape)

        xe51 = relu(self.e51(xp4))
        # print("xe51: ", xe51.shape)
        
        # Decoder
        xu1 = self.upconv1(xe51)
        # print("xu1: ", xu1.shape)
        xu11 = torch.cat([xu1, xe41], dim=1)
        # print("xu11: ", xu11.shape)
        xd11 = relu(self.d11(xu11))
        # print("xd11: ", xd11.shape)

        xu2 = self.upconv2(xd11)
        # print("xu2: ", xu2.shape)
        xu22 = torch.cat([xu2, xe31], dim=1)
        # print("xu22: ", xu22.shape)
        xd21 = relu(self.d21(xu22))
        # print("xd21: ", xd21.shape)

        xu3 = self.upconv3(xd21)
        # print("xu3: ", xu3.shape)
        xu33 = torch.cat([xu3, xe21], dim=1)
        # print("xu33: ", xu33.shape)
        xd31 = relu(self.d31(xu33))
        # print("xd31: ", xd31.shape)

        xu4 = self.upconv4(xd31)
        # print("xu4: ", xu4.shape)
        xu44 = torch.cat([xu4, xe11], dim=1)
        # print("xu44: ", xu44.shape)
        xd41 = relu(self.d41(xu44))
        # print("xd41: ", xd41.shape)

        # Output layer
        out = self.outconv(xd41)

        return out
    

if __name__ == "__main__":
    input = torch.Tensor(1, 1, 3, 256)
    model = UNet(3)
    output = model(input[0])
    print(output)