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
        self.e11 = nn.Conv1d(3, 48, kernel_size=9, padding=4, padding_mode='replicate') # output: 1x256x48
        self.e12 = nn.Conv1d(48, 48, kernel_size=9, padding=4, padding_mode='replicate') # output: 1x256x48
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x128x48

        # input: 1x128x48
        self.e21 = nn.Conv1d(48, 96, kernel_size=7, padding=3, padding_mode='replicate') # output: 1x128x96
        self.e22 = nn.Conv1d(96, 96, kernel_size=7, padding=3, padding_mode='replicate') # output: 1x128x96
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x64x96

        # input: 1x64x128
        self.e31 = nn.Conv1d(96, 192, kernel_size=7, padding=3, padding_mode='replicate') # output: 1x64x192
        self.e32 = nn.Conv1d(192, 192, kernel_size=7, padding=3, padding_mode='replicate') # output: 1x64x192
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x32x192

        # input: 1x32x192
        self.e41 = nn.Conv1d(192, 384, kernel_size=5, padding=2, padding_mode='replicate') # output: 1x32x384
        self.e42 = nn.Conv1d(384, 384, kernel_size=5, padding=2, padding_mode='replicate') # output: 1x32x384
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x16x384

        # input: 1x16x384
        self.e51 = nn.Conv1d(384, 768, kernel_size=5, padding=2, padding_mode='replicate') # output: 1x16x768
        self.e52 = nn.Conv1d(768, 768, kernel_size=5, padding=2, padding_mode='replicate') # output: 1x16x768
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2) # output: 1x8x768

        # input: 1x8x1536
        self.e61 = nn.Conv1d(768, 1536, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x8x1536
        self.e62 = nn.Conv1d(1536, 1536, kernel_size=3, padding=1, padding_mode='replicate') # output: 1x8x1536

        # Decoder
        self.upconv0 = nn.ConvTranspose1d(1536, 768, kernel_size=2, stride=2)
        self.d01 = nn.Conv1d(1536, 768, kernel_size=5, padding=2, padding_mode='replicate')
        self.d02 = nn.Conv1d(768, 768, kernel_size=5, padding=2, padding_mode='replicate')

        self.upconv1 = nn.ConvTranspose1d(768, 384, kernel_size=2, stride=2)
        self.d11 = nn.Conv1d(768, 384, kernel_size=5, padding=2, padding_mode='replicate')
        self.d12 = nn.Conv1d(384, 384, kernel_size=5, padding=2, padding_mode='replicate')

        self.upconv2 = nn.ConvTranspose1d(384, 192, kernel_size=2, stride=2)
        self.d21 = nn.Conv1d(384, 192, kernel_size=7, padding=3, padding_mode='replicate')
        self.d22 = nn.Conv1d(192, 192, kernel_size=7, padding=3, padding_mode='replicate')

        self.upconv3 = nn.ConvTranspose1d(192, 96, kernel_size=2, stride=2)
        self.d31 = nn.Conv1d(192, 96, kernel_size=7, padding=3, padding_mode='replicate')
        self.d32 = nn.Conv1d(96, 96, kernel_size=7, padding=3, padding_mode='replicate')

        self.upconv4 = nn.ConvTranspose1d(96, 48, kernel_size=2, stride=2)
        self.d41 = nn.Conv1d(96, 48, kernel_size=9, padding=4, padding_mode='replicate')
        self.d42 = nn.Conv1d(48, 48, kernel_size=9, padding=4, padding_mode='replicate')

        # Output layer
        self.outconv = nn.Conv1d(48, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        # print("xe11: ", xe11.shape)
        xe12 = relu(self.e12(xe11))
        # print("xe12: ", xe12.shape)
        xp1 = self.pool1(xe12)
        # print("xp1: ", xp1.shape)

        xe21 = relu(self.e21(xp1))
        # print("xe21: ", xe21.shape)
        xe22 = relu(self.e22(xe21))
        # print("xe22: ", xe22.shape)
        xp2 = self.pool2(xe22)
        # print("xp2: ", xp2.shape)

        xe31 = relu(self.e31(xp2))
        # print("xe31: ", xe31.shape)
        xe32 = relu(self.e32(xe31))
        # print("xe32: ", xe32.shape)
        xp3 = self.pool3(xe32)
        # print("xp3: ", xp3.shape)

        xe41 = relu(self.e41(xp3))
        # print("xe41: ", xe41.shape)
        xe42 = relu(self.e42(xe41))
        # print("xe42: ", xe42.shape)
        xp4 = self.pool4(xe42)
        # print("xp4: ", xp4.shape)

        xe51 = relu(self.e51(xp4))
        # print("xe51: ", xe51.shape)
        xe52 = relu(self.e52(xe51))
        # print("xe52: ", xe52.shape)
        xp5 = self.pool5(xe52)
        # print("xp4: ", xp4.shape)
        
        xe61 = relu(self.e61(xp5))
        # print("xe51: ", xe51.shape)
        xe62 = relu(self.e62(xe61))
        # print("xe52: ", xe52.shape)

        # Decoder
        xu0 = self.upconv0(xe62)
        # print("xu1: ", xu1.shape)
        xu01 = torch.cat([xu0, xe52], dim=1)
        # print("xu11: ", xu11.shape)
        xd01 = relu(self.d01(xu01))
        # print("xd11: ", xd11.shape)
        xd02 = relu(self.d02(xd01))
        # print("xd12: ", xd12.shape)

        xu1 = self.upconv1(xd02)
        # print("xu1: ", xu1.shape)
        xu11 = torch.cat([xu1, xe42], dim=1)
        # print("xu11: ", xu11.shape)
        xd11 = relu(self.d11(xu11))
        # print("xd11: ", xd11.shape)
        xd12 = relu(self.d12(xd11))
        # print("xd12: ", xd12.shape)

        xu2 = self.upconv2(xd12)
        # print("xu2: ", xu2.shape)
        xu22 = torch.cat([xu2, xe32], dim=1)
        # print("xu22: ", xu22.shape)
        xd21 = relu(self.d21(xu22))
        # print("xd21: ", xd21.shape)
        xd22 = relu(self.d22(xd21))
        # print("xd22: ", xd22.shape)

        xu3 = self.upconv3(xd22)
        # print("xu3: ", xu3.shape)
        xu33 = torch.cat([xu3, xe22], dim=1)
        # print("xu33: ", xu33.shape)
        xd31 = relu(self.d31(xu33))
        # print("xd31: ", xd31.shape)
        xd32 = relu(self.d32(xd31))
        # print("xd32: ", xd32.shape)

        xu4 = self.upconv4(xd32)
        # print("xu4: ", xu4.shape)
        xu44 = torch.cat([xu4, xe12], dim=1)
        # print("xu44: ", xu44.shape)
        xd41 = relu(self.d41(xu44))
        # print("xd41: ", xd41.shape)
        xd42 = relu(self.d42(xd41))
        # print("xd42: ", xd42.shape)

        # Output layer
        out = self.outconv(xd42)

        return out
    

if __name__ == "__main__":
    input = torch.Tensor(1, 1, 3, 256)
    model = UNet(3)
    output = model(input[0])
    print(output)