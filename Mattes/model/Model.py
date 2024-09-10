# Author: Mattes Warning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, BatchNorm2d, Linear, AvgPool1d


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)                                                     
        )

    def forward(self, inputs):

        conv_output = self.conv_layer2(self.conv_layer1(inputs))

        return conv_output

class Encoder(nn.Module):
    def __init__(self, input_channels) -> None:
        super(Encoder, self).__init__()
        
        self.ConvUnit1 = ConvUnit(input_channels,16) # 28x28
        self.ConvUnit2 = ConvUnit(16,32) # 14x14
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2) 
        
        self.fc = nn.Sequential(nn.Linear(32*7*7, 1024),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2))
        
        
    def forward(self, inputs):
        
        out_conv1 = self.maxpool(self.ConvUnit1(inputs)) # 28x28 -> 14x14
        out_conv2 = self.maxpool(self.ConvUnit2(out_conv1)) # 14x14 -> 7x7
        
        flattened = torch.flatten(out_conv2, 1)
        out_fc = self.fc(flattened)
        
        return out_fc
        
        
        
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.ImageEncoder = Encoder(input_channels=3)
        self.CannyEncoder = Encoder(input_channels=3)
        
        self.CrossAttention = nn.MultiheadAttention(256, num_heads=16)
        
        self.key_transform = nn.Linear(256, 256)
        self.query_transform = nn.Linear(256, 256)
        self.value_transform = nn.Linear(256, 256)
        
        self.fc = nn.Linear(256,7)
        self.out = nn.Linear(512,7)
                              




    def forward(self, imgs, canny_imgs):
        
        img_encoding = self.ImageEncoder(imgs)
        canny_encoding = self.CannyEncoder(canny_imgs)
        
        #key = self.key_transform(img_encoding)
        #query = self.key_transform(canny_encoding)
        #value = self.key_transform(img_encoding)
        
        #cross_attn_output, attn_output_weights = self.CrossAttention(query, key, value)
        
        #cross_attn_output.flatten()
        #out = self.fc(cross_attn_output)
        
        combined = torch.cat((img_encoding, canny_encoding), dim=1)
        out = self.out(combined)

        return out
    
    
class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        
        self.ImageEncoder = Encoder(input_channels=3)
        
        self.fc = nn.Linear(256,7)
        
    def forward(self, imgs):
        
        out = self.fc(self.ImageEncoder(imgs))
        return out
