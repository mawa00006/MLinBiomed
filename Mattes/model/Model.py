# Author: Mattes Warning
import torch
import torch.nn as nn


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUnit, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)                                                     
        )

    def forward(self, inputs):
        conv_output = self.conv_layer2(self.conv_layer1(inputs))
        return conv_output


class Encoder(nn.Module):
    def __init__(self, input_channels) -> None:
        super(Encoder, self).__init__()
        
        self.ConvUnit1 = ConvUnit(input_channels, 16) # 28x28
        self.ConvUnit2 = ConvUnit(16, 32) # 14x14
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
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
        
        out_conv1 = self.maxpool(self.ConvUnit1(inputs))  # 28x28 -> 14x14
        out_conv2 = self.maxpool(self.ConvUnit2(out_conv1))  # 14x14 -> 7x7
        
        flattened = torch.flatten(out_conv2, 1)
        out_fc = self.fc(flattened)
        
        return out_fc
        

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.ImageEncoder = Encoder(input_channels=3)
        self.CannyEncoder = Encoder(input_channels=3)
        
        self.out = nn.Linear(512, 7)

    def forward(self, imgs, canny_imgs):
        
        img_encoding = self.ImageEncoder(imgs)
        canny_encoding = self.CannyEncoder(canny_imgs)
        
        combined = torch.cat((img_encoding, canny_encoding), dim=1)
        out = self.out(combined)

        return out
    
    
class Model2(nn.Module):

    def __init__(self):
        super(Model2, self).__init__()
        
        self.ImageEncoder = Encoder(input_channels=3)
        
        self.fc = nn.Linear(256, 7)
        
    def forward(self, imgs):
        
        out = self.fc(self.ImageEncoder(imgs))
        return out
