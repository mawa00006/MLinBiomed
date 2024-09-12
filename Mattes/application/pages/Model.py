import pandas as pd
import streamlit as st


st.title("Model Architecture")

st.subheader("Dual-Modality CNN")
st.markdown(""" Our proposed architecture combines two image modalities. It used the original image as well as 
an edge representation of the image calculated with the Canny Edge algorithm. Both are encoded using an [Encoder](#encoder).""")
st.code("""class Model(nn.Module):
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
        return out""")

st.subheader("Encoder")

st.code("""class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        
        self.ConvUnit1 = ConvUnit(input_channels,16)
        self.ConvUnit2 = ConvUnit(16,32) 
        
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
        
        return out_fc""")



st.markdown(""" ### ConvUnit\n
    The Basic building block of our [Encoder](#encoder) CNN is the ConvUnit. It consists of two convolutional layers, each with a 
    filter size of 3x3 and stride 1. Zero padding of size one is added to each side of the input to preserve dimensionality.
    The convolutional operations are followed by a ReLU activation function. To regularize 
    the model during the training process a dropout of 0.2 is used.""")

st.code("""class ConvUnit(nn.Module):
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
        return conv_output""")

st.title("Training Process")

hyperparameters = pd.DataFrame({"Parameter": ["Learning Rate"], "Value": [0.1]})
st.table(hyperparameters)


st.title("Evaluation")

df = pd.read_csv("Accuracies.csv")

st.bar_chart(df, x="Class", y="Accuracy", horizontal=True)
