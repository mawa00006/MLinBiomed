import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.switch_page_button import switch_page

# with st.sidebar:
#     selected = option_menu(
#         menu_title=None,
#         options=["Home", "DermaMNIST", "Model"]
#     )
# if selected == "DermaMNIST":
#     switch_page("dermamnist")
# if selected == "Home":
#     switch_page("main")
#
# selected2 = option_menu( menu_title=None,
#         options=["Home", "DermaMNIST", "Model"],
#         orientation="horizontal"
#     )
# selected2


st.title("Model Architecture")

st.subheader("Dual-Modality CNN")
st.markdown("""We propose a dual-modality convolutional neural network (CNN) that integrates two distinct image 
representations to enhance feature extraction. The model leverages both the raw input image and its corresponding 
edge map, which is computed using the [Canny edge detection algorithm](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html). These two modalities are processed 
independently through separate [Encoders](#encoder). \n

The architecture consists of two encoder modules: one for the original RGB image and another for the edge 
representation. The outputs of both encoders are concatenated and passed through a fully connected layer for 
classification.""")
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
st.markdown("""
The Encoder class extracts features from images through two convolutional layers followed 
by max-pooling, progressively reducing spatial dimensions. The output is flattened and passed through fully connected 
layers with dropout for regularization, producing a compact feature representation. This structure captures both 
local and abstract features of the input image.""")

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

hyperparameters = pd.DataFrame({"Parameter": ["Epochs", "Optimizer", "Learning Rate"], "Value": [200, "Adam", 0.001]})
st.table(hyperparameters)

col1, col2 = st.columns(2)
with col1:
    st.image('imgs/train_loss.png', use_column_width=True)
with col2:
    st.image('imgs/eval_acc.png', use_column_width=True)

st.title("Evaluation")

df = pd.read_csv("Accuracies.csv")

st.bar_chart(df, x="Class", y="Accuracy", horizontal=True)

# st.download_button("Download predictions", data=, file_name="predictions.csv", mime="text/csv")
