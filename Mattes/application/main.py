import streamlit as st
import zipfile
from PIL import Image
import numpy as np
import io
import pandas as pd
import cv2
import base64
from io import BytesIO

from model.Model import Model

import torch

model = Model()
model.train()

class_dict = {0: "Actinic keratoses and intraepithelial carcinoma",
              1: "Basal cell carcinoma",
              2: "Benign keratosis-like lesions",
              3: "Dermatofibroma",
              4: "Melanoma",
              5: "Melanocytic nevi",
              6: "Vascular lesions"}


# Function to convert PIL image to base64 string for HTML rendering
def pil_image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")

st.title('Machine Learning in Biomedicine, SS24')

st.markdown("""
 * Mattes Warning
 """)

image_names = []
predicted_classes = []
images = []
images_base64 = []

zip_file = st.file_uploader("Upload images in a zip file.", type="zip")

if zip_file:
    # Open the uploaded zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # List all files in the zip archive
        file_names = zip_ref.namelist()

        # Loop through files and check if they are images
        for file_name in file_names:
            if ".jpg" in file_name or ".png" in file_name:
                # Open each file in the zip
                with zip_ref.open(file_name) as file:
                    image = Image.open(file)
                    images.append(image)
                    images_base64.append(pil_image_to_base64(image))
                    img = np.array(image).astype("float32")
                    canny = cv2.Canny(np.uint8(img), 50, 150).tolist()
                    canny = list(canny)
                    canny_img = torch.as_tensor(canny)
                    #canny_img = torch.as_tensor(canny_img)
                    canny_img = canny_img.repeat(3, 1, 1).reshape(1,3,28,28).to(torch.float)

                    image = list(img.tolist())
                    tensor = torch.as_tensor(image).reshape(1,3,28,28)


                    out = model(tensor, canny_img)

                    predicted_class = torch.argmax(out).item()

                    # Store image name, predicted class, and image data
                    image_names.append(file_name)
                    predicted_classes.append(class_dict[predicted_class])


    # Create output datafram
    df = pd.DataFrame({
        "Image Name": image_names,
        "Predicted Class": predicted_classes
    })

    df = convert_df(df)

    st.download_button("Download predictions", data=df, file_name="predictions.csv", mime="text/csv")

    # Create an HTML table to display the image name, image, and predicted class
    table_html = """
<div style="display: flex; justify-content: center;">
    <table>
        <tr><th>Image Name</th><th>Image</th><th>Predicted Class</th></tr>
"""


    for i in range(len(image_names)):
        img_html = f'<img src="data:image/png;base64,{images_base64[i]}" width="100">'
        table_html += f"<tr><td>{image_names[i]}</td><td>{img_html}</td><td>{predicted_classes[i]}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True, )
