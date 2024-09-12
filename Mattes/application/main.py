# Author: Mattes Warning
import cv2
import torch
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image
from utils import *
from model.Model import Model

# Initialize model
model = Model()
model = load_model_ckpt(model)
model.train()

st.title('Skin Lesion Analysis')

st.markdown(""" * Mattes Warning """)

st.markdown(""" Upload a ZIP file containing .jpg or .png images of skin lesions you want to analyse.""")

image_names = []
predicted_classes = []
images_base64 = []

# Create widget to upload zip file
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
                    # I had some issues with the numpy to torch conversion, probably due to version conflicts
                    # that I couldn't resolve but this strange way of doing it worked

                    # Read image
                    try:
                        image = Image.open(file)
                    except:
                        print(file)
                        continue
                    # Append to output to write to display in table later
                    images_base64.append(pil_image_to_base64(image))

                    # Convert image to tensor
                    img = np.array(image).astype("float32")
                    image = list(img.tolist())
                    tensor = torch.as_tensor(image).reshape(1, 3, 28, 28)

                    # Get edges and convert to tensor
                    canny = cv2.Canny(np.uint8(img), 50, 150).tolist()
                    canny = list(canny)
                    canny_img = torch.as_tensor(canny)
                    # Since the canny image only has 1 channel we copy it 3 times and then reshape
                    canny_img = canny_img.repeat(3, 1, 1).reshape(1, 3, 28, 28).to(torch.float)

                    # Make predictions
                    out = model(tensor, canny_img)

                    # Get the highest probability class index
                    predicted_class = torch.argmax(out).item()

                    # Store image name, predicted class, and image data
                    image_names.append(file_name)
                    predicted_classes.append(class_dict[predicted_class])

    # Create output dataframe
    df = pd.DataFrame({
        "Image Name": image_names,
        "Predicted Class": predicted_classes
    })

    # Convert to be able to download
    df = df.to_csv().encode("utf-8")

    # Button to download predictions as csv file
    st.download_button("Download predictions", data=df, file_name="predictions.csv", mime="text/csv")

    # Create an HTML table to display the image name, image, and predicted class
    table_html = """
    <div style="display: flex; justify-content: center;">
        <table>
            <tr><th>Image Name</th><th>Image</th><th>Predicted Class</th></tr>
    """

    # Display HTML table, this way we can print the images inside the table
    for i in range(len(image_names)):
        img_html = f'<img src="data:image/png;base64,{images_base64[i]}" width="100">'
        table_html += f"<tr><td>{image_names[i]}</td><td>{img_html}</td><td>{predicted_classes[i]}</td></tr>"

    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True, )
