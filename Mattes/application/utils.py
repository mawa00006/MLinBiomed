# Author: Mattes Warning
import torch
import base64

from io import BytesIO

# Dictionary that maps labels indexes to class names
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


def load_model_ckpt(model):
    ckpt = torch.load('weights.pt', map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    return model
