# app/utils.py
"""Utility functions for the Streamlit application.
Provides image preprocessing to match the MNIST format used during training.
"""
import io
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# Same normalization as used for training (mean=0.5, std=0.5) to map [0,1] -> [-1,1]
_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def preprocess_image(uploaded_file) -> torch.Tensor:
    """Convert uploaded image file to a normalized torch tensor.
    Args:
        uploaded_file: Streamlit UploadedFile object (bytes). It can be a file-like object.
    Returns:
        Tensor of shape (1, 1, 28, 28) ready for model inference.
    """
    # Read image bytes
    if isinstance(uploaded_file, bytes):
        image_bytes = uploaded_file
    else:
        # Streamlit's UploadedFile provides a read() method
        image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  # ensure grayscale
    tensor = _transform(image).unsqueeze(0)  # shape (1,1,28,28)
    return tensor

def load_image_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to a PIL image for display.
    The tensor is expected to be in range [-1, 1].
    """
    # De‑normalize
    tensor = tensor.clone()
    tensor = tensor * 0.5 + 0.5  # back to [0,1]
    array = tensor.squeeze().cpu().numpy() * 255
    array = array.astype(np.uint8)
    return Image.fromarray(array, mode='L')
