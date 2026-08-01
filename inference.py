import os
import tifffile
import numpy as np
import torch
from model import ResNet50UNet

def preprocess_tiff(image_path):
    """
    Loads and normalizes the 12-channel TIFF image exactly as done during training.
    """
    image = tifffile.imread(image_path).astype(np.float32)
    
    # Ensure shape is Channel-First (12, 128, 128)
    if image.shape[0] == 12:
        pass
    elif image.shape[2] == 12:
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Expected 12 channels, got shape {image.shape}")
        
    normalized_image = np.zeros_like(image)
    
    # Normalize first 11 bands
    for c in range(11):
        band = image[c, :, :]
        b_min, b_max = band.min(), band.max()
        denom = b_max - b_min
        if denom == 0:
            denom = 1e-8
        normalized_image[c, :, :] = (band - b_min) / denom
        
    # Normalize 12th band
    normalized_image[11, :, :] = image[11, :, :] / 100.0
    
    return normalized_image, image


def load_model(weights_path, device='cpu'):
    """
    Instantiates the ResNet-50 U-Net model and loads the trained weights.
    """
    model = ResNet50UNet(out_channels=1, pretrained=False)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_mask(model, preprocessed_image, device='cpu'):
    """
    Runs inference on a preprocessed 12-channel image and returns a binary mask.
    """
    # Add batch dimension: (1, 12, 128, 128)
    input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
    # Extract 2D mask
    mask = preds.squeeze(0).squeeze(0).cpu().numpy()
    
    # Convert mask to 0 (land) and 255 (water) for visual representation
    return (mask * 255).astype(np.uint8)