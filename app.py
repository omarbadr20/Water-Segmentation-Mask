import io
import os
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from inference import preprocess_tiff, load_model, predict_mask

app = Flask(__name__)

# Server configuration
MODEL_WEIGHTS_PATH = "best_unet_water_model.pth"
DEVICE = "cpu" # Default to CPU for safe local execution

# Preload model on server startup
model = None
if os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"Loading pre-trained ResNet-50 model weights from {MODEL_WEIGHTS_PATH}...")
    model = load_model(MODEL_WEIGHTS_PATH, device=DEVICE)
else:
    print(f"Warning: Model weights not found at {MODEL_WEIGHTS_PATH}. API will run in mock mode.")


def generate_rgb_preview(raw_image):
    """
    Generates an RGB composite preview PNG from Channels 3 (Red), 2 (Green), 
    and 1 (Blue) of your 12-channel TIFF file.
    """
    # Extract visible light bands
    r = raw_image[3, :, :]
    g = raw_image[2, :, :]
    b = raw_image[1, :, :]
    
    # Normalize channels to 0-255 for web display
    def norm_channel(ch):
        c_min, c_max = ch.min(), ch.max()
        denom = c_max - c_min
        if denom == 0:
            denom = 1e-8
        return ((ch - c_min) / denom * 255).astype(np.uint8)
    
    rgb = np.stack([norm_channel(r), norm_channel(g), norm_channel(b)], axis=-1)
    img = Image.fromarray(rgb)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


@app.route("/", methods=["GET"])
def index():
    """Renders the HTML Frontend."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an uploaded TIFF file, runs inference, and returns 
    the base64-encoded water mask and RGB preview.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    if not file.filename.endswith(('.tif', '.tiff')):
        return jsonify({"error": "Only .tif or .tiff images are supported"}), 400

    try:
        # Save uploaded file temporarily
        temp_path = "temp_upload.tif"
        file.save(temp_path)
        
        # Preprocess the TIFF file
        preprocessed_img, raw_img = preprocess_tiff(temp_path)
        
        # Run inference if model was successfully loaded
        if model is not None:
            mask = predict_mask(model, preprocessed_img, device=DEVICE)
        else:
            # Fallback mock mask (empty)
            mask = np.zeros((128, 128), dtype=np.uint8)
            
        # Convert output mask to a base64 PNG
        mask_img = Image.fromarray(mask)
        buffered_mask = io.BytesIO()
        mask_img.save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode('utf-8')
        
        # Generate raw composite RGB preview as base64
        rgb_base64 = generate_rgb_preview(raw_img)
        
        # Clean up temporary storage
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            "success": True,
            "mask_image": mask_base64,
            "rgb_preview": rgb_base64
        })
        
    except Exception as e:
        # Prevent server crashes on erroneous uploads
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": f"Inference processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)