import boto3
import json
import base64
import io
import zipfile
from datetime import datetime
from PIL import Image

def encode_image(image):
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def validate_image(uploaded_file):
    """Validate image size constraints."""
    try:
        img = Image.open(uploaded_file)
        width, height = img.size
        
        if min(width, height) < 256:
            return False, f"Too small (min: 256px, got: {min(width, height)}px)"
        elif max(width, height) > 4096:
            return False, f"Too large (max: 4096px, got: {max(width, height)}px)"
        else:
            return True, f"Valid ({width}×{height}px)"
    except Exception:
        return False, "Invalid image file"

def create_zip_download(all_images):
    """Create ZIP file from list of (filename, image) tuples."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image in all_images:
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            zip_file.writestr(filename, img_buffer.getvalue())
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return zip_buffer.getvalue(), f"generated_images_{timestamp}.zip"

