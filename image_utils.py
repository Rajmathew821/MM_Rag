import io
import base64
from PIL import Image

def compress_and_encode_image(image_path, resize_width=512):
    try:
        img = Image.open(image_path)
        if img.width > resize_width:
            ratio = resize_width / img.width
            img = img.resize((resize_width, int(img.height * ratio)))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        raise RuntimeError(f"Error compressing/encoding image {image_path}: {e}")
