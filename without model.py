import fitz  # PyMuPDF for PDF to image conversion
import os
from PIL import Image, ExifTags
from pyzbar.pyzbar import decode
import numpy as np

# Paths
pdf_path = 'screenshot.pdf'
output_dir = 'temp_barcodes1'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def correct_image_orientation(img):
    """Corrects the orientation of an image based on its EXIF data."""
    try:
        exif = img._getexif()
        if exif:
            orientation_tag = next(
                (tag for tag, name in ExifTags.TAGS.items() if name == 'Orientation'), None)
            if orientation_tag and orientation_tag in exif:
                orientation = exif[orientation_tag]
                if orientation == 3:  # Upside down
                    img = img.rotate(180, expand=True)
                elif orientation == 6:  # Rotated 90 degrees
                    img = img.rotate(270, expand=True)
                elif orientation == 8:  # Rotated -90 degrees
                    img = img.rotate(90, expand=True)
    except Exception as e:
        print(f"Error correcting orientation: {e}")
    return img

def decode_barcodes_from_image(image):
    """Decodes barcodes from an image and returns the decoded data."""
    barcodes = decode(image)
    decoded_data = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        decoded_data.append(barcode_data)
    return decoded_data

# Convert PDF to images
doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Correct orientation
    img = correct_image_orientation(img)
    
    # Decode barcodes
    barcodes = decode_barcodes_from_image(img)
    
    # Save detected barcode images
    for i, barcode in enumerate(barcodes):
        # For demonstration, save the full image with detected barcodes
        barcode_img_path = os.path.join(output_dir, f'barcode_page{page_num}_{i}.png')
        img.save(barcode_img_path)
        print(f"Saved barcode image with decoded data: {barcode} at {barcode_img_path}")

print("Barcode detection and extraction completed.")
