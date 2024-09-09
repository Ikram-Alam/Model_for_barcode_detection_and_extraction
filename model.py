from ultralytics import YOLO
import fitz  # PyMuPDF for PDF to image conversion
import os
from PIL import Image, ImageOps
import numpy as np

# Paths
# model_path = 'best.pt'  # Ensure this is a simple string, not a tuple
pdf_path = 'Catalog (230 pages) .pdf'
output_dir = 'temp_barcodes'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
model = YOLO('best.pt', 'last.pt')

# Convert PDF to images
doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert image to numpy array for model processing
    img_np = np.array(img)
    
    # Detect barcodes using YOLO model
    results = model.predict(img_np)
    
    # Save detected barcode images
    for i, bbox in enumerate(results[0].boxes.xyxy):  # Assuming 'xyxy' format output
        x_min, y_min, x_max, y_max = map(int, bbox)
        barcode_img = img.crop((x_min, y_min, x_max, y_max))
        
        # Check if the barcode is vertically oriented, rotate it if needed
        if barcode_img.height > barcode_img.width:
            barcode_img = barcode_img.rotate(270, expand=True)  # Rotate to make it horizontal
        
        # Save barcode to temp directory
        barcode_img.save(os.path.join(output_dir, f'barcode_page{page_num}_{i}.png'))

print("Barcode detection and extraction completed.")













# from ultralytics import YOLO
# import fitz  # PyMuPDF for PDF to image conversion
# import os
# from PIL import Image, ExifTags
# import numpy as np

# # Paths
# # model_path = 'best.pt'  # Path to your YOLO model
# pdf_path = 'test.pdf'
# output_dir = 'temp_barcodes'

# # Ensure output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Load YOLO model
# model = YOLO('best.pt', 'last.pt')

# def correct_image_orientation(img):
#     """Corrects the orientation of an image based on its EXIF data."""
#     try:
#         exif = img._getexif()
#         if exif:
#             orientation_tag = next(
#                 (tag for tag, name in ExifTags.TAGS.items() if name == 'Orientation'), None)
#             if orientation_tag and orientation_tag in exif:
#                 orientation = exif[orientation_tag]
#                 if orientation == 3:  # Upside down
#                     img = img.rotate(180, expand=True)
#                 elif orientation == 6:  # Rotated 90 degrees
#                     img = img.rotate(270, expand=True)
#                 elif orientation == 8:  # Rotated -90 degrees
#                     img = img.rotate(90, expand=True)
#     except Exception as e:
#         print(f"Error correcting orientation: {e}")
#     return img

# # Convert PDF to images
# doc = fitz.open(pdf_path)
# for page_num in range(len(doc)):
#     page = doc.load_page(page_num)
#     pix = page.get_pixmap()
#     img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
#     # Convert image to numpy array for model processing
#     img_np = np.array(img)
    
#     # Detect barcodes using YOLO model
#     results = model.predict(img_np)
    
#     # Save detected barcode images
#     for i, bbox in enumerate(results[0].boxes.xyxy):  # Assuming 'xyxy' format output
#         x_min, y_min, x_max, y_max = map(int, bbox)
#         barcode_img = img.crop((x_min, y_min, x_max, y_max))
        
#         # Correct orientation if the barcode image is upside down
#         barcode_img = correct_image_orientation(barcode_img)
        
#         # Rotate barcode image if it's vertically oriented
#         if barcode_img.height > barcode_img.width:
#             barcode_img = barcode_img.rotate(90, expand=True)  # Rotate to make it horizontal
        
#         # Save barcode to temp directory
#         barcode_img.save(os.path.join(output_dir, f'barcode_page{page_num}_{i}.png'))

# print("Barcode detection and extraction completed.")






