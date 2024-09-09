import os
import cv2
import pyzbar.pyzbar as pyzbar
import fitz  # PyMuPDF
import tempfile
from ultralytics import YOLO

# Load models
model_best = YOLO("best.pt")
model_last = YOLO("last.pt")

# Convert PDF pages to images using PyMuPDF (fitz)
def convert_pdf(file_path):
    temp_dir = tempfile.mkdtemp()
    pdf_document = fitz.open(file_path)

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Adjust DPI if necessary
        image_path = os.path.join(temp_dir, f'page_{page_num + 1}.jpg')
        pix.save(image_path)

    pdf_document.close()
    return temp_dir

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image at {image_path} could not be loaded.")
        return None
    return img

def detect_barcodes_with_model(image, model):
    results = model(image)
    barcodes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            barcode_img = image[y1:y2, x1:x2]  # Crop barcode image
            barcodes.append(barcode_img)
    return barcodes

def decode_barcodes(image):
    barcodes = pyzbar.decode(image)
    decoded_barcodes = [barcode.data.decode('utf-8') for barcode in barcodes]
    return decoded_barcodes, len(decoded_barcodes)

def barcode_process_images_in_directory(directory):
    bars = []
    total_count = 0

    image_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = preprocess_image(image_path)
        if image is not None:
            detected_barcodes_best = detect_barcodes_with_model(image, model_best)
            detected_barcodes_last = detect_barcodes_with_model(image, model_last)
            
            for barcode_img in detected_barcodes_best + detected_barcodes_last:
                decoded_barcodes, count = decode_barcodes(barcode_img)
                bars.extend(decoded_barcodes)
                total_count += count
    
    return bars, total_count

# Main function to convert PDF, detect barcodes, and decode them
def process_pdf(file_path):
    directory = convert_pdf(file_path)
    barcodes, count = barcode_process_images_in_directory(directory)
    
    print(f"Decoded Barcodes: {barcodes}")
    print(f"Total Barcodes Found: {count}")

# Example usage
file_path = 'screenshot.pdf'
process_pdf(file_path)
