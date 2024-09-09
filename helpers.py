from functools import wraps
import os
import re
from flask import redirect, session
from pdf2image import convert_from_path
import cv2
import pyzbar.pyzbar as pyzbar
import pdfplumber
import tempfile
import numpy as np
import logging
from ultralytics import YOLO

# file_path = 'screenshot.pdf'


# LOGIN
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/")
        return f(*args, **kwargs)
    return decorated_function

# Load models
model_best = YOLO("best.pt")
model_last = YOLO("last.pt")

# UPC

def convert_pdf(file_path):
    temp_dir = tempfile.mkdtemp()
    images = convert_from_path(file_path, dpi=400) 
    
    for i, image in enumerate(images):
        image_path = os.path.join(temp_dir, f'page_{i + 1}.jpg')
        image.save(image_path, 'JPEG', quality=100, optimize=True)
    return temp_dir

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image at {image_path} could not be loaded.")
        return None
    
    return img  # No additional preprocessing needed since YOLO handles this

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
    decoded_upcs = [barcode.data.decode('utf-8') for barcode in barcodes]
    return decoded_upcs, len(decoded_upcs)

def barcode_process_images_in_directory(directory):
    bars = []
    total_count = 0

    all_files = os.listdir(directory)
    image_files = [file for file in all_files if file.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = preprocess_image(image_path)
        if image is not None:
            detected_barcodes_best = detect_barcodes_with_model(image, model_best)
            detected_barcodes_last = detect_barcodes_with_model(image, model_last)
            
            for barcode_img in detected_barcodes_best + detected_barcodes_last:
                decoded_upcs, count = decode_barcodes(barcode_img)
                bars.extend(decoded_upcs)
                total_count += count
    return bars, total_count

def calculate_check_digit(upc):
    upc = upc.zfill(11)
    multipliers = [3 if i % 2 == 0 else 1 for i in range(11)]
    total = sum(m * int(digit) for m, digit in zip(multipliers, upc))
    return str((10 - total % 10) % 10)

def fix_upc(upc_list):
    correct_upcs = []
    for upc in upc_list:
        if len(upc) == 12:
            correct_upcs.append(upc)
        elif len(upc) == 11:
            correct_upcs.append(upc + calculate_check_digit(upc))
        elif len(upc) == 10:
            for first_digit in '01678':
                full_upc = first_digit + upc
                full_upc += calculate_check_digit(full_upc)
                correct_upcs.append(full_upc)
    return correct_upcs

def upc_process_pdf(file_path):
    upc_regex = re.compile(r'\b\d{10,13}\b')
    upcs = []
    count = 0
    page_number = 0
    pages_without_upcs = 0  
    max_consecutive_pages_without_upcs = 15
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.WARNING)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber.utils").setLevel(logging.WARNING)

    if not file_path.lower().endswith('.pdf'):
        print(f"Error: Provided path {file_path} is not a PDF.")
        return [], 0

    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_number += 1
                text = page.extract_text()
                if text:
                    found_upcs = upc_regex.findall(text)
                    if found_upcs:
                        upcs.extend(fix_upc(found_upcs))
                        pages_without_upcs = 0  
                    else:
                        pages_without_upcs += 1 
                        if pages_without_upcs > max_consecutive_pages_without_upcs:
                            print(f"No UPCs found for {pages_without_upcs} consecutive pages. Stopping process.")
                            break
                else:
                    pages_without_upcs += 1  
                    if pages_without_upcs > max_consecutive_pages_without_upcs:
                        print(f"No text found for {pages_without_upcs} consecutive pages. Stopping process.")
                        break
        count = len(upcs)
    except Exception as e:
        print(f"An error occurred while processing the PDF file: {e}")
        return [], 0

    return upcs, count


def allowed_file(file):
    allowed_extensions = {'pdf'}
    max_file_size = 500 * 1024 * 1024  # 500mb

    def is_allowed_filename(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)

    if file_length > max_file_size:
        return False, "File size exceeds the maximum limit of 500 MB."

    if not is_allowed_filename(file.filename):
        return False, "Invalid file type. Only PDF files are allowed."

    return True, "File is allowed."



