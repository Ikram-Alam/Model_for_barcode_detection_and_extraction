import cv2
import fitz  # PyMuPDF
import pytesseract
from ultralytics import YOLO
import os
import re

# Path to Tesseract executable (adjust according to your OS)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust if necessary

# Load your trained YOLO model
model = YOLO('best.pt')

# Folder to save extracted images
output_folder = 'extracted_images'

# Create folder if not exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Regex pattern for 12-digit UPC codes
upc_pattern = r'\b\d{12}\b'

# Path to the PDF file
pdf_path = 'emdcatalog (1).pdf'

# Function to extract UPC numbers from detected regions
def detect_upc_from_pdf(pdf_path):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()  # Get the page as an image
        img_path = os.path.join(output_folder, f'page_{page_num}.png')
        
        # Save the page as an image
        pix.save(img_path)
        
        # Read the saved image using OpenCV
        img = cv2.imread(img_path)
        
        # Use your YOLO model to detect regions of interest (ROIs)
        results = model.predict(source=img, save=False)
        
        for i, (box, conf, class_id) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
            x1, y1, x2, y2 = map(int, box)  # Get bounding box coordinates
            roi = img[y1:y2, x1:x2]  # Extract the region of interest (ROI)

            # Perform OCR on the detected ROI
            roi_text = pytesseract.image_to_string(roi)
            print(f"OCR Result from Page {page_num + 1}, Region {i}: {roi_text}")

            # Search for UPC numbers in the extracted text
            upcs = re.findall(upc_pattern, roi_text)
            
            if upcs:
                print(f"Detected UPC numbers in Region {i}: {upcs}")
            else:
                print(f"No UPC numbers detected in Region {i}")
    
    pdf_document.close()

# Run the function to detect and extract UPC numbers
detect_upc_from_pdf(pdf_path)
