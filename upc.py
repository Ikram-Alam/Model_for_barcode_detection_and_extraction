import fitz  # PyMuPDF to read PDFs
import easyocr  # EasyOCR for OCR-based text extraction
import re  # Regular expression for matching UPC numbers
import cv2
import numpy as np

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define a regex pattern for a 12-digit UPC number
upc_pattern = re.compile(r'\b\d{12}\b')

# Convert each page of the PDF to an image
def pdf_to_images(pdf_path):
    images = []
    
    # Open the PDF document using PyMuPDF
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)  # Load each page
            pix = page.get_pixmap()  # Render page to an image (pixmap)
            
            # Convert pixmap to a NumPy array (OpenCV format)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # If there are alpha channels (RGBA), convert to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            images.append(img)
    
    return images

# Perform OCR on images and extract text
def ocr_extract_text_from_images(images):
    extracted_text = []
    
    # Perform OCR for each image
    for img in images:
        # Use EasyOCR to extract text
        results = reader.readtext(img, detail=0)  # detail=0 to get only text, not bounding box
        extracted_text.append(" ".join(results))  # Combine text in one string for each page
    
    return extracted_text

# Extract UPC numbers using regex from the text
def extract_upc_from_text(texts):
    upc_numbers = []
    
    # Find all UPC numbers in the extracted text using regex
    for text in texts:
        found_upc = upc_pattern.findall(text)
        upc_numbers.extend(found_upc)  # Add found UPC numbers to the list
    
    return upc_numbers

# End-to-end function to extract UPC numbers from a PDF
def extract_upc_from_pdf(pdf_path):
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    
    # Extract text from the images using OCR
    texts = ocr_extract_text_from_images(images)
    
    # Extract UPC numbers from the text
    upc_numbers = extract_upc_from_text(texts)
    
    return upc_numbers

# Test the function on your PDF
pdf_path = 'emdcatalog (1).pdf'  # Replace with your PDF path
upc_numbers = extract_upc_from_pdf(pdf_path)

# Print the extracted UPC numbers
if upc_numbers:
    print(f"Extracted UPC numbers: {upc_numbers}")
else:
    print("No UPC numbers found in the PDF.")
