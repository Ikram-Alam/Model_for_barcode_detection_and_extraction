# import fitz  # PyMuPDF for PDF handling
# import onnxruntime as ort  # ONNX Runtime for inference
# import numpy as np
# import cv2
# import re

# # Load ONNX model
# def load_onnx_model(onnx_model_path):
#     ort_session = ort.InferenceSession(onnx_model_path)
#     return ort_session

# # Preprocess the image to fit the input format of your model
# def preprocess_image(image):
#     image = cv2.resize(image, (800, 800))  # Resize if necessary
#     image = image.astype(np.float32)
#     image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# # Convert PDF pages to images using PyMuPDF
# def pdf_to_images(pdf_path):
#     images = []
    
#     # Open the PDF file
#     with fitz.open(pdf_path) as pdf_document:
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)  # Load each page
#             pix = page.get_pixmap()  # Render page to an image
            
#             # Convert pixmap to NumPy array (OpenCV format)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             if pix.n == 4:  # Convert RGBA to RGB
#                 img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#             images.append(img)
    
#     return images

# # Perform inference using ONNX Runtime
# def detect_text_with_onnx(onnx_session, image):
#     # Preprocess the image
#     input_image = preprocess_image(image)

#     # Run inference
#     ort_inputs = {onnx_session.get_inputs()[0].name: input_image}
#     ort_outs = onnx_session.run(None, ort_inputs)
    
#     return ort_outs

# # Extract UPC numbers from detected text using regex
# def extract_upc_numbers(detected_texts):
#     upc_pattern = re.compile(r'\b\d{9,13}\b')  # Regex for 9 to 13 digit numbers
    
#     upc_numbers = []
#     for text in detected_texts:
#         found_upc = upc_pattern.findall(text)
#         upc_numbers.extend(found_upc)
    
#     return upc_numbers

# # Main function to extract UPC numbers from a PDF
# def extract_upc_from_pdf(pdf_path, onnx_model_path):
#     # Step 1: Load the ONNX model
#     onnx_model = load_onnx_model(onnx_model_path)
    
#     # Step 2: Convert PDF pages to images
#     images = pdf_to_images(pdf_path)
    
#     upc_numbers = []
    
#     # Step 3: Loop over each image (page), detect text, and extract UPC numbers
#     for image in images:
#         # Step 4: Perform text detection using the ONNX model
#         prediction = detect_text_with_onnx(onnx_model, image)
        
#         # Step 5: Extract detected text (adjust this part based on model output)
#         detected_texts = [str(pred) for pred in prediction]  # Placeholder for actual post-processing
#         upc_numbers.extend(extract_upc_numbers(detected_texts))
    
#     return upc_numbers

# # Test the function
# pdf_path = 'DSD Charlotte (S  P).pdf'  # Replace with your actual PDF path
# onnx_model_path = 'craft_ic15_20k.onnx'  # Replace with the path to your ONNX model

# upc_numbers = extract_upc_from_pdf(pdf_path, onnx_model_path)

# # Print the extracted UPC numbers
# if upc_numbers:
#     print(f"Extracted UPC numbers: {upc_numbers}")
# else:
#     print("No UPC numbers found in the PDF.")





# import fitz  # PyMuPDF for PDF to image conversion
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import cv2
# import numpy as np
# import re

# # Load the trained model (CRAFT) for text detection
# def load_craft_model(model_path):
#     model = torch.hub.load('clovaai/CRAFT-pytorch', 'craft', pretrained=False)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

# # Convert PDF to images using PyMuPDF
# def pdf_to_images(pdf_path):
#     images = []
    
#     # Open the PDF
#     with fitz.open(pdf_path) as pdf_document:
#         for page_num in range(len(pdf_document)):
#             page = pdf_document.load_page(page_num)
#             pix = page.get_pixmap()
            
#             # Convert pixmap to a NumPy array (OpenCV format)
#             img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#             if pix.n == 4:  # Convert RGBA to RGB
#                 img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#             images.append(img)
    
#     return images

# # Perform text detection using the CRAFT model
# def detect_text_with_craft(model, image):
#     # Convert the image to a format suitable for CRAFT
#     transform = transforms.Compose([transforms.ToTensor()])
#     image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
#     with torch.no_grad():
#         prediction = model(image_tensor)
    
#     # Return the regions where text was detected
#     return prediction

# # Post-process the text detections and extract text regions
# def extract_text_regions(prediction, image):
#     # Placeholder: this part would extract and return detected text boxes
#     # You would need to process the 'prediction' to get bounding boxes of text regions.
#     # Implement this based on how your CRAFT model returns the detections.
#     return []

# # Extract UPC numbers from detected text using regex
# def extract_upc_numbers(detected_texts):
#     # Define regex pattern for UPC (9 to 13 digits)
#     upc_pattern = re.compile(r'\b\d{9,13}\b')
    
#     upc_numbers = []
#     for text in detected_texts:
#         found_upc = upc_pattern.findall(text)
#         upc_numbers.extend(found_upc)
    
#     return upc_numbers

# # Main function to extract UPC numbers from a PDF
# def extract_upc_from_pdf(pdf_path, model_path):
#     # Step 1: Load the CRAFT model
#     craft_model = load_craft_model(model_path)
    
#     # Step 2: Convert PDF pages to images
#     images = pdf_to_images(pdf_path)
    
#     upc_numbers = []
    
#     # Step 3: Loop over each image (page), detect text and extract UPC numbers
#     for image in images:
#         # Step 4: Perform text detection
#         prediction = detect_text_with_craft(craft_model, image)
        
#         # Step 5: Extract detected text regions (this part is custom to your model)
#         detected_texts = extract_text_regions(prediction, image)
        
#         # Step 6: Use regex to extract UPC numbers from the detected text
#         upc_numbers.extend(extract_upc_numbers(detected_texts))
    
#     return upc_numbers

# # Test the function with your PDF and model
# pdf_path = 'DSD Charlotte (S  P).pdf'  # Replace with your actual PDF path
# model_path = 'craft_ic15_20k.pth'  # Replace with your model path

# upc_numbers = extract_upc_from_pdf(pdf_path, model_path)

# # Print the extracted UPC numbers
# if upc_numbers:
#     print(f"Extracted UPC numbers: {upc_numbers}")
# else:
#     print("No UPC numbers found in the PDF.")

























import fitz  # PyMuPDF
import cv2
import os
from ultralytics import YOLO  # Assuming you're using YOLO for UPC text detection
import pytesseract


# Load your trained model for detecting UPC numbers
model = YOLO('best.pt')  # Replace 'your_trained_model.pt' with your actual model

# Function to convert PDF pages to images
def pdf_to_images(pdf_path, output_folder):
    doc = fitz.open(pdf_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = f"{output_folder}/page_{page_num}.png"
        pix.save(image_path)
        images.append(image_path)
    
    return images

# Function to detect UPC text regions using your trained model
def detect_upc_text(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    upc_text_regions = []
    
    # Assuming the model detects bounding boxes around UPC text regions
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        upc_text_img = img[y1:y2, x1:x2]  # Crop the region containing UPC text
        upc_text_regions.append(upc_text_img)
    
    return upc_text_regions

# Function to extract text from the detected regions (if needed)
# You may not need this function if your model directly outputs UPC numbers
def extract_upc_from_region(upc_text_img):
    # Placeholder for any post-processing if required
    # If your model directly detects UPC numbers, you can skip this
    return pytesseract.image_to_string(upc_text_img, config='--psm 7').strip()  # For isolated text blocks

# End-to-End Processing
def process_pdf_for_upc(pdf_path):
    output_folder = 'temp_images'
    images = pdf_to_images(pdf_path, output_folder)
    
    all_upc_numbers = []
    for image in images:
        upc_text_regions = detect_upc_text(image)
        for region in upc_text_regions:
            upc_text = extract_upc_from_region(region)  # Skip if the model already gives text output
            all_upc_numbers.append(upc_text)
    
    return all_upc_numbers

# Example usage
pdf_path = 'DSD Charlotte (S  P).pdf'  # Path to your input PDF file
upc_numbers = process_pdf_for_upc(pdf_path)

print("Detected UPC Numbers:", upc_numbers)
