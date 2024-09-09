import cv2
import easyocr
import re

# Load image
image_path = '251EB9E2-0D10-46E5-A0E4-4ED0FA91A373.png'
image = cv2.imread(image_path)

# Convert to grayscale for better OCR results
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Perform OCR on the grayscale image
result = reader.readtext(gray_image)

# Extract and filter UPC numbers
upc_numbers = []
for detection in result:
    text = detection[1]
    # Regex to match 12-digit UPC numbers
    if re.match(r'^\d{12}$', text):
        upc_numbers.append(text)

# Print detected UPC numbers
print("Detected UPC numbers:", upc_numbers)
