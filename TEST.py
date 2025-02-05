import cv2
import numpy as np
import pytesseract
import re

def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding to binarize the image
    _, binary_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to close gaps in lines and enhance table structure
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_close = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Dilate the image to strengthen table lines
    dilated = cv2.dilate(morph_close, kernel, iterations=1)

    return dilated

def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(processed_image, config='--psm 6')
    return extracted_text

def process_extracted_text(text):
    lines = text.strip().split("\n")  # Split the extracted text into lines
    processed_lines = []
    last_numeric_line_index = -1  # Track the last numeric line
    
    for i, line in enumerate(lines):
        if re.search(r'[a-zA-Z]', line):  # Check if the line contains alphabets
            if last_numeric_line_index != -1:
                # Modify the last numeric line by adding '@' at the start
                processed_lines[last_numeric_line_index] = "@ " + processed_lines[last_numeric_line_index]
        else:
            last_numeric_line_index = len(processed_lines)  # Update last numeric line index
        
        processed_lines.append(line)  # Store the processed line
    
    # Ensure the last numeric line is marked with '@' if it wasn't already
    if last_numeric_line_index != -1 and not processed_lines[last_numeric_line_index].startswith("@"): 
        processed_lines[last_numeric_line_index] = "@ " + processed_lines[last_numeric_line_index]
    
    return "\n".join(processed_lines)  # Join the processed lines back into text

# Example usage
def main(image_path, output_file):
    extracted_text = extract_text_from_image(image_path)
    processed_text = process_extracted_text(extracted_text)
    
    # Save the processed text into a file instead of printing
    with open(output_file, "w") as file:
        file.write(processed_text)

# Example call
main("OCR/nia.png", "output.txt")
