import cv2
import easyocr
import os
from ultralytics import YOLO
import torch.multiprocessing as mp
import torch
import numpy as np
import time
import pytesseract
from PIL import Image
import re

mp.set_start_method('spawn', force=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize EasyOCR and Tesseract
reader = easyocr.Reader(['en'])
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

model_path = os.path.join('ML', 'runs', 'detect', 'train2', 'weights', 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

infer = YOLO(model_path)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open video stream")
    exit()

frame_image_path = "Main/frame.jpg"
text_file_path = "Main/plates.txt"

conf_threshold = 0.8
conf_drop_threshold = 0.5
save_duration = 3
timer_started = False
ocr_done = False
timer_start_time = 0

with open(text_file_path, 'a') as f:
    pass

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def perform_ocr(image):
    # Preprocess the image
    preprocessed = preprocess_image(image)
    sharpened = sharpen_image(preprocessed)
    
    # Perform OCR using EasyOCR
    easyocr_result = reader.readtext(sharpened)
    easyocr_text = ' '.join([text for _, text, _ in easyocr_result]) if easyocr_result else ''
    
    # Perform OCR using Tesseract with different PSM modes
    tesseract_text_6 = pytesseract.image_to_string(Image.fromarray(sharpened), config='--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    tesseract_text_7 = pytesseract.image_to_string(Image.fromarray(sharpened), config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    tesseract_text_8 = pytesseract.image_to_string(Image.fromarray(sharpened), config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Combine results
    combined_text = f"{easyocr_text} {tesseract_text_6} {tesseract_text_7} {tesseract_text_8}".upper()
    
    # Clean and format the result
    cleaned_text = re.sub(r'[^A-Z0-9]', '', combined_text)
    
    # Try to match Indian license plate format
    match = re.search(r'([A-Z]{2}\s?[0-9]{1,2}\s?[A-Z]{1,2}\s?[0-9]{4})', cleaned_text)
    if match:
        formatted_text = match.group(1)
        formatted_text = re.sub(r'(\w{2})(\d{1,2})(\w{1,2})(\d{4})', r'\1 \2 \3 \4', formatted_text)
    else:
        # If no match, use the entire cleaned text
        formatted_text = ' '.join(cleaned_text[i:i+2] for i in range(0, len(cleaned_text), 2))
    
    # Check for 'IND' in the text
    if 'IND' in formatted_text:
        formatted_text = formatted_text.replace('IND', '').strip()
        formatted_text += ' (IND)'
    
    return formatted_text if formatted_text else 'No text detected'

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Error: Failed to capture image")
        break
    
    cv2.imwrite(frame_image_path, frame)
    results = infer(frame_image_path)
    
    for r in results:
        if isinstance(r, torch.Tensor):
            boxes = r.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Conf: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                if conf >= conf_threshold and not ocr_done:
                    if not timer_started:
                        timer_started = True
                        timer_start_time = time.time()
                    
                    if time.time() - timer_start_time >= save_duration:
                        cv2.imwrite(frame_image_path, frame)
                        license_plate = frame[y1:y2, x1:x2]
                        ocr_text = perform_ocr(license_plate)
                        
                        with open(text_file_path, 'a') as f:
                            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            f.write(f"{current_time}, Plate: {ocr_text}, Confidence: {conf:.2f}\n")
                        
                        ocr_done = True
                        timer_started = False

                elif conf < conf_drop_threshold:
                    timer_started = False
                    ocr_done = False
    
    cv2.imshow('License Plate Detection with Confidence Display', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

if os.path.exists(frame_image_path):
    os.remove(frame_image_path)