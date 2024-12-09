import cv2
import easyocr
import os
from ultralytics import YOLO
import torch
import numpy as np
import time
import sys

# Ensure proper multiprocessing start method
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define model path
model_path = os.path.join('ML', 'runs', 'detect', 'train2', 'weights', 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

# Load YOLO model
infer = YOLO(model_path)

# Get image path from command-line arguments
if len(sys.argv) != 2:
    raise ValueError("Usage: python misc.py <image_path>")

image_path = sys.argv[1]
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

# Define output paths
frame_image_path = "Main/frame.jpg"
text_file_path = "Main/plates.txt"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(text_file_path), exist_ok=True)

# Open text file for appending
with open(text_file_path, 'a') as f:
    pass

# Define thresholds
conf_threshold = 0.5
conf_drop_threshold = 0.5

# Read image
frame = cv2.imread(image_path)
if frame is None:
    raise ValueError("Error: Failed to load image")

# Save the frame to process later
cv2.imwrite(frame_image_path, frame)

# Perform inference
results = infer(frame_image_path)

# Process results
ocr_done = False
for r in results:
    if isinstance(r, torch.Tensor):
        boxes = r.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"Conf: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]
            cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Perform OCR if confidence is high enough
            if conf >= conf_threshold and not ocr_done:
                license_plate = frame[y1:y2, x1:x2]
                ocr_result = reader.readtext(license_plate)
                ocr_text = ' '.join([text for _, text, _ in ocr_result]) if ocr_result else 'No text detected'

                # Write results to text file
                with open(text_file_path, 'a') as f:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    f.write(f"{current_time}, Plate: {ocr_text}, Confidence: {conf:.2f}\n")

                ocr_done = True

# Display processed image
cv2.imshow('License Plate Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cleanup
if os.path.exists(frame_image_path):
    os.remove(frame_image_path)