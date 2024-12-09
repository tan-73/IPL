import cv2
import easyocr
import os
from ultralytics import YOLO
import torch.multiprocessing as mp
import torch
import numpy as np

mp.set_start_method('spawn', force=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

reader = easyocr.Reader(['en'])
model_path = os.path.join('ML', 'runs', 'detect', 'train2', 'weights', 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

infer = YOLO(model_path)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open video stream")
    exit()

frame_image_path = "frame.jpg"

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

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color (BGR format)
                
                license_plate = frame[y1:y2, x1:x2]
                ocr_result = reader.readtext(license_plate)
                
                ocr_text = ' '.join([text for _, text, _ in ocr_result]) if ocr_result else 'No text detected'
                label = f"Conf: {conf:.2f}, Plate: {ocr_text}"
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1]
                
                cv2.rectangle(frame, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (x1, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    cv2.imshow('License Plate Detection with OCR', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

if os.path.exists(frame_image_path):
    os.remove(frame_image_path)