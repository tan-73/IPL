# Working 

import cv2
import easyocr
import os
from ultralytics import YOLO
import torch.multiprocessing as mp
import torch
import numpy as np
import time

mp.set_start_method('spawn', force=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

reader = easyocr.Reader(['en'])
model_path = os.path.join('Main', 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model file not found: {model_path}')

infer = YOLO(model_path)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open video stream")
    exit()

frame_image_path = "Main/frame.jpg"
text_file_path = "Main/plates.txt"

conf_threshold = 0.7
conf_drop_threshold = 0.6
save_duration = 2 
timer_started = False
ocr_done = False  
timer_start_time = 0

with open(text_file_path, 'a') as f:
    pass

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
                        ocr_result = reader.readtext(license_plate)
                        ocr_text = ' '.join([text for _, text, _ in ocr_result]) if ocr_result else 'No text detected'
                        
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
