# YOLOv4 Vehicle Detection and Tracking with Deep Sort and Speed Calculation
This repository contains a Python script for vehicle detection, tracking, and speed calculation using YOLOv4, Deep Sort, and a vehicle speed estimation model. The code is designed to work with TensorFlow and includes options for tracking, speed estimation, and video output.

weights - https://drive.google.com/drive/folders/1Ld_FgcD9x1Q7HfC0ZxiNeVf2M4-Y4EHt?usp=sharing

## Commands to run - 
### save yolov4 model
python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4 --model yolov4
### Run yolov4 deep sort object tracker on video
python object_tracker.py --weights ./checkpoints/yolov4 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars.avi

### save yolov4-tiny model
python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --model yolov4 --tiny
### Run yolov4-tiny object tracker
python object_tracker.py --weights ./checkpoints/yolov4-tiny-416 --model yolov4 --video ./data/video/cars.mp4 --output ./outputs/cars_tiny.avi --tiny