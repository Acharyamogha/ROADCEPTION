# Overview
This project combines a Convolutional Neural Network (CNN) for lane detection and a YOLOv5 model for object detection to process and analyze video footage. The system identifies and highlights road lanes and objects (like vehicles and pedestrians) in real-time. This application is particularly useful for automated driving systems.

# Repository Structure
The repository contains the following folders:

## 1. Frontend
Contains all files for the user interface or visualization aspects of the project.
Includes HTML, CSS, JavaScript files, and any frontend-related assets.
Helps in visualizing the output or providing an interface for input video uploads and result display.
## 2. Model Code
Contains Python scripts for lane detection and object detection.
Includes:
  Model loading and inference code.
  Scripts for processing video frames.
  Pretrained weights for both CNN and YOLOv5.

# Features
Lane Detection: Detects road lanes in video using a custom-trained Convolutional Neural Network (CNN).
Object Detection: Identifies and localizes objects like vehicles and pedestrians using YOLOv5.
Frontend Interface: User-friendly interface for video uploads and displaying results.
Real-Time Processing: Simultaneously processes lane detection and object detection in video footage.
Region of Interest (ROI): Focuses processing on road areas to improve detection accuracy.
Annotated Output: Saves processed videos with lane markings and detected objects.

## Technologies Used
-Backend (Model Code)
-Programming Language: Python
## Libraries:
-PyTorch (for the CNN model)
-OpenCV (for video processing and visualization)
-NumPy
-Ultralytics YOLOv5 (pretrained model)
## Models:
-Custom CNN for lane detection.
-Pre-trained YOLOv5 for object detection.
## Frontend
-Programming Languages: HTML, CSS, JavaScript
-Frameworks/Libraries: Bootstrap, optional frameworks for visualizations (e.g., Chart.js).

# Installation and Setup
## Prerequisites
-Python (3.8+): Install Python from python.org.
-Dependencies: Install required Python libraries:

pip install opencv-python-headless numpy torch torchvision ultralytics

-Frontend Requirements: A modern web browser is sufficient.

## Clone the Repository

git clone https://github.com/Acharyamogha/ROADCEPTION.git
cd ROADCEPTION

## Usage
1. Frontend Interface
Navigate to the frontend folder.
Open index.html in a web browser.
Upload your video file through the interface.
View processed results directly.

## Input and Output
Input: Place your input video (e.g., vehicle.mp4) in the videos folder.
Output: Processed videos will be saved as:
lane_detection_output.mp4 (Lane detection results)
object_detection_output.mp4 (Object detection results)

# Dataset
## Lane Detection
The CNN model is trained on a custom dataset of road images with labeled lane markings.
Preprocessing includes resizing to 64x64 and applying augmentations for robustness.

## Object Detection
The YOLOv5 model is pre-trained on the COCO dataset, enabling it to detect common objects like cars, pedestrians, etc.

# Results
Lane Detection: The CNN achieves high accuracy in detecting lanes under clear conditions.
Object Detection: YOLOv5 performs real-time detection of vehicles, pedestrians, and other objects.

# Limitations
## Lane Detection:
Performance may degrade under poor lighting or occluded lanes.
Limited to static lane markings.

## Object Detection:
May struggle with overlapping objects in crowded scenes.

# Future Enhancements
Improve the robustness of lane detection under varying weather and lighting conditions.
Integrate lane curvature estimation for dynamic tracking.
Extend object detection to multi-class tracking.
Add a dashboard in the frontend for real-time visualization and statistics.

# Contributors
Amogha Acharya - Mangalore Institute of Technology & Engineering
Anubhav Gour - Mangalore Institute of Technology & Engineering
Deviprasad N Shetty - Mangalore Institute of Technology & Engineering
N Shreyas Nayak - Mangalore Institute of Technology & Engineering
