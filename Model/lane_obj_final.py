import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO

class LaneDetectionCNN(nn.Module):
    def __init__(self):
        super(LaneDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 64 * 64)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).view(-1, 1, 64, 64)


def load_trained_model(model_path):
    model = LaneDetectionCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    frame = transform(frame).unsqueeze(0)
    return frame

def detect_objects(frame, yolo_model):
    results = yolo_model(frame)  
    detections = results[0]  

    
    annotated_frame = frame.copy()
    for box in detections.boxes:  
        x1, y1, x2, y2 = map(int, box.xyxy[0])  
        confidence = box.conf[0]  
        class_id = int(box.cls[0])  
        label = f"{yolo_model.names[class_id]}: {confidence:.2f}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return annotated_frame

def process_video_with_cnn_and_object_detection(input_video_path, model_path, yolo_model_path):
    cap = cv2.VideoCapture(input_video_path)
    model = load_trained_model(model_path)
    yolo_model = YOLO(yolo_model_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_lane = cv2.VideoWriter('lane_detection_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    out_objects = cv2.VideoWriter('object_detection_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
     
        original_frame = frame.copy()

        height, width = frame.shape[:2]
        roi_vertices = [
            (0, height),
            (0, int(height * 0.5)),
            (width, int(height * 0.5)),
            (width, height)
        ]
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], (255, 255, 255))
        masked_frame = cv2.bitwise_and(frame, mask)
        x, y, w, h = cv2.boundingRect(np.array([roi_vertices], np.int32))
        roi_frame = masked_frame[y:y+h, x:x+w]
        preprocessed_frame = preprocess_frame(roi_frame)

        with torch.no_grad():
            mask_prediction = model(preprocessed_frame).squeeze().numpy()
        mask_resized = cv2.resize(mask_prediction, (w, h))
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        lower_road = np.array([0, 0, 50])
        upper_road = np.array([180, 50, 255])
        road_mask = cv2.inRange(hsv, lower_road, upper_road)
        road_mask_resized = cv2.resize(road_mask, (binary_mask.shape[1], binary_mask.shape[0]))
        combined_mask = cv2.bitwise_and(binary_mask, road_mask_resized)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.zeros_like(roi_frame)
        cv2.drawContours(contour_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        blue_color = np.array([255, 0, 0], dtype=np.uint8)
        roi_colored = np.where(contour_mask == 255, blue_color, roi_frame)
        edges = cv2.Canny(contour_mask, 100, 200)
        roi_colored[edges != 0] = [0, 0, 255]
        frame[y:y+h, x:x+w] = roi_colored

        out_lane.write(frame)

        detected_frame = detect_objects(original_frame, yolo_model)

        out_objects.write(detected_frame)

        cv2.imshow('Processed Frame with CNN Lane Detection', frame)
        cv2.imshow('Processed Frame with Object Detection', detected_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_lane.release()
    out_objects.release()
    cv2.destroyAllWindows()

# Set paths and run the video processing
input_video_path = "C://lane_detection//videos//vehicle.mp4"
model_path = "C://lane_detection//model//cnn_lane_model.pth"
yolo_model_path = "C://lane_detection//codes//yolov5s.pt"  # Pre-trained YOLOv5 model
process_video_with_cnn_and_object_detection(input_video_path, model_path, yolo_model_path)
