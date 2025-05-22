import torch
import numpy as np
import cv2 as cv
from ultralytics.utils.plotting import Colors


class YOLOObjectDetector:

    def __init__(self, detection_quality_threshold: float):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")
        self.colors = Colors()
        self.detection_quality_threshold = detection_quality_threshold
        self.model.eval()
        # Set the confidence threshold attribute in the model
        self.model.conf = self.detection_quality_threshold

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        # Ensure the model's confidence threshold is set before inference
        self.model.conf = self.detection_quality_threshold
        with torch.no_grad():
            # The YOLO model includes built-in preprocessing that can accept a list of
            # OpenCV-style images directly, as long as we do BGR to RGB conversion
            result = self.model([cv.cvtColor(img, cv.COLOR_BGR2RGB)])
        
        # Process detections
        predictions = result.pandas().xyxy[0]  # Get the pandas DataFrame with results
        
        # Draw bounding boxes for each detection
        for i, row in predictions.iterrows():
            # Extract bounding box coordinates and convert to integers
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            # Get class information
            cls = int(row['class'])
            cls_name = row['name']
            conf = row['confidence']
            
            # Get color for this class
            color = self.colors(cls, True)  # True for BGR format
            
            # Draw rectangle around the object using class-specific color and LINE_AA
            cv.rectangle(img, (x1, y1), (x2, y2), color, 1, cv.LINE_AA)
            
            # Add label text with class name and confidence score, 5 pixels above the rectangle
            label_text = f"{cls_name} {conf:.2f}"
            cv.putText(img, label_text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

        return img
