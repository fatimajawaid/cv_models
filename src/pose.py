import torch
import numpy as np
import cv2 as cv
import torchvision


class PoseEstimator:
    coco_keypoints = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]

    coco_skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 7),
        (5, 6),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]

    def __init__(self, detection_quality_threshold: float, keypoint_quality_threshold: float):
        self.weight = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=self.weight)
        self.transforms = self.weight.transforms()
        self.detection_quality_threshold = detection_quality_threshold
        self.keypoint_quality_threshold = keypoint_quality_threshold
        self.model.eval()

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        torch_img = (
            torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()
        )
        torch_img = self.transforms(torch_img)
        with torch.no_grad():
            result = self.model([torch_img])[0]
        
        # Process each detection that meets the quality threshold
        for idx, score in enumerate(result["scores"]):
            # Check if detection score is above threshold
            if score >= self.detection_quality_threshold:
                # Get the bounding box and convert to integers
                box = result["boxes"][idx].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                
                # Get the label and keypoints
                label = int(result["labels"][idx].cpu().numpy())
                keypoints = result["keypoints"][idx].cpu().numpy()
                keypoint_scores = result["keypoints_scores"][idx].cpu().numpy()
                
                # Draw red rectangle around the detected object
                cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1, cv.LINE_AA)
                
                # Add label text at top-left corner with exact formatting from assignment
                label_text = f"person {score:.2f}"
                cv.putText(img, label_text, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                
                # If the label is "person" (COCO class 1), draw the skeleton
                if label == 1:  # 1 is the COCO id for person
                    # Draw keypoints that meet confidence threshold
                    for kp_idx, (kp, kp_score) in enumerate(zip(keypoints, keypoint_scores)):
                        if kp_score >= self.keypoint_quality_threshold:
                            # Draw circle at keypoint location with parameters from assignment
                            x, y = int(kp[0]), int(kp[1])
                            cv.circle(img, (x, y), 3, (0, 255, 0), 1, cv.LINE_AA)
                    
                    # Draw skeleton lines between keypoints
                    for (start_idx, end_idx) in self.coco_skeleton:
                        # Only draw if both keypoints meet the threshold
                        if (keypoint_scores[start_idx] >= self.keypoint_quality_threshold and 
                            keypoint_scores[end_idx] >= self.keypoint_quality_threshold):
                            # Get keypoint coordinates
                            x1, y1 = int(keypoints[start_idx][0]), int(keypoints[start_idx][1]) 
                            x2, y2 = int(keypoints[end_idx][0]), int(keypoints[end_idx][1])
                            # Draw line with parameters from assignment
                            cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1, cv.LINE_AA)

        return img
