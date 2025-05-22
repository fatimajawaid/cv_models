import cv2 as cv
import numpy as np
from pathlib import Path


class FaceDetector:
    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = cv.CascadeClassifier()
        self.model.load(str(model_path))

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_gray = cv.equalizeHist(img_gray)

        # Detect faces using the Haar cascade classifier
        faces = self.model.detectMultiScale(img_gray)
        
        # Draw rectangles around detected faces
        for (x, y, width, height) in faces:
            # Draw a red rectangle with 1-pixel width using LINE_AA for smoother lines
            cv.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 1, cv.LINE_AA)

        return img
