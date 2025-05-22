import numpy as np
import cv2 as cv
import os
import shutil


class SunglassesAnnotator:
    def __init__(self, sunglasses_path: str, keypoint_model_path: str = "download"):
        # Using OpenCV's face detector (Haar cascade)
        self.face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
        
        # Read the sunglasses image with alpha channel
        self.sunglasses = cv.imread(sunglasses_path, cv.IMREAD_UNCHANGED)
        
        # Check if sunglasses image exists and has alpha channel
        if self.sunglasses is None:
            raise ValueError(f"Could not read sunglasses image: {sunglasses_path}")
        
        # If the image doesn't have an alpha channel, add one
        if self.sunglasses.shape[2] == 3:
            self.sunglasses = cv.cvtColor(self.sunglasses, cv.COLOR_BGR2BGRA)

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        # Convert to grayscale for face detection
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Create a copy of the input image for output
        output_img = img.copy()
        
        # Detect faces using Haar cascade
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # For each detected face, apply sunglasses
        for (x, y, w, h) in faces:
            # Calculate position for sunglasses
            # Position the glasses on the upper part of the face, where eyes would be
            glasses_width = int(w * 1.0)  # Make glasses slightly smaller than face width
            
            # Scale height proportionally to maintain aspect ratio
            scale_factor = glasses_width / self.sunglasses.shape[1]
            glasses_height = int(self.sunglasses.shape[0] * scale_factor)
            
            # Position glasses at approximately eye level (around 25% from the top of the face)
            x_offset = x + int((w - glasses_width) / 2)  # Center horizontally
            y_offset = y + int(h * 0.25)  # Position at approximately eye level
            
            # Resize sunglasses to fit the face
            resized_glasses = cv.resize(self.sunglasses, (glasses_width, glasses_height))
            
            # Apply the sunglasses overlay to the image
            self.overlay_transparent(output_img, resized_glasses, x_offset, y_offset)
            
        return output_img
    
    def overlay_transparent(self, background_img, overlay_img, x_offset, y_offset):
        """
        Overlay a transparent PNG image onto another image
        """
        # Get dimensions of overlay image
        h, w = overlay_img.shape[:2]
        
        # Ensure overlay fits within the background image
        if x_offset < 0:
            overlay_img = overlay_img[:, abs(x_offset):]
            w = overlay_img.shape[1]
            x_offset = 0
        if y_offset < 0:
            overlay_img = overlay_img[abs(y_offset):, :]
            h = overlay_img.shape[0]
            y_offset = 0
        
        if x_offset + w > background_img.shape[1]:
            overlay_img = overlay_img[:, :background_img.shape[1] - x_offset]
            w = overlay_img.shape[1]
        if y_offset + h > background_img.shape[0]:
            overlay_img = overlay_img[:background_img.shape[0] - y_offset, :]
            h = overlay_img.shape[0]
        
        # Extract region of interest from background
        if h > 0 and w > 0:
            roi = background_img[y_offset:y_offset + h, x_offset:x_offset + w]
            
            # Skip if dimensions don't match
            if roi.shape[:2] != overlay_img.shape[:2]:
                return
            
            # Split alpha channel from overlay
            overlay_rgb = overlay_img[:, :, :3]  # RGB channels
            
            # Handle case where overlay might not have 4 channels
            if overlay_img.shape[2] == 4:
                overlay_alpha = overlay_img[:, :, 3] / 255.0  # Alpha channel, normalized
                
                # Reshape alpha for broadcasting
                alpha = overlay_alpha[:, :, np.newaxis]
                
                # Blend the background with the overlay based on alpha
                blended = (1.0 - alpha) * roi + alpha * overlay_rgb
                
                # Replace the region in the output image with the blended region
                background_img[y_offset:y_offset + h, x_offset:x_offset + w] = blended
