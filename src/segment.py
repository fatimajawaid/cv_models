import torch
import numpy as np
import cv2 as cv
import torchvision
from matplotlib import cm


class SemanticSegmenter:
    def __init__(self, alpha=0.8):
        self.weight = torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        self.model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(weights=self.weight)
        self.transforms = self.weight.transforms()
        self.alpha = alpha
        self.model.eval()

        self.color_per_class_bgr = np.concatenate([
            np.array([[0, 0, 0]]),  # background
            np.array(cm.get_cmap('tab20').colors)[:, ::-1] * 255
        ], axis=0).astype(np.uint8)

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        torch_img = torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()
        torch_img = self.transforms(torch_img)
        with torch.no_grad():
            result = self.model(torch.stack([torch_img], dim=0))
        
        # Extract the "out" key from the result dictionary which contains the logits
        logits = result["out"][0]  # Shape: [num_classes, height, width]
        
        # Get the most likely class for each pixel
        class_predictions = torch.argmax(logits, dim=0).cpu().numpy()  # Shape: [height, width]
        
        # Create a color map by mapping each class ID to its corresponding color
        colored_segmentation = np.zeros((class_predictions.shape[0], class_predictions.shape[1], 3), dtype=np.uint8)
        
        # Apply the color mapping using vectorized operations
        for class_id in range(len(self.color_per_class_bgr)):
            mask = (class_predictions == class_id)
            if np.any(mask):
                colored_segmentation[mask] = self.color_per_class_bgr[class_id]
        
        # Resize the colored segmentation to match the original image dimensions if needed
        if colored_segmentation.shape[:2] != img.shape[:2]:
            colored_segmentation = cv.resize(
                colored_segmentation, 
                (img.shape[1], img.shape[0]), 
                interpolation=cv.INTER_NEAREST
            )
        
        # Blend the original image with the colorized output using alpha blending
        # img × (1−self.alpha) + colored_segmentation × self.alpha
        blended_image = cv.addWeighted(
            img, 1 - self.alpha,
            colored_segmentation, self.alpha,
            0
        )
        
        return blended_image
