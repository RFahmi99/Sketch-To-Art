"""
Sketch-to-Image Dataset
-----------------------
Custom Dataset for paired sketch/original images

Features:
- Automatic image pairing
- BGR to RGB conversion
- Joint transformations
"""

from torch.utils.data import Dataset
import cv2
import os
from PIL import Image

class SketchToImageDataset(Dataset):
    def __init__(self, sketch_dir, original_dir, transform=None):
        """
        Args:
            sketch_dir: Directory with sketch images
            original_dir: Directory with corresponding target images
            transform: Albumentations transform for both images
        """
        self.sketch_dir = sketch_dir
        self.original_dir = original_dir
        self.image_files = sorted(os.listdir(sketch_dir))  # Ensure matching order
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get paired paths
        sketch_path = os.path.join(self.sketch_dir, self.image_files[idx])
        original_path = os.path.join(self.original_dir, self.image_files[idx])

        # Read images
        sketch = cv2.imread(sketch_path)
        original = cv2.imread(original_path)

        # Convert BGR to RGB
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Apply identical transforms to both images
        if self.transform:
            transformed = self.transform(image=sketch, mask=original)
            sketch = transformed["image"]
            original = transformed["mask"]

        return sketch, original  # (input, target)