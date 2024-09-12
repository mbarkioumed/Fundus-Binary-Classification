import os
from PIL import Image
from torch.utils.data import Dataset
import cv2
import numpy as np

class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['normal', 'abnormal']
        self.image_paths = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Apply CLAHE
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        image_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # Convert numpy array back to PIL image
        image = Image.fromarray(image_np)

        if self.transform:
            image = self.transform(image)

        return image, label