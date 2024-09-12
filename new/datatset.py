import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2

# Cropping
def crop_black_background(image):
    gray = np.array(image.convert('L'))  # grayscale
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # Threshold to identify dark pixels
    coords = np.column_stack(np.where(thresh > 0))  # Find coordinates of non-black pixels
    
    if coords.size == 0:  # if  image is all black report error
        return "bruh"
    
    # Determine the bounding box of the non-black area and crop the image
    x0, y0, x1, y1 = coords[:, 1].min(), coords[:, 0].min(), coords[:, 1].max(), coords[:, 0].max()
    cropped_image = image.crop((x0, y0, x1, y1))
    return cropped_image

# Apply CLAHE
def apply_clahe(image):
    img_np = np.array(image)  # Convert to numpy array
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)  # Convert RGB to LAB color space
    l, a, b = cv2.split(lab)  # Split LAB image into channels
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    cl = clahe.apply(l)  # Apply CLAHE to the L-channel
    
    limg = cv2.merge((cl, a, b))  # Merge enhanced L-channel with a and b channels
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)  # Convert LAB image back to RGB
    final_image = Image.fromarray(final)  # Convert numpy array back to PIL image
    return final_image

# Custom dataset class for loading and preprocessing fundus images
class FundusDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Load images and assign labels
        normal_dir = os.path.join(img_dir, 'normal')
        abnormal_dir = os.path.join(img_dir, 'abnormal')

        for f in os.listdir(normal_dir):
            if os.path.isfile(os.path.join(normal_dir, f)):
                self.image_files.append(os.path.join(normal_dir, f))
                self.labels.append(0)  # Label normal  as 0

        for f in os.listdir(abnormal_dir):
            if os.path.isfile(os.path.join(abnormal_dir, f)):
                self.image_files.append(os.path.join(abnormal_dir, f))
                self.labels.append(1)  # Label abnormal as 1

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        # Preprocessingg
        image = crop_black_background(image)
        image = apply_clahe(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

# Define augmentation steps and resize for VGG. 224x224
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats????
])

# initialize
img_dir = 'C:\\Users\\pc\\Desktop\\messidor\\new\\data2' 
dataset = FundusDataset(img_dir=img_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)