import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from attention_model import AttnVGG
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from dataset import FundusDataset  # Import the FundusDataset class

# Set device
device = "cpu"

# Data preparation with augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Load dataset
test_dataset = FundusDataset(root_dir='data', transform=transform)
data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Initialize the model
model = AttnVGG(num_classes=1, normalize_attn=True, dropout=0.5)
model.load_state_dict(torch.load('models/fundus_model3.pth', map_location=device))
model = model.to(device)
model.eval()

# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_labels)

# Evaluate the model
all_preds = []
all_targets = []
misclassified_images = []

with torch.no_grad():
    for X, y in tqdm(data_loader):
        print(f"Processing batch with {X.size(0)} images")
        X, y = X.to(device), y.to(device)
        y_pred, a1, a2 = model(X)
        y_pred = y_pred.squeeze()
        y_pred_probs = torch.sigmoid(y_pred).detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()
        
        all_preds.append(y_pred_probs)
        all_targets.append(y_true)
        
        # Identify misclassified images
        y_pred_labels = (y_pred_probs > 0.5).astype(int)
        misclassified_indices = np.where(y_pred_labels != y_true)[0]
        for idx in misclassified_indices:
            misclassified_images.append((X[idx].cpu(), y_true[idx], y_pred_probs[idx], a1[idx].cpu(), a2[idx].cpu()))

# Check if all_preds and all_targets are not empty before concatenating
if all_preds and all_targets:
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    accuracy = calculate_accuracy(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_preds)

    print(f'Accuracy: {accuracy:.8f}')
    print(f'AUC: {auc:.8f}')
else:
    print("No predictions were made. Please check the dataset and DataLoader.")

# Function to unnormalize and display images
def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean  # Unnormalize
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.show()

# Function to visualize attention maps
def visualize_attention(img, attn_map, title):
    img = img.numpy().transpose((1, 2, 0))
    attn_map = attn_map.numpy().squeeze()
    attn_map = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
    attn_map = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    attn_map = cv2.cvtColor(attn_map, cv2.COLOR_BGR2RGB)
    vis = 0.6 * img + 0.4 * attn_map / 255
    vis = np.clip(vis, 0, 1)
    plt.imshow(vis)
    plt.title(title)
    plt.show()

print(f'Total misclassified images: {len(misclassified_images)}')
for img, true_label, pred_prob, a1, a2 in misclassified_images:
    imshow(img, f'True: {true_label}, Pred: {pred_prob:.2f}')
    visualize_attention(img, a1, 'Attention Map 1')
    visualize_attention(img, a2, 'Attention Map 2')