import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datatset import FundusDataset, preprocess  # Corrected import
from train import VGGWithAttention  # Import your VGG model with CBAM
import matplotlib.pyplot as plt

# Configuration
img_dir = 'C:\\Users\\pc\\Desktop\\messidor\\new\\data2'  # Directory with 'normal' and 'abnormal' subfolders
model_path = 'C:\\Users\\pc\\Desktop\\messidor\\best_model_with_attention.pth'  # Path to the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the test dataset with the correct transformation
test_dataset = FundusDataset(img_dir=img_dir, transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)  # Batch size of 1 for visualization

# Load the trained model
model = VGGWithAttention(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Register hooks to extract attention maps
attention_maps = []

def get_attention_maps_hook(module, input, output):
    attention_maps.append(output)

# Assuming your model has cbam layers named cbam1, cbam2, etc.
model.cbam1.register_forward_hook(get_attention_maps_hook)
model.cbam2.register_forward_hook(get_attention_maps_hook)
model.cbam3.register_forward_hook(get_attention_maps_hook)
model.cbam4.register_forward_hook(get_attention_maps_hook)

# Function to visualize images
def visualize_images(original_img, preprocessed_img, attention_maps):
    fig, axs = plt.subplots(1, len(attention_maps) + 2, figsize=(15, 5))

    # Display the original image
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    # Display the preprocessed image
    preprocessed_img_np = preprocessed_img.permute(1, 2, 0).cpu().numpy()
    preprocessed_img_np = (preprocessed_img_np - preprocessed_img_np.min()) / (preprocessed_img_np.max() - preprocessed_img_np.min())  # Normalize for display
    axs[1].imshow(preprocessed_img_np)  # Convert from CxHxW to HxWxC
    axs[1].set_title("Preprocessed Image")
    axs[1].axis('off')

    # Display attention maps
    for i, att_map in enumerate(attention_maps):
        # Normalize the attention map
        att_map = att_map.squeeze(0).mean(dim=0).cpu().numpy()  # Mean across channels
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())  # Normalize to [0, 1]

        axs[i+2].imshow(att_map, cmap='jet')  # Display attention map
        axs[i+2].set_title(f"Attention Map {i+1}")
        axs[i+2].axis('off')

    plt.tight_layout()
    plt.show()

# Test on a few images
def test_on_few_images(model, test_loader):
    data_iter = iter(test_loader)
    for _ in range(3):  # Test on 3 random images
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)

        # Clear previous attention maps
        global attention_maps
        attention_maps = []

        # Forward pass to generate attention maps
        with torch.no_grad():
            outputs = model(images)

        # Convert the original image back to CPU numpy format for visualization
        original_image = images.cpu().squeeze(0).permute(1, 2, 0).numpy()
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())  # Normalize for display

        # Visualize the preprocessed image and attention maps
        visualize_images(original_image, images.cpu().squeeze(0), attention_maps)

if __name__ == "__main__":
    test_on_few_images(model, test_loader)