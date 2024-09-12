import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as utils
import torch.nn.functional as F
import cv2
import numpy as np
from attention_model import AttnVGG

# Load the trained model
model = AttnVGG(num_classes=1, normalize_attn=True, dropout=0.5)
model.load_state_dict(torch.load('models/fundus_model3.pth'))
model.eval()

# Function to apply CLAHE
def apply_clahe(image):
    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(image_np)

# Function to visualize attention maps
def visualize_attention(I_train, a, up_factor, no_attention=False):
    img = I_train.permute((1, 2, 0)).cpu().numpy()
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=8, normalize=True, scale_each=True)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    attn = cv2.resize(attn, (img.shape[1], img.shape[0]))

    if no_attention:
        return torch.from_numpy(attn)
    else:
        vis = 0.6 * img + 0.4 * attn
        return torch.from_numpy(vis)

# Function to process and visualize attention maps for an image
def process_and_visualize(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the original image for visualization
    original_image = Image.open(image_path).convert('RGB')
    
    # Apply CLAHE preprocessing to the image before feeding it to the model
    clahe_image = apply_clahe(original_image)
    image_tensor = transform(clahe_image).unsqueeze(0)

    with torch.no_grad():
        block1 = model.conv_block1(image_tensor)
        pool1 = F.max_pool2d(block1, 2, 2)
        block2 = model.conv_block2(pool1)
        pool2 = F.max_pool2d(block2, 2, 2)
        block3 = model.conv_block3(pool2)
        pool3 = F.max_pool2d(block3, 2, 2)
        block4 = model.conv_block4(pool3)
        pool4 = F.max_pool2d(block4, 2, 2)
        block5 = model.conv_block5(pool4)
        pool5 = F.max_pool2d(block5, 2, 2)
        
        g = model.pool(pool5).view(pool5.size(0), 512)
        a1, g1 = model.attn1(pool3, pool5)
        a2, g2 = model.attn2(pool4, pool5)
        
        # Concatenate g, g1, and g2 to form g_hat
        g_hat = torch.cat((g, g1, g2), dim=1)
        
        # Get the model's prediction using the correct final layer attribute
        prediction = model.cls(g_hat)
        prediction = torch.sigmoid(prediction)

    I_train = image_tensor.squeeze(0)
    orig = visualize_attention(I_train, a1, up_factor=2, no_attention=True)
    first = visualize_attention(I_train, a1, up_factor=2, no_attention=False)
    second = visualize_attention(I_train, a2, up_factor=4, no_attention=False)
    attention_only = visualize_attention(I_train, a1, up_factor=2, no_attention=True)

    # Corrected dimension ordering for plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))
    ax1.imshow(np.array(clahe_image))  # Display the CLAHE image
    ax2.imshow(first.numpy())
    ax3.imshow(second.numpy())
    ax4.imshow(attention_only.numpy())  # Display the attention map only
    ax1.title.set_text('CLAHE Image')
    ax2.title.set_text('pool-3 attention')
    ax3.title.set_text('pool-4 attention')
    ax4.title.set_text('Attention Map Only')
    
    # Display the prediction
    plt.suptitle(f'Prediction: {prediction.item():.4f}', fontsize=16)
    plt.show()

# Example usage
process_and_visualize('data/abnormal/20051020_58214_0100_PP.tif')