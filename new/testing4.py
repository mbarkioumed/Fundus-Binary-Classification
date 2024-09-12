import torch
from torch.utils.data import DataLoader
from datatset import FundusDataset, preprocess 
from train import VGGWithAttention 

# Setup dHYperParameters
img_dir = 'C:\\Users\\pc\\Desktop\\messidor\\data' 
model_path = 'C:\\Users\\pc\\Desktop\\messidor\\best_model_with_attention2.pth' 
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_transform = preprocess

test_dataset = FundusDataset(img_dir=img_dir, transform=test_transform)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load model
model = VGGWithAttention(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()  # evaluation mode


def test_model_accuracy(model, test_loader):
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():  # Disable gradient calculation for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Index of max probality 0 or  1.
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    accuracy = correct_preds / total_preds
    print(f'Test Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    test_model_accuracy(model, test_loader)
