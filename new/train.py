import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from datatset import FundusDataset, preprocess 
from cbam import CBAM  #CBAM attentinon

# Configuration
img_dir = 'C:\\Users\\pc\\Desktop\\messidor\\new\\data2'  
num_classes = 2  # Binary classification
batch_size = 32
num_epochs = 10
learning_rate = 0.001 #TRY TO LOWER THIS to 0.0001 SO WE DONT MISS THE OPTIMAL SOL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # for optimizing. maybe install cuda??
model_save_path = 'best_model_with_attention2.pth'  
validation_split = 0.2  # Fraction of the data to be used for validation

# Load the dataset
dataset = FundusDataset(img_dir=img_dir, transform=preprocess)

# Split the dataset into training and validation sets
total_size = len(dataset)
val_size = int(validation_split * total_size)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Initialize the VGG model (pretrained on ImageNet)
class VGGWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(VGGWithAttention, self).__init__()
        self.vgg = models.vgg16(pretrained=True)

        # Add CBAM after some convolutional layers
        self.cbam1 = CBAM(in_channels=64)
        self.cbam2 = CBAM(in_channels=128)
        self.cbam3 = CBAM(in_channels=256)
        self.cbam4 = CBAM(in_channels=512)

        # Modify the final layer for binary classification
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, num_classes)

    def forward(self, x):
        # VGG's features block
        x = self.vgg.features[0:5](x)   # First two conv layers
        x = self.cbam1(x)               # Apply CBAM after the first block
        x = self.vgg.features[5:10](x)  # Next two conv layers
        x = self.cbam2(x)               # Apply CBAM after the second block
        x = self.vgg.features[10:17](x) # Next three conv layers
        x = self.cbam3(x)               # Apply CBAM after the third block
        x = self.vgg.features[17:24](x) # Next three conv layers
        x = self.cbam4(x)               # Apply CBAM after the fourth block
        x = self.vgg.features[24:](x)   # Remaining layers of VGG
        x = self.vgg.avgpool(x)         # Average pooling layer
        x = torch.flatten(x, 1)
        x = self.vgg.classifier(x)      # Fully connected layers
        return x

# Instantiate the model
model = VGGWithAttention(num_classes=num_classes).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to evaluate the model on the validation set
def evaluate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct_preds / total_preds
    return val_loss, val_accuracy

# Training loop 
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()  #training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # reset adam
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # BackPropagation and optimizatio
            loss.backward()
            optimizer.step()

            #  running loss
            running_loss += loss.item()

        # Calculate validation loss and accuracy
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Save the model if it has the best validation accuracy so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation accuracy: {val_accuracy:.4f}')

    print("Training complete.")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path)
