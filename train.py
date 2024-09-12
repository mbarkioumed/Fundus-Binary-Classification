import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from dataset import FundusDataset
from attention_model import AttnVGG
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import time
from tqdm import tqdm


batch_size = 32
learning_rate = 0.0001
epochs = 10
validation_split = 0.2


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


dataset = FundusDataset(root_dir='data', transform=transform)


train_size = int((1 - validation_split) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


model = AttnVGG(num_classes=1, normalize_attn=True, dropout=0.5)


device = "cpu"
model = model.to(device)  


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


criterion = FocalLoss(logits=True)  # Set logits=True for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)


def calculate_accuracy(y_true, y_pred):
    y_pred_labels = (y_pred > 0.5).astype(int)
    return accuracy_score(y_true, y_pred_labels)


start_time = time.time()

train_losses = []
train_auc = []
val_auc = []
val_accuracies = []

for epoch in range(epochs):
    model.train()
    train_preds = []
    train_targets = []
    auc_train = []
    loss_epoch_train = []

    
    for b, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X_train, y_train = X_train.to(device), y_train.to(device)  
        y_pred, _, _ = model(X_train)
        y_pred = y_pred.squeeze() 
        loss = criterion(torch.sigmoid(y_pred), y_train.float())   
        loss_epoch_train.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_preds.append(torch.sigmoid(y_pred).detach().cpu().numpy())
        train_targets.append(y_train.detach().cpu().numpy())

    train_preds = np.concatenate(train_preds)
    train_targets = np.concatenate(train_targets)
    auc_train = roc_auc_score(train_targets, train_preds)
    train_losses.append(np.mean(loss_epoch_train))
    train_auc.append(auc_train)
    print(f'Epoch: {epoch+1}/{epochs}, Loss: {np.mean(loss_epoch_train):.8f}, AUC: {auc_train:.8f}')

    
    model.eval()
    val_preds = []
    val_targets = []
    loss_epoch_val = []

    with torch.no_grad():
        for b, (X_val, y_val) in enumerate(val_loader):
            X_val, y_val = X_val.to(device), y_val.to(device) 
            y_pred, _, _ = model(X_val)
            y_pred = y_pred.squeeze()  
            loss = criterion(torch.sigmoid(y_pred), y_val.float())
            loss_epoch_val.append(loss.item())

            val_preds.append(torch.sigmoid(y_pred).detach().cpu().numpy())
            val_targets.append(y_val.detach().cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    auc_val = roc_auc_score(val_targets, val_preds)
    val_accuracy = calculate_accuracy(val_targets, val_preds)
    val_auc.append(auc_val)
    val_accuracies.append(val_accuracy)
    print(f'Epoch: {epoch+1}, Val Loss: {np.mean(loss_epoch_val):.8f}, AUC: {auc_val:.8f}, Accuracy: {val_accuracy:.8f}')

    
    scheduler.step()

print(f'\nDuration: {time.time() - start_time:.0f} seconds') 

# Save the model
torch.save(model.state_dict(), 'models/fundus_model5.pth')