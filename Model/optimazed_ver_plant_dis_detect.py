import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
import zipfile
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from torch.optim.lr_scheduler import StepLR

# Set optimal CPU settings
torch.set_num_threads(torch.get_num_threads())  # Use all available CPU cores
torch.backends.mkldnn.enabled = True  # Enable Intel MKL-DNN acceleration

# Define paths for local execution
zip_path = 'data/Plant_leaf_diseases_dataset_with_augmentation.zip'
extracted_path = 'data/extracted/Plant_Leaf_Dataset'

# Extract the ZIP file if not already done
if not os.path.exists(extracted_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    print(f"Dataset extracted to {extracted_path}")
else:
    print(f"Dataset already extracted at {extracted_path}")

# Optimized image transformations - smaller input size for faster processing
transform_train = transforms.Compose([
    transforms.Resize(128),  # Reduced from 255
    transforms.CenterCrop(112),  # Reduced from 224
    transforms.RandomHorizontalFlip(0.3),  # Light augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

transform_val = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets with different transforms
train_dataset = datasets.ImageFolder(extracted_path, transform=transform_train)
val_dataset = datasets.ImageFolder(extracted_path, transform=transform_val)

print(f"Dataset loaded with {len(train_dataset)} images and {len(train_dataset.classes)} classes")

# Optimized dataset splitting
total_size = len(train_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Create indices for splitting
indices = torch.randperm(total_size).tolist()
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

# Optimized MobileNet-inspired CNN model
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes):
        super(OptimizedCNN, self).__init__()
        
        # Depthwise separable convolutions for efficiency
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Depthwise separable block 1
            self._make_depthwise_block(32, 64, 1),
            self._make_depthwise_block(64, 128, 2),
            self._make_depthwise_block(128, 128, 1),
            self._make_depthwise_block(128, 256, 2),
            self._make_depthwise_block(256, 256, 1),
            
            # Global average pooling instead of large dense layers
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Much smaller classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_depthwise_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Set device (CPU optimized)
device = torch.device("cpu")
print(f"Using device: {device}")

# Initialize optimized model
num_classes = len(train_dataset.classes)
model = OptimizedCNN(num_classes)
model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Optimized training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Optimized data loaders with increased batch size for CPU
batch_size = 128  # Larger batch size for CPU efficiency
num_workers = min(4, torch.get_num_threads() // 2)  # Optimal for CPU

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, 
    sampler=SubsetRandomSampler(train_indices),
    num_workers=num_workers, pin_memory=False, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=batch_size,
    sampler=SubsetRandomSampler(val_indices),
    num_workers=num_workers, pin_memory=False, persistent_workers=True
)

test_loader = DataLoader(
    val_dataset, batch_size=batch_size,
    sampler=SubsetRandomSampler(test_indices),
    num_workers=num_workers, pin_memory=False, persistent_workers=True
)

# Optimized training function with mixed precision and gradient accumulation
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=20):
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Gradient accumulation steps for memory efficiency
    accumulation_steps = 2
    
    for epoch in range(epochs):
        start_time = datetime.now()
        
        # Training phase
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
        
        # Handle remaining gradients
        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        # Update learning rate
        scheduler.step()
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print progress
        elapsed_time = datetime.now() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Time: {elapsed_time}")
    
    return train_losses, val_losses, val_accuracies

# Optimized evaluation function
def evaluate_model(model, test_loader, device, classes):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

# Training
print("Starting optimized training...")
epochs = 20  # Reduced epochs with better model
train_losses, val_losses, val_accuracies = train_model(
    model, criterion, optimizer, scheduler, train_loader, val_loader, epochs
)

# Save the optimized model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'classes': train_dataset.classes
}, 'optimized_plant_disease_model.pt')
print("Optimized model saved!")

# Evaluate on test set
print("\nEvaluating on test set...")
test_metrics = evaluate_model(model, test_loader, device, train_dataset.classes)

# Plot training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nOptimization complete! Model has {trainable_params:,} parameters vs {50176*1024 + 1024*num_classes:,} in original.")