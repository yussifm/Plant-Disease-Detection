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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
from torch.optim.lr_scheduler import StepLR
import multiprocessing
import sys

# Define local paths
zip_path = 'data/Plant_leaf_diseases_dataset_with_augmentation.zip'  # Update this to your ZIP file location
extracted_base = 'data/extracted/Plant_Leaf_Dataset'  #

def extract_dataset():
    """Extract the dataset if not already extracted"""
    # Create extraction directory if it doesn't exist
    os.makedirs(extracted_base, exist_ok=True)

    # Extract the zip file if not already extracted
    if not os.listdir(extracted_base):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_base)
            print(f"Dataset extracted to {extracted_base}")
        except Exception as e:
            print(f"Extraction failed: {e}")
            return None
    else:
        print(f"Dataset already extracted at {extracted_base}")
    
    return extracted_base

def find_dataset_root(start_path):
    """Function to find the dataset root directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    def is_image_file(filename):
        return any(filename.lower().endswith(ext) for ext in image_extensions)
    
    def has_image_files(directory):
        try:
            return any(is_image_file(f) for f in os.listdir(directory))
        except:
            return False
    
    subdirs = [d for d in os.listdir(start_path) if os.path.isdir(os.path.join(start_path, d))]
    
    if subdirs and all(has_image_files(os.path.join(start_path, d)) for d in subdirs):
        return start_path
    
    if len(subdirs) == 1:
        return find_dataset_root(os.path.join(start_path, subdirs[0]))
    
    return None

def explore_dataset(path):
    """Explore dataset structure"""
    classes = os.listdir(path)
    print(f"Found {len(classes)} classes:")
    total_images = 0
    for cls in classes[:10]:
        cls_path = os.path.join(path, cls)
        if os.path.isdir(cls_path):
            count = len(os.listdir(cls_path))
            total_images += count
            print(f"  {cls}: {count} images")
    if len(classes) > 10:
        print(f"  ... and {len(classes) - 10} more classes")
    print(f"Total images: {total_images}")

# Define the optimized CNN model
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes, input_size=112):
        super(OptimizedCNN, self).__init__()
        self.input_size = input_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            self._make_depthwise_block(32, 64, 1),
            self._make_depthwise_block(64, 128, 2),
            self._make_depthwise_block(128, 128, 1),
            self._make_depthwise_block(128, 256, 2),
            self._make_depthwise_block(256, 256, 1),
            self._make_depthwise_block(256, 512, 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _make_depthwise_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, 
                     groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=15):
    """Training function"""
    train_losses, val_losses, val_accuracies, val_top3_accuracies = [], [], [], []
    learning_rates, train_times = [], []
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("STARTING TRAINING...")
    print("="*60)
    sys.stdout.flush()  # Force output
    
    for epoch in range(epochs):
        print(f"\n--- EPOCH {epoch+1}/{epochs} ---")
        sys.stdout.flush()
        
        start_time = datetime.now()
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress tracking for training
        total_batches = len(train_loader)
        print(f"Training on {total_batches} batches...")
        sys.stdout.flush()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                current_loss = running_loss / (batch_idx + 1)
                current_acc = train_correct / train_total
                print(f"  Batch [{batch_idx+1}/{total_batches}] - Loss: {current_loss:.4f}, Acc: {current_acc:.4f}")
                sys.stdout.flush()
        
        train_loss = running_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        print(f"Training completed - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print("Starting validation...")
        sys.stdout.flush()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_correct_top3 = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                _, top3_pred = torch.topk(outputs, 3, dim=1)
                val_correct_top3 += (top3_pred == targets.view(-1, 1)).any(dim=1).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_top3_acc = val_correct_top3 / val_total
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        elapsed_time = datetime.now() - start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_top3_accuracies.append(val_top3_acc)
        learning_rates.append(current_lr)
        train_times.append(elapsed_time.total_seconds())
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print("*** NEW BEST MODEL SAVED! ***")
        
        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Top-3 Acc: {val_top3_acc:.4f}")
        print(f"  Time: {elapsed_time} | LR: {current_lr:.6f}")
        print(f"  Best Val Acc So Far: {best_val_acc:.4f}")
        print("-" * 50)
        sys.stdout.flush()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    sys.stdout.flush()
    
    return train_losses, val_losses, val_accuracies, val_top3_accuracies, learning_rates, train_times

def evaluate_model(model, test_loader, device, classes, criterion):
    """Evaluation function"""
    model.eval()
    all_preds, all_targets = [], []
    test_loss = 0.0
    correct_top3 = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            _, top3_pred = torch.topk(outputs, 3, dim=1)
            correct_top3 += (top3_pred == targets.view(-1, 1)).any(dim=1).sum().item()
    
    test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    top3_accuracy = correct_top3 / len(all_targets)
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    for t, p in zip(all_targets, all_preds):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    class_accuracy = [c / t if t > 0 else 0 for c, t in zip(class_correct, class_total)]
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=classes))
    
    return accuracy, top3_accuracy, class_accuracy, all_preds, all_targets

def create_plots(train_losses, val_losses, val_accuracies, val_top3_accuracies, 
                learning_rates, train_times, class_accuracy, classes, all_targets, all_preds):
    """Create and save plots"""
    # Training curves
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Top-1 Accuracy', color='green')
    plt.plot(val_top3_accuracies, label='Top-3 Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

    # Training time
    plt.figure(figsize=(8, 4))
    plt.plot(train_times, label='Training Time per Epoch', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_time.png')
    plt.close()

    # Class-wise accuracy and confusion matrix (only for <= 20 classes)
    if len(classes) <= 20:
        plt.figure(figsize=(12, 6))
        plt.bar(classes, class_accuracy)
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.title('Class-wise Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_accuracy.png')
        plt.close()
        
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[cls[:15] for cls in classes],
                    yticklabels=[cls[:15] for cls in classes])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()

def main():
    global device
    
    # Extract dataset
    if not extract_dataset():
        print("Cannot proceed due to dataset extraction issues.")
        return

    # Find the dataset root
    dataset_root = find_dataset_root(extracted_base)
    if dataset_root is None:
        print("Error: Could not find the dataset root directory.")
        print("Ensure the ZIP file contains class folders with images.")
        return
    else:
        print(f"Dataset root found at: {dataset_root}")

    # Explore dataset structure
    explore_dataset(dataset_root)

    # Check device availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Device-specific optimizations
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(torch.get_num_threads())
        torch.backends.mkldnn.enabled = True

    # Image transformations
    img_size = 128 if device.type == 'cuda' else 96
    crop_size = 112 if device.type == 'cuda' else 84

    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(dataset_root, transform=transform_train)
    val_dataset = datasets.ImageFolder(dataset_root, transform=transform_val)
    
    print(f"Dataset loaded with {len(train_dataset)} images and {len(train_dataset.classes)} classes")
    print(f"Classes: {train_dataset.classes[:5]}..." if len(train_dataset.classes) > 5 else train_dataset.classes)

    # Dataset splitting
    total_size = len(train_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    print(f"\nDataset split:")
    print(f"  Training: {train_size:,} ({train_size/total_size*100:.1f}%)")
    print(f"  Validation: {val_size:,} ({val_size/total_size*100:.1f}%)")
    print(f"  Test: {test_size:,} ({test_size/total_size*100:.1f}%)")

    torch.manual_seed(42)
    indices = torch.randperm(total_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Initialize model
    num_classes = len(train_dataset.classes)
    model = OptimizedCNN(num_classes, crop_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.5)

    batch_size = 64 if device.type == 'cuda' else 32
    # Set num_workers to 0 for Windows to avoid multiprocessing issues
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices),
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices),
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(test_indices),
        num_workers=num_workers, pin_memory=(device.type == 'cuda')
    )

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")
    print(f"  Total training batches per epoch: {len(DataLoader(train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices)))}")
    print(f"  Total validation batches per epoch: {len(DataLoader(val_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices)))}")
    
    # Force output buffer flush
    sys.stdout.flush()

    # Train the model
    epochs = 15
    print(f"\nAbout to start training for {epochs} epochs...")
    sys.stdout.flush()
    
    train_losses, val_losses, val_accuracies, val_top3_accuracies, learning_rates, train_times = train_model(
        model, criterion, optimizer, scheduler, train_loader, val_loader, epochs
    )

    # Evaluate with the best model
    model.load_state_dict(torch.load('best_model.pt'))
    accuracy, top3_accuracy, class_accuracy, all_preds, all_targets = evaluate_model(
        model, test_loader, device, train_dataset.classes, criterion
    )

    # Create plots
    create_plots(train_losses, val_losses, val_accuracies, val_top3_accuracies, 
                learning_rates, train_times, class_accuracy, train_dataset.classes, 
                all_targets, all_preds)

    # Save the final model
    final_model_path = 'optimized_plant_disease_model.pt'
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to: {final_model_path}")
    print(f"Training completed! Best validation accuracy: {max(val_accuracies):.4f}")

    # Sample prediction
    model.eval()
    with torch.no_grad():
        sample_idx = np.random.randint(0, len(test_indices))
        image, true_label = val_dataset[test_indices[sample_idx]]
        image_batch = image.unsqueeze(0).to(device)
        output = model(image_batch)
        _, predicted = torch.max(output, 1)
        
        predicted_class = train_dataset.classes[predicted.item()]
        true_class = train_dataset.classes[true_label]
        confidence = torch.softmax(output, 1)[0][predicted].item()
        
        print(f"\nSample Prediction:")
        print(f"  True class: {true_class}")
        print(f"  Predicted class: {predicted_class}")
        print(f"  Confidence: {confidence:.4f}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Required for Windows
    main()