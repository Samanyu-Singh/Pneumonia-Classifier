import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(root_dir, label)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(0 if label == 'NORMAL' else 1)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Separate transforms for training (with augmentation) and validation/testing
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Larger resize for random crop
    transforms.RandomCrop(224),     # Random crop for augmentation
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color variation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Clean transforms for validation/testing (no augmentation)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = PneumoniaDataset(root_dir='chest_xray/train', transform=train_transform)
test_dataset = PneumoniaDataset(root_dir='chest_xray/test', transform=val_test_transform)
val_dataset = PneumoniaDataset(root_dir='chest_xray/val', transform=val_test_transform)

# Optimize worker distribution for 12 CPU cores
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# Stronger regularization to prevent overfitting
model.fc = nn.Sequential(
    nn.Dropout(0.7),  # 70% dropout to prevent memorization
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),  # Additional dropout layer
    nn.Linear(512, 2)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# Stronger regularization to prevent overfitting
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
# Add learning rate scheduler to prevent overfitting
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

num_epochs = 10

if __name__ == '__main__':
    print("=== PNEUMONIA CLASSIFIER WITH ANTI-OVERFITTING MEASURES ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Track latest non-overfitted model
    latest_non_overfit_acc = 0
    latest_non_overfit_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training...")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

        print("Validating...")
        model.eval()
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                
                # Print validation progress
                if (batch_idx + 1) % 2 == 0:
                    print(f"  Validation batch {batch_idx + 1}/{len(val_loader)}")

        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f'Validation accuracy: {val_accuracy:.4f}')
        
        # Update learning rate based on validation performance
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # Track latest non-overfitted model
        if val_accuracy < 1.0:
            latest_non_overfit_acc = val_accuracy
            latest_non_overfit_epoch = epoch + 1
            # Save latest non-overfitted model
            torch.save(model.state_dict(), 'latest_non_overfit_model.pth')
            print(f'✅ Epoch {epoch+1}: Validation {val_accuracy:.4f} - saved (non-overfitted)')
        else:
            print(f'⚠️  Epoch {epoch+1}: Perfect validation {val_accuracy:.4f} - overfitting, not saving')

    print("\nTraining completed! Running final test...")
    
    # Load latest non-overfitted model for testing
    if latest_non_overfit_epoch > 0:
        model.load_state_dict(torch.load('latest_non_overfit_model.pth'))
        print(f"✅ Loaded latest non-overfitted model from epoch {latest_non_overfit_epoch}")
        print(f"   Validation accuracy: {latest_non_overfit_acc:.4f}")
    else:
        print("⚠️  No non-overfitted model found - using final model (may be overfitted)")
    
    model.eval()
    test_labels = []
    test_preds = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            test_labels.extend(labels.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            
            # Print test progress
            if (batch_idx + 1) % 5 == 0:
                print(f"  Test batch {batch_idx + 1}/{len(test_loader)}")

    test_accuracy = accuracy_score(test_labels, test_preds)
    print(f'Final test accuracy: {test_accuracy:.4f}')
    print("Saving model...")
    torch.save(model.state_dict(), 'pneumonia_detection_model.pth')
    print("Model saved successfully!")