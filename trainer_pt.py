import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image dimensions
IMG_HEIGHT = 500
IMG_WIDTH = 400
IMG_CHANNELS = 3
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 10

# Paths
train_data_dir = "Data/spectrogram_images"  # Change to your directory path

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = ImageFolder(train_data_dir, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

validation_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

validation_dataset = ImageFolder(train_data_dir, transform=validation_transform)

validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(IMG_CHANNELS, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(32 * (IMG_HEIGHT // 8) * (IMG_WIDTH // 8), 64)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(32, 16)
        self.dropout3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(16, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.bn1(x)

        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.bn2(x)

        x = self.pool3(nn.functional.relu(self.conv3(x)))
        x = self.bn3(x)

        x = self.flatten(x)

        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)

        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)

        x = nn.functional.relu(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)

        return x

if __name__ == '__main__':
    # Create an instance of the model and move it to the GPU
    model = Net().to(device)

    # Define the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(validation_loader)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Model summary
    print(model)

    print("Training completed!")