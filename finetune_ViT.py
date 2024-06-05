import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dataset import GTZAN
from vit_model import VIT_LRP as ViT, default_cfgs

class FineTuneViT:
    def __init__(self, model_name, num_classes, dataset_path, batch_size=32, learning_rate=1e-4, num_epochs=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Load the dataset
        self.train_loader, self.val_loader = self.load_dataset()

        # Prepare the model
        self.model = self.prepare_model(model_name, num_classes)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = GTZAN(self.dataset_path, transform=transform)
        val_dataset = GTZAN(self.dataset_path, transform=transform)  # Placeholder for validation dataset

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def prepare_model(self, model_name, num_classes):
        model_cfg = default_cfgs[model_name]
        model = ViT(model_cfg)
        model.head = nn.Linear(model.head.in_features, num_classes)  # Replace the head for the num of classes in GTZAN
        model.to(self.device)
        return model

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            self.validate()

    def validate(self):
        self.model.eval()
        running_corrects = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        accuracy = running_corrects.double() / len(self.val_loader.dataset)
        print(f"Validation Accuracy: {accuracy:.4f}")