import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from dataset.GTZAN import GTZAN
from vit_model.VIT_LRP import vit_base_patch16_224 as vit_LRP
from tqdm import tqdm


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
        self.model = self.prepare_model(num_classes)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_dataset(self):
        train_dataset = GTZAN()
        val_dataset = GTZAN()  # Placeholder for validation dataset

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def prepare_model(self, num_classes):
        model = vit_LRP(pretrained=True)
        model.head = nn.Linear(model.head.in_features, num_classes)  # Replace the head for the num of classes in GTZAN
        model.to(self.device)
        return model

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            
            # Use tqdm to wrap the data loader for progress tracking
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", unit="batch")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

                # Update progress bar description with the current loss
                progress_bar.set_postfix(loss=loss.item())

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

if __name__ == '__main__':
    finetuner = FineTuneViT(model_name='vit_base_patch16_224', num_classes=10, dataset_path='../data/images_original')
    finetuner.train()