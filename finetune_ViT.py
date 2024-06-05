import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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

        self.train_loader, self.val_loader, self.test_loader = self.load_dataset()

        self.model = self.prepare_model(num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.best_val_accuracy = 0.0  # Track the best validation accuracy

    def load_dataset(self):
        ds = GTZAN()

        total_len = len(ds)
        train_len = int(0.7 * total_len)
        val_len = int(0.15 * total_len)
        test_len = total_len - train_len - val_len
        
        train_set, val_set, test_set = random_split(ds, [train_len, val_len, test_len])
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def prepare_model(self, num_classes):
        model = vit_LRP(pretrained=True).to(self.device)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model.to(self.device)
        return model

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", unit="batch")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * images.size(0)

                progress_bar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / len(self.train_loader.dataset)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            val_accuracy = self.validate()

            # Save the model if the validation accuracy is the best we've seen so far
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.save_model(f"best_model_epoch_{epoch + 1}.pth")

    def validate(self):
        self.model.eval()
        running_corrects = 0
        progress_bar = tqdm(self.val_loader, desc="Validation", unit="batch")
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(accuracy=running_corrects.double() / len(self.val_loader.dataset))

        accuracy = running_corrects.double() / len(self.val_loader.dataset)
        print(f"Validation Accuracy: {accuracy:.4f}")
        return accuracy

    def test(self):
        self.model.eval()
        running_corrects = 0
        progress_bar = tqdm(self.test_loader, desc="Test", unit="batch")
        with torch.no_grad():
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

                progress_bar.set_postfix(accuracy=running_corrects.double() / len(self.test_loader.dataset))

        accuracy = running_corrects.double() / len(self.test_loader.dataset)
        print(f"Test Accuracy: {accuracy:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

if __name__ == '__main__':
    finetuner = FineTuneViT(model_name='vit_base_patch16_224', num_classes=10, dataset_path='../data/images_original')
    finetuner.train()
    finetuner.test()
