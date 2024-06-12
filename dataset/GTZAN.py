import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GTZAN(Dataset):
    def __init__(self, mode):
        super(GTZAN, self).__init__()

        self.path = f'./data/images_1600_224/{mode}'
        self.data = []
        self.labels = []
        self.label_map = {}

        genres = sorted(os.listdir(self.path))
        #genres = os.listdir(self.path)
        
        self.label_map = {genre: idx for idx, genre in enumerate(genres)}
        self.genres = genres
        for genre in genres:
            genre_path = os.path.join(self.path, genre)
            if os.path.isdir(genre_path):
                n_sample = len(os.listdir(genre_path))
                for img_file in os.listdir(genre_path):
                    if img_file.endswith('.png'):
                        self.data.append(os.path.join(genre_path, img_file))
                        self.labels.append(self.label_map[genre])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img_path = self.data[item]
        label = self.labels[item]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()