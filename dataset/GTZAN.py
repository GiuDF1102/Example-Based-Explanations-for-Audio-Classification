import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GTZAN(Dataset):
    def __init__(self, path):
        super(self).__init__()

        self.path = path
        self.data = []
        self.labels = []
        self.label_map = {}

        genres = os.listdir(path)
        self.label_map = {genre: idx for idx, genre in enumerate(genres)}

        for genre in genres:
            genre_path = os.path.join(path, genre)
            if os.path.isdir(genre_path):
                for img_file in os.listdir(genre_path):
                    if img_file.endswith('.png'):
                        self.data.append(os.path.join(genre_path, img_file))
                        self.labels.append(self.label_map[genre])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
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

if __name__ == '__main__':
    ds = GTZAN('../data/images_original')
    sample_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=5,
        shuffle=False
    )

    iterator = iter(sample_loader)
    images, labels = next(iterator)

    print(images.shape)
    print(labels.shape)
    print(len(ds))
