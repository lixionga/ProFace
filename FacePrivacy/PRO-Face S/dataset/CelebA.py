import os
from PIL import Image
from torch.utils.data import Dataset

class CelebAImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, img_name)
            for img_name in os.listdir(root_dir)
            if img_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
