import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from torchvision.io import read_image, ImageReadMode
from pathlib import Path


class SimpleDataset(Dataset):
    def __init__(self, 
                 root, 
                 input_size=None, 
                 classes=None, 
                 images_subdir="images", 
                 transform=None,
                 include_metadata=False):
        self.root = Path(root)
        self.images_dir = self.root / images_subdir
        self._classes = classes
        self.include_metadata = include_metadata

        self.samples = []

        self.transform = transform
        if self.transform is None:
            self.transform = v2.Compose(
                [
                    v2.Resize(input_size if input_size is not None else 256, antialias=True),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.CenterCrop(input_size if input_size is not None else 224),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])

        samples = list(self.images_dir.glob("*/*.*"))
        print(f"Found {len(samples)} image files in {self.images_dir}")
        print(f"First 5 samples: {samples[:5]}")
        for file in samples:
            label = file.parent.name
            filename = file.name

            self.samples.append(
                {
                    "label": label,
                    "filename": filename,
                    "path": file.resolve(),
                }
            )

        if self._classes is None:
            seen = dict.fromkeys(s["label"] for s in self.samples)  # preserves insertion order
            self._classes = list(seen)

        self._class_to_idx = {c: i for i, c in enumerate(self._classes)}
        print(f"Loaded {len(self.samples)} samples from {self.images_dir}")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = read_image(str(sample["path"]), mode=ImageReadMode.RGB)

        if self.transform:
            img = self.transform(img)

        label = sample["label"]
        label_idx = self._class_to_idx[label]

        filename = sample["filename"]
        path = sample["path"]

        if self.include_metadata:
            return (
                img, 
                label_idx, 
                {
                    "filename": filename,
                    "path": str(path)
                }
            )
        else:
            return (
                img,
                label_idx
            )
            
    def __len__(self):
        return len(self.samples)

    def classes(self):
        return self._classes
