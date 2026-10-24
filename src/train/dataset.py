from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def get_dataloaders(
    val_split: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""
    full_train = datasets.FashionMNIST(root="data", train=True, download=True, transform=_TRANSFORM)
    test_ds = datasets.FashionMNIST(root="data", train=False, download=True, transform=_TRANSFORM)

    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def preprocess_image(img: "PIL.Image.Image") -> torch.Tensor:
    """Preprocess a PIL image the same way training data is preprocessed."""
    from PIL import Image
    img = img.convert("L").resize((28, 28), Image.LANCZOS)
    return _TRANSFORM(img).unsqueeze(0)
