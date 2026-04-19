import os
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from config import CONFIG

LABEL_NAMES = {0: "Real", 3: "SD3", 4: "DALLE3"}

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

transform_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
])

def collate_train(batch):
    images   = [transform_train(row["Image"].convert("RGB")) for row in batch]
    labels_a = [row["Label_A"] for row in batch]
    labels_b = [row["Label_B"] for row in batch]
    return torch.stack(images), torch.tensor(labels_a), torch.tensor(labels_b)

def collate_eval(batch):
    images   = [transform_eval(row["Image"].convert("RGB")) for row in batch]
    labels_a = [row["Label_A"] for row in batch]
    labels_b = [row["Label_B"] for row in batch]
    return torch.stack(images), torch.tensor(labels_a), torch.tensor(labels_b)

def get_dataloaders():
    dataset = load_from_disk(CONFIG["save_path"])

    train_loader = DataLoader(
        dataset["train"],
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_train,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_eval,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader