"""Readable utilities reused after the first explicit training-loop lessons.

Open this file while studying: these are ordinary PyTorch functions, not a
training framework. All digit lessons use the same stratified 70/15/15 split.
"""
from pathlib import Path
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SMOKE_TEST = os.environ.get("COURSE_SMOKE_TEST", "0") == "1"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def seed_everything(seed=42):
    """Reduce run-to-run variation; hardware/library changes may still differ."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Small teaching models are usually faster without dozens of CPU threads.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def get_device():
    """COURSE_DEVICE=cpu is useful for testing the portable fallback."""
    requested = os.environ.get("COURSE_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def digit_splits(seed=42):
    """Return normalized images and labels; split BEFORE fitting any model."""
    digits = load_digits()
    images = (digits.images / 16.0).astype(np.float32)
    labels = digits.target.astype(np.int64)
    indices = np.arange(len(labels))
    train, held = train_test_split(indices, test_size=0.30,
                                   stratify=labels, random_state=seed)
    val, test = train_test_split(held, test_size=0.50,
                                stratify=labels[held], random_state=seed)
    result = {}
    for name, idx in (("train", train), ("val", val), ("test", test)):
        limit = 128 if name == "train" else 64
        if SMOKE_TEST and len(idx) > limit:
            idx, _ = train_test_split(idx, train_size=limit,
                                      stratify=labels[idx], random_state=seed)
        result[name] = (images[idx], labels[idx])
    return result


class DigitDataset(Dataset):
    def __init__(self, images, labels, image_size=8, rgb=False, transform=None):
        self.images = torch.from_numpy(images).unsqueeze(1)
        if image_size != 8:
            self.images = nn.functional.interpolate(
                self.images, size=(image_size, image_size),
                mode="bilinear", align_corners=False)
        if rgb:
            self.images = self.images.repeat(1, 3, 1, 1)
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.images[index].clone()
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[index]


def get_digits_loaders(batch_size=64, image_size=8, rgb=False, seed=42,
                       train_transform=None):
    loaders = []
    for name, (images, labels) in digit_splits(seed).items():
        dataset = DigitDataset(images, labels, image_size, rgb,
                               train_transform if name == "train" else None)
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train",
            num_workers=0, pin_memory=get_device().type == "cuda",
            generator=torch.Generator().manual_seed(seed)))
    return tuple(loaders)


class TinyCNN(nn.Module):
    """Compact classifier for 8x8 or larger images. Returns raw logits."""
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)))
        self.classifier = nn.Linear(32 * 2 * 2, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


@torch.inference_mode()
def evaluate(model, loader, device=None):
    device = get_device() if device is None else torch.device(device)
    model.to(device).eval()
    loss_sum, count = 0.0, 0
    all_logits, all_targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        count += y.size(0)
        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())
    if not count:
        raise ValueError("Evaluation requires at least one sample.")
    logits, targets = torch.cat(all_logits), torch.cat(all_targets)
    return {"loss": loss_sum / count,
            "accuracy": (logits.argmax(1) == targets).float().mean().item(),
            "logits": logits, "targets": targets}


def fit(model, train_loader, val_loader, epochs=5, lr=1e-3,
        weight_decay=0.0, device=None):
    """Train, track sample-weighted metrics, restore best validation-loss state."""
    device = get_device() if device is None else torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay)
    history, best_state, best_loss = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, count = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += y.size(0)
        if count == 0:
            raise ValueError("Training requires at least one sample.")
        validation = evaluate(model, val_loader, device)
        row = dict(epoch=epoch, train_loss=total_loss / count,
                   train_acc=correct / count, val_loss=validation["loss"],
                   val_acc=validation["accuracy"])
        history.append(row)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:02d} | train loss {row['train_loss']:.3f} | "
              f"val loss {row['val_loss']:.3f} | val acc {row['val_acc']:.1%}")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history
