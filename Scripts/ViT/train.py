import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import CONFIG, LABEL_NAMES, device
from data_loader import get_dataloaders
from model import get_vit_model

# Required for num_workers > 0 on Windows
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────
    model     = get_vit_model(CONFIG["num_classes"], CONFIG["freeze_backbone"]).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["lr"],
    weight_decay=CONFIG["weight_decay"]   # penalizes large weights
)

    scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",       # minimize val loss
    factor=0.5,       # halve lr when plateauing
    patience=2,       # wait 2 epochs before dropping lr
)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ── Train & eval functions ─────────────────────────────
    def train_epoch(model, loader):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels_a, labels_b in tqdm(loader, desc="Training"):
            images, labels_a = images.to(device), labels_a.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels_a)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (outputs.argmax(1) == labels_a).sum().item()
            total      += labels_a.size(0)

        return total_loss / len(loader), correct / total

    def eval_epoch(model, loader, desc="Validation"):
        model.eval()
        total_loss, correct, total = 0, 0, 0
        label_b_correct, label_b_total = {}, {}

        with torch.no_grad():
            for images, labels_a, labels_b in tqdm(loader, desc=desc):
                images, labels_a = images.to(device), labels_a.to(device)

                outputs = model(images)
                loss    = criterion(outputs, labels_a)
                preds   = outputs.argmax(1)

                total_loss += loss.item()
                correct    += (preds == labels_a).sum().item()
                total      += labels_a.size(0)

                for pred, true_a, true_b in zip(preds.cpu(), labels_a.cpu(), labels_b):
                    true_b = true_b.item()
                    label_b_correct[true_b] = label_b_correct.get(true_b, 0) + (pred == true_a).item()
                    label_b_total[true_b]   = label_b_total.get(true_b, 0) + 1

        label_b_acc = {k: label_b_correct[k] / label_b_total[k] for k in label_b_total}
        return total_loss / len(loader), correct / total, label_b_acc

    # ── Training loop ─────────────────────────────────────
    os.makedirs(os.path.dirname(CONFIG["checkpoint_path"]), exist_ok=True)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "label_b_acc": []
    }

    best_val_loss     = float("inf")
    best_val_acc      = 0.0
    epochs_no_improve = 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        train_loss, train_acc          = train_epoch(model, train_loader)
        val_loss, val_acc, label_b_acc = eval_epoch(model, val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["label_b_acc"].append({str(k): v for k, v in label_b_acc.items()})

        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}")
        print(f"  Val accuracy by source:")
        for label, acc in label_b_acc.items():
            print(f"    {LABEL_NAMES.get(label, label)}: {acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            best_val_acc      = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"  Saved best model (val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{CONFIG['patience']} epochs")
            if epochs_no_improve >= CONFIG["patience"]:
                print(f"\n Early stopping triggered at epoch {epoch+1}")
                break

    # Save history to JSON so notebook can plot it later
    history_path = os.path.join(os.path.dirname(CONFIG["checkpoint_path"]), "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    # ── Test evaluation ────────────────────────────────────
    print("\nRunning test evaluation...")
    model.load_state_dict(torch.load(CONFIG["checkpoint_path"]))
    test_loss, test_acc, test_label_b_acc = eval_epoch(model, test_loader, desc="Testing")

    print(f"\nTest loss: {test_loss:.4f}  acc: {test_acc:.4f}")
    print("\nTest accuracy by source:")
    for label, acc in test_label_b_acc.items():
        print(f"  {LABEL_NAMES.get(label, label)}: {acc:.4f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}  acc: {best_val_acc:.4f}")