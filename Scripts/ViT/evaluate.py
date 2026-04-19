import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
from config import CONFIG, LABEL_NAMES, device
from data_loader import get_dataloaders
from model import get_vit_model

if __name__ == "__main__":

    # ── Load model ────────────────────────────────────────
    model = get_vit_model(CONFIG["num_classes"], freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(CONFIG["checkpoint_path"], map_location=device))
    model.eval()

    _, _, test_loader = get_dataloaders()

    # ── Collect predictions ───────────────────────────────
    all_preds    = []
    all_labels_a = []
    all_labels_b = []
    all_probs    = []

    with torch.no_grad():
        for images, labels_a, labels_b in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs   = torch.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels_a.extend(labels_a.numpy())
            all_labels_b.extend(labels_b.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # prob of AI class

    all_preds    = np.array(all_preds)
    all_labels_a = np.array(all_labels_a)
    all_labels_b = np.array(all_labels_b)
    all_probs    = np.array(all_probs)

    # ── Overall metrics ───────────────────────────────────
    accuracy  = accuracy_score(all_labels_a, all_preds)
    precision = precision_score(all_labels_a, all_preds)
    recall    = recall_score(all_labels_a, all_preds)
    f1        = f1_score(all_labels_a, all_preds)
    roc_auc   = roc_auc_score(all_labels_a, all_probs)

    print("=" * 50)
    print("OVERALL TEST METRICS")
    print("=" * 50)
    print(f"  Accuracy:  {accuracy:.4f}  (human baseline: 0.62)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}  (target: 0.75)")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # ── Per source metrics ────────────────────────────────
    print("\n" + "=" * 50)
    print("PER SOURCE METRICS (post-hoc)")
    print("=" * 50)

    results = {}
    for label, name in LABEL_NAMES.items():
        mask = all_labels_b == label

        src_preds    = all_preds[mask]
        src_labels_a = all_labels_a[mask]
        src_probs    = all_probs[mask]

        src_acc  = accuracy_score(src_labels_a, src_preds)
        src_prec = precision_score(src_labels_a, src_preds, zero_division=0)
        src_rec  = recall_score(src_labels_a, src_preds, zero_division=0)
        src_f1   = f1_score(src_labels_a, src_preds, zero_division=0)

        # ROC-AUC only works if both classes present in subset
        if len(np.unique(src_labels_a)) > 1:
            src_auc = roc_auc_score(src_labels_a, src_probs)
        else:
            src_auc = float("nan")

        results[name] = {
            "accuracy":  src_acc,
            "precision": src_prec,
            "recall":    src_rec,
            "f1":        src_f1,
            "roc_auc":   src_auc,
        }

        print(f"\n  {name}:")
        print(f"    Accuracy:  {src_acc:.4f}")
        print(f"    Precision: {src_prec:.4f}")
        print(f"    Recall:    {src_rec:.4f}")
        print(f"    F1:        {src_f1:.4f}")
        print(f"    ROC-AUC:   {src_auc:.4f}")

    # ── Save metrics to JSON for sharing with teammates ───
    output_dir = os.path.dirname(CONFIG["checkpoint_path"])
    metrics = {
        "overall": {
            "accuracy":  accuracy,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "roc_auc":   roc_auc,
        },
        "per_source": results
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {output_dir}/metrics.json")

    # ── Confusion matrix ──────────────────────────────────
    cm = confusion_matrix(all_labels_a, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Real", "AI"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — ViT Binary Classification")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.show()
    print("Confusion matrix saved")

    # ── ROC curve ─────────────────────────────────────────
    fpr, tpr, _ = roc_curve(all_labels_a, all_probs)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"ViT (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")
    ax.axvline(x=0.62, color="gray", linestyle=":", label="Human baseline (0.62 acc)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — ViT Binary Classification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.show()
    print("ROC curve saved")

    # ── Per source accuracy bar chart ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    f1s    = [results[n]["f1"] for n in names]
    x      = np.arange(len(names))
    width  = 0.35

    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, f1s,  width, label="F1 Score")
    ax.axhline(y=0.62, color="gray", linestyle="--", label="Human baseline")
    ax.axhline(y=0.75, color="red",  linestyle="--", label="Target (0.75)")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per Source Accuracy & F1 — ViT")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_source_metrics.png"), dpi=150)
    plt.show()
    print("Per source chart saved")