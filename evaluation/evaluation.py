import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

def evaluate(model, dataloader, device="cuda", topk=(1,5)):
    model.eval()
    model.to(device)

    correct_k = {k: 0 for k in topk}
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, pred_topk = logits.topk(max(topk), dim=1)  # [B, maxk]

            total += labels.size(0)

            for k in topk:
                # Check if label in top-k predictions
                correct_k[k] += (pred_topk[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            all_preds.extend(pred_topk[:, 0].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc_k = {f"top_{k}": correct_k[k] / total for k in topk}
    cm = confusion_matrix(all_labels, all_preds)

    return acc_k, cm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm
import numpy as np
import time

def evaluate_full(model, dataloader, class_names, device="cuda"):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    elapsed = time.time() - start_time

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Basic accuracy
    acc = accuracy_score(all_labels, all_preds)

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=range(len(class_names))
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Text summary
    report = classification_report(all_labels, all_preds, target_names=class_names)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "time": elapsed
    }
