import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_curve
)
import numpy as np

def evaluate(model, loader, device, verbose=False):
    """Evaluate model on validation/test set"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.flatten())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Binary predictions (threshold 0.5)
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if verbose:
        print(f"  AUROC:      {auroc:.4f}")
        print(f"  Accuracy:   {acc:.4f}")
        print(f"  Precision:  {precision:.4f}")
        print(f"  Recall:     {recall:.4f} (Sensitivity)")
        print(f"  F1 Score:   {f1:.4f}")
        print(f"  Specificity:{specificity:.4f}")
    
    return auroc, acc, precision, recall, f1, specificity, sensitivity
