import torch
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def evaluate(model, loader, device):
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
    
    # Calculate metrics
    auroc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    
    return auroc, acc
