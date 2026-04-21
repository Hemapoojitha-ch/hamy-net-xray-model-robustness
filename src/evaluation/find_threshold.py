import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
import matplotlib.pyplot as plt

def find_optimal_threshold(model, loader, device, metric='f1'):
    """Find optimal classification threshold"""
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
    
    # Calculate metrics for different thresholds
    thresholds = np.linspace(0, 1, 101)
    f1_scores = []
    recalls = []
    precisions = []
    specificities = []
    
    for threshold in thresholds:
        preds_binary = (all_preds > threshold).astype(int)
        
        # F1 Score
        f1 = f1_score(all_labels, preds_binary, zero_division=0)
        f1_scores.append(f1)
        
        # Recall (Sensitivity)
        tp = np.sum((preds_binary == 1) & (all_labels == 1))
        fn = np.sum((preds_binary == 0) & (all_labels == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)
        
        # Precision
        fp = np.sum((preds_binary == 1) & (all_labels == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precisions.append(precision)
        
        # Specificity
        tn = np.sum((preds_binary == 0) & (all_labels == 0))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    
    # Find optimal threshold
    if metric == 'f1':
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'recall':
        optimal_idx = np.argmax(recalls)
    else:  # balanced
        # Maximize sensitivity while keeping specificity > 0.80
        balanced_scores = []
        for i, threshold in enumerate(thresholds):
            if specificities[i] >= 0.80:
                balanced_scores.append((recalls[i] + specificities[i]) / 2)
            else:
                balanced_scores.append(-1)
        optimal_idx = np.argmax(balanced_scores)
    
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
    plt.axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid()
    
    plt.subplot(1, 2, 2)
    plt.plot(thresholds, recalls, 'g-', label='Recall (Sensitivity)')
    plt.plot(thresholds, precisions, 'b-', label='Precision')
    plt.plot(thresholds, specificities, 'r-', label='Specificity')
    plt.axvline(optimal_threshold, color='k', linestyle='--', label=f'Optimal: {optimal_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Threshold')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig('results/baseline/threshold_analysis.png', dpi=100)
    print(f"📊 Threshold analysis saved to results/baseline/threshold_analysis.png")
    
    return optimal_threshold, f1_scores, recalls, precisions
