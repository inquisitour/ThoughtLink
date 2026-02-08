"""
Utilities for data loading, metrics, and visualization
"""
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def load_npz_data(data_dir):
    """Load all .npz files from dataset directory"""
    data_path = Path(data_dir)
    
    # Check if data is in subdirectory
    if (data_path / 'data').exists():
        data_path = data_path / 'data'
    
    npz_files = sorted(data_path.glob("*.npz"))
    
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {data_path}")
    
    samples = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        
        eeg = data['feature_eeg']
        fnirs = data['feature_moments']
        label_info = data['label'].item()
        
        samples.append((eeg, fnirs, label_info))
    
    return samples


def extract_labels(samples):
    """Extract label strings from sample list"""
    return [s[2]['label'] for s in samples]


def compute_metrics(y_true, y_pred, classes):
    """
    Compute comprehensive evaluation metrics
    
    Returns:
        Dict with accuracy, per-class accuracy, confusion matrix
    """
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Per-class accuracy
    per_class_acc = {}
    for i, cls in enumerate(classes):
        if cm[i].sum() > 0:
            per_class_acc[cls] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[cls] = 0.0
    
    return {
        'accuracy': acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report': report
    }


def plot_confusion_matrix(cm, classes, save_path=None):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def plot_per_class_accuracy(per_class_acc, save_path=None):
    """Plot per-class accuracy bar chart"""
    classes = list(per_class_acc.keys())
    accuracies = list(per_class_acc.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, accuracies)
    
    # Color bars
    for i, bar in enumerate(bars):
        if accuracies[i] >= 0.7:
            bar.set_color('green')
        elif accuracies[i] >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.2, color='gray', linestyle='--', label='Chance (20%)')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def print_metrics(metrics):
    """Pretty print metrics"""
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
    print(f"\nPer-Class Accuracy:")
    for cls, acc in metrics['per_class_accuracy'].items():
        print(f"  {cls:20s}: {acc:.2%}")
    print("="*50 + "\n")
