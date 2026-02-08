"""
Training script for ThoughtLink Brain-Robot Challenge
"""
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

from src.preprocessing import preprocess_sample
from src.features import MultimodalFeatureExtractor
from src.models import create_model
from src.utils import load_npz_data, extract_labels, compute_metrics, print_metrics, plot_confusion_matrix


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("ThoughtLink Brain-Robot Training")
    print("="*50)
    
    # Load dataset
    print("Loading dataset...")
    data_dir = Path(config['dataset']['cache_dir']) / 'robot_control'
    samples = load_npz_data(data_dir)
    print(f"Loaded {len(samples)} samples")
    
    # Preprocess all samples
    print("\nPreprocessing signals...")
    preprocessed = []
    for i, (eeg, fnirs, label_info) in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(samples)}")
        eeg_clean, fnirs_clean = preprocess_sample(eeg, fnirs, config)
        preprocessed.append((eeg_clean, fnirs_clean, label_info))
    
    # Extract features
    print("\nExtracting features...")
    feature_extractor = MultimodalFeatureExtractor(config)
    
    X_list = []
    y_list = []
    for eeg, fnirs, label_info in preprocessed:
        features = feature_extractor.extract(eeg, fnirs)
        X_list.append(features)
        y_list.append(label_info['label'])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {np.unique(y, return_counts=True)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )
    
    # Fit PCA on training features
    print("\nFitting PCA on fNIRS features...")
    feature_extractor.fit_pca(X_train)
    
    # Apply PCA transformation
    X_train_pca = np.array([feature_extractor.apply_pca(x) for x in X_train])
    X_test_pca = np.array([feature_extractor.apply_pca(x) for x in X_test])

    # Clean NaN/Inf values
    X_train_pca = np.nan_to_num(X_train_pca, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_pca = np.nan_to_num(X_test_pca, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Reduced feature shape: {X_train_pca.shape}")
    
    # Train model
    print("\nTraining model...")
    model = create_model(config, config['model']['type'])
    model.fit(X_train_pca, y_train)
    
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_pca)
    
    metrics = compute_metrics(y_test, y_pred, config['classes'])
    print_metrics(metrics)
    
    # Save model and feature extractor
    model_dir = Path(config['logging']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving model...")
    model.save(model_dir / 'model.pkl')
    joblib.dump(feature_extractor, model_dir / 'feature_extractor.pkl')
    joblib.dump(config, model_dir / 'config.pkl')
    
    # Save confusion matrix
    results_dir = Path(config['logging']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        config['classes'],
        save_path=results_dir / 'confusion_matrix.png'
    )
    
    print(f"\n✓ Model saved to {model_dir}")
    print(f"✓ Results saved to {results_dir}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
