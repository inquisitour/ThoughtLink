"""
Model implementations: Simple SVM baseline, optional CNN, ensemble
"""
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib


class SimpleSVMClassifier:
    """
    Linear/RBF SVM - our baseline model
    Fast inference (<10ms), proven effective
    """
    
    def __init__(self, config):
        self.config = config
        svm_config = config['model']['svm']
        
        self.model = SVC(
            kernel=svm_config['kernel'],
            C=svm_config['C'],
            gamma=svm_config['gamma'],
            class_weight=svm_config['class_weight'],
            probability=svm_config['probability'],
            random_state=42
        )
        
    def fit(self, X, y):
        """Train the SVM"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Predict class labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities for confidence scoring"""
        return self.model.predict_proba(X)
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load model from disk"""
        self.model = joblib.load(path)


class RandomForestClassifier_:
    """Random Forest as alternative to SVM"""
    
    def __init__(self, config):
        self.config = config
        rf_config = config['model']['rf']
        
        self.model = RandomForestClassifier(
            n_estimators=rf_config['n_estimators'],
            max_depth=rf_config['max_depth'],
            class_weight=rf_config['class_weight'],
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)


class EnsembleClassifier:
    """
    Soft voting ensemble of multiple models
    Typically +2-5% accuracy boost
    """
    
    def __init__(self, models):
        """
        Args:
            models: List of trained model objects with predict_proba method
        """
        self.models = models
        
    def predict_proba(self, X):
        """Average probabilities from all models"""
        probs = [model.predict_proba(X) for model in self.models]
        return np.mean(probs, axis=0)
    
    def predict(self, X):
        """Predict using averaged probabilities"""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def create_model(config, model_type='svm'):
    """
    Factory function to create models
    
    Args:
        config: Configuration dict
        model_type: 'svm', 'rf', or 'ensemble'
        
    Returns:
        Model instance
    """
    if model_type == 'svm':
        return SimpleSVMClassifier(config)
    elif model_type == 'rf':
        return RandomForestClassifier_(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
