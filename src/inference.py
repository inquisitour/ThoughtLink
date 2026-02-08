"""
Real-time inference engine with confidence thresholding and temporal smoothing
ThoughtLink constraint: <50ms latency
"""
import time
import numpy as np
from collections import deque


class RealtimePredictor:
    """
    Fast prediction with confidence filtering and temporal smoothing
    Prevents false triggers and oscillation
    """
    
    def __init__(self, model, feature_extractor, config):
        """
        Args:
            model: Trained classifier with predict_proba method
            feature_extractor: MultimodalFeatureExtractor instance
            config: Configuration dict
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        
        self.confidence_threshold = config['inference']['confidence_threshold']
        self.use_smoothing = config['inference']['temporal_smoothing']
        self.smoothing_window = config['inference']['smoothing_window']
        
        # Prediction history for smoothing
        self.history = deque(maxlen=self.smoothing_window)
        
        # Class names
        self.classes = config['classes']
        
    def predict(self, eeg_data, fnirs_data, return_latency=False):
        """
        Predict brain intent from raw signals
        
        Args:
            eeg_data: (7499, 6) preprocessed EEG
            fnirs_data: (72, 40, 2, 2, 3) preprocessed fNIRS
            return_latency: If True, return (prediction, latency_ms)
            
        Returns:
            Dict with 'command', 'confidence', and optionally 'latency_ms'
        """
        start_time = time.perf_counter()
        
        # Extract features
        features = self.feature_extractor.extract(eeg_data, fnirs_data)
        
        # Apply PCA if fitted
        features = self.feature_extractor.apply_pca(features)

        # Clean NaN/Inf before prediction
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Predict probabilities
        probs = self.model.predict_proba(features.reshape(1, -1))[0]
        
        # Get class with highest probability
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            command = "REJECTED"
            confidence = 0.0
        else:
            command = self.classes[pred_idx]
        
        # Temporal smoothing (moving average)
        if self.use_smoothing and command != "REJECTED":
            self.history.append(pred_idx)
            if len(self.history) >= 2:
                # Majority vote
                from collections import Counter
                most_common = Counter(self.history).most_common(1)[0][0]
                command = self.classes[most_common]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        result = {
            'command': command,
            'confidence': float(confidence),
            'all_probs': probs.tolist(),
        }
        
        if return_latency:
            result['latency_ms'] = latency_ms
            
        return result
    
    def reset_history(self):
        """Clear prediction history (e.g., between trials)"""
        self.history.clear()


class BatchPredictor:
    """Batch prediction for offline evaluation"""
    
    def __init__(self, model, feature_extractor, config):
        self.model = model
        self.feature_extractor = feature_extractor
        self.config = config
        self.classes = config['classes']
        
    def predict_batch(self, X_features):
        """
        Predict on batch of features
        
        Args:
            X_features: (n_samples, n_features) array
            
        Returns:
            predictions, confidences
        """
        probs = self.model.predict_proba(X_features)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        
        predictions = [self.classes[idx] for idx in pred_indices]
        
        return predictions, confidences
