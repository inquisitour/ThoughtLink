"""
Prediction interface for hackathon robot simulation integration
"""
import numpy as np
import joblib
from pathlib import Path

from src.preprocessing import preprocess_sample
from src.inference import RealtimePredictor


class BrainRobotPredictor:
    """
    Standardized interface for hackathon integration
    Loads trained model and provides clean predict() method
    """
    
    def __init__(self, model_dir='./models'):
        """Load trained model and components"""
        model_path = Path(model_dir)
        
        # Load model, feature extractor, and config
        self.model = joblib.load(model_path / 'model.pkl')
        self.feature_extractor = joblib.load(model_path / 'feature_extractor.pkl')
        self.config = joblib.load(model_path / 'config.pkl')
        
        # Create realtime predictor
        self.predictor = RealtimePredictor(
            self.model,
            self.feature_extractor,
            self.config
        )
        
        print("✓ Model loaded successfully")
        print(f"  Classes: {self.config['classes']}")
        
    def predict(self, eeg_data, fnirs_data):
        """
        Predict robot command from brain signals
        
        Args:
            eeg_data: (7499, 6) raw EEG array
            fnirs_data: (72, 40, 3, 2, 3) raw TD-NIRS array
            
        Returns:
            dict with 'command', 'confidence', 'latency_ms'
        """
        # Preprocess
        eeg_clean, fnirs_clean = preprocess_sample(eeg_data, fnirs_data, self.config)
        
        # Predict with latency tracking
        result = self.predictor.predict(eeg_clean, fnirs_clean, return_latency=True)
        
        return result
    
    def predict_from_file(self, npz_path):
        """
        Predict from .npz file
        
        Args:
            npz_path: Path to .npz file with 'feature_eeg' and 'feature_moments'
            
        Returns:
            dict with 'command', 'confidence', 'latency_ms'
        """
        data = np.load(npz_path, allow_pickle=True)
        eeg = data['feature_eeg']
        fnirs = data['feature_moments']
        
        return self.predict(eeg, fnirs)


# Example usage for hackathon integration
if __name__ == "__main__":
    import sys
    
    # Initialize predictor
    predictor = BrainRobotPredictor(model_dir='./models')
    
    # Test on sample file
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"\nTesting on: {test_file}")
        
        result = predictor.predict_from_file(test_file)
        
        print("\nPrediction Result:")
        print(f"  Command:    {result['command']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Latency:    {result['latency_ms']:.1f} ms")
        
        if result['latency_ms'] > 50:
            print("\n  Latency exceeds 50ms target")
        else:
            print("\n  Success: Latency within real-time constraint")
    else:
        print("\nUsage: python predict.py <path_to_npz_file>")
