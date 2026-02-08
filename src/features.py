"""
Feature extraction from preprocessed signals
Focuses on M2 variance (our edge) and motor-relevant EEG bands
"""
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA


class FNIRSFeatureExtractor:
    """Extract features from TD-NIRS moments - emphasis on M2 variance"""
    
    def __init__(self, config):
        self.config = config
        self.peak_start = config['fnirs']['peak_window_start']
        self.peak_end = config['fnirs']['peak_window_end']
        self.use_moments = config['features']['fnirs']['use_moments']
        self.use_sds = config['features']['fnirs']['use_sds']
        self.use_wavelengths = config['features']['fnirs']['use_wavelengths']
        self.pca_components = config['features']['fnirs']['pca_components']
        self.pca = None
        
    def extract_peak_window_mean(self, data):
        """
        Extract mean from peak hemodynamic window (t=8-12s)
        Most discriminative single feature
        """
        peak_data = data[self.peak_start:self.peak_end, ...]
        return peak_data.mean(axis=0)
    
    def extract_features(self, fnirs_data):
        """
        Extract features from preprocessed TD-NIRS data
        
        Args:
            fnirs_data: (72, 40, 2, 2, 3) preprocessed array
                72 samples, 40 modules, 2 SDS (0=medium, 1=long), 2 wavelengths, 3 moments
                
        Returns:
            Feature vector focused on M2 variance
        """
        # Extract peak window mean
        peak_mean = self.extract_peak_window_mean(fnirs_data)
        
        # use_sds=2 means we want the long channels (index 1 after preprocessing)
        # After preprocessing: SDS index 0=medium, 1=long
        sds_idx = 1  # Long channels
        
        features = []
        for moment_idx in self.use_moments:
            for wavelength_idx in self.use_wavelengths:
                # Shape: (40 modules,)
                feat = peak_mean[:, sds_idx, wavelength_idx, moment_idx]
                features.append(feat)
        
        # Flatten: 40 modules × 2 moments × 2 wavelengths = 160 features
        features = np.concatenate(features)
        
        return features
    
    def fit_pca(self, features_list):
        """Fit PCA on training data"""
        X = np.array(features_list)
        self.pca = PCA(n_components=self.pca_components, random_state=42)
        self.pca.fit(X)
        
    def transform_pca(self, features):
        """Apply PCA transform"""
        if self.pca is None:
            return features
        return self.pca.transform(features.reshape(1, -1))[0]


class EEGFeatureExtractor:
    """Extract bandpower features from motor-relevant channels"""
    
    def __init__(self, config):
        self.config = config
        self.fs = config['eeg']['sampling_rate']
        self.use_channels = config['features']['eeg']['use_channels']
        self.bands = config['features']['eeg']['bands']
        self.active_start = config['eeg']['active_start']
        
        # Map channel names to indices
        all_channels = config['eeg']['channels']
        self.channel_indices = [all_channels.index(ch) for ch in self.use_channels]
        
    def compute_bandpower(self, data, low, high):
        """Compute average power in frequency band using Welch's method"""
        freqs, psd = welch(data, fs=self.fs, nperseg=256, axis=0)
        
        # Select frequency range
        mask = (freqs >= low) & (freqs <= high)
        bandpower = psd[mask, :].mean(axis=0)
        
        return bandpower
    
    def extract_features(self, eeg_data):
        """
        Extract bandpower features from motor-relevant channels
        
        Args:
            eeg_data: (7499, 6) preprocessed EEG
            
        Returns:
            Feature vector: 2 channels × 2 bands = 4 features
        """
        # Use only active period (after stimulus)
        active_data = eeg_data[self.active_start:, :]
        
        # Select motor-relevant channels (FCz, CPz)
        active_data = active_data[:, self.channel_indices]
        
        features = []
        for band_name, (low, high) in self.bands.items():
            bp = self.compute_bandpower(active_data, low, high)
            features.extend(bp)
        
        return np.array(features)


class MultimodalFeatureExtractor:
    """Combine EEG and fNIRS features"""
    
    def __init__(self, config):
        self.eeg_extractor = EEGFeatureExtractor(config)
        self.fnirs_extractor = FNIRSFeatureExtractor(config)
        
    def extract(self, eeg_data, fnirs_data):
        """
        Extract and concatenate features from both modalities
        
        Returns:
            Combined feature vector
        """
        eeg_feat = self.eeg_extractor.extract_features(eeg_data)
        fnirs_feat = self.fnirs_extractor.extract_features(fnirs_data)
        
        # Concatenate: 4 (EEG) + 160 (fNIRS) = 164 features before PCA
        return np.concatenate([eeg_feat, fnirs_feat])
    
    def fit_pca(self, features_list):
        """Fit PCA on fNIRS features only"""
        # Separate fNIRS features (last 160 elements)
        fnirs_features = [f[4:] for f in features_list]
        self.fnirs_extractor.fit_pca(fnirs_features)
    
    def apply_pca(self, features):
        """Apply PCA to fNIRS portion"""
        eeg_part = features[:4]
        fnirs_part = features[4:]
        
        fnirs_reduced = self.fnirs_extractor.transform_pca(fnirs_part)
        
        # Concatenate: 4 (EEG) + 30 (fNIRS after PCA) = 34 features
        return np.concatenate([eeg_part, fnirs_reduced])
