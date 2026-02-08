"""
Preprocessing for EEG and TD-NIRS signals
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class EEGPreprocessor:
    """EEG preprocessing pipeline for 6-channel frontal setup"""
    
    def __init__(self, config):
        self.config = config
        self.fs = config['eeg']['sampling_rate']
        self.baseline_start = config['eeg']['baseline_start']
        self.baseline_end = config['eeg']['baseline_end']
        
    def bandpass_filter(self, data, lowcut, highcut, order=5):
        """Zero-phase Butterworth bandpass filter"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    def notch_filter(self, data, freq, Q=30):
        """Notch filter for line noise removal"""
        nyq = 0.5 * self.fs
        w0 = freq / nyq
        b, a = iirnotch(w0, Q)
        return filtfilt(b, a, data, axis=0)
    
    def common_average_reference(self, data):
        """CAR - only viable re-referencing for sparse montage"""
        return data - data.mean(axis=1, keepdims=True)
    
    def baseline_normalize(self, data):
        """Z-score normalization using pre-stimulus baseline"""
        baseline = data[self.baseline_start:self.baseline_end, :]
        mean = baseline.mean(axis=0, keepdims=True)
        std = baseline.std(axis=0, keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)  # Avoid division by zero
        return (data - mean) / std
    
    def preprocess(self, eeg_data):
        """
        Full EEG preprocessing pipeline
        
        Args:
            eeg_data: (7499, 6) array of EEG voltage data
            
        Returns:
            Preprocessed EEG data
        """
        # Step 1: Bandpass filter (0.5-45 Hz)
        data = self.bandpass_filter(
            eeg_data,
            self.config['eeg']['bandpass_low'],
            self.config['eeg']['bandpass_high']
        )
        
        # Step 2: Notch filter (50/60 Hz)
        data = self.notch_filter(
            data,
            self.config['eeg']['notch_freq'],
            self.config['eeg']['notch_q']
        )
        
        # Step 3: Common Average Reference
        data = self.common_average_reference(data)
        
        # Step 4: Baseline normalization (z-score)
        data = self.baseline_normalize(data)

        # Clean NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data


class FNIRSPreprocessor:
    """
    TD-NIRS preprocessing with short-channel regression
    CRITICAL COMPETITIVE ADVANTAGE - most teams will miss this
    """
    
    def __init__(self, config):
        self.config = config
        self.fs = config['fnirs']['sampling_rate']
        self.baseline_start = config['fnirs']['baseline_start']
        self.baseline_end = config['fnirs']['baseline_end']
        self.use_scr = config['fnirs']['use_short_channel_regression']
        
    def short_channel_regression(self, long_data, short_data):
        """
        Remove systemic physiology using short-channel regression
        This is THE key preprocessing step that gives us an edge
        
        Args:
            long_data: (72, 40, 2, 3) - medium/long channels
            short_data: (72, 40, 2, 3) - short channels (scalp only)
            
        Returns:
            Corrected long_data with scalp signals removed
        """
        # Demean both signals
        s = short_data - short_data.mean(axis=0, keepdims=True)
        l = long_data - long_data.mean(axis=0, keepdims=True)
        
        # Compute regression coefficients per module/wavelength/moment
        numerator = (s * l).sum(axis=0, keepdims=True)
        denominator = (s * s).sum(axis=0, keepdims=True)
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)
        beta = numerator / denominator
        
        # Subtract regressed scalp signal
        return long_data - beta * short_data
    
    def bandpass_filter(self, data, lowcut, highcut, order=4):
        """Zero-phase Butterworth bandpass for hemodynamic signals"""
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        
        # Filter along time axis (axis=0)
        # Using padlen=0 to avoid error with short signals
        filtered = np.zeros_like(data)
        for i in range(data.shape[1]):  # modules
            for j in range(data.shape[2]):  # wavelengths
                for k in range(data.shape[3]):  # moments
                    try:
                        filtered[:, i, j, k] = filtfilt(b, a, data[:, i, j, k], padlen=0)
                    except:
                        # If still fails, skip filtering for this channel
                        filtered[:, i, j, k] = data[:, i, j, k]
        return filtered
    
    def baseline_correct(self, data):
            """Subtract baseline mean"""
            baseline = data[self.baseline_start:self.baseline_end, ...]
            baseline_mean = baseline.mean(axis=0, keepdims=True)
            return data - baseline_mean
        
    def preprocess(self, fnirs_data):
        """Full TD-NIRS preprocessing pipeline"""
        # Extract short and long channels
        short = fnirs_data[:, :, 0, :, :]
        medium = fnirs_data[:, :, 1, :, :]
        long = fnirs_data[:, :, 2, :, :]
        
        # Step 1: Short-channel regression
        if self.use_scr:
            medium = self.short_channel_regression(medium, short)
            long = self.short_channel_regression(long, short)
        
        data = np.stack([medium, long], axis=2)
        
        # Step 2: Bandpass filter (SKIP - causes NaN with short signals)
        # if self.config['fnirs'].get('use_bandpass', True):
        #     data = self.bandpass_filter(data, ...)
        
        # Step 3: Baseline correction
        data = self.baseline_correct(data)
        
        # Replace any NaN/Inf with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data


def preprocess_sample(eeg_data, fnirs_data, config):
    """
    Preprocess a single trial
    
    Args:
        eeg_data: (7499, 6) EEG array
        fnirs_data: (72, 40, 3, 2, 3) TD-NIRS array
        config: Configuration dictionary
        
    Returns:
        Tuple of (preprocessed_eeg, preprocessed_fnirs)
    """
    eeg_proc = EEGPreprocessor(config)
    fnirs_proc = FNIRSPreprocessor(config)
    
    eeg_clean = eeg_proc.preprocess(eeg_data)
    fnirs_clean = fnirs_proc.preprocess(fnirs_data)
    
    return eeg_clean, fnirs_clean
