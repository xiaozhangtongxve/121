"""
Frequency Domain Feature Extraction for Ship Steering Gear Fault Diagnosis
Implements FFT-based spectral features for signal analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class FrequencyDomainFeatures:
    """Extract frequency domain features from signals using FFT analysis"""
    
    def __init__(self, sampling_rate: float = 100.0):
        """
        Initialize frequency domain feature extractor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.feature_names = [
            'dominant_frequency', 'centroid_frequency', 'frequency_std',
            'spectral_rolloff', 'spectral_flux', 'spectral_flatness',
            'spectral_bandwidth', 'spectral_contrast', 'zero_crossing_rate_freq'
        ]
    
    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of signal
        
        Args:
            signal: Input time series signal
            
        Returns:
            Tuple of (frequencies, magnitude spectrum)
        """
        # Remove DC component
        signal = signal - np.mean(signal)
        
        # Apply window to reduce spectral leakage
        windowed_signal = signal * np.hanning(len(signal))
        
        # Compute FFT
        fft_values = fft(windowed_signal)
        freqs = fftfreq(len(signal), 1/self.sampling_rate)
        
        # Take positive frequencies only
        positive_freq_idx = freqs >= 0
        freqs = freqs[positive_freq_idx]
        magnitude = np.abs(fft_values[positive_freq_idx])
        
        return freqs, magnitude
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features from a signal
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Compute FFT
        freqs, magnitude = self.compute_fft(signal)
        
        # Normalize magnitude spectrum
        if np.sum(magnitude) > 0:
            magnitude_norm = magnitude / np.sum(magnitude)
        else:
            magnitude_norm = magnitude
        
        # Dominant frequency (frequency with maximum magnitude)
        if len(magnitude) > 0:
            dominant_idx = np.argmax(magnitude)
            features['dominant_frequency'] = freqs[dominant_idx]
            features['dominant_magnitude'] = magnitude[dominant_idx]
        else:
            features['dominant_frequency'] = 0
            features['dominant_magnitude'] = 0
        
        # Spectral centroid (center of mass of spectrum)
        if np.sum(magnitude) > 0:
            features['centroid_frequency'] = np.sum(freqs * magnitude) / np.sum(magnitude)
        else:
            features['centroid_frequency'] = 0
        
        # Spectral spread (standard deviation around centroid)
        if np.sum(magnitude) > 0:
            features['frequency_std'] = np.sqrt(
                np.sum(((freqs - features['centroid_frequency'])**2) * magnitude) / np.sum(magnitude)
            )
        else:
            features['frequency_std'] = 0
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(magnitude**2)
        total_energy = cumulative_energy[-1] if len(cumulative_energy) > 0 else 0
        
        if total_energy > 0:
            rolloff_threshold = 0.85 * total_energy
            rolloff_idx = np.where(cumulative_energy >= rolloff_threshold)[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = freqs[-1] if len(freqs) > 0 else 0
        else:
            features['spectral_rolloff'] = 0
        
        # Spectral flux (measure of how quickly the power spectrum changes)
        if len(magnitude) > 1:
            spectral_diff = np.diff(magnitude)
            features['spectral_flux'] = np.sum(spectral_diff**2)
        else:
            features['spectral_flux'] = 0
        
        # Spectral flatness (measure of how noise-like vs. tone-like the spectrum is)
        if len(magnitude) > 0 and np.all(magnitude > 0):
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            if arithmetic_mean > 0:
                features['spectral_flatness'] = geometric_mean / arithmetic_mean
            else:
                features['spectral_flatness'] = 0
        else:
            features['spectral_flatness'] = 0
        
        # Spectral bandwidth (weighted standard deviation around centroid)
        features['spectral_bandwidth'] = features['frequency_std']
        
        # Spectral contrast (difference between peaks and valleys in spectrum)
        if len(magnitude) > 0:
            features['spectral_contrast'] = np.max(magnitude) - np.min(magnitude)
        else:
            features['spectral_contrast'] = 0
        
        # Zero crossing rate in frequency domain
        if len(magnitude) > 1:
            zero_crossings = np.where(np.diff(np.signbit(magnitude - np.mean(magnitude))))[0]
            features['zero_crossing_rate_freq'] = len(zero_crossings) / len(magnitude)
        else:
            features['zero_crossing_rate_freq'] = 0
        
        # Additional spectral features
        features['spectral_energy'] = np.sum(magnitude**2)
        features['spectral_entropy'] = self._compute_spectral_entropy(magnitude_norm)
        features['spectral_kurtosis'] = self._compute_spectral_kurtosis(freqs, magnitude)
        features['spectral_skewness'] = self._compute_spectral_skewness(freqs, magnitude)
        
        # Peak features
        peak_features = self._extract_peak_features(freqs, magnitude)
        features.update(peak_features)
        
        # Band power features
        band_features = self._extract_band_power_features(freqs, magnitude)
        features.update(band_features)
        
        return features
    
    def _compute_spectral_entropy(self, magnitude_norm: np.ndarray) -> float:
        """Compute spectral entropy"""
        # Remove zeros to avoid log(0)
        magnitude_norm = magnitude_norm[magnitude_norm > 0]
        if len(magnitude_norm) > 0:
            return -np.sum(magnitude_norm * np.log2(magnitude_norm + 1e-10))
        else:
            return 0
    
    def _compute_spectral_kurtosis(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Compute spectral kurtosis"""
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            variance = np.sum(((freqs - centroid)**2) * magnitude) / np.sum(magnitude)
            if variance > 0:
                fourth_moment = np.sum(((freqs - centroid)**4) * magnitude) / np.sum(magnitude)
                return fourth_moment / (variance**2) - 3
            else:
                return 0
        else:
            return 0
    
    def _compute_spectral_skewness(self, freqs: np.ndarray, magnitude: np.ndarray) -> float:
        """Compute spectral skewness"""
        if np.sum(magnitude) > 0:
            centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            variance = np.sum(((freqs - centroid)**2) * magnitude) / np.sum(magnitude)
            if variance > 0:
                third_moment = np.sum(((freqs - centroid)**3) * magnitude) / np.sum(magnitude)
                return third_moment / (variance**(3/2))
            else:
                return 0
        else:
            return 0
    
    def _extract_peak_features(self, freqs: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
        """Extract features related to spectral peaks"""
        features = {}
        
        if len(magnitude) > 2:
            # Find peaks
            peaks, properties = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
            
            features['num_peaks'] = len(peaks)
            
            if len(peaks) > 0:
                # Peak frequencies
                peak_freqs = freqs[peaks]
                peak_magnitudes = magnitude[peaks]
                
                features['first_peak_freq'] = peak_freqs[0]
                features['first_peak_magnitude'] = peak_magnitudes[0]
                
                if len(peaks) > 1:
                    features['second_peak_freq'] = peak_freqs[1]
                    features['second_peak_magnitude'] = peak_magnitudes[1]
                    features['peak_freq_ratio'] = peak_freqs[1] / peak_freqs[0] if peak_freqs[0] > 0 else 0
                else:
                    features['second_peak_freq'] = 0
                    features['second_peak_magnitude'] = 0
                    features['peak_freq_ratio'] = 0
                
                # Peak statistics
                features['mean_peak_freq'] = np.mean(peak_freqs)
                features['std_peak_freq'] = np.std(peak_freqs)
                features['mean_peak_magnitude'] = np.mean(peak_magnitudes)
                features['std_peak_magnitude'] = np.std(peak_magnitudes)
            else:
                # No peaks found
                for key in ['first_peak_freq', 'first_peak_magnitude', 'second_peak_freq', 
                           'second_peak_magnitude', 'peak_freq_ratio', 'mean_peak_freq',
                           'std_peak_freq', 'mean_peak_magnitude', 'std_peak_magnitude']:
                    features[key] = 0
        else:
            features['num_peaks'] = 0
            for key in ['first_peak_freq', 'first_peak_magnitude', 'second_peak_freq', 
                       'second_peak_magnitude', 'peak_freq_ratio', 'mean_peak_freq',
                       'std_peak_freq', 'mean_peak_magnitude', 'std_peak_magnitude']:
                features[key] = 0
        
        return features
    
    def _extract_band_power_features(self, freqs: np.ndarray, magnitude: np.ndarray) -> Dict[str, float]:
        """Extract power in different frequency bands"""
        features = {}
        
        # Define frequency bands (adjust based on your application)
        bands = {
            'low': (0, 5),      # 0-5 Hz
            'mid': (5, 20),     # 5-20 Hz
            'high': (20, 50)    # 20-50 Hz (up to Nyquist)
        }
        
        total_power = np.sum(magnitude**2)
        
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.sum(magnitude[band_mask]**2)
            
            features[f'{band_name}_band_power'] = band_power
            
            if total_power > 0:
                features[f'{band_name}_band_power_ratio'] = band_power / total_power
            else:
                features[f'{band_name}_band_power_ratio'] = 0
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                      signal_columns: List[str]) -> pd.DataFrame:
        """
        Extract frequency domain features from multiple signals in a DataFrame
        
        Args:
            df: DataFrame containing signal data
            signal_columns: List of column names containing signals
            
        Returns:
            DataFrame with extracted features
        """
        feature_data = []
        
        for idx, row in df.iterrows():
            sample_features = {'sample_id': row.get('sample_id', idx)}
            
            # Add metadata if available
            if 'condition' in row:
                sample_features['condition'] = row['condition']
            if 'label' in row:
                sample_features['label'] = row['label']
            if 'noise_level' in row:
                sample_features['noise_level'] = row['noise_level']
            
            # Extract features for each signal
            for signal_col in signal_columns:
                if signal_col in row and isinstance(row[signal_col], np.ndarray):
                    signal_features = self.extract_features(row[signal_col])
                    
                    # Add signal prefix to feature names
                    for feature_name, feature_value in signal_features.items():
                        sample_features[f'{signal_col}_{feature_name}'] = feature_value
            
            feature_data.append(sample_features)
        
        return pd.DataFrame(feature_data)
    
    def get_feature_names(self, signal_columns: List[str]) -> List[str]:
        """Get list of all feature names for given signal columns"""
        base_features = [
            'dominant_frequency', 'dominant_magnitude', 'centroid_frequency', 'frequency_std',
            'spectral_rolloff', 'spectral_flux', 'spectral_flatness', 'spectral_bandwidth',
            'spectral_contrast', 'zero_crossing_rate_freq', 'spectral_energy',
            'spectral_entropy', 'spectral_kurtosis', 'spectral_skewness', 'num_peaks',
            'first_peak_freq', 'first_peak_magnitude', 'second_peak_freq', 'second_peak_magnitude',
            'peak_freq_ratio', 'mean_peak_freq', 'std_peak_freq', 'mean_peak_magnitude',
            'std_peak_magnitude', 'low_band_power', 'low_band_power_ratio',
            'mid_band_power', 'mid_band_power_ratio', 'high_band_power', 'high_band_power_ratio'
        ]
        
        all_features = []
        for signal_col in signal_columns:
            for feature_name in base_features:
                all_features.append(f'{signal_col}_{feature_name}')
        
        return all_features
