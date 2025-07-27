"""
Time Domain Feature Extraction for Ship Steering Gear Fault Diagnosis
Implements statistical features for time-series signal analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeDomainFeatures:
    """Extract time domain statistical features from signals"""
    
    def __init__(self):
        self.feature_names = [
            'mean', 'std', 'variance', 'rms', 'kurtosis', 'skewness',
            'peak_to_peak', 'margin_factor', 'waveform_factor', 'crest_factor',
            'clearance_factor', 'impulse_factor', 'shape_factor'
        ]
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features from a signal
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['variance'] = np.var(signal)
        features['rms'] = np.sqrt(np.mean(signal**2))
        
        # Higher order moments
        features['kurtosis'] = stats.kurtosis(signal)
        features['skewness'] = stats.skew(signal)
        
        # Peak and range features
        features['peak_to_peak'] = np.ptp(signal)
        features['max_value'] = np.max(signal)
        features['min_value'] = np.min(signal)
        
        # Derived features
        abs_signal = np.abs(signal)
        mean_abs = np.mean(abs_signal)
        
        # Margin factor
        if mean_abs > 0:
            features['margin_factor'] = np.max(abs_signal) / mean_abs
        else:
            features['margin_factor'] = 0
        
        # Waveform factor
        if features['mean'] != 0:
            features['waveform_factor'] = features['rms'] / abs(features['mean'])
        else:
            features['waveform_factor'] = 0
        
        # Crest factor
        if features['rms'] > 0:
            features['crest_factor'] = np.max(abs_signal) / features['rms']
        else:
            features['crest_factor'] = 0
        
        # Clearance factor
        sqrt_mean_sqrt = np.mean(np.sqrt(abs_signal))**2
        if sqrt_mean_sqrt > 0:
            features['clearance_factor'] = np.max(abs_signal) / sqrt_mean_sqrt
        else:
            features['clearance_factor'] = 0
        
        # Impulse factor
        if mean_abs > 0:
            features['impulse_factor'] = np.max(abs_signal) / mean_abs
        else:
            features['impulse_factor'] = 0
        
        # Shape factor
        if mean_abs > 0:
            features['shape_factor'] = features['rms'] / mean_abs
        else:
            features['shape_factor'] = 0
        
        # Additional statistical features
        features['median'] = np.median(signal)
        features['mad'] = np.median(np.abs(signal - features['median']))  # Median Absolute Deviation
        features['iqr'] = np.percentile(signal, 75) - np.percentile(signal, 25)  # Interquartile Range
        
        # Energy and power features
        features['energy'] = np.sum(signal**2)
        features['power'] = features['energy'] / len(signal)
        
        # Zero crossing rate
        zero_crossings = np.where(np.diff(np.signbit(signal)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / len(signal)
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                      signal_columns: List[str]) -> pd.DataFrame:
        """
        Extract time domain features from multiple signals in a DataFrame
        
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
        all_features = []
        for signal_col in signal_columns:
            for feature_name in self.feature_names:
                all_features.append(f'{signal_col}_{feature_name}')
            # Add additional features
            additional_features = ['median', 'mad', 'iqr', 'energy', 'power', 
                                 'zero_crossing_rate', 'max_value', 'min_value']
            for feature_name in additional_features:
                all_features.append(f'{signal_col}_{feature_name}')
        return all_features


class AdvancedTimeDomainFeatures:
    """Advanced time domain feature extraction"""
    
    def __init__(self):
        pass
    
    def extract_autoregressive_features(self, signal: np.ndarray, order: int = 4) -> Dict[str, float]:
        """Extract autoregressive model coefficients as features"""
        from scipy.signal import lfilter
        
        try:
            # Fit AR model using Yule-Walker equations
            r = np.correlate(signal, signal, mode='full')
            r = r[r.size // 2:]
            
            # Solve Yule-Walker equations
            R = np.zeros((order, order))
            for i in range(order):
                for j in range(order):
                    R[i, j] = r[abs(i - j)]
            
            if np.linalg.det(R) != 0:
                ar_coeffs = np.linalg.solve(R, r[1:order+1])
            else:
                ar_coeffs = np.zeros(order)
            
            features = {}
            for i, coeff in enumerate(ar_coeffs):
                features[f'ar_coeff_{i+1}'] = coeff
            
            return features
        except:
            # Return zeros if AR fitting fails
            return {f'ar_coeff_{i+1}': 0.0 for i in range(order)}
    
    def extract_trend_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract trend-based features"""
        features = {}
        
        # Linear trend
        x = np.arange(len(signal))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, signal)
        
        features['trend_slope'] = slope
        features['trend_intercept'] = intercept
        features['trend_r_squared'] = r_value**2
        features['trend_p_value'] = p_value
        features['trend_std_err'] = std_err
        
        # Detrended signal features
        detrended = signal - (slope * x + intercept)
        features['detrended_std'] = np.std(detrended)
        features['detrended_var'] = np.var(detrended)
        
        return features
    
    def extract_entropy_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract entropy-based features"""
        features = {}
        
        # Sample entropy (approximate)
        def sample_entropy(data, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                C = np.sum([np.sum([_maxdist(patterns[i], patterns[j], m) <= r 
                                   for j in range(len(patterns)) if i != j]) 
                           for i in range(len(patterns))])
                return C / (len(patterns) * (len(patterns) - 1))
            
            try:
                phi_m = _phi(m)
                phi_m1 = _phi(m + 1)
                if phi_m > 0 and phi_m1 > 0:
                    return -np.log(phi_m1 / phi_m)
                else:
                    return 0
            except:
                return 0
        
        features['sample_entropy'] = sample_entropy(signal)
        
        # Spectral entropy (simplified)
        try:
            from scipy.signal import periodogram
            freqs, psd = periodogram(signal)
            psd_norm = psd / np.sum(psd)
            psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
            features['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        except:
            features['spectral_entropy'] = 0
        
        return features
