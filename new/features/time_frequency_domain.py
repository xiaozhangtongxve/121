"""
Time-Frequency Domain Feature Extraction for Ship Steering Gear Fault Diagnosis
Implements Wavelet Packet Transform (WPT) and Variational Mode Decomposition (VMD) features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import pywt
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


class WaveletPacketFeatures:
    """Extract time-frequency features using Wavelet Packet Transform"""
    
    def __init__(self, wavelet: str = 'db4', levels: int = 4):
        """
        Initialize WPT feature extractor
        
        Args:
            wavelet: Wavelet type (e.g., 'db4', 'haar', 'coif2')
            levels: Number of decomposition levels
        """
        self.wavelet = wavelet
        self.levels = levels
        self.feature_names = []
        
        # Generate feature names for each level and node
        for level in range(1, levels + 1):
            for node in range(2**level):
                self.feature_names.extend([
                    f'wpt_L{level}_N{node}_energy',
                    f'wpt_L{level}_N{node}_energy_ratio',
                    f'wpt_L{level}_N{node}_std',
                    f'wpt_L{level}_N{node}_mean',
                    f'wpt_L{level}_N{node}_max'
                ])
    
    def wavelet_packet_decomposition(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform wavelet packet decomposition
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary of wavelet packet coefficients
        """
        # Ensure signal length is power of 2 for WPT
        signal_length = len(signal)
        next_power_of_2 = 2**int(np.ceil(np.log2(signal_length)))
        
        if signal_length < next_power_of_2:
            # Pad signal with zeros
            padded_signal = np.pad(signal, (0, next_power_of_2 - signal_length), 'constant')
        else:
            padded_signal = signal
        
        # Create wavelet packet tree
        wp = pywt.WaveletPacket(data=padded_signal, wavelet=self.wavelet, mode='symmetric')
        
        # Extract coefficients from all nodes
        coefficients = {}
        
        for level in range(1, self.levels + 1):
            for node in range(2**level):
                # Get node path
                node_path = self._get_node_path(level, node)
                try:
                    node_coeffs = wp[node_path].data
                    coefficients[f'L{level}_N{node}'] = node_coeffs
                except:
                    # If node doesn't exist, create empty array
                    coefficients[f'L{level}_N{node}'] = np.array([])
        
        return coefficients
    
    def _get_node_path(self, level: int, node: int) -> str:
        """Convert level and node number to wavelet packet path"""
        if level == 0:
            return ''
        
        # Convert node number to binary path
        binary_path = format(node, f'0{level}b')
        # Convert binary to 'a' (approximation) and 'd' (detail)
        path = ''.join(['a' if bit == '0' else 'd' for bit in binary_path])
        return path
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract wavelet packet features from a signal
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Perform wavelet packet decomposition
        coefficients = self.wavelet_packet_decomposition(signal)
        
        # Calculate total energy for normalization
        total_energy = sum([np.sum(coeffs**2) for coeffs in coefficients.values() if len(coeffs) > 0])
        
        # Extract features from each node
        for node_key, coeffs in coefficients.items():
            if len(coeffs) > 0:
                # Energy features
                energy = np.sum(coeffs**2)
                features[f'wpt_{node_key}_energy'] = energy
                
                # Energy ratio
                if total_energy > 0:
                    features[f'wpt_{node_key}_energy_ratio'] = energy / total_energy
                else:
                    features[f'wpt_{node_key}_energy_ratio'] = 0
                
                # Statistical features
                features[f'wpt_{node_key}_std'] = np.std(coeffs)
                features[f'wpt_{node_key}_mean'] = np.mean(coeffs)
                features[f'wpt_{node_key}_max'] = np.max(np.abs(coeffs))
                features[f'wpt_{node_key}_rms'] = np.sqrt(np.mean(coeffs**2))
                features[f'wpt_{node_key}_kurtosis'] = self._safe_kurtosis(coeffs)
                features[f'wpt_{node_key}_skewness'] = self._safe_skewness(coeffs)
            else:
                # Empty coefficients
                for suffix in ['energy', 'energy_ratio', 'std', 'mean', 'max', 'rms', 'kurtosis', 'skewness']:
                    features[f'wpt_{node_key}_{suffix}'] = 0
        
        # Additional wavelet-based features
        features.update(self._extract_wavelet_entropy_features(coefficients))
        features.update(self._extract_relative_energy_features(coefficients))
        
        return features
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Safely compute kurtosis"""
        try:
            from scipy.stats import kurtosis
            return kurtosis(data)
        except:
            return 0
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Safely compute skewness"""
        try:
            from scipy.stats import skew
            return skew(data)
        except:
            return 0
    
    def _extract_wavelet_entropy_features(self, coefficients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract entropy-based features from wavelet coefficients"""
        features = {}
        
        # Calculate relative energies
        energies = []
        for coeffs in coefficients.values():
            if len(coeffs) > 0:
                energies.append(np.sum(coeffs**2))
            else:
                energies.append(0)
        
        energies = np.array(energies)
        total_energy = np.sum(energies)
        
        if total_energy > 0:
            relative_energies = energies / total_energy
            # Remove zeros for entropy calculation
            relative_energies = relative_energies[relative_energies > 0]
            
            if len(relative_energies) > 0:
                features['wavelet_entropy'] = -np.sum(relative_energies * np.log2(relative_energies))
            else:
                features['wavelet_entropy'] = 0
        else:
            features['wavelet_entropy'] = 0
        
        return features
    
    def _extract_relative_energy_features(self, coefficients: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract relative energy features across frequency bands"""
        features = {}
        
        # Group coefficients by level (frequency bands)
        level_energies = {}
        for level in range(1, self.levels + 1):
            level_energy = 0
            for node in range(2**level):
                node_key = f'L{level}_N{node}'
                if node_key in coefficients and len(coefficients[node_key]) > 0:
                    level_energy += np.sum(coefficients[node_key]**2)
            level_energies[level] = level_energy
        
        total_energy = sum(level_energies.values())
        
        # Calculate relative energies per level
        for level, energy in level_energies.items():
            if total_energy > 0:
                features[f'level_{level}_energy_ratio'] = energy / total_energy
            else:
                features[f'level_{level}_energy_ratio'] = 0
        
        return features
    
    def extract_features_from_dataframe(self, df: pd.DataFrame, 
                                      signal_columns: List[str]) -> pd.DataFrame:
        """
        Extract WPT features from multiple signals in a DataFrame
        
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


class VariationalModeDecomposition:
    """Simplified Variational Mode Decomposition implementation"""
    
    def __init__(self, n_modes: int = 4, alpha: float = 2000, tau: float = 0.0, 
                 tolerance: float = 1e-7, max_iterations: int = 500):
        """
        Initialize VMD parameters
        
        Args:
            n_modes: Number of modes to decompose
            alpha: Balancing parameter
            tau: Time-step of dual ascent
            tolerance: Tolerance for convergence
            max_iterations: Maximum number of iterations
        """
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def vmd_decomposition(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified VMD implementation
        Note: This is a basic implementation. For production use, consider using
        a more robust VMD implementation from specialized libraries.
        """
        # This is a simplified version - in practice, you might want to use
        # a more sophisticated VMD implementation
        
        # For now, we'll use EMD-like decomposition as a placeholder
        modes = []
        residual = signal.copy()
        
        for i in range(self.n_modes):
            if len(residual) < 10:  # Too short for further decomposition
                break
                
            # Simple high-pass filtering to extract mode
            from scipy.signal import butter, filtfilt
            
            # Define frequency bands
            nyquist = 0.5
            low_freq = (i + 1) * nyquist / (self.n_modes + 1)
            high_freq = (i + 2) * nyquist / (self.n_modes + 1)
            
            if high_freq >= nyquist:
                high_freq = nyquist - 0.01
            
            try:
                b, a = butter(4, [low_freq, high_freq], btype='band', fs=1.0)
                mode = filtfilt(b, a, residual)
                modes.append(mode)
                residual = residual - mode
            except:
                # If filtering fails, use residual as mode
                modes.append(residual)
                break
        
        # Add residual as final mode
        if len(modes) < self.n_modes:
            modes.append(residual)
        
        return np.array(modes), np.arange(len(modes))
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Extract features from VMD modes"""
        features = {}
        
        try:
            modes, _ = self.vmd_decomposition(signal)
            
            # Extract features from each mode
            for i, mode in enumerate(modes):
                if len(mode) > 0:
                    features[f'vmd_mode_{i}_energy'] = np.sum(mode**2)
                    features[f'vmd_mode_{i}_std'] = np.std(mode)
                    features[f'vmd_mode_{i}_mean'] = np.mean(mode)
                    features[f'vmd_mode_{i}_max'] = np.max(np.abs(mode))
                    features[f'vmd_mode_{i}_rms'] = np.sqrt(np.mean(mode**2))
                else:
                    for suffix in ['energy', 'std', 'mean', 'max', 'rms']:
                        features[f'vmd_mode_{i}_{suffix}'] = 0
            
            # Calculate relative energies
            total_energy = sum([np.sum(mode**2) for mode in modes if len(mode) > 0])
            for i, mode in enumerate(modes):
                if len(mode) > 0 and total_energy > 0:
                    features[f'vmd_mode_{i}_energy_ratio'] = np.sum(mode**2) / total_energy
                else:
                    features[f'vmd_mode_{i}_energy_ratio'] = 0
            
        except Exception as e:
            # If VMD fails, return zero features
            for i in range(self.n_modes):
                for suffix in ['energy', 'std', 'mean', 'max', 'rms', 'energy_ratio']:
                    features[f'vmd_mode_{i}_{suffix}'] = 0
        
        return features
