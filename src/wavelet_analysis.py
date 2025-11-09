"""
Wavelet analysis module for time series.

This module provides comprehensive wavelet transform analysis capabilities
including Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pywt
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns


class WaveletAnalyzer:
    """
    Comprehensive wavelet analysis for time series data.
    
    This class provides methods for both Continuous Wavelet Transform (CWT)
    and Discrete Wavelet Transform (DWT) analysis with various wavelets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the wavelet analyzer.
        
        Args:
            config: Configuration dictionary containing wavelet parameters
        """
        self.config = config
        self.wavelet_config = config.get('wavelet', {})
        
        # Default parameters
        self.wavelet_type = self.wavelet_config.get('wavelet_type', 'morl')
        self.scales_min = self.wavelet_config.get('scales_min', 1)
        self.scales_max = self.wavelet_config.get('scales_max', 100)
        self.sampling_period = self.wavelet_config.get('sampling_period', 0.01)
        
        logger.info(f"WaveletAnalyzer initialized with wavelet: {self.wavelet_type}")
    
    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive wavelet analysis on time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Starting wavelet analysis")
        
        # Extract time series values
        if 'value' in data.columns:
            signal = data['value'].values
        else:
            signal = data.iloc[:, 1].values  # Assume second column is the signal
        
        # Remove NaN values
        signal = signal[~np.isnan(signal)]
        
        # Perform CWT analysis
        cwt_results = self._perform_cwt(signal)
        
        # Perform DWT analysis
        dwt_results = self._perform_dwt(signal)
        
        # Calculate wavelet-based features
        features = self._extract_features(signal, cwt_results, dwt_results)
        
        results = {
            'cwt': cwt_results,
            'dwt': dwt_results,
            'features': features,
            'signal': signal,
            'config': {
                'wavelet_type': self.wavelet_type,
                'scales_range': (self.scales_min, self.scales_max),
                'sampling_period': self.sampling_period
            }
        }
        
        logger.info("Wavelet analysis completed")
        return results
    
    def _perform_cwt(self, signal: np.ndarray) -> Dict:
        """
        Perform Continuous Wavelet Transform analysis.
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary containing CWT results
        """
        logger.info("Performing Continuous Wavelet Transform")
        
        # Define scales
        scales = np.arange(self.scales_min, self.scales_max + 1)
        
        # Perform CWT
        try:
            coeffs, freqs = pywt.cwt(
                signal, 
                scales, 
                wavelet=self.wavelet_type, 
                sampling_period=self.sampling_period
            )
            
            # Calculate magnitude and phase
            magnitude = np.abs(coeffs)
            phase = np.angle(coeffs)
            
            # Find dominant frequencies
            dominant_freqs = self._find_dominant_frequencies(magnitude, freqs)
            
            # Calculate energy distribution
            energy_distribution = np.sum(magnitude**2, axis=1)
            
            cwt_results = {
                'coefficients': coeffs,
                'frequencies': freqs,
                'scales': scales,
                'magnitude': magnitude,
                'phase': phase,
                'dominant_frequencies': dominant_freqs,
                'energy_distribution': energy_distribution
            }
            
            logger.info("CWT analysis completed successfully")
            
        except Exception as e:
            logger.error(f"CWT analysis failed: {e}")
            cwt_results = {
                'coefficients': None,
                'frequencies': None,
                'scales': None,
                'magnitude': None,
                'phase': None,
                'dominant_frequencies': None,
                'energy_distribution': None,
                'error': str(e)
            }
        
        return cwt_results
    
    def _perform_dwt(self, signal: np.ndarray) -> Dict:
        """
        Perform Discrete Wavelet Transform analysis.
        
        Args:
            signal: Input time series signal
            
        Returns:
            Dictionary containing DWT results
        """
        logger.info("Performing Discrete Wavelet Transform")
        
        try:
            # Choose appropriate wavelet for DWT
            dwt_wavelet = 'db4' if self.wavelet_type == 'morl' else self.wavelet_type
            
            # Perform multi-level DWT decomposition
            coeffs = pywt.wavedec(signal, dwt_wavelet, level=6)
            
            # Separate approximation and detail coefficients
            cA = coeffs[0]  # Approximation coefficients
            cD = coeffs[1:]  # Detail coefficients
            
            # Reconstruct components
            reconstructed_components = []
            for i, c in enumerate(coeffs):
                # Create coefficient list for reconstruction
                coeffs_recon = [None] * len(coeffs)
                coeffs_recon[i] = c
                
                # Reconstruct component
                component = pywt.waverec(coeffs_recon, dwt_wavelet)
                reconstructed_components.append(component)
            
            # Calculate energy in each level
            energy_levels = [np.sum(c**2) for c in coeffs]
            
            # Calculate relative energy
            total_energy = sum(energy_levels)
            relative_energy = [e / total_energy for e in energy_levels]
            
            dwt_results = {
                'coefficients': coeffs,
                'approximation': cA,
                'details': cD,
                'reconstructed_components': reconstructed_components,
                'energy_levels': energy_levels,
                'relative_energy': relative_energy,
                'wavelet': dwt_wavelet,
                'levels': len(coeffs) - 1
            }
            
            logger.info("DWT analysis completed successfully")
            
        except Exception as e:
            logger.error(f"DWT analysis failed: {e}")
            dwt_results = {
                'coefficients': None,
                'approximation': None,
                'details': None,
                'reconstructed_components': None,
                'energy_levels': None,
                'relative_energy': None,
                'wavelet': None,
                'levels': 0,
                'error': str(e)
            }
        
        return dwt_results
    
    def _find_dominant_frequencies(self, magnitude: np.ndarray, freqs: np.ndarray) -> List[float]:
        """
        Find dominant frequencies in the wavelet transform.
        
        Args:
            magnitude: Magnitude of wavelet coefficients
            freqs: Frequency array
            
        Returns:
            List of dominant frequencies
        """
        # Calculate average magnitude across time for each frequency
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Find peaks in the magnitude spectrum
        from scipy.signal import find_peaks
        
        peaks, _ = find_peaks(avg_magnitude, height=np.max(avg_magnitude) * 0.1)
        
        # Get corresponding frequencies
        dominant_freqs = freqs[peaks].tolist()
        
        return dominant_freqs
    
    def _extract_features(self, signal: np.ndarray, cwt_results: Dict, dwt_results: Dict) -> Dict:
        """
        Extract wavelet-based features from the analysis results.
        
        Args:
            signal: Original signal
            cwt_results: CWT analysis results
            dwt_results: DWT analysis results
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        # Basic signal statistics
        features['signal_length'] = len(signal)
        features['signal_mean'] = np.mean(signal)
        features['signal_std'] = np.std(signal)
        features['signal_energy'] = np.sum(signal**2)
        
        # CWT-based features
        if cwt_results['magnitude'] is not None:
            magnitude = cwt_results['magnitude']
            features['cwt_max_magnitude'] = np.max(magnitude)
            features['cwt_mean_magnitude'] = np.mean(magnitude)
            features['cwt_energy_concentration'] = np.sum(magnitude**2)
            features['cwt_frequency_bandwidth'] = np.std(cwt_results['frequencies'])
        
        # DWT-based features
        if dwt_results['energy_levels'] is not None:
            energy_levels = dwt_results['energy_levels']
            features['dwt_total_energy'] = sum(energy_levels)
            features['dwt_energy_concentration'] = max(dwt_results['relative_energy'])
            features['dwt_approximation_energy'] = energy_levels[0] / sum(energy_levels)
        
        # Spectral features
        if cwt_results['dominant_frequencies'] is not None:
            features['n_dominant_frequencies'] = len(cwt_results['dominant_frequencies'])
            if cwt_results['dominant_frequencies']:
                features['max_dominant_frequency'] = max(cwt_results['dominant_frequencies'])
                features['min_dominant_frequency'] = min(cwt_results['dominant_frequencies'])
        
        return features
    
    def plot_scalogram(self, cwt_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot the wavelet scalogram (time-frequency representation).
        
        Args:
            cwt_results: CWT analysis results
            save_path: Optional path to save the plot
        """
        if cwt_results['magnitude'] is None:
            logger.warning("Cannot plot scalogram: CWT results are None")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot scalogram
        magnitude = cwt_results['magnitude']
        scales = cwt_results['scales']
        
        plt.imshow(
            magnitude, 
            extent=[0, len(magnitude[0]) * self.sampling_period, scales[0], scales[-1]], 
            cmap='jet', 
            aspect='auto', 
            origin='lower'
        )
        
        plt.title(f'Wavelet Scalogram ({self.wavelet_type.upper()} Wavelet)')
        plt.xlabel('Time')
        plt.ylabel('Scale')
        plt.colorbar(label='Magnitude')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_dwt_decomposition(self, dwt_results: Dict, signal: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot DWT decomposition components.
        
        Args:
            dwt_results: DWT analysis results
            signal: Original signal
            save_path: Optional path to save the plot
        """
        if dwt_results['reconstructed_components'] is None:
            logger.warning("Cannot plot DWT decomposition: DWT results are None")
            return
        
        components = dwt_results['reconstructed_components']
        levels = dwt_results['levels']
        
        fig, axes = plt.subplots(levels + 2, 1, figsize=(12, 2 * (levels + 2)))
        
        # Plot original signal
        axes[0].plot(signal)
        axes[0].set_title('Original Signal')
        axes[0].grid(True)
        
        # Plot approximation
        axes[1].plot(components[0])
        axes[1].set_title('Approximation (A6)')
        axes[1].grid(True)
        
        # Plot details
        for i in range(1, len(components)):
            axes[i + 1].plot(components[i])
            axes[i + 1].set_title(f'Detail (D{levels - i + 1})')
            axes[i + 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
