"""
Anomaly detection module for time series analysis.

This module provides comprehensive anomaly detection capabilities using
multiple methods including Isolation Forest, Autoencoders, and statistical approaches.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Anomaly detection libraries
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, autoencoder anomaly detection disabled")

from scipy import stats
from scipy.signal import find_peaks


class AnomalyDetector:
    """
    Comprehensive anomaly detection for time series data.
    
    This class provides multiple anomaly detection methods including statistical,
    machine learning, and deep learning approaches.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the anomaly detector.
        
        Args:
            config: Configuration dictionary containing anomaly detection parameters
        """
        self.config = config
        self.anomaly_config = config.get('anomaly_detection', {})
        
        # Method configurations
        self.isolation_forest_config = self.anomaly_config.get('isolation_forest', {})
        self.autoencoder_config = self.anomaly_config.get('autoencoder', {})
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
        logger.info("AnomalyDetector initialized")
    
    def detect(self, data: pd.DataFrame) -> Dict:
        """
        Perform comprehensive anomaly detection using multiple methods.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dictionary containing anomaly detection results from all methods
        """
        logger.info("Starting anomaly detection analysis")
        
        # Prepare data
        if 'value' in data.columns:
            ts_data = data['value'].dropna()
        else:
            ts_data = data.iloc[:, 1].dropna()
        
        # Ensure we have enough data
        if len(ts_data) < 10:
            logger.warning("Insufficient data for anomaly detection")
            return {'error': 'Insufficient data'}
        
        results = {}
        
        # Statistical anomaly detection
        try:
            statistical_results = self._detect_statistical(ts_data)
            results['statistical'] = statistical_results
            logger.info("Statistical anomaly detection completed")
        except Exception as e:
            logger.error(f"Statistical anomaly detection failed: {e}")
            results['statistical'] = {'error': str(e)}
        
        # Isolation Forest
        try:
            isolation_results = self._detect_isolation_forest(ts_data)
            results['isolation_forest'] = isolation_results
            logger.info("Isolation Forest anomaly detection completed")
        except Exception as e:
            logger.error(f"Isolation Forest anomaly detection failed: {e}")
            results['isolation_forest'] = {'error': str(e)}
        
        # Autoencoder (if PyTorch is available)
        if TORCH_AVAILABLE:
            try:
                autoencoder_results = self._detect_autoencoder(ts_data)
                results['autoencoder'] = autoencoder_results
                logger.info("Autoencoder anomaly detection completed")
            except Exception as e:
                logger.error(f"Autoencoder anomaly detection failed: {e}")
                results['autoencoder'] = {'error': str(e)}
        
        # Wavelet-based anomaly detection
        try:
            wavelet_results = self._detect_wavelet_based(ts_data)
            results['wavelet'] = wavelet_results
            logger.info("Wavelet-based anomaly detection completed")
        except Exception as e:
            logger.error(f"Wavelet-based anomaly detection failed: {e}")
            results['wavelet'] = {'error': str(e)}
        
        # Combine results
        results['combined'] = self._combine_results(results, ts_data)
        
        logger.info("Anomaly detection analysis completed")
        return results
    
    def _detect_statistical(self, ts_data: pd.Series) -> Dict:
        """
        Detect anomalies using statistical methods.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Dictionary containing statistical anomaly detection results
        """
        logger.info("Performing statistical anomaly detection")
        
        # Z-score method
        z_scores = np.abs(stats.zscore(ts_data))
        z_threshold = 3
        z_anomalies = z_scores > z_threshold
        
        # Modified Z-score method (using median)
        median = np.median(ts_data)
        mad = np.median(np.abs(ts_data - median))
        modified_z_scores = 0.6745 * (ts_data - median) / mad
        modified_z_anomalies = np.abs(modified_z_scores) > 3.5
        
        # Interquartile Range (IQR) method
        Q1 = ts_data.quantile(0.25)
        Q3 = ts_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_anomalies = (ts_data < lower_bound) | (ts_data > upper_bound)
        
        # Rolling statistics method
        window = min(30, len(ts_data) // 4)
        rolling_mean = ts_data.rolling(window=window, center=True).mean()
        rolling_std = ts_data.rolling(window=window, center=True).std()
        
        # Detect points that are more than 2 standard deviations from rolling mean
        rolling_anomalies = np.abs(ts_data - rolling_mean) > 2 * rolling_std
        
        # Count anomalies
        anomaly_counts = {
            'z_score': np.sum(z_anomalies),
            'modified_z_score': np.sum(modified_z_anomalies),
            'iqr': np.sum(iqr_anomalies),
            'rolling_statistics': np.sum(rolling_anomalies)
        }
        
        return {
            'z_score_anomalies': z_anomalies,
            'modified_z_score_anomalies': modified_z_anomalies,
            'iqr_anomalies': iqr_anomalies,
            'rolling_anomalies': rolling_anomalies,
            'anomaly_counts': anomaly_counts,
            'z_scores': z_scores,
            'modified_z_scores': modified_z_scores,
            'iqr_bounds': (lower_bound, upper_bound)
        }
    
    def _detect_isolation_forest(self, ts_data: pd.Series) -> Dict:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Dictionary containing Isolation Forest results
        """
        logger.info("Performing Isolation Forest anomaly detection")
        
        # Prepare features
        features = self._create_features(ts_data)
        
        # Initialize Isolation Forest
        contamination = self.isolation_forest_config.get('contamination', 0.1)
        random_state = self.isolation_forest_config.get('random_state', 42)
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        
        # Fit and predict
        anomaly_labels = iso_forest.fit_predict(features)
        anomaly_scores = iso_forest.decision_function(features)
        
        # Convert to boolean (True for anomalies)
        is_anomaly = anomaly_labels == -1
        
        # Count anomalies
        n_anomalies = np.sum(is_anomaly)
        anomaly_percentage = n_anomalies / len(ts_data) * 100
        
        return {
            'anomaly_labels': is_anomaly,
            'anomaly_scores': anomaly_scores,
            'n_anomalies': n_anomalies,
            'anomaly_percentage': anomaly_percentage,
            'model': iso_forest,
            'features': features
        }
    
    def _detect_autoencoder(self, ts_data: pd.Series) -> Dict:
        """
        Detect anomalies using Autoencoder.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Dictionary containing Autoencoder results
        """
        logger.info("Performing Autoencoder anomaly detection")
        
        # Prepare data
        sequence_length = 10
        data_scaled = self.scaler.fit_transform(ts_data.values.reshape(-1, 1))
        
        # Create sequences
        sequences = []
        for i in range(sequence_length, len(data_scaled)):
            sequences.append(data_scaled[i-sequence_length:i].flatten())
        
        sequences = np.array(sequences)
        
        # Initialize autoencoder
        input_dim = sequences.shape[1]
        encoding_dim = self.autoencoder_config.get('encoding_dim', 32)
        
        autoencoder = AutoencoderModel(input_dim, encoding_dim)
        
        # Training parameters
        epochs = self.autoencoder_config.get('epochs', 100)
        batch_size = self.autoencoder_config.get('batch_size', 32)
        
        # Convert to PyTorch tensors
        sequences_tensor = torch.FloatTensor(sequences)
        
        # Train autoencoder
        autoencoder.train()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for i in range(0, len(sequences_tensor), batch_size):
                batch = sequences_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed = autoencoder(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
        
        # Detect anomalies
        autoencoder.eval()
        with torch.no_grad():
            reconstructed = autoencoder(sequences_tensor)
            reconstruction_errors = torch.mean((sequences_tensor - reconstructed)**2, dim=1)
        
        # Convert to numpy
        reconstruction_errors = reconstruction_errors.numpy()
        
        # Determine threshold (using 95th percentile)
        threshold = np.percentile(reconstruction_errors, 95)
        is_anomaly = reconstruction_errors > threshold
        
        # Pad with False for the first sequence_length points
        full_anomaly_labels = np.concatenate([
            np.zeros(sequence_length, dtype=bool),
            is_anomaly
        ])
        
        return {
            'anomaly_labels': full_anomaly_labels,
            'reconstruction_errors': reconstruction_errors,
            'threshold': threshold,
            'model': autoencoder,
            'n_anomalies': np.sum(is_anomaly)
        }
    
    def _detect_wavelet_based(self, ts_data: pd.Series) -> Dict:
        """
        Detect anomalies using wavelet-based methods.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Dictionary containing wavelet-based anomaly detection results
        """
        logger.info("Performing wavelet-based anomaly detection")
        
        try:
            import pywt
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(ts_data.values, 'db4', level=4)
            
            # Reconstruct detail coefficients
            detail_coeffs = []
            for i in range(1, len(coeffs)):
                coeffs_recon = [None] * len(coeffs)
                coeffs_recon[i] = coeffs[i]
                detail = pywt.waverec(coeffs_recon, 'db4')
                detail_coeffs.append(detail)
            
            # Combine detail coefficients
            combined_details = np.sum(detail_coeffs, axis=0)
            
            # Detect anomalies in detail coefficients
            detail_std = np.std(combined_details)
            detail_mean = np.mean(combined_details)
            threshold = 3 * detail_std
            
            is_anomaly = np.abs(combined_details - detail_mean) > threshold
            
            return {
                'anomaly_labels': is_anomaly,
                'detail_coefficients': combined_details,
                'threshold': threshold,
                'n_anomalies': np.sum(is_anomaly)
            }
            
        except ImportError:
            logger.warning("PyWavelets not available for wavelet-based detection")
            return {'error': 'PyWavelets not available'}
    
    def _create_features(self, ts_data: pd.Series) -> np.ndarray:
        """
        Create features for machine learning-based anomaly detection.
        
        Args:
            ts_data: Time series data
            
        Returns:
            Feature matrix
        """
        features = []
        
        # Basic statistical features
        features.append(ts_data.values)
        
        # Rolling statistics
        window = min(10, len(ts_data) // 4)
        rolling_mean = ts_data.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        rolling_std = ts_data.rolling(window=window, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        features.append(rolling_mean.values)
        features.append(rolling_std.values)
        
        # Differences
        features.append(np.concatenate([[0], np.diff(ts_data.values)]))
        
        # Second differences
        features.append(np.concatenate([[0, 0], np.diff(ts_data.values, n=2)]))
        
        # Lagged values
        lag1 = ts_data.shift(1).fillna(method='bfill').values
        lag2 = ts_data.shift(2).fillna(method='bfill').values
        
        features.append(lag1)
        features.append(lag2)
        
        # Combine features
        feature_matrix = np.column_stack(features)
        
        return feature_matrix
    
    def _combine_results(self, results: Dict, ts_data: pd.Series) -> Dict:
        """
        Combine results from multiple anomaly detection methods.
        
        Args:
            results: Results from all anomaly detection methods
            ts_data: Original time series data
            
        Returns:
            Dictionary containing combined results
        """
        logger.info("Combining anomaly detection results")
        
        # Collect anomaly labels from all methods
        anomaly_labels = {}
        method_weights = {}
        
        for method_name, method_results in results.items():
            if 'error' in method_results:
                continue
            
            if 'anomaly_labels' in method_results:
                anomaly_labels[method_name] = method_results['anomaly_labels']
                
                # Assign weights based on method reliability
                if method_name == 'statistical':
                    method_weights[method_name] = 0.3
                elif method_name == 'isolation_forest':
                    method_weights[method_name] = 0.4
                elif method_name == 'autoencoder':
                    method_weights[method_name] = 0.2
                elif method_name == 'wavelet':
                    method_weights[method_name] = 0.1
        
        if not anomaly_labels:
            return {'error': 'No valid anomaly detection results'}
        
        # Normalize weights
        total_weight = sum(method_weights.values())
        method_weights = {k: v/total_weight for k, v in method_weights.items()}
        
        # Combine using weighted voting
        combined_scores = np.zeros(len(ts_data))
        
        for method_name, labels in anomaly_labels.items():
            weight = method_weights[method_name]
            combined_scores += weight * labels.astype(float)
        
        # Determine final anomalies (threshold at 0.5)
        final_anomalies = combined_scores > 0.5
        
        # Calculate consensus metrics
        consensus_metrics = {}
        for method_name, labels in anomaly_labels.items():
            agreement = np.sum(labels == final_anomalies) / len(labels)
            consensus_metrics[method_name] = agreement
        
        return {
            'combined_anomalies': final_anomalies,
            'combined_scores': combined_scores,
            'method_weights': method_weights,
            'consensus_metrics': consensus_metrics,
            'n_anomalies': np.sum(final_anomalies),
            'anomaly_percentage': np.sum(final_anomalies) / len(ts_data) * 100
        }


class AutoencoderModel(nn.Module):
    """
    Autoencoder model for anomaly detection.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int):
        """
        Initialize autoencoder model.
        
        Args:
            input_dim: Input dimension
            encoding_dim: Encoding dimension
        """
        super(AutoencoderModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed tensor
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
