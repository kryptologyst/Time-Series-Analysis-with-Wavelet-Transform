"""
Data generation and loading utilities for time series analysis.

This module provides functionality to generate synthetic time series data
and load real-world datasets for analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger
import requests
from io import StringIO


class SyntheticDataGenerator:
    """
    Generate realistic synthetic time series data with various characteristics.
    
    This class creates time series with trends, seasonality, noise, and anomalies
    to test and demonstrate analysis capabilities.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration dictionary containing data generation parameters
        """
        self.config = config
        self.synthetic_config = config.get('data', {}).get('synthetic', {})
        
        # Default parameters
        self.n_samples = self.synthetic_config.get('n_samples', 1000)
        self.noise_level = self.synthetic_config.get('noise_level', 0.1)
        self.trend_strength = self.synthetic_config.get('trend_strength', 0.5)
        self.seasonality_strength = self.synthetic_config.get('seasonality_strength', 0.3)
        self.anomaly_probability = self.synthetic_config.get('anomaly_probability', 0.05)
        
        logger.info("SyntheticDataGenerator initialized")
    
    def generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic time series data with multiple components.
        
        Returns:
            DataFrame with time series data including timestamp, value, trend, seasonality, and noise
        """
        logger.info(f"Generating synthetic data with {self.n_samples} samples")
        
        # Create time index
        t = np.linspace(0, 10, self.n_samples)
        
        # Generate base components
        trend = self._generate_trend(t)
        seasonality = self._generate_seasonality(t)
        noise = self._generate_noise(t)
        anomalies = self._generate_anomalies(t)
        
        # Combine components
        signal = trend + seasonality + noise + anomalies
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=self.n_samples, freq='H'),
            'value': signal,
            'trend': trend,
            'seasonality': seasonality,
            'noise': noise,
            'anomaly': anomalies != 0,
            'time': t
        })
        
        logger.info("Synthetic data generation completed")
        return data
    
    def _generate_trend(self, t: np.ndarray) -> np.ndarray:
        """Generate trend component."""
        # Linear trend with some curvature
        trend = (self.trend_strength * t + 
                 0.1 * self.trend_strength * t**2 + 
                 0.01 * self.trend_strength * np.sin(0.5 * t))
        return trend
    
    def _generate_seasonality(self, t: np.ndarray) -> np.ndarray:
        """Generate seasonal component."""
        # Multiple seasonal patterns
        daily_season = self.seasonality_strength * np.sin(2 * np.pi * t / 24)  # Daily
        weekly_season = 0.5 * self.seasonality_strength * np.sin(2 * np.pi * t / (24 * 7))  # Weekly
        monthly_season = 0.3 * self.seasonality_strength * np.sin(2 * np.pi * t / (24 * 30))  # Monthly
        
        return daily_season + weekly_season + monthly_season
    
    def _generate_noise(self, t: np.ndarray) -> np.ndarray:
        """Generate noise component."""
        # White noise with some autocorrelation
        noise = np.random.normal(0, self.noise_level, len(t))
        
        # Add some autocorrelation
        for i in range(1, len(noise)):
            noise[i] += 0.1 * noise[i-1]
        
        return noise
    
    def _generate_anomalies(self, t: np.ndarray) -> np.ndarray:
        """Generate anomaly component."""
        anomalies = np.zeros_like(t)
        
        # Randomly place anomalies
        n_anomalies = int(self.anomaly_probability * len(t))
        anomaly_indices = np.random.choice(len(t), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            # Random anomaly magnitude and type
            magnitude = np.random.uniform(2, 5) * self.noise_level
            anomaly_type = np.random.choice(['spike', 'drop', 'shift'])
            
            if anomaly_type == 'spike':
                anomalies[idx] = magnitude
            elif anomaly_type == 'drop':
                anomalies[idx] = -magnitude
            else:  # shift
                # Create a temporary shift
                shift_duration = min(10, len(t) - idx)
                anomalies[idx:idx+shift_duration] = magnitude * 0.5
        
        return anomalies
    
    def load_real_data(self, data_type: str) -> pd.DataFrame:
        """
        Load real-world time series data.
        
        Args:
            data_type: Type of data to load ("energy", "stock", "weather")
            
        Returns:
            DataFrame with real time series data
        """
        logger.info(f"Loading real data: {data_type}")
        
        real_data_config = self.config.get('data', {}).get('real_data', {})
        sources = real_data_config.get('sources', [])
        
        # Find the requested data source
        source = None
        for s in sources:
            if s['name'] == data_type:
                source = s
                break
        
        if source is None:
            raise ValueError(f"Data source '{data_type}' not found in configuration")
        
        try:
            # Download data
            response = requests.get(source['url'])
            response.raise_for_status()
            
            # Parse CSV data
            data = pd.read_csv(StringIO(response.text))
            
            # Standardize column names and add timestamp
            if 'date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['date'])
            elif 'Date' in data.columns:
                data['timestamp'] = pd.to_datetime(data['Date'])
            else:
                # Create synthetic timestamp if none exists
                data['timestamp'] = pd.date_range('2020-01-01', periods=len(data), freq='D')
            
            # Use the first numeric column as value
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data['value'] = data[numeric_cols[0]]
            else:
                raise ValueError("No numeric columns found in the data")
            
            # Clean and prepare data
            data = data[['timestamp', 'value']].dropna()
            data = data.reset_index(drop=True)
            
            logger.info(f"Real data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            logger.info("Falling back to synthetic data")
            return self.generate_synthetic_data()
    
    def generate_chirp_signal(self, duration: float = 1.0, n_samples: int = 400) -> pd.DataFrame:
        """
        Generate a chirp signal (frequency sweep) for wavelet analysis demonstration.
        
        Args:
            duration: Duration of the signal in seconds
            n_samples: Number of samples
            
        Returns:
            DataFrame with chirp signal data
        """
        logger.info(f"Generating chirp signal: {duration}s, {n_samples} samples")
        
        t = np.linspace(0, duration, n_samples)
        
        # Create chirp signal with increasing frequency
        f0, f1 = 5, 20  # Start and end frequencies
        signal = np.sin(2 * np.pi * f0 * t + 2 * np.pi * (f1 - f0) * t**2 / (2 * duration))
        
        # Add some noise
        noise = np.random.normal(0, 0.1, len(signal))
        signal += noise
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='S'),
            'value': signal,
            'time': t,
            'frequency': f0 + (f1 - f0) * t / duration
        })
        
        logger.info("Chirp signal generation completed")
        return data
