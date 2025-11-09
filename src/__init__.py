"""
Time Series Analysis with Wavelet Transform

This module provides comprehensive time series analysis capabilities including
wavelet transforms, forecasting, anomaly detection, and visualization.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from loguru import logger
import yaml
from pathlib import Path

from .data_generator import SyntheticDataGenerator
from .wavelet_analysis import WaveletAnalyzer
from .forecasting import ForecastingPipeline
from .anomaly_detection import AnomalyDetector
from .visualization import TimeSeriesVisualizer


class TimeSeriesAnalysisPipeline:
    """
    Main pipeline for comprehensive time series analysis.
    
    This class orchestrates the entire analysis workflow including data generation,
    wavelet analysis, forecasting, anomaly detection, and visualization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        self.data_generator = SyntheticDataGenerator(self.config)
        self.wavelet_analyzer = WaveletAnalyzer(self.config)
        self.forecasting_pipeline = ForecastingPipeline(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.visualizer = TimeSeriesVisualizer(self.config)
        
        self.data: Optional[pd.DataFrame] = None
        self.wavelet_results: Optional[Dict] = None
        self.forecast_results: Optional[Dict] = None
        self.anomaly_results: Optional[Dict] = None
        
        logger.info("TimeSeriesAnalysisPipeline initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data': {
                'synthetic': {
                    'n_samples': 1000,
                    'noise_level': 0.1,
                    'trend_strength': 0.5,
                    'seasonality_strength': 0.3,
                    'anomaly_probability': 0.05
                }
            },
            'wavelet': {
                'wavelet_type': 'morl',
                'scales_min': 1,
                'scales_max': 100,
                'sampling_period': 0.01
            }
        }
    
    def generate_data(self, data_type: str = "synthetic") -> pd.DataFrame:
        """
        Generate or load time series data.
        
        Args:
            data_type: Type of data to generate ("synthetic", "energy", "stock")
            
        Returns:
            DataFrame with time series data
        """
        logger.info(f"Generating {data_type} data")
        
        if data_type == "synthetic":
            self.data = self.data_generator.generate_synthetic_data()
        else:
            self.data = self.data_generator.load_real_data(data_type)
        
        logger.info(f"Data generated with shape: {self.data.shape}")
        return self.data
    
    def perform_wavelet_analysis(self) -> Dict:
        """
        Perform wavelet transform analysis on the time series.
        
        Returns:
            Dictionary containing wavelet analysis results
        """
        if self.data is None:
            raise ValueError("No data available. Call generate_data() first.")
        
        logger.info("Performing wavelet analysis")
        self.wavelet_results = self.wavelet_analyzer.analyze(self.data)
        logger.info("Wavelet analysis completed")
        return self.wavelet_results
    
    def perform_forecasting(self) -> Dict:
        """
        Perform forecasting using multiple models.
        
        Returns:
            Dictionary containing forecasting results
        """
        if self.data is None:
            raise ValueError("No data available. Call generate_data() first.")
        
        logger.info("Performing forecasting analysis")
        self.forecast_results = self.forecasting_pipeline.forecast(self.data)
        logger.info("Forecasting analysis completed")
        return self.forecast_results
    
    def detect_anomalies(self) -> Dict:
        """
        Detect anomalies in the time series.
        
        Returns:
            Dictionary containing anomaly detection results
        """
        if self.data is None:
            raise ValueError("No data available. Call generate_data() first.")
        
        logger.info("Performing anomaly detection")
        self.anomaly_results = self.anomaly_detector.detect(self.data)
        logger.info("Anomaly detection completed")
        return self.anomaly_results
    
    def visualize_results(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualizations of all analysis results.
        
        Args:
            save_path: Optional path to save plots
        """
        if self.data is None:
            raise ValueError("No data available. Call generate_data() first.")
        
        logger.info("Creating visualizations")
        
        # Create comprehensive dashboard
        self.visualizer.create_dashboard(
            data=self.data,
            wavelet_results=self.wavelet_results,
            forecast_results=self.forecast_results,
            anomaly_results=self.anomaly_results,
            save_path=save_path
        )
        
        logger.info("Visualizations completed")
    
    def run_full_analysis(self, data_type: str = "synthetic") -> Dict:
        """
        Run the complete analysis pipeline.
        
        Args:
            data_type: Type of data to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting full analysis pipeline")
        
        # Generate data
        self.generate_data(data_type)
        
        # Perform analyses
        wavelet_results = self.perform_wavelet_analysis()
        forecast_results = self.perform_forecasting()
        anomaly_results = self.detect_anomalies()
        
        # Create visualizations
        self.visualize_results()
        
        # Compile results
        results = {
            'data': self.data,
            'wavelet': wavelet_results,
            'forecasting': forecast_results,
            'anomaly_detection': anomaly_results
        }
        
        logger.info("Full analysis pipeline completed")
        return results


def main():
    """Main function to run the analysis pipeline."""
    # Initialize pipeline
    pipeline = TimeSeriesAnalysisPipeline()
    
    # Run full analysis
    results = pipeline.run_full_analysis("synthetic")
    
    # Print summary
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("="*50)
    print(f"Data shape: {results['data'].shape}")
    print(f"Wavelet analysis: {'Completed' if results['wavelet'] else 'Not performed'}")
    print(f"Forecasting: {'Completed' if results['forecasting'] else 'Not performed'}")
    print(f"Anomaly detection: {'Completed' if results['anomaly_detection'] else 'Not performed'}")
    print("="*50)


if __name__ == "__main__":
    main()
