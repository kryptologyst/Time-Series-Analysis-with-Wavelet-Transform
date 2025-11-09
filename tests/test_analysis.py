"""
Unit tests for the time series analysis project.

This module contains comprehensive tests for all major components
including data generation, wavelet analysis, forecasting, and anomaly detection.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_generator import SyntheticDataGenerator
from src.wavelet_analysis import WaveletAnalyzer
from src.forecasting import ForecastingPipeline
from src.anomaly_detection import AnomalyDetector
from src.visualization import TimeSeriesVisualizer


class TestSyntheticDataGenerator:
    """Test cases for SyntheticDataGenerator."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'data': {
                'synthetic': {
                    'n_samples': 100,
                    'noise_level': 0.1,
                    'trend_strength': 0.5,
                    'seasonality_strength': 0.3,
                    'anomaly_probability': 0.05
                }
            }
        }
    
    @pytest.fixture
    def generator(self, config):
        """SyntheticDataGenerator instance."""
        return SyntheticDataGenerator(config)
    
    def test_generate_synthetic_data(self, generator):
        """Test synthetic data generation."""
        data = generator.generate_synthetic_data()
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'timestamp' in data.columns
        assert 'value' in data.columns
        assert 'trend' in data.columns
        assert 'seasonality' in data.columns
        assert 'noise' in data.columns
        assert 'anomaly' in data.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(data['timestamp'])
        assert pd.api.types.is_numeric_dtype(data['value'])
        assert pd.api.types.is_bool_dtype(data['anomaly'])
        
        # Check for anomalies
        assert data['anomaly'].sum() > 0  # Should have some anomalies
    
    def test_generate_chirp_signal(self, generator):
        """Test chirp signal generation."""
        data = generator.generate_chirp_signal(duration=1.0, n_samples=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'value' in data.columns
        assert 'frequency' in data.columns
        
        # Check frequency increases over time
        frequencies = data['frequency'].values
        assert frequencies[0] < frequencies[-1]  # Frequency should increase
    
    def test_data_properties(self, generator):
        """Test data properties and statistics."""
        data = generator.generate_synthetic_data()
        
        # Check that components add up
        reconstructed = data['trend'] + data['seasonality'] + data['noise'] + data['anomaly']
        np.testing.assert_allclose(data['value'], reconstructed, rtol=1e-10)


class TestWaveletAnalyzer:
    """Test cases for WaveletAnalyzer."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'wavelet': {
                'wavelet_type': 'morl',
                'scales_min': 1,
                'scales_max': 50,
                'sampling_period': 0.01
            }
        }
    
    @pytest.fixture
    def analyzer(self, config):
        """WaveletAnalyzer instance."""
        return WaveletAnalyzer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample time series data for testing."""
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value': signal
        })
    
    def test_analyze(self, analyzer, sample_data):
        """Test wavelet analysis."""
        results = analyzer.analyze(sample_data)
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'cwt' in results
        assert 'dwt' in results
        assert 'features' in results
        assert 'signal' in results
        
        # Check CWT results
        cwt_results = results['cwt']
        assert 'coefficients' in cwt_results
        assert 'frequencies' in cwt_results
        assert 'scales' in cwt_results
        assert 'magnitude' in cwt_results
        
        # Check DWT results
        dwt_results = results['dwt']
        assert 'coefficients' in dwt_results
        assert 'approximation' in dwt_results
        assert 'details' in dwt_results
    
    def test_cwt_properties(self, analyzer, sample_data):
        """Test CWT properties."""
        results = analyzer.analyze(sample_data)
        cwt_results = results['cwt']
        
        if cwt_results['magnitude'] is not None:
            magnitude = cwt_results['magnitude']
            scales = cwt_results['scales']
            
            # Check dimensions
            assert magnitude.shape[0] == len(scales)
            assert magnitude.shape[1] == len(sample_data)
            
            # Check magnitude is non-negative
            assert np.all(magnitude >= 0)
    
    def test_dwt_properties(self, analyzer, sample_data):
        """Test DWT properties."""
        results = analyzer.analyze(sample_data)
        dwt_results = results['dwt']
        
        if dwt_results['coefficients'] is not None:
            coeffs = dwt_results['coefficients']
            
            # Check that we have approximation and details
            assert len(coeffs) > 1
            assert len(coeffs[0]) > 0  # Approximation should not be empty


class TestForecastingPipeline:
    """Test cases for ForecastingPipeline."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'models': {
                'arima': {
                    'auto_arima': True,
                    'seasonal': True,
                    'max_p': 3,
                    'max_q': 3
                },
                'lstm': {
                    'sequence_length': 10,
                    'hidden_units': 20,
                    'epochs': 5,
                    'batch_size': 16
                }
            }
        }
    
    @pytest.fixture
    def forecaster(self, config):
        """ForecastingPipeline instance."""
        return ForecastingPipeline(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample time series data for testing."""
        np.random.seed(42)
        t = np.arange(100)
        trend = 0.1 * t
        seasonality = 2 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 0.5, 100)
        signal = trend + seasonality + noise
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='D'),
            'value': signal
        })
    
    def test_forecast_baseline(self, forecaster, sample_data):
        """Test baseline forecasting methods."""
        results = forecaster.forecast(sample_data, forecast_horizon=10)
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'baseline' in results
        
        baseline_results = results['baseline']
        assert 'moving_average' in baseline_results
        assert 'linear_trend' in baseline_results
        assert 'naive' in baseline_results
        assert 'seasonal_naive' in baseline_results
        
        # Check forecast lengths
        for method, forecast in baseline_results.items():
            assert len(forecast) == 10
    
    def test_forecast_arima(self, forecaster, sample_data):
        """Test ARIMA forecasting (if available)."""
        try:
            results = forecaster.forecast(sample_data, forecast_horizon=10)
            
            if 'arima' in results and 'error' not in results['arima']:
                arima_results = results['arima']
                assert 'forecast' in arima_results
                assert 'training_mae' in arima_results
                assert 'training_rmse' in arima_results
                assert len(arima_results['forecast']) == 10
        except ImportError:
            pytest.skip("pmdarima not available")


class TestAnomalyDetector:
    """Test cases for AnomalyDetector."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'anomaly_detection': {
                'isolation_forest': {
                    'contamination': 0.1,
                    'random_state': 42
                }
            }
        }
    
    @pytest.fixture
    def detector(self, config):
        """AnomalyDetector instance."""
        return AnomalyDetector(config)
    
    @pytest.fixture
    def sample_data_with_anomalies(self):
        """Sample time series data with known anomalies."""
        np.random.seed(42)
        t = np.arange(100)
        signal = np.sin(2 * np.pi * t / 20) + 0.1 * np.random.randn(100)
        
        # Add some anomalies
        signal[20] += 3  # Spike
        signal[50] -= 2  # Drop
        signal[80:85] += 1.5  # Shift
        
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value': signal
        })
    
    def test_detect_statistical(self, detector, sample_data_with_anomalies):
        """Test statistical anomaly detection."""
        results = detector.detect(sample_data_with_anomalies)
        
        assert isinstance(results, dict)
        assert 'statistical' in results
        
        statistical_results = results['statistical']
        assert 'z_score_anomalies' in statistical_results
        assert 'iqr_anomalies' in statistical_results
        assert 'anomaly_counts' in statistical_results
        
        # Check that anomalies are detected
        assert statistical_results['anomaly_counts']['z_score'] > 0
    
    def test_detect_isolation_forest(self, detector, sample_data_with_anomalies):
        """Test Isolation Forest anomaly detection."""
        results = detector.detect(sample_data_with_anomalies)
        
        assert 'isolation_forest' in results
        
        if_results = results['isolation_forest']
        if 'error' not in if_results:
            assert 'anomaly_labels' in if_results
            assert 'anomaly_scores' in if_results
            assert 'n_anomalies' in if_results
    
    def test_combine_results(self, detector, sample_data_with_anomalies):
        """Test result combination."""
        results = detector.detect(sample_data_with_anomalies)
        
        assert 'combined' in results
        
        combined_results = results['combined']
        if 'error' not in combined_results:
            assert 'combined_anomalies' in combined_results
            assert 'combined_scores' in combined_results
            assert 'method_weights' in combined_results


class TestTimeSeriesVisualizer:
    """Test cases for TimeSeriesVisualizer."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'visualization': {
                'figure_size': [10, 6],
                'style': 'seaborn-v0_8',
                'color_palette': 'husl'
            }
        }
    
    @pytest.fixture
    def visualizer(self, config):
        """TimeSeriesVisualizer instance."""
        return TimeSeriesVisualizer(config)
    
    @pytest.fixture
    def sample_data(self):
        """Sample time series data for testing."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=100, freq='H'),
            'value': np.random.randn(100).cumsum()
        })
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization."""
        assert visualizer.config is not None
        assert visualizer.figure_size == (10, 6)
        assert visualizer.style == 'seaborn-v0_8'
    
    def test_create_interactive_dashboard(self, visualizer, sample_data):
        """Test interactive dashboard creation."""
        fig = visualizer.create_interactive_dashboard(sample_data)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def config(self):
        """Default configuration for testing."""
        return {
            'data': {
                'synthetic': {
                    'n_samples': 50,
                    'noise_level': 0.1,
                    'trend_strength': 0.3,
                    'seasonality_strength': 0.2,
                    'anomaly_probability': 0.1
                }
            },
            'wavelet': {
                'wavelet_type': 'morl',
                'scales_min': 1,
                'scales_max': 20,
                'sampling_period': 0.01
            },
            'models': {
                'lstm': {
                    'sequence_length': 5,
                    'hidden_units': 10,
                    'epochs': 2,
                    'batch_size': 8
                }
            }
        }
    
    def test_full_pipeline(self, config):
        """Test the complete analysis pipeline."""
        from src import TimeSeriesAnalysisPipeline
        
        # Initialize pipeline
        pipeline = TimeSeriesAnalysisPipeline()
        pipeline.config = config
        
        # Generate data
        data = pipeline.generate_data("synthetic")
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 50
        
        # Perform wavelet analysis
        wavelet_results = pipeline.perform_wavelet_analysis()
        assert isinstance(wavelet_results, dict)
        
        # Perform forecasting
        forecast_results = pipeline.perform_forecasting()
        assert isinstance(forecast_results, dict)
        
        # Detect anomalies
        anomaly_results = pipeline.detect_anomalies()
        assert isinstance(anomaly_results, dict)
        
        # Create visualizations
        pipeline.visualize_results()
        
        # Run full analysis
        results = pipeline.run_full_analysis("synthetic")
        assert isinstance(results, dict)
        assert 'data' in results
        assert 'wavelet' in results
        assert 'forecasting' in results
        assert 'anomaly_detection' in results


if __name__ == "__main__":
    pytest.main([__file__])
