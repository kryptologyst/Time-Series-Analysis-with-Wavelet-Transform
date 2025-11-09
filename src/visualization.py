"""
Visualization module for time series analysis.

This module provides comprehensive visualization capabilities for time series
analysis including plots for trends, seasonality, anomalies, forecasts, and wavelet analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class TimeSeriesVisualizer:
    """
    Comprehensive visualization for time series analysis results.
    
    This class provides methods to create various plots including time series plots,
    wavelet scalograms, forecasting results, anomaly detection, and interactive dashboards.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary containing visualization parameters
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        
        # Visualization parameters
        self.figure_size = tuple(self.viz_config.get('figure_size', [12, 8]))
        self.style = self.viz_config.get('style', 'seaborn-v0_8')
        self.color_palette = self.viz_config.get('color_palette', 'husl')
        
        # Set matplotlib style
        plt.style.use(self.style)
        sns.set_palette(self.color_palette)
        
        logger.info("TimeSeriesVisualizer initialized")
    
    def create_dashboard(self, data: pd.DataFrame, wavelet_results: Optional[Dict] = None,
                        forecast_results: Optional[Dict] = None, 
                        anomaly_results: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with all analysis results.
        
        Args:
            data: Time series data
            wavelet_results: Wavelet analysis results
            forecast_results: Forecasting results
            anomaly_results: Anomaly detection results
            save_path: Optional path to save the dashboard
        """
        logger.info("Creating comprehensive dashboard")
        
        # Determine number of subplots needed
        n_plots = 1  # Basic time series plot
        
        if wavelet_results is not None:
            n_plots += 2  # Scalogram and DWT decomposition
        
        if forecast_results is not None:
            n_plots += 1  # Forecasting plot
        
        if anomaly_results is not None:
            n_plots += 1  # Anomaly detection plot
        
        # Create subplots
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Basic time series plot
        self._plot_time_series(data, axes[plot_idx])
        plot_idx += 1
        
        # 2. Wavelet analysis plots
        if wavelet_results is not None:
            if plot_idx < n_plots:
                self._plot_wavelet_scalogram(wavelet_results, axes[plot_idx])
                plot_idx += 1
            
            if plot_idx < n_plots:
                self._plot_dwt_decomposition(wavelet_results, data, axes[plot_idx])
                plot_idx += 1
        
        # 3. Forecasting plot
        if forecast_results is not None and plot_idx < n_plots:
            self._plot_forecasting_results(forecast_results, data, axes[plot_idx])
            plot_idx += 1
        
        # 4. Anomaly detection plot
        if anomaly_results is not None and plot_idx < n_plots:
            self._plot_anomaly_detection(anomaly_results, data, axes[plot_idx])
            plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        
        plt.show()
    
    def _plot_time_series(self, data: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot the basic time series.
        
        Args:
            data: Time series data
            ax: Matplotlib axes object
        """
        if 'timestamp' in data.columns:
            time_col = 'timestamp'
        else:
            time_col = data.columns[0]
        
        if 'value' in data.columns:
            value_col = 'value'
        else:
            value_col = data.columns[1]
        
        ax.plot(data[time_col], data[value_col], linewidth=1, alpha=0.8)
        ax.set_title('Time Series Data', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Add trend line if available
        if 'trend' in data.columns:
            ax.plot(data[time_col], data['trend'], '--', alpha=0.7, label='Trend')
            ax.legend()
    
    def _plot_wavelet_scalogram(self, wavelet_results: Dict, ax: plt.Axes) -> None:
        """
        Plot wavelet scalogram.
        
        Args:
            wavelet_results: Wavelet analysis results
            ax: Matplotlib axes object
        """
        if wavelet_results['cwt']['magnitude'] is None:
            ax.text(0.5, 0.5, 'Wavelet Scalogram\n(Data not available)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Wavelet Scalogram', fontsize=14, fontweight='bold')
            return
        
        magnitude = wavelet_results['cwt']['magnitude']
        scales = wavelet_results['cwt']['scales']
        
        im = ax.imshow(
            magnitude,
            extent=[0, magnitude.shape[1], scales[0], scales[-1]],
            cmap='jet',
            aspect='auto',
            origin='lower'
        )
        
        ax.set_title('Wavelet Scalogram (CWT)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Scale')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Magnitude')
    
    def _plot_dwt_decomposition(self, wavelet_results: Dict, data: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot DWT decomposition components.
        
        Args:
            wavelet_results: Wavelet analysis results
            data: Original time series data
            ax: Matplotlib axes object
        """
        if wavelet_results['dwt']['reconstructed_components'] is None:
            ax.text(0.5, 0.5, 'DWT Decomposition\n(Data not available)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('DWT Decomposition', fontsize=14, fontweight='bold')
            return
        
        components = wavelet_results['dwt']['reconstructed_components']
        
        # Plot approximation and first few details
        ax.plot(components[0], label='Approximation', linewidth=2)
        
        for i in range(1, min(4, len(components))):
            ax.plot(components[i], label=f'Detail {i}', alpha=0.7)
        
        ax.set_title('DWT Decomposition Components', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_forecasting_results(self, forecast_results: Dict, data: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot forecasting results.
        
        Args:
            forecast_results: Forecasting results
            data: Original time series data
            ax: Matplotlib axes object
        """
        # Plot original data
        if 'value' in data.columns:
            ax.plot(data['value'].values, label='Original', alpha=0.7)
        else:
            ax.plot(data.iloc[:, 1].values, label='Original', alpha=0.7)
        
        # Plot forecasts from different models
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        color_idx = 0
        
        for model_name, model_results in forecast_results.items():
            if 'error' in model_results or 'forecast' not in model_results:
                continue
            
            forecast = model_results['forecast']
            if isinstance(forecast, (list, np.ndarray)):
                # Plot forecast
                forecast_start = len(data) - len(forecast) if len(forecast) < len(data) else len(data)
                forecast_x = range(forecast_start, forecast_start + len(forecast))
                
                ax.plot(forecast_x, forecast, 
                       label=f'{model_name.title()} Forecast', 
                       color=colors[color_idx % len(colors)],
                       linewidth=2)
                
                # Plot confidence intervals if available
                if 'confidence_interval' in model_results:
                    ci = model_results['confidence_interval']
                    ax.fill_between(forecast_x, ci[:, 0], ci[:, 1], 
                                   alpha=0.2, color=colors[color_idx % len(colors)])
                
                color_idx += 1
        
        ax.set_title('Forecasting Results', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_anomaly_detection(self, anomaly_results: Dict, data: pd.DataFrame, ax: plt.Axes) -> None:
        """
        Plot anomaly detection results.
        
        Args:
            anomaly_results: Anomaly detection results
            data: Original time series data
            ax: Matplotlib axes object
        """
        # Plot original data
        if 'value' in data.columns:
            ax.plot(data['value'].values, label='Original', alpha=0.7, color='blue')
            values = data['value'].values
        else:
            ax.plot(data.iloc[:, 1].values, label='Original', alpha=0.7, color='blue')
            values = data.iloc[:, 1].values
        
        # Plot anomalies from different methods
        colors = ['red', 'green', 'orange', 'purple']
        color_idx = 0
        
        for method_name, method_results in anomaly_results.items():
            if 'error' in method_results or 'anomaly_labels' not in method_results:
                continue
            
            anomaly_labels = method_results['anomaly_labels']
            if len(anomaly_labels) == len(values):
                anomaly_indices = np.where(anomaly_labels)[0]
                if len(anomaly_indices) > 0:
                    ax.scatter(anomaly_indices, values[anomaly_indices], 
                             color=colors[color_idx % len(colors)], 
                             label=f'{method_name.title()} Anomalies',
                             s=20, alpha=0.8)
                    color_idx += 1
        
        # Plot combined anomalies if available
        if 'combined' in anomaly_results and 'combined_anomalies' in anomaly_results['combined']:
            combined_anomalies = anomaly_results['combined']['combined_anomalies']
            if len(combined_anomalies) == len(values):
                combined_indices = np.where(combined_anomalies)[0]
                if len(combined_indices) > 0:
                    ax.scatter(combined_indices, values[combined_indices], 
                             color='black', marker='x', s=50,
                             label='Combined Anomalies', alpha=0.9)
        
        ax.set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_interactive_dashboard(self, data: pd.DataFrame, wavelet_results: Optional[Dict] = None,
                                   forecast_results: Optional[Dict] = None,
                                   anomaly_results: Optional[Dict] = None) -> go.Figure:
        """
        Create an interactive Plotly dashboard.
        
        Args:
            data: Time series data
            wavelet_results: Wavelet analysis results
            forecast_results: Forecasting results
            anomaly_results: Anomaly detection results
            
        Returns:
            Plotly figure object
        """
        logger.info("Creating interactive Plotly dashboard")
        
        # Determine number of subplots
        n_subplots = 1
        if wavelet_results is not None:
            n_subplots += 1
        if forecast_results is not None:
            n_subplots += 1
        if anomaly_results is not None:
            n_subplots += 1
        
        # Create subplots
        fig = make_subplots(
            rows=n_subplots, cols=1,
            subplot_titles=['Time Series', 'Wavelet Scalogram', 'Forecasting', 'Anomaly Detection'][:n_subplots],
            vertical_spacing=0.1
        )
        
        row_idx = 1
        
        # 1. Time series plot
        if 'timestamp' in data.columns:
            time_col = 'timestamp'
        else:
            time_col = data.columns[0]
        
        if 'value' in data.columns:
            value_col = 'value'
        else:
            value_col = data.columns[1]
        
        fig.add_trace(
            go.Scatter(x=data[time_col], y=data[value_col], 
                      mode='lines', name='Time Series', line=dict(color='blue')),
            row=row_idx, col=1
        )
        row_idx += 1
        
        # 2. Wavelet scalogram
        if wavelet_results is not None and wavelet_results['cwt']['magnitude'] is not None:
            magnitude = wavelet_results['cwt']['magnitude']
            scales = wavelet_results['cwt']['scales']
            
            fig.add_trace(
                go.Heatmap(z=magnitude, x=list(range(magnitude.shape[1])), 
                          y=scales, colorscale='Jet', name='Scalogram'),
                row=row_idx, col=1
            )
            row_idx += 1
        
        # 3. Forecasting
        if forecast_results is not None:
            # Add original data
            fig.add_trace(
                go.Scatter(x=data[time_col], y=data[value_col], 
                          mode='lines', name='Original', line=dict(color='blue')),
                row=row_idx, col=1
            )
            
            # Add forecasts
            for model_name, model_results in forecast_results.items():
                if 'error' not in model_results and 'forecast' in model_results:
                    forecast = model_results['forecast']
                    if isinstance(forecast, (list, np.ndarray)):
                        forecast_start = len(data) - len(forecast) if len(forecast) < len(data) else len(data)
                        forecast_x = data[time_col].iloc[forecast_start:forecast_start + len(forecast)]
                        
                        fig.add_trace(
                            go.Scatter(x=forecast_x, y=forecast, 
                                      mode='lines', name=f'{model_name.title()} Forecast'),
                            row=row_idx, col=1
                        )
            
            row_idx += 1
        
        # 4. Anomaly detection
        if anomaly_results is not None:
            # Add original data
            fig.add_trace(
                go.Scatter(x=data[time_col], y=data[value_col], 
                          mode='lines', name='Original', line=dict(color='blue')),
                row=row_idx, col=1
            )
            
            # Add anomalies
            for method_name, method_results in anomaly_results.items():
                if 'error' not in method_results and 'anomaly_labels' in method_results:
                    anomaly_labels = method_results['anomaly_labels']
                    if len(anomaly_labels) == len(data):
                        anomaly_indices = np.where(anomaly_labels)[0]
                        if len(anomaly_indices) > 0:
                            fig.add_trace(
                                go.Scatter(x=data[time_col].iloc[anomaly_indices], 
                                         y=data[value_col].iloc[anomaly_indices],
                                         mode='markers', name=f'{method_name.title()} Anomalies',
                                         marker=dict(size=8, symbol='x')),
                                row=row_idx, col=1
                            )
        
        # Update layout
        fig.update_layout(
            height=300 * n_subplots,
            title_text="Time Series Analysis Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        logger.info("Interactive dashboard created")
        return fig
    
    def plot_wavelet_features(self, wavelet_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot wavelet-based features and statistics.
        
        Args:
            wavelet_results: Wavelet analysis results
            save_path: Optional path to save the plot
        """
        logger.info("Creating wavelet features plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Energy distribution across scales
        if wavelet_results['cwt']['energy_distribution'] is not None:
            energy_dist = wavelet_results['cwt']['energy_distribution']
            scales = wavelet_results['cwt']['scales']
            
            axes[0, 0].plot(scales, energy_dist)
            axes[0, 0].set_title('Energy Distribution Across Scales')
            axes[0, 0].set_xlabel('Scale')
            axes[0, 0].set_ylabel('Energy')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Dominant frequencies
        if wavelet_results['cwt']['dominant_frequencies'] is not None:
            dominant_freqs = wavelet_results['cwt']['dominant_frequencies']
            axes[0, 1].bar(range(len(dominant_freqs)), dominant_freqs)
            axes[0, 1].set_title('Dominant Frequencies')
            axes[0, 1].set_xlabel('Frequency Index')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # DWT energy levels
        if wavelet_results['dwt']['energy_levels'] is not None:
            energy_levels = wavelet_results['dwt']['energy_levels']
            levels = range(len(energy_levels))
            
            axes[1, 0].bar(levels, energy_levels)
            axes[1, 0].set_title('DWT Energy Levels')
            axes[1, 0].set_xlabel('Decomposition Level')
            axes[1, 0].set_ylabel('Energy')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Relative energy distribution
        if wavelet_results['dwt']['relative_energy'] is not None:
            relative_energy = wavelet_results['dwt']['relative_energy']
            levels = range(len(relative_energy))
            
            axes[1, 1].pie(relative_energy, labels=[f'Level {i}' for i in levels], autopct='%1.1f%%')
            axes[1, 1].set_title('Relative Energy Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Wavelet features plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, forecast_results: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot model comparison results.
        
        Args:
            forecast_results: Forecasting results
            save_path: Optional path to save the plot
        """
        logger.info("Creating model comparison plot")
        
        # Extract model metrics
        models = []
        mae_scores = []
        rmse_scores = []
        
        for model_name, model_results in forecast_results.items():
            if 'error' not in model_results and 'training_mae' in model_results:
                models.append(model_name.title())
                mae_scores.append(model_results['training_mae'])
                rmse_scores.append(model_results['training_rmse'])
        
        if not models:
            logger.warning("No valid model results for comparison")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAE comparison
        bars1 = ax1.bar(models, mae_scores, color='skyblue', alpha=0.7)
        ax1.set_title('Model Comparison - MAE', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, mae_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')
        
        # RMSE comparison
        bars2 = ax2.bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        ax2.set_title('Model Comparison - RMSE', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Root Mean Square Error')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
