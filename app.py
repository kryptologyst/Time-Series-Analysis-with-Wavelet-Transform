"""
Streamlit web interface for time series analysis.

This module provides an interactive web interface for exploring time series
analysis results including wavelet transforms, forecasting, and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src import TimeSeriesAnalysisPipeline


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Time Series Analysis Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Time Series Analysis Dashboard")
    st.markdown("Comprehensive time series analysis with wavelet transforms, forecasting, and anomaly detection")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data type selection
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["synthetic", "energy", "stock"],
        help="Choose the type of time series data to analyze"
    )
    
    # Analysis options
    st.sidebar.header("Analysis Options")
    
    perform_wavelet = st.sidebar.checkbox("Wavelet Analysis", value=True)
    perform_forecasting = st.sidebar.checkbox("Forecasting", value=True)
    perform_anomaly_detection = st.sidebar.checkbox("Anomaly Detection", value=True)
    
    # Forecasting parameters
    if perform_forecasting:
        st.sidebar.subheader("Forecasting Parameters")
        forecast_horizon = st.sidebar.slider("Forecast Horizon", 10, 100, 30)
        
        # Model selection
        st.sidebar.subheader("Forecasting Models")
        use_arima = st.sidebar.checkbox("ARIMA", value=True)
        use_prophet = st.sidebar.checkbox("Prophet", value=True)
        use_lstm = st.sidebar.checkbox("LSTM", value=True)
    
    # Wavelet parameters
    if perform_wavelet:
        st.sidebar.subheader("Wavelet Parameters")
        wavelet_type = st.sidebar.selectbox(
            "Wavelet Type",
            ["morl", "cmor", "cgau", "mexh", "gaus"],
            help="Type of wavelet to use for analysis"
        )
        scales_max = st.sidebar.slider("Maximum Scale", 50, 200, 100)
    
    # Run analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        run_analysis(data_type, perform_wavelet, perform_forecasting, 
                    perform_anomaly_detection, wavelet_type, scales_max)
    
    # Display instructions
    st.markdown("""
    ### How to Use This Dashboard
    
    1. **Select Data Type**: Choose between synthetic data or real-world datasets
    2. **Configure Analysis**: Select which analyses to perform
    3. **Adjust Parameters**: Fine-tune analysis parameters in the sidebar
    4. **Run Analysis**: Click the "Run Analysis" button to start processing
    5. **Explore Results**: View interactive plots and analysis results below
    
    ### Features
    
    - **Wavelet Analysis**: Time-frequency analysis using Continuous and Discrete Wavelet Transforms
    - **Forecasting**: Multiple forecasting models including ARIMA, Prophet, and LSTM
    - **Anomaly Detection**: Statistical and machine learning-based anomaly detection
    - **Interactive Visualizations**: Explore results with interactive Plotly charts
    """)


def run_analysis(data_type: str, perform_wavelet: bool, perform_forecasting: bool,
                perform_anomaly_detection: bool, wavelet_type: str = "morl", 
                scales_max: int = 100):
    """
    Run the time series analysis and display results.
    
    Args:
        data_type: Type of data to analyze
        perform_wavelet: Whether to perform wavelet analysis
        perform_forecasting: Whether to perform forecasting
        perform_anomaly_detection: Whether to perform anomaly detection
        wavelet_type: Type of wavelet to use
        scales_max: Maximum scale for wavelet analysis
    """
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize pipeline
        status_text.text("Initializing analysis pipeline...")
        progress_bar.progress(10)
        
        pipeline = TimeSeriesAnalysisPipeline()
        
        # Update wavelet configuration
        if perform_wavelet:
            pipeline.config['wavelet']['wavelet_type'] = wavelet_type
            pipeline.config['wavelet']['scales_max'] = scales_max
        
        # Generate data
        status_text.text("Generating/loading data...")
        progress_bar.progress(20)
        
        data = pipeline.generate_data(data_type)
        
        # Display data summary
        st.subheader("📊 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Data Points", len(data))
        
        with col2:
            st.metric("Mean Value", f"{data['value'].mean():.2f}")
        
        with col3:
            st.metric("Std Deviation", f"{data['value'].std():.2f}")
        
        with col4:
            st.metric("Data Range", f"{data['value'].min():.2f} - {data['value'].max():.2f}")
        
        # Display basic time series plot
        st.subheader("📈 Time Series Plot")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['value'],
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=1)
        ))
        
        fig_ts.update_layout(
            title="Time Series Data",
            xaxis_title="Time",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Perform wavelet analysis
        wavelet_results = None
        if perform_wavelet:
            status_text.text("Performing wavelet analysis...")
            progress_bar.progress(40)
            
            wavelet_results = pipeline.perform_wavelet_analysis()
            
            # Display wavelet results
            st.subheader("🌊 Wavelet Analysis Results")
            
            if wavelet_results['cwt']['magnitude'] is not None:
                # Wavelet scalogram
                magnitude = wavelet_results['cwt']['magnitude']
                scales = wavelet_results['cwt']['scales']
                
                fig_scalogram = go.Figure(data=go.Heatmap(
                    z=magnitude,
                    x=list(range(magnitude.shape[1])),
                    y=scales,
                    colorscale='Jet',
                    name='Scalogram'
                ))
                
                fig_scalogram.update_layout(
                    title=f"Wavelet Scalogram ({wavelet_type.upper()} Wavelet)",
                    xaxis_title="Time",
                    yaxis_title="Scale",
                    height=400
                )
                
                st.plotly_chart(fig_scalogram, use_container_width=True)
                
                # Wavelet features
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Max Magnitude", f"{np.max(magnitude):.2f}")
                    st.metric("Mean Magnitude", f"{np.mean(magnitude):.2f}")
                
                with col2:
                    if wavelet_results['cwt']['dominant_frequencies']:
                        st.metric("Dominant Frequencies", len(wavelet_results['cwt']['dominant_frequencies']))
                        st.metric("Max Frequency", f"{max(wavelet_results['cwt']['dominant_frequencies']):.2f}")
        
        # Perform forecasting
        forecast_results = None
        if perform_forecasting:
            status_text.text("Performing forecasting analysis...")
            progress_bar.progress(60)
            
            forecast_results = pipeline.perform_forecasting()
            
            # Display forecasting results
            st.subheader("🔮 Forecasting Results")
            
            # Create forecasting plot
            fig_forecast = go.Figure()
            
            # Add original data
            fig_forecast.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['value'],
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1)
            ))
            
            # Add forecasts
            colors = ['red', 'green', 'orange', 'purple']
            color_idx = 0
            
            for model_name, model_results in forecast_results.items():
                if 'error' not in model_results and 'forecast' in model_results:
                    forecast = model_results['forecast']
                    if isinstance(forecast, (list, np.ndarray)):
                        # Create future timestamps
                        last_timestamp = data['timestamp'].iloc[-1]
                        future_timestamps = pd.date_range(
                            start=last_timestamp + pd.Timedelta(days=1),
                            periods=len(forecast),
                            freq='D'
                        )
                        
                        fig_forecast.add_trace(go.Scatter(
                            x=future_timestamps,
                            y=forecast,
                            mode='lines',
                            name=f'{model_name.title()} Forecast',
                            line=dict(color=colors[color_idx % len(colors)], width=2)
                        ))
                        
                        # Add confidence intervals if available
                        if 'confidence_interval' in model_results:
                            ci = model_results['confidence_interval']
                            fig_forecast.add_trace(go.Scatter(
                                x=future_timestamps,
                                y=ci[:, 1],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
                            fig_forecast.add_trace(go.Scatter(
                                x=future_timestamps,
                                y=ci[:, 0],
                                mode='lines',
                                line=dict(width=0),
                                fill='tonexty',
                                fillcolor=f'rgba({colors[color_idx % len(colors)]}, 0.2)',
                                name=f'{model_name.title()} CI',
                                showlegend=False
                            ))
                        
                        color_idx += 1
            
            fig_forecast.update_layout(
                title="Forecasting Results",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            # Model comparison
            st.subheader("📊 Model Comparison")
            
            comparison_data = []
            for model_name, model_results in forecast_results.items():
                if 'error' not in model_results and 'training_mae' in model_results:
                    comparison_data.append({
                        'Model': model_name.title(),
                        'MAE': model_results['training_mae'],
                        'RMSE': model_results['training_rmse']
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Model comparison chart
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    name='MAE',
                    x=comparison_df['Model'],
                    y=comparison_df['MAE'],
                    marker_color='skyblue'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='RMSE',
                    x=comparison_df['Model'],
                    y=comparison_df['RMSE'],
                    marker_color='lightcoral'
                ))
                
                fig_comparison.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Model",
                    yaxis_title="Error",
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Perform anomaly detection
        anomaly_results = None
        if perform_anomaly_detection:
            status_text.text("Performing anomaly detection...")
            progress_bar.progress(80)
            
            anomaly_results = pipeline.detect_anomalies()
            
            # Display anomaly detection results
            st.subheader("🚨 Anomaly Detection Results")
            
            # Create anomaly plot
            fig_anomaly = go.Figure()
            
            # Add original data
            fig_anomaly.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['value'],
                mode='lines',
                name='Original Data',
                line=dict(color='blue', width=1)
            ))
            
            # Add anomalies
            colors = ['red', 'green', 'orange', 'purple']
            color_idx = 0
            
            for method_name, method_results in anomaly_results.items():
                if 'error' not in method_results and 'anomaly_labels' in method_results:
                    anomaly_labels = method_results['anomaly_labels']
                    if len(anomaly_labels) == len(data):
                        anomaly_indices = np.where(anomaly_labels)[0]
                        if len(anomaly_indices) > 0:
                            fig_anomaly.add_trace(go.Scatter(
                                x=data['timestamp'].iloc[anomaly_indices],
                                y=data['value'].iloc[anomaly_indices],
                                mode='markers',
                                name=f'{method_name.title()} Anomalies',
                                marker=dict(
                                    size=8,
                                    color=colors[color_idx % len(colors)],
                                    symbol='x'
                                )
                            ))
                            color_idx += 1
            
            # Add combined anomalies if available
            if 'combined' in anomaly_results and 'combined_anomalies' in anomaly_results['combined']:
                combined_anomalies = anomaly_results['combined']['combined_anomalies']
                if len(combined_anomalies) == len(data):
                    combined_indices = np.where(combined_anomalies)[0]
                    if len(combined_indices) > 0:
                        fig_anomaly.add_trace(go.Scatter(
                            x=data['timestamp'].iloc[combined_indices],
                            y=data['value'].iloc[combined_indices],
                            mode='markers',
                            name='Combined Anomalies',
                            marker=dict(size=10, color='black', symbol='x')
                        ))
            
            fig_anomaly.update_layout(
                title="Anomaly Detection Results",
                xaxis_title="Time",
                yaxis_title="Value",
                height=500
            )
            
            st.plotly_chart(fig_anomaly, use_container_width=True)
            
            # Anomaly statistics
            st.subheader("📈 Anomaly Statistics")
            
            anomaly_stats = []
            for method_name, method_results in anomaly_results.items():
                if 'error' not in method_results and 'anomaly_labels' in method_results:
                    anomaly_labels = method_results['anomaly_labels']
                    n_anomalies = np.sum(anomaly_labels)
                    anomaly_percentage = n_anomalies / len(data) * 100
                    
                    anomaly_stats.append({
                        'Method': method_name.title(),
                        'Anomalies': n_anomalies,
                        'Percentage': f"{anomaly_percentage:.2f}%"
                    })
            
            if anomaly_stats:
                anomaly_df = pd.DataFrame(anomaly_stats)
                st.dataframe(anomaly_df, use_container_width=True)
        
        # Analysis complete
        status_text.text("Analysis completed successfully!")
        progress_bar.progress(100)
        
        st.success("✅ Analysis completed successfully!")
        
        # Display summary
        st.subheader("📋 Analysis Summary")
        
        summary_data = {
            'Analysis': ['Data Generation', 'Wavelet Analysis', 'Forecasting', 'Anomaly Detection'],
            'Status': [
                '✅ Completed' if data is not None else '❌ Failed',
                '✅ Completed' if wavelet_results is not None else '❌ Skipped',
                '✅ Completed' if forecast_results is not None else '❌ Skipped',
                '✅ Completed' if anomaly_results is not None else '❌ Skipped'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
