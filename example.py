#!/usr/bin/env python3
"""
Example script demonstrating time series analysis with wavelet transforms.

This script shows how to use the time series analysis pipeline for
comprehensive analysis including wavelet transforms, forecasting, and anomaly detection.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src import TimeSeriesAnalysisPipeline


def main():
    """Main function to run the analysis example."""
    print("🚀 Time Series Analysis with Wavelet Transform")
    print("=" * 50)
    
    # Initialize the analysis pipeline
    print("📊 Initializing analysis pipeline...")
    pipeline = TimeSeriesAnalysisPipeline()
    
    # Run complete analysis
    print("🔄 Running complete analysis...")
    results = pipeline.run_full_analysis("synthetic")
    
    # Display summary
    print("\n📋 Analysis Summary:")
    print(f"  Data points: {len(results['data'])}")
    print(f"  Data range: {results['data']['value'].min():.2f} to {results['data']['value'].max():.2f}")
    print(f"  Anomalies detected: {results['data']['anomaly'].sum()}")
    
    # Wavelet analysis results
    if results['wavelet']:
        wavelet_features = results['wavelet']['features']
        print(f"\n🌊 Wavelet Analysis:")
        print(f"  Signal energy: {wavelet_features.get('signal_energy', 'N/A'):.4f}")
        print(f"  Dominant frequencies: {wavelet_features.get('n_dominant_frequencies', 'N/A')}")
    
    # Forecasting results
    if results['forecasting']:
        print(f"\n🔮 Forecasting Results:")
        for model_name, model_results in results['forecasting'].items():
            if 'error' not in model_results and 'training_mae' in model_results:
                print(f"  {model_name.title()} MAE: {model_results['training_mae']:.4f}")
    
    # Anomaly detection results
    if results['anomaly_detection']:
        print(f"\n🚨 Anomaly Detection:")
        for method_name, method_results in results['anomaly_detection'].items():
            if 'error' not in method_results and 'anomaly_labels' in method_results:
                n_anomalies = method_results['anomaly_labels'].sum()
                percentage = n_anomalies / len(results['data']) * 100
                print(f"  {method_name.title()}: {n_anomalies} anomalies ({percentage:.2f}%)")
    
    print("\n✅ Analysis completed successfully!")
    print("\n💡 Next steps:")
    print("  - Run 'streamlit run app.py' for interactive web interface")
    print("  - Open notebooks/ for detailed analysis")
    print("  - Check data/processed/ for saved results")


if __name__ == "__main__":
    main()
