# Time Series Analysis with Wavelet Transform

A comprehensive time series analysis project featuring wavelet transforms, forecasting, anomaly detection, and interactive visualization. This project demonstrates state-of-the-art methods for analyzing non-stationary time series data with time-localized frequency analysis.

## Features

### Core Analysis
- **Wavelet Transform Analysis**: Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT) with multiple wavelet types
- **Advanced Forecasting**: ARIMA, Prophet, and LSTM models with automatic model selection
- **Anomaly Detection**: Statistical methods, Isolation Forest, and Autoencoder-based detection
- **Interactive Visualization**: Comprehensive dashboards with Plotly and Matplotlib

### Data Sources
- **Synthetic Data**: Realistic time series with trends, seasonality, noise, and anomalies
- **Real Data**: Energy consumption, stock prices, and weather data
- **Customizable**: Configurable parameters for data generation and analysis

### User Interface
- **Streamlit Web App**: Interactive web interface for exploring results
- **Jupyter Notebooks**: For detailed analysis and experimentation
- **Command Line Interface**: For batch processing and automation

## Installation

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kryptologyst/Time-Series-Analysis-with-Wavelet-Transform.git
cd Time-Series-Analysis-with-Wavelet-Transform
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, install additional packages:

```bash
# For GPU acceleration (PyTorch)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Prophet (if not already installed)
pip install prophet

# For additional visualization
pip install plotly-dash
```

## Quick Start

### 1. Command Line Usage

Run the complete analysis pipeline:

```bash
python -m src
```

### 2. Streamlit Web Interface

Launch the interactive web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### 3. Jupyter Notebook

For detailed analysis and experimentation:

```bash
jupyter notebook notebooks/
```

## Usage Examples

### Basic Analysis

```python
from src import TimeSeriesAnalysisPipeline

# Initialize pipeline
pipeline = TimeSeriesAnalysisPipeline()

# Run complete analysis
results = pipeline.run_full_analysis("synthetic")

# Access results
data = results['data']
wavelet_results = results['wavelet']
forecast_results = results['forecasting']
anomaly_results = results['anomaly_detection']
```

### Custom Configuration

```python
import yaml

# Load custom configuration
with open('config/custom_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize with custom config
pipeline = TimeSeriesAnalysisPipeline('config/custom_config.yaml')
```

### Individual Components

```python
from src.data_generator import SyntheticDataGenerator
from src.wavelet_analysis import WaveletAnalyzer
from src.forecasting import ForecastingPipeline
from src.anomaly_detection import AnomalyDetector

# Generate synthetic data
generator = SyntheticDataGenerator(config)
data = generator.generate_synthetic_data()

# Perform wavelet analysis
analyzer = WaveletAnalyzer(config)
wavelet_results = analyzer.analyze(data)

# Generate forecasts
forecaster = ForecastingPipeline(config)
forecast_results = forecaster.forecast(data)

# Detect anomalies
detector = AnomalyDetector(config)
anomaly_results = detector.detect(data)
```

## Project Structure

```
time-series-analysis/
├── src/                          # Source code
│   ├── __init__.py              # Main pipeline
│   ├── data_generator.py        # Data generation and loading
│   ├── wavelet_analysis.py     # Wavelet transform analysis
│   ├── forecasting.py          # Forecasting models
│   ├── anomaly_detection.py    # Anomaly detection methods
│   └── visualization.py        # Visualization utilities
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration
├── data/                        # Data storage
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── models/                      # Model storage
│   └── checkpoints/            # Model checkpoints
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Unit tests
├── logs/                        # Log files
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Key configuration options:

### Data Generation
```yaml
data:
  synthetic:
    n_samples: 1000
    noise_level: 0.1
    trend_strength: 0.5
    seasonality_strength: 0.3
    anomaly_probability: 0.05
```

### Wavelet Analysis
```yaml
wavelet:
  wavelet_type: "morl"
  scales_min: 1
  scales_max: 100
  sampling_period: 0.01
```

### Forecasting Models
```yaml
models:
  arima:
    auto_arima: true
    seasonal: true
    max_p: 5
    max_q: 5
  
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
  
  lstm:
    sequence_length: 60
    hidden_units: 50
    epochs: 100
```

## Advanced Features

### Wavelet Analysis

The project supports multiple wavelet types and analysis methods:

- **Continuous Wavelet Transform (CWT)**: Time-frequency analysis
- **Discrete Wavelet Transform (DWT)**: Multi-resolution analysis
- **Wavelet Types**: Morlet, Mexican Hat, Gaussian, Complex Morlet
- **Feature Extraction**: Energy distribution, dominant frequencies

### Forecasting Models

Multiple state-of-the-art forecasting methods:

- **ARIMA**: Auto-ARIMA with automatic parameter selection
- **Prophet**: Facebook's forecasting tool with seasonality handling
- **LSTM**: Deep learning approach with PyTorch
- **Baseline Methods**: Moving average, linear trend, naive forecasts

### Anomaly Detection

Comprehensive anomaly detection using multiple approaches:

- **Statistical Methods**: Z-score, Modified Z-score, IQR, Rolling statistics
- **Machine Learning**: Isolation Forest
- **Deep Learning**: Autoencoder-based detection
- **Wavelet-based**: Frequency domain anomaly detection

## Performance and Optimization

### Memory Optimization
- Efficient data structures and lazy loading
- Configurable batch sizes for large datasets
- Memory-mapped file support for large time series

### Computational Optimization
- Parallel processing for multiple models
- GPU acceleration support (PyTorch)
- Caching of intermediate results

### Scalability
- Modular architecture for easy extension
- Plugin system for custom models
- Distributed processing support

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_wavelet_analysis.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linting
black src/
flake8 src/
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{time_series_analysis,
  title={Time Series Analysis with Wavelet Transform},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Time-Series-Analysis-with-Wavelet-Transform}
}
```

## Acknowledgments

- PyWavelets for wavelet transform implementation
- Prophet team for the forecasting library
- PyTorch team for deep learning capabilities
- Streamlit team for the web interface framework

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
2. **Memory Issues**: Reduce batch sizes or use data chunking
3. **GPU Issues**: Install appropriate PyTorch version for your system
4. **Configuration Errors**: Validate YAML syntax in config files

### Getting Help

- Check the [Issues](https://github.com/kryptologyst/Time-Series-Analysis-with-Wavelet-Transform/issues) page
- Review the documentation in the `docs/` directory
- Join our community discussions

## Roadmap

- [ ] Add more forecasting models (Transformer, GRU)
- [ ] Implement real-time streaming analysis
- [ ] Add more visualization options
- [ ] Support for multivariate time series
- [ ] Integration with cloud platforms
- [ ] Mobile application interface
# Time-Series-Analysis-with-Wavelet-Transform
