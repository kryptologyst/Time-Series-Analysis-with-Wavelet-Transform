"""
Forecasting module for time series analysis.

This module provides comprehensive forecasting capabilities using multiple
state-of-the-art methods including ARIMA, Prophet, LSTM, and other models.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Forecasting libraries
try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    logger.warning("pmdarima not available, ARIMA forecasting disabled")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("prophet not available, Prophet forecasting disabled")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not available, LSTM forecasting disabled")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


class ForecastingPipeline:
    """
    Comprehensive forecasting pipeline using multiple models.
    
    This class provides forecasting capabilities using ARIMA, Prophet, LSTM,
    and other state-of-the-art methods with automatic model selection and evaluation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config: Configuration dictionary containing forecasting parameters
        """
        self.config = config
        self.models_config = config.get('models', {})
        
        # Model configurations
        self.arima_config = self.models_config.get('arima', {})
        self.prophet_config = self.models_config.get('prophet', {})
        self.lstm_config = self.models_config.get('lstm', {})
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        
        logger.info("ForecastingPipeline initialized")
    
    def forecast(self, data: pd.DataFrame, forecast_horizon: int = 30) -> Dict:
        """
        Perform forecasting using multiple models.
        
        Args:
            data: DataFrame with time series data
            forecast_horizon: Number of steps to forecast ahead
            
        Returns:
            Dictionary containing forecasting results from all models
        """
        logger.info(f"Starting forecasting analysis with horizon: {forecast_horizon}")
        
        # Prepare data
        if 'value' in data.columns:
            ts_data = data['value'].dropna()
        else:
            ts_data = data.iloc[:, 1].dropna()
        
        # Ensure we have enough data
        if len(ts_data) < 50:
            logger.warning("Insufficient data for forecasting, using synthetic extension")
            ts_data = self._extend_data(ts_data, 100)
        
        results = {}
        
        # ARIMA forecasting
        if PMDARIMA_AVAILABLE:
            try:
                arima_results = self._forecast_arima(ts_data, forecast_horizon)
                results['arima'] = arima_results
                logger.info("ARIMA forecasting completed")
            except Exception as e:
                logger.error(f"ARIMA forecasting failed: {e}")
                results['arima'] = {'error': str(e)}
        
        # Prophet forecasting
        if PROPHET_AVAILABLE:
            try:
                prophet_results = self._forecast_prophet(ts_data, forecast_horizon)
                results['prophet'] = prophet_results
                logger.info("Prophet forecasting completed")
            except Exception as e:
                logger.error(f"Prophet forecasting failed: {e}")
                results['prophet'] = {'error': str(e)}
        
        # LSTM forecasting
        if TORCH_AVAILABLE:
            try:
                lstm_results = self._forecast_lstm(ts_data, forecast_horizon)
                results['lstm'] = lstm_results
                logger.info("LSTM forecasting completed")
            except Exception as e:
                logger.error(f"LSTM forecasting failed: {e}")
                results['lstm'] = {'error': str(e)}
        
        # Simple baseline methods
        baseline_results = self._forecast_baseline(ts_data, forecast_horizon)
        results['baseline'] = baseline_results
        
        # Model comparison
        results['comparison'] = self._compare_models(results, ts_data)
        
        logger.info("Forecasting analysis completed")
        return results
    
    def _forecast_arima(self, ts_data: pd.Series, horizon: int) -> Dict:
        """
        Perform ARIMA forecasting using auto_arima.
        
        Args:
            ts_data: Time series data
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing ARIMA results
        """
        logger.info("Training ARIMA model")
        
        # Auto ARIMA parameters
        auto_arima_params = {
            'seasonal': self.arima_config.get('seasonal', True),
            'max_p': self.arima_config.get('max_p', 5),
            'max_q': self.arima_config.get('max_q', 5),
            'max_P': self.arima_config.get('max_P', 2),
            'max_Q': self.arima_config.get('max_Q', 2),
            'suppress_warnings': True,
            'stepwise': True
        }
        
        # Fit model
        model = auto_arima(ts_data, **auto_arima_params)
        
        # Generate forecasts
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
        
        # Calculate metrics on training data
        train_pred = model.predict_in_sample()
        train_mae = mean_absolute_error(ts_data, train_pred)
        train_rmse = np.sqrt(mean_squared_error(ts_data, train_pred))
        
        return {
            'model': model,
            'forecast': forecast,
            'confidence_interval': conf_int,
            'training_mae': train_mae,
            'training_rmse': train_rmse,
            'aic': model.aic(),
            'bic': model.bic(),
            'order': model.order,
            'seasonal_order': model.seasonal_order
        }
    
    def _forecast_prophet(self, ts_data: pd.Series, horizon: int) -> Dict:
        """
        Perform Prophet forecasting.
        
        Args:
            ts_data: Time series data
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing Prophet results
        """
        logger.info("Training Prophet model")
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D'),
            'y': ts_data.values
        })
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=self.prophet_config.get('yearly_seasonality', True),
            weekly_seasonality=self.prophet_config.get('weekly_seasonality', True),
            daily_seasonality=self.prophet_config.get('daily_seasonality', False)
        )
        
        # Fit model
        model.fit(df_prophet)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast['yhat'].iloc[-horizon:].values
        forecast_lower = forecast['yhat_lower'].iloc[-horizon:].values
        forecast_upper = forecast['yhat_upper'].iloc[-horizon:].values
        
        # Calculate training metrics
        train_pred = forecast['yhat'].iloc[:-horizon].values
        train_mae = mean_absolute_error(ts_data.values, train_pred)
        train_rmse = np.sqrt(mean_squared_error(ts_data.values, train_pred))
        
        return {
            'model': model,
            'forecast': forecast_values,
            'forecast_lower': forecast_lower,
            'forecast_upper': forecast_upper,
            'training_mae': train_mae,
            'training_rmse': train_rmse,
            'components': forecast[['trend', 'seasonal', 'yearly', 'weekly']].iloc[-horizon:]
        }
    
    def _forecast_lstm(self, ts_data: pd.Series, horizon: int) -> Dict:
        """
        Perform LSTM forecasting.
        
        Args:
            ts_data: Time series data
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing LSTM results
        """
        logger.info("Training LSTM model")
        
        # Prepare data
        sequence_length = self.lstm_config.get('sequence_length', 60)
        data_scaled = self.scaler.fit_transform(ts_data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(data_scaled, sequence_length)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Initialize model
        model = LSTMModel(
            input_size=1,
            hidden_size=self.lstm_config.get('hidden_units', 50),
            num_layers=2,
            dropout=self.lstm_config.get('dropout', 0.2)
        )
        
        # Training parameters
        epochs = self.lstm_config.get('epochs', 100)
        batch_size = self.lstm_config.get('batch_size', 32)
        
        # Train model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Generate forecasts
        model.eval()
        with torch.no_grad():
            # Use last sequence_length points to predict future
            last_sequence = X_test[-1:].unsqueeze(0)
            forecasts = []
            
            for _ in range(horizon):
                pred = model(last_sequence)
                forecasts.append(pred.item())
                
                # Update sequence
                last_sequence = torch.cat([last_sequence[:, 1:, :], pred.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Inverse transform forecasts
        forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        # Calculate training metrics
        train_pred = model(X_train).detach().numpy()
        train_pred = self.scaler.inverse_transform(train_pred).flatten()
        train_actual = self.scaler.inverse_transform(y_train.numpy().reshape(-1, 1)).flatten()
        
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        
        return {
            'model': model,
            'forecast': forecasts,
            'training_mae': train_mae,
            'training_rmse': train_rmse,
            'sequence_length': sequence_length
        }
    
    def _forecast_baseline(self, ts_data: pd.Series, horizon: int) -> Dict:
        """
        Perform baseline forecasting using simple methods.
        
        Args:
            ts_data: Time series data
            horizon: Forecast horizon
            
        Returns:
            Dictionary containing baseline results
        """
        logger.info("Generating baseline forecasts")
        
        # Simple moving average
        window = min(30, len(ts_data) // 4)
        ma_forecast = [ts_data.rolling(window=window).mean().iloc[-1]] * horizon
        
        # Linear trend
        x = np.arange(len(ts_data))
        coeffs = np.polyfit(x, ts_data.values, 1)
        trend_forecast = [coeffs[0] * (len(ts_data) + i) + coeffs[1] for i in range(1, horizon + 1)]
        
        # Naive forecast (last value)
        naive_forecast = [ts_data.iloc[-1]] * horizon
        
        # Seasonal naive (if we have enough data)
        seasonal_period = 7  # Weekly seasonality
        if len(ts_data) > seasonal_period:
            seasonal_forecast = []
            for i in range(horizon):
                seasonal_forecast.append(ts_data.iloc[-(seasonal_period - i % seasonal_period)])
        else:
            seasonal_forecast = naive_forecast
        
        return {
            'moving_average': ma_forecast,
            'linear_trend': trend_forecast,
            'naive': naive_forecast,
            'seasonal_naive': seasonal_forecast
        }
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Input data
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _extend_data(self, ts_data: pd.Series, target_length: int) -> pd.Series:
        """
        Extend short time series with synthetic data.
        
        Args:
            ts_data: Original time series
            target_length: Target length
            
        Returns:
            Extended time series
        """
        if len(ts_data) >= target_length:
            return ts_data
        
        # Generate synthetic continuation
        last_value = ts_data.iloc[-1]
        trend = np.mean(np.diff(ts_data.values))
        noise_std = np.std(ts_data.values)
        
        synthetic_values = []
        for i in range(target_length - len(ts_data)):
            new_value = last_value + trend + np.random.normal(0, noise_std * 0.1)
            synthetic_values.append(new_value)
            last_value = new_value
        
        # Combine original and synthetic data
        extended_data = pd.concat([ts_data, pd.Series(synthetic_values)])
        return extended_data
    
    def _compare_models(self, results: Dict, ts_data: pd.Series) -> Dict:
        """
        Compare forecasting models and rank them.
        
        Args:
            results: Results from all forecasting models
            ts_data: Original time series data
            
        Returns:
            Dictionary containing model comparison results
        """
        comparison = {}
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                continue
            
            if 'training_mae' in model_results:
                comparison[model_name] = {
                    'mae': model_results['training_mae'],
                    'rmse': model_results['training_rmse']
                }
        
        # Rank models by MAE
        if comparison:
            ranked_models = sorted(comparison.items(), key=lambda x: x[1]['mae'])
            comparison['ranking'] = [model[0] for model in ranked_models]
        
        return comparison


class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 50, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out
