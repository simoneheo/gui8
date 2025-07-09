
import numpy as np
from arch import arch_model
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

@register_step
class garch_forecast_step(BaseStep):
    name = "garch_forecast"
    category = "Forecasting"
    description = """Fit a GARCH(1,1) model to the signal and forecast volatility.

This step fits a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model
to estimate and forecast volatility in time series data.

Useful for:
• Financial time series volatility modeling
• Risk assessment and forecasting
• Time-varying volatility analysis
• Quantitative finance applications"""
    tags = ["finance", "garch", "forecast", "volatility", "time-series"]
    params = [
        {
            "name": "horizon",
            "type": "int",
            "default": 10,
            "help": "Forecast horizon (steps ahead, must be positive)"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — GARCH(1,1) volatility forecast (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input signal data"""
        if len(y) < 50:
            raise ValueError("Signal too short for GARCH modeling (minimum 50 samples recommended)")
        if np.all(np.isnan(y)):
            raise ValueError("Signal contains only NaN values")
        if np.all(y == y[0]):
            raise ValueError("Signal has no variation (constant values)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate parameters and business rules"""
        horizon = params.get("horizon")
        
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if horizon > 100:
            raise ValueError("horizon too large (maximum 100 steps)")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, forecast: np.ndarray) -> None:
        """Validate output data"""
        if forecast.size == 0:
            raise ValueError("GARCH forecast produced empty results")
        if np.any(forecast < 0):
            raise ValueError("GARCH forecast contains negative variance values")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        for param in cls.params:
            name = param["name"]
            val = user_input.get(name, param.get("default"))
            try:
                if val == "":
                    parsed[name] = None
                elif param["type"] == "float":
                    parsed[name] = float(val)
                elif param["type"] == "int":
                    parsed[name] = int(val)
                else:
                    parsed[name] = val
            except ValueError as e:
                if "could not convert" in str(e) or "invalid literal" in str(e):
                    raise ValueError(f"{name} must be a valid {param['type']}")
                raise e
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> list:
        """Apply GARCH forecasting to the channel data."""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            forecast, x_new = cls.script(y, params)
            
            # Validate output data
            cls._validate_output_data(y, forecast)
            
            # Create forecast channel
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=forecast,
                params=params,
                suffix="GARCHForecast"
            )
            
            # Set channel properties
            new_channel.tags = ["finance", "garch", "forecast", "volatility"]
            new_channel.legend_label = f"{channel.legend_label} (GARCH Forecast)"
            
            return [new_channel]
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"GARCH forecasting failed: {str(e)}")

    @classmethod
    def script(cls, y: np.ndarray, params: dict) -> tuple:
        """Core processing logic for GARCH forecasting"""
        horizon = params.get("horizon", 10)
        
        # Ensure y is numpy array
        y = np.asarray(y)
        
        # Remove any NaN values for GARCH fitting
        y_clean = y[~np.isnan(y)]
        if len(y_clean) < 50:
            raise ValueError("Insufficient non-NaN data for GARCH modeling")
        
        # Fit GARCH(1,1) model
        model = arch_model(y_clean, vol="Garch", p=1, q=1, rescale=False)
        res = model.fit(disp="off")
        
        # Generate forecast
        forecast = res.forecast(horizon=horizon)
        fcast = forecast.variance.values[-1, :]
        
        # Create new x-axis for forecast period
        x_new = np.arange(len(y), len(y) + horizon)
        
        return fcast, x_new
