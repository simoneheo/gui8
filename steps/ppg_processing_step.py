import numpy as np
import warnings
import heartpy as hp
from steps.process_registry import register_step
from steps.base_step import BaseStep
from channel import Channel

# Suppress deprecated pkg_resources warning from heartpy
warnings.filterwarnings("ignore", category=UserWarning, module="heartpy")

@register_step
class ppg_processing_step(BaseStep):
    name = "ppg_processing"
    category = "heartpy"
    description = "Process PPG signal to extract heart rate and HRV features using HeartPy."
    tags = ["biosignal", "ppg", "heartpy", "photoplethysmography", "heart-rate", "hrv", "variability", 
            "peaks", "cardiac", "rhythm", "frequency", "time-domain", "preprocessing", "filtering","time-series"]
    
    params = [
        {
            "name": "sample_rate",
            "type": "float",
            "default": "",
            "description": "Sampling rate of the PPG signal in Hz (auto-detected if not specified)",
            "help": "HeartPy requires accurate sampling rate for proper processing"
        },
        {
            "name": "windowsize",
            "type": "float", 
            "default": 0.75,
            "description": "Window size for peak detection (in seconds)",
            "help": "Larger windows = more robust detection, smaller windows = better temporal resolution"
        },
        {
            "name": "bpmmin",
            "type": "float",
            "default": "40.0",
            "description": "Minimum heart rate in BPM",
            "help": "Lower bound for heart rate detection (default: 40 BPM)"
        },
        {
            "name": "bpmmax",
            "type": "float",
            "default": "200.0",
            "description": "Maximum heart rate in BPM",
            "help": "Upper bound for heart rate detection (default: 200 BPM)"
        },
        {
            "name": "report_time",
            "type": "bool",
            "default": True,
            "description": "Whether to calculate time-domain HRV measures",
            "help": "Includes RMSSD, pNN50, SDNN, and other time-domain metrics"
        },
        {
            "name": "report_freq",
            "type": "bool", 
            "default": True,
            "description": "Whether to calculate frequency-domain HRV measures",
            "help": "Includes LF, HF, LF/HF ratio, and spectral power analysis"
        },
        {
            "name": "freq_method",
            "type": "select",
            "options": ["fft", "welch", "lomb"],
            "default": "fft",
            "description": "Method for frequency domain analysis",
            "help": "FFT is fastest, Welch is most robust, Lomb handles irregularly sampled data"
        },
        {
            "name": "interp_clipping",
            "type": "bool",
            "default": True,
            "description": "Whether to apply interpolation clipping to remove artifacts",
            "help": "Removes physiologically impossible heart rate values"
        },
        {
            "name": "clipping_scale",
            "type": "float",
            "default": 1.3,
            "description": "Clipping scale factor for artifact removal",
            "help": "Higher values = more permissive clipping, lower values = more aggressive"
        },
        {
            "name": "interp_threshold",
            "type": "float",
            "default": 1020,
            "description": "Threshold for interpolation (in ms)",
            "help": "RR intervals above this value will be interpolated"
        },
        {
            "name": "hampel_correct",
            "type": "bool",
            "default": True,
            "description": "Whether to apply Hampel correction for outlier removal",
            "help": "Removes statistical outliers from RR interval series"
        },
        {
            "name": "hampel_threshold",
            "type": "float",
            "default": 3.0,
            "description": "Hampel filter threshold (standard deviations)",
            "help": "Lower values = more aggressive outlier removal"
        }
    ]

    @classmethod
    def get_info(cls):
        return f"{cls.name} — {cls.description} (Category: {cls.category})"

    @classmethod
    def get_prompt(cls):
        return {"info": cls.description, "params": cls.params}

    @classmethod
    def _get_channel_fs(cls, channel: Channel) -> float:
        """Get sampling frequency from channel"""
        if hasattr(channel, 'fs_median') and channel.fs_median:
            return float(channel.fs_median)
        elif hasattr(channel, 'fs') and channel.fs:
            return float(channel.fs)
        else:
            return 100.0  # Default fallback

    @classmethod
    def _validate_input_data(cls, y: np.ndarray) -> None:
        """Validate input data is suitable for PPG processing"""
        if len(y) == 0:
            raise ValueError("Input PPG signal is empty")
        
        # Check for minimum signal length
        if len(y) < 1000:
            raise ValueError("PPG signal too short for reliable processing (minimum 1000 samples)")
        
        # Check for non-finite values
        if not np.isfinite(y).all():
            raise ValueError("PPG signal contains NaN or infinite values")
        
        # Check for constant signal
        if np.std(y) == 0:
            raise ValueError("PPG signal is constant (no variability)")
        
        # Check for reasonable signal range
        if np.max(y) - np.min(y) < 1e-6:
            raise ValueError("PPG signal has very low amplitude (potential scaling issue)")

    @classmethod
    def _validate_parameters(cls, params: dict) -> None:
        """Validate PPG processing parameters"""
        sample_rate = params.get('sample_rate')
        if sample_rate is not None:
            if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
                raise ValueError("Sample rate must be a positive number")
            if sample_rate < 10 or sample_rate > 10000:
                raise ValueError("Sample rate must be between 10 and 10000 Hz")
        
        windowsize = params.get('windowsize', 0.75)
        if not isinstance(windowsize, (int, float)) or windowsize <= 0:
            raise ValueError("Window size must be a positive number")
        if windowsize < 0.1 or windowsize > 10.0:
            raise ValueError("Window size must be between 0.1 and 10.0 seconds")
        
        bpmmin = params.get('bpmmin', 40.0)
        if not isinstance(bpmmin, (int, float)) or bpmmin <= 0:
            raise ValueError("Minimum BPM must be a positive number")
        if bpmmin < 20 or bpmmin > 100:
            raise ValueError("Minimum BPM must be between 20 and 100")
        
        bpmmax = params.get('bpmmax', 200.0)
        if not isinstance(bpmmax, (int, float)) or bpmmax <= 0:
            raise ValueError("Maximum BPM must be a positive number")
        if bpmmax < 100 or bpmmax > 300:
            raise ValueError("Maximum BPM must be between 100 and 300")
        
        if bpmmin >= bpmmax:
            raise ValueError("Minimum BPM must be less than maximum BPM")

    @classmethod
    def _validate_output_data(cls, y_original: np.ndarray, hr_times: np.ndarray, hr_values: np.ndarray) -> None:
        """Validate output heart rate data"""
        if len(hr_times) == 0 or len(hr_values) == 0:
            raise ValueError("PPG processing failed to extract heart rate data")
        
        if len(hr_times) != len(hr_values):
            raise ValueError("Heart rate times and values have different lengths")
        
        if not np.isfinite(hr_times).all() or not np.isfinite(hr_values).all():
            raise ValueError("Output heart rate data contains NaN or infinite values")
        
        if np.any(hr_values <= 0) or np.any(hr_values > 300):
            raise ValueError("Heart rate values out of physiological range (0-300 bpm)")

    @classmethod
    def parse_input(cls, user_input: dict) -> dict:
        """Parse and validate user input parameters"""
        parsed = {}
        # Parse the parameters that are actually used by HeartPy
        for param in cls.params:
            name = param["name"]
            if name == "sample_rate":
                continue  # Skip sample_rate as it's injected from channel
            elif name in ["windowsize", "bpmmin", "bpmmax"]:
                val = user_input.get(name, param.get("default"))
                try:
                    if val == "":
                        parsed[name] = float(param.get("default", 0.75))
                    else:
                        parsed[name] = float(val)
                except ValueError as e:
                    if "could not convert" in str(e) or "invalid literal" in str(e):
                        raise ValueError(f"{name} must be a valid float")
                    raise e
            # Skip other parameters for now as they're not supported by current HeartPy version
        return parsed

    @classmethod
    def apply(cls, channel: Channel, params: dict) -> Channel:
        """Apply PPG processing pipeline to the channel data"""
        try:
            x = channel.xdata
            y = channel.ydata
            
            # Get sampling frequency from channel
            fs = cls._get_channel_fs(channel)
            if fs is None:
                fs = 100.0  # Default PPG sampling rate
            
            # Inject sampling frequency into params
            params["sample_rate"] = fs
            
            # Validate input data and parameters
            cls._validate_input_data(y)
            cls._validate_parameters(params)
            
            # Process the data
            x_new, y_new, metadata = cls.script(x, y, fs, params)
            
            # Validate output data
            cls._validate_output_data(y, x_new, y_new)
            
            # Create new channel with results
            new_channel = cls.create_new_channel(
                parent=channel,
                xdata=x_new,
                ydata=y_new,
                params=params,
                suffix="PPG_HR"
            )
            
            # Add metadata
            new_channel.metadata.update(metadata)
            
            return new_channel
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            else:
                raise ValueError(f"PPG processing failed: {str(e)}")

    @classmethod
    def script(cls, x: np.ndarray, y: np.ndarray, fs: float, params: dict) -> tuple:
        """Core PPG processing logic"""
        
        # Parse parameters with defaults
        # Note: We'll use only the core parameters that are definitely supported by HeartPy
        parsed_params = {
            'sample_rate': fs,
            'windowsize': float(params.get('windowsize', 0.75)),
            'bpmmin': float(params.get('bpmmin', 40.0)),
            'bpmmax': float(params.get('bpmmax', 200.0))
        }
        
        try:
            # Preprocess the signal to improve HeartPy's ability to detect heart rate
            y_processed = cls._preprocess_ppg_signal(y, fs)
            
            # Run HeartPy processing with parsed parameters
            # Note: Some parameters might not be supported in all HeartPy versions
            # We'll use only the core parameters that are definitely supported
            wd, m = hp.process(
                y_processed,
                sample_rate=parsed_params['sample_rate'],
                windowsize=parsed_params['windowsize'],
                bpmmin=parsed_params['bpmmin'],
                bpmmax=parsed_params['bpmmax']
            )
            
            # Extract heart rate time series
            if 'hrv_time' in wd and 'time' in wd['hrv_time'] and 'hr' in wd['hrv_time']:
                new_x = np.array(wd['hrv_time']['time'])
                new_y = np.array(wd['hrv_time']['hr'])
            else:
                raise ValueError("HeartPy processing failed to generate heart rate time series")
            
            # Store metadata
            metadata = {
                'heartpy_summary': m,
                'heartpy_working_data': wd,
                'processing_params': parsed_params,
                'preprocessing_applied': True
            }
            
            return new_x, new_y, metadata
            
        except Exception as e:
            # Try with original signal if preprocessing fails
            try:
                wd, m = hp.process(
                    y,
                    sample_rate=parsed_params['sample_rate'],
                    windowsize=parsed_params['windowsize'],
                    bpmmin=parsed_params['bpmmin'],
                    bpmmax=parsed_params['bpmmax']
                )
                
                if 'hrv_time' in wd and 'time' in wd['hrv_time'] and 'hr' in wd['hrv_time']:
                    new_x = np.array(wd['hrv_time']['time'])
                    new_y = np.array(wd['hrv_time']['hr'])
                else:
                    raise ValueError("HeartPy processing failed to generate heart rate time series")
                
                metadata = {
                    'heartpy_summary': m,
                    'heartpy_working_data': wd,
                    'processing_params': parsed_params,
                    'preprocessing_applied': False
                }
                
                return new_x, new_y, metadata
                
            except Exception as e2:
                raise ValueError(f"PPG processing failed with both preprocessed and original signals: {e2}")

    @classmethod
    def _preprocess_ppg_signal(cls, y: np.ndarray, fs: float) -> np.ndarray:
        """Preprocess PPG signal to improve HeartPy detection"""
        from scipy.signal import butter, filtfilt
        
        # 1. Remove DC component (baseline)
        y_detrended = y - np.mean(y)
        
        # 2. Apply bandpass filter to focus on heart rate frequencies (0.5-5 Hz)
        # This corresponds to 30-300 BPM
        lowcut = 0.5  # Hz
        highcut = 5.0  # Hz
        
        # Design Butterworth filter
        nyquist = fs / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure frequencies are within valid range
        low = max(0.001, min(low, 0.99))
        high = max(0.001, min(high, 0.99))
        
        if low < high:
            b, a = butter(4, [low, high], btype='band')
            y_filtered = filtfilt(b, a, y_detrended)
        else:
            # If filter design fails, use detrended signal
            y_filtered = y_detrended
        
        # 3. Normalize the signal
        if np.std(y_filtered) > 0:
            y_normalized = (y_filtered - np.mean(y_filtered)) / np.std(y_filtered)
        else:
            y_normalized = y_filtered
        
        return y_normalized
