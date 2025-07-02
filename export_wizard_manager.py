from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox
from export_wizard_window import ExportWizardWindow
import numpy as np
import pandas as pd
import csv
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

class ExportWizardManager(QObject):
    """
    Manager for the export wizard that handles:
    - Channel data export to various formats
    - File format conversion
    - Export progress and error handling
    - State management and statistics tracking
    """
    
    export_complete = Signal(str)  # Emits the exported file path
    export_started = Signal(str)   # Emits when export begins
    export_progress = Signal(int)  # Progress percentage (0-100)
    state_changed = Signal(str)    # General state changes
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers with validation
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state tracking
        self._stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'total_channels_exported': 0,
            'last_export_path': None,
            'last_export_time': None,
            'session_start': time.time()
        }
        
        # Export history tracking
        self._export_history = []  # List of export records
        
        # Validate initialization
        if not self._validate_managers():
            raise ValueError("Required managers not available for ExportWizardManager")
        
        # Create window after validation
        self.window = ExportWizardWindow(
            file_manager=self.file_manager,
            channel_manager=self.channel_manager,
            signal_bus=self.signal_bus,
            parent=self.parent_window
        )
        
        # Set bidirectional reference
        self.window.export_manager = self
        
        # Connect signals
        self._connect_signals()
        
        # Log initialization
        self._log_state_change("Export manager initialized successfully")
        
    def _validate_managers(self) -> bool:
        """Validate that required managers are available and functional"""
        if not self.file_manager:
            print("[ExportWizardManager] ERROR: File manager not provided")
            return False
            
        if not self.channel_manager:
            print("[ExportWizardManager] ERROR: Channel manager not provided")
            return False
            
        # Validate manager functionality
        try:
            self.file_manager.get_file_count()
            self.channel_manager.get_channel_count()
            return True
        except Exception as e:
            print(f"[ExportWizardManager] ERROR: Manager validation failed: {e}")
            return False
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[ExportWizardManager {timestamp}] {message}")
        self.state_changed.emit(message)
        
    def get_stats(self) -> Dict:
        """Get comprehensive export statistics"""
        return {
            **self._stats,
            'export_history_count': len(self._export_history),
            'success_rate': (
                self._stats['successful_exports'] / max(1, self._stats['total_exports']) * 100
            ),
            'session_duration': time.time() - self._stats['session_start']
        }
        
    def _connect_signals(self):
        """Connect window signals to manager methods"""
        self.window.export_complete.connect(self._on_export_requested)
        
    def show(self):
        """Show the export wizard window"""
        self.window.show()
        
    def close(self):
        """Close the export wizard"""
        if self.window:
            self.window.close()
            
    def _on_export_requested(self, export_info):
        """Handle export request from the window"""
        try:
            channels = export_info['channels']
            export_format = export_info['format']
            file_path = export_info['file_path']
            
            # Update statistics
            self._stats['total_exports'] += 1
            self._log_state_change(f"Starting export of {len(channels)} channels to {file_path}")
            
            # Emit start signal
            self.export_started.emit(file_path)
            
            # Perform the export
            success = self._export_channels(channels, file_path, export_format)
            
            if success:
                # Update success statistics
                self._stats['successful_exports'] += 1
                self._stats['total_channels_exported'] += len(channels)
                self._stats['last_export_path'] = file_path
                self._stats['last_export_time'] = time.time()
                
                # Record export in history
                export_record = {
                    'timestamp': time.time(),
                    'file_path': file_path,
                    'format': export_format,
                    'channel_count': len(channels),
                    'success': True
                }
                self._export_history.append(export_record)
                
                # Show success message
                QMessageBox.information(
                    self.window,
                    "Export Complete",
                    f"Successfully exported {len(channels)} channel(s) to:\n{file_path}"
                )
                
                # Emit completion signal
                self.export_complete.emit(file_path)
                self._log_state_change(f"Export completed successfully: {file_path}")
                
                # Close the wizard window
                self.window.close()
            else:
                # Update failure statistics
                self._stats['failed_exports'] += 1
                
                # Record failed export
                export_record = {
                    'timestamp': time.time(),
                    'file_path': file_path,
                    'format': export_format,
                    'channel_count': len(channels),
                    'success': False,
                    'error': 'Export operation failed'
                }
                self._export_history.append(export_record)
                
                self._log_state_change(f"Export failed: {file_path}")
                QMessageBox.critical(
                    self.window,
                    "Export Failed",
                    "Failed to export data. Please check the file path and try again."
                )
                
        except Exception as e:
            # Update failure statistics
            self._stats['failed_exports'] += 1
            
            # Record error in history
            export_record = {
                'timestamp': time.time(),
                'file_path': export_info.get('file_path', 'unknown'),
                'format': export_info.get('format', 'unknown'),
                'channel_count': len(export_info.get('channels', [])),
                'success': False,
                'error': str(e)
            }
            self._export_history.append(export_record)
            
            self._log_state_change(f"Export error: {str(e)}")
            QMessageBox.critical(
                self.window,
                "Export Error",
                f"An error occurred during export:\n\n{str(e)}"
            )
            
    def _export_channels(self, channels, file_path, export_format):
        """Export channels to the specified file format"""
        try:
            if export_format.lower() == 'csv':
                return self._export_to_csv(channels, file_path)
            elif export_format.lower() == 'txt':
                return self._export_to_txt(channels, file_path)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
                
        except Exception as e:
            print(f"[ExportWizard] Error in _export_channels: {str(e)}")
            raise
            
    def _export_to_csv(self, channels, file_path):
        """Export channels to CSV format"""
        try:
            # Prepare data for export
            export_data = self._prepare_export_data(channels)
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Write to CSV
            df.to_csv(file_path, index=False)
            
            print(f"[ExportWizard] Successfully exported to CSV: {file_path}")
            return True
            
        except Exception as e:
            print(f"[ExportWizard] CSV export error: {str(e)}")
            raise
            
    def _export_to_txt(self, channels, file_path):
        """Export channels to tab-separated text format"""
        try:
            # Prepare data for export
            export_data = self._prepare_export_data(channels)
            
            # Create DataFrame
            df = pd.DataFrame(export_data)
            
            # Write to tab-separated text file
            df.to_csv(file_path, index=False, sep='\t')
            
            print(f"[ExportWizard] Successfully exported to TXT: {file_path}")
            return True
            
        except Exception as e:
            print(f"[ExportWizard] TXT export error: {str(e)}")
            raise
            
    def _prepare_export_data(self, channels):
        """Prepare channel data for export"""
        try:
            export_data = {}
            
            # Process each channel
            for i, channel in enumerate(channels):
                print(f"[ExportWizard] Processing channel {i+1}/{len(channels)}: {channel.legend_label}")
                
                # Check if this is a spectrogram channel
                is_spectrogram = "spectrogram" in getattr(channel, 'tags', [])
                
                if is_spectrogram:
                    print(f"[ExportWizard] Detected spectrogram channel: {channel.legend_label}")
                    self._add_spectrogram_data(channel, export_data)
                else:
                    print(f"[ExportWizard] Processing regular channel: {channel.legend_label}")
                    self._add_regular_channel_data(channel, export_data)
            
            # Handle case where channels have different lengths
            # Find the maximum length
            if export_data:
                max_length = max(len(data) for data in export_data.values())
                print(f"[ExportWizard] Maximum data length: {max_length}")
                
                # Pad shorter arrays with NaN
                for col_name, data in export_data.items():
                    if len(data) < max_length:
                        padded_data = np.full(max_length, np.nan)
                        padded_data[:len(data)] = data
                        export_data[col_name] = padded_data
                        print(f"[ExportWizard] Padded column {col_name} from {len(data)} to {max_length}")
            
            return export_data
            
        except Exception as e:
            print(f"[ExportWizard] Error preparing export data: {str(e)}")
            raise
            
    def _add_regular_channel_data(self, channel, export_data):
        """Add regular (non-spectrogram) channel data to export dictionary"""
        # Get channel data
        xdata = channel.xdata
        ydata = channel.ydata
        
        if xdata is None or ydata is None:
            print(f"[ExportWizard] Warning: Channel {channel.legend_label} has no data")
            return
        
        # Convert to numpy arrays if they aren't already
        if not isinstance(xdata, np.ndarray):
            xdata = np.array(xdata)
        if not isinstance(ydata, np.ndarray):
            ydata = np.array(ydata)
        
        # Create column names
        x_col_name = f"{channel.legend_label}_{channel.xlabel}" if channel.xlabel else f"{channel.legend_label}_X"
        y_col_name = f"{channel.legend_label}_{channel.ylabel}" if channel.ylabel else f"{channel.legend_label}_Y"
        
        # Handle duplicate column names
        x_col_name = self._ensure_unique_column_name(x_col_name, export_data)
        y_col_name = self._ensure_unique_column_name(y_col_name, export_data)
        
        # Add data to export dictionary
        export_data[x_col_name] = xdata
        export_data[y_col_name] = ydata
        
        print(f"[ExportWizard] Added columns: {x_col_name}, {y_col_name}")
        
    def _add_spectrogram_data(self, channel, export_data):
        """Add spectrogram channel data to export dictionary"""
        # Get spectrogram data from metadata
        if not hasattr(channel, 'metadata') or not channel.metadata:
            print(f"[ExportWizard] Warning: Spectrogram channel {channel.legend_label} has no metadata")
            return
            
        zxx_data = channel.metadata.get('Zxx')
        if zxx_data is None:
            print(f"[ExportWizard] Warning: Spectrogram channel {channel.legend_label} has no Zxx data in metadata")
            return
            
        # Get time and frequency axes
        time_data = channel.xdata  # Time axis
        freq_data = channel.ydata  # Frequency axis
        
        if time_data is None or freq_data is None:
            print(f"[ExportWizard] Warning: Spectrogram channel {channel.legend_label} missing time or frequency data")
            return
            
        # Convert to numpy arrays
        if not isinstance(time_data, np.ndarray):
            time_data = np.array(time_data)
        if not isinstance(freq_data, np.ndarray):
            freq_data = np.array(freq_data)
        if not isinstance(zxx_data, np.ndarray):
            zxx_data = np.array(zxx_data)
            
        print(f"[ExportWizard] Spectrogram shape: {zxx_data.shape} (freq x time)")
        print(f"[ExportWizard] Time axis length: {len(time_data)}, Freq axis length: {len(freq_data)}")
        
        # Create base column names
        base_name = channel.legend_label.replace(" ", "_")
        time_col = self._ensure_unique_column_name(f"{base_name}_Time", export_data)
        freq_col = self._ensure_unique_column_name(f"{base_name}_Frequency", export_data)
        
        # Flatten the spectrogram data for export
        # Each row will represent one time-frequency point
        n_freq, n_time = zxx_data.shape
        total_points = n_freq * n_time
        
        # Create flattened arrays
        time_flat = np.repeat(time_data, n_freq)  # Repeat each time point for all frequencies
        freq_flat = np.tile(freq_data, n_time)    # Tile frequency array for each time point
        power_flat = zxx_data.flatten(order='F')  # Flatten column-wise (frequency first)
        
        # Add the three columns to export data
        export_data[time_col] = time_flat
        export_data[freq_col] = freq_flat
        
        # Add power/intensity column
        power_col = self._ensure_unique_column_name(f"{base_name}_Power", export_data)
        export_data[power_col] = power_flat
        
        print(f"[ExportWizard] Added spectrogram columns: {time_col}, {freq_col}, {power_col}")
        print(f"[ExportWizard] Total spectrogram data points: {len(power_flat)}")
        
    def _ensure_unique_column_name(self, base_name, export_data):
        """Ensure column name is unique by adding suffix if needed"""
        if base_name not in export_data:
            return base_name
            
        counter = 2
        while f"{base_name}_{counter}" in export_data:
            counter += 1
        return f"{base_name}_{counter}"
            
    def _validate_export_path(self, file_path):
        """Validate the export file path"""
        try:
            # Check if directory exists
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                
            # Check if we can write to the location
            # Try to create a temporary file to test write permissions
            test_path = file_path + ".tmp"
            try:
                with open(test_path, 'w') as f:
                    f.write("test")
                os.remove(test_path)
                return True
            except:
                return False
                
        except Exception as e:
            print(f"[ExportWizard] Path validation error: {str(e)}")
            return False
            
    def get_export_summary(self, channels):
        """Get a summary of what will be exported"""
        try:
            summary = {
                'channel_count': len(channels),
                'total_data_points': 0,
                'files_involved': set(),
                'channel_types': set()
            }
            
            for channel in channels:
                if channel.ydata is not None:
                    summary['total_data_points'] += len(channel.ydata)
                
                summary['files_involved'].add(channel.filename)
                
                if channel.type:
                    summary['channel_types'].add(channel.type.value)
            
            summary['files_involved'] = list(summary['files_involved'])
            summary['channel_types'] = list(summary['channel_types'])
            
            return summary
            
        except Exception as e:
            print(f"[ExportWizard] Error creating export summary: {str(e)}")
            return {
                'channel_count': len(channels),
                'total_data_points': 0,
                'files_involved': [],
                'channel_types': []
            } 