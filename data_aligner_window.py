from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSpinBox, 
    QDoubleSpinBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt, Signal
from typing import Dict, Any, Optional
from channel import Channel


class DataAlignerWidget(QWidget):
    """
    Reusable data alignment UI widget for both Signal Mixer and Comparison Wizards.
    
    Provides a consistent interface for configuring data alignment parameters
    that work directly with the DataAligner class.
    """
    
    # Signal emitted when alignment parameters change
    parameters_changed = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize UI
        self._build_ui()
        
        # Connect signals
        self._connect_signals()
        
    def _build_ui(self):
        """Build the alignment UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main alignment group
        self.alignment_group = QGroupBox("Data Alignment")
        group_layout = QVBoxLayout(self.alignment_group)
        
        # Alignment mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Alignment Method:"))
        self.alignment_method_combo = QComboBox()
        self.alignment_method_combo.addItems(["time", "index"])
        mode_layout.addWidget(self.alignment_method_combo)
        group_layout.addLayout(mode_layout)
        
        # Index-based options
        self.index_group = QGroupBox("Index-Based Alignment")
        index_layout = QFormLayout(self.index_group)
        
        # Index mode
        self.index_mode_combo = QComboBox()
        self.index_mode_combo.addItems(["truncate", "custom"])
        index_layout.addRow("Mode:", self.index_mode_combo)
        
        # Index range controls
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(0, 9999999)
        self.start_index_spin.setValue(0)
        index_layout.addRow("Start Index:", self.start_index_spin)
        
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(0, 9999999)
        self.end_index_spin.setValue(1000)
        index_layout.addRow("End Index:", self.end_index_spin)
        
        # Index offset
        self.index_offset_spin = QSpinBox()
        self.index_offset_spin.setRange(-9999999, 9999999)
        self.index_offset_spin.setValue(0)
        self.index_offset_spin.setToolTip("Positive: shift test data forward, Negative: shift reference data forward")
        index_layout.addRow("Offset:", self.index_offset_spin)
        
        group_layout.addWidget(self.index_group)
        
        # Time-based options
        self.time_group = QGroupBox("Time-Based Alignment")
        time_layout = QFormLayout(self.time_group)
        
        # Time mode
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems(["overlap", "custom"])
        time_layout.addRow("Mode:", self.time_mode_combo)
        
        # Time range controls
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(-999999999.0, 999999999.0)
        self.start_time_spin.setDecimals(6)
        self.start_time_spin.setValue(0.0)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-999999999.0, 999999999.0)
        self.end_time_spin.setDecimals(6)
        self.end_time_spin.setValue(10.0)
        time_layout.addRow("End Time:", self.end_time_spin)
        
        # Time offset
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-999999999.0, 999999999.0)
        self.time_offset_spin.setDecimals(6)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setToolTip("Time offset to apply to test data")
        time_layout.addRow("Time Offset:", self.time_offset_spin)
        
        # Interpolation method
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["linear", "nearest", "cubic"])
        time_layout.addRow("Interpolation:", self.interpolation_combo)
        
        # Time resolution
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.000001, 1000.0)
        self.resolution_spin.setDecimals(6)
        self.resolution_spin.setValue(0.1)
        self.resolution_spin.setToolTip("Time grid resolution (smaller = more points)")
        time_layout.addRow("Resolution:", self.resolution_spin)
        
        group_layout.addWidget(self.time_group)
        
        # Alignment status
        self.status_label = QLabel("Ready for alignment")
        self.status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        group_layout.addWidget(self.status_label)
        
        layout.addWidget(self.alignment_group)
        
        # Set initial visibility
        self._on_alignment_method_changed("time")
        
    def _connect_signals(self):
        """Connect UI signals to parameter change handler"""
        # Alignment method change
        self.alignment_method_combo.currentTextChanged.connect(self._on_alignment_method_changed)
        
        # Index parameter changes
        self.index_mode_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.start_index_spin.valueChanged.connect(self._on_parameter_changed)
        self.end_index_spin.valueChanged.connect(self._on_parameter_changed)
        self.index_offset_spin.valueChanged.connect(self._on_parameter_changed)
        
        # Time parameter changes
        self.time_mode_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.start_time_spin.valueChanged.connect(self._on_parameter_changed)
        self.end_time_spin.valueChanged.connect(self._on_parameter_changed)
        self.time_offset_spin.valueChanged.connect(self._on_parameter_changed)
        self.interpolation_combo.currentTextChanged.connect(self._on_parameter_changed)
        self.resolution_spin.valueChanged.connect(self._on_parameter_changed)
        
    def _on_alignment_method_changed(self, method: str):
        """Handle alignment method change"""
        if method == "index":
            self.index_group.setVisible(True)
            self.time_group.setVisible(False)
            self.status_label.setText("Index-based alignment configured")
        else:  # time
            self.index_group.setVisible(False)
            self.time_group.setVisible(True)
            self.status_label.setText("Time-based alignment configured")
        
        # Emit parameter change
        self._on_parameter_changed()
        
    def _on_parameter_changed(self):
        """Handle any parameter change"""
        params = self.get_alignment_parameters()
        self.parameters_changed.emit(params)
        
    def get_alignment_parameters(self) -> Dict[str, Any]:
        """Get current alignment parameters in DataAligner format"""
        alignment_method = self.alignment_method_combo.currentText()
        
        if alignment_method == "index":
            return {
                'alignment_method': 'index',
                'mode': self.index_mode_combo.currentText(),
                'start_index': self.start_index_spin.value(),
                'end_index': self.end_index_spin.value(),
                'offset': self.index_offset_spin.value()
            }
        else:  # time
            return {
                'alignment_method': 'time',
                'mode': self.time_mode_combo.currentText(),
                'start_time': self.start_time_spin.value(),
                'end_time': self.end_time_spin.value(),
                'offset': self.time_offset_spin.value(),
                'interpolation': self.interpolation_combo.currentText(),
                'resolution': self.resolution_spin.value()
            }
    
    def set_alignment_parameters(self, params: Dict[str, Any]):
        """Set alignment parameters from a dictionary"""
        try:
            alignment_method = params.get('alignment_method', 'time')
            
            # Set alignment method
            index = self.alignment_method_combo.findText(alignment_method)
            if index >= 0:
                self.alignment_method_combo.setCurrentIndex(index)
            
            if alignment_method == "index":
                # Set index parameters
                mode = params.get('mode', 'truncate')
                index = self.index_mode_combo.findText(mode)
                if index >= 0:
                    self.index_mode_combo.setCurrentIndex(index)
                
                self.start_index_spin.setValue(params.get('start_index', 0))
                self.end_index_spin.setValue(params.get('end_index', 1000))
                self.index_offset_spin.setValue(params.get('offset', 0))
                
            else:  # time
                # Set time parameters
                mode = params.get('mode', 'overlap')
                index = self.time_mode_combo.findText(mode)
                if index >= 0:
                    self.time_mode_combo.setCurrentIndex(index)
                
                self.start_time_spin.setValue(params.get('start_time', 0.0))
                self.end_time_spin.setValue(params.get('end_time', 10.0))
                self.time_offset_spin.setValue(params.get('offset', 0.0))
                
                # Set interpolation
                interpolation = params.get('interpolation', 'linear')
                index = self.interpolation_combo.findText(interpolation)
                if index >= 0:
                    self.interpolation_combo.setCurrentIndex(index)
                
                self.resolution_spin.setValue(params.get('resolution', 0.1))
                
        except Exception as e:
            print(f"[DataAlignerWidget] Error setting parameters: {e}")
    
    def auto_configure_for_channels(self, ref_channel: Optional[Channel], test_channel: Optional[Channel]):
        """Auto-configure alignment parameters based on selected channels"""
        if not ref_channel or not test_channel:
            self.status_label.setText("Select channels to auto-configure alignment")
            return
        
        try:
            alignment_method = self.alignment_method_combo.currentText()
            
            if alignment_method == "index":
                # Auto-configure index-based alignment
                len_ref = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
                len_test = len(test_channel.ydata) if test_channel.ydata is not None else 0
                
                if len_ref > 0 and len_test > 0:
                    # Set range to minimum of both channels
                    max_index = min(len_ref, len_test) - 1
                    self.start_index_spin.setValue(0)
                    self.end_index_spin.setValue(max_index)
                    self.end_index_spin.setMaximum(max_index)
                    
                    # Update status
                    self.status_label.setText(f"Auto-configured: ref({len_ref}) test({len_test}) → range [0:{max_index}]")
                else:
                    self.status_label.setText("Channels have no data for alignment")
            
            else:  # time
                # Auto-configure time-based alignment
                has_time_ref = hasattr(ref_channel, 'xdata') and ref_channel.xdata is not None
                has_time_test = hasattr(test_channel, 'xdata') and test_channel.xdata is not None
                
                if has_time_ref and has_time_test:
                    # Find overlap region
                    ref_start, ref_end = float(ref_channel.xdata[0]), float(ref_channel.xdata[-1])
                    test_start, test_end = float(test_channel.xdata[0]), float(test_channel.xdata[-1])
                    
                    overlap_start = max(ref_start, test_start)
                    overlap_end = min(ref_end, test_end)
                    
                    if overlap_start < overlap_end:
                        self.start_time_spin.setValue(overlap_start)
                        self.end_time_spin.setValue(overlap_end)
                        self.status_label.setText(f"Auto-configured: overlap [{overlap_start:.3f}:{overlap_end:.3f}]")
                    else:
                        self.status_label.setText("No time overlap between channels")
                else:
                    self.status_label.setText("Channels missing time data - will create synthetic time")
        
        except Exception as e:
            print(f"[DataAlignerWidget] Error auto-configuring: {e}")
            self.status_label.setText(f"Auto-configuration failed: {e}")
    
    def validate_parameters(self) -> tuple[bool, str]:
        """Validate current alignment parameters"""
        try:
            params = self.get_alignment_parameters()
            alignment_method = params['alignment_method']
            
            if alignment_method == "index":
                start_idx = params['start_index']
                end_idx = params['end_index']
                
                if start_idx >= end_idx:
                    return False, "Start index must be less than end index"
                
                if start_idx < 0 or end_idx < 0:
                    return False, "Indices must be non-negative"
                    
            else:  # time
                start_time = params['start_time']
                end_time = params['end_time']
                resolution = params['resolution']
                
                if start_time >= end_time:
                    return False, "Start time must be less than end time"
                
                if resolution <= 0:
                    return False, "Resolution must be positive"
            
            return True, "Parameters are valid"
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def set_status_message(self, message: str, is_error: bool = False):
        """Set status message with optional error styling"""
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red; font-size: 10px; padding: 5px;")
        else:
            self.status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
    
    def get_alignment_summary(self) -> str:
        """Get a human-readable summary of current alignment configuration"""
        params = self.get_alignment_parameters()
        method = params['alignment_method']
        
        if method == "index":
            mode = params['mode']
            if mode == "truncate":
                return f"Index-based: truncate to shortest"
            else:
                return f"Index-based: range [{params['start_index']}:{params['end_index']}], offset {params['offset']}"
        else:
            mode = params['mode']
            if mode == "overlap":
                return f"Time-based: overlap region, {params['interpolation']} interpolation"
            else:
                return f"Time-based: range [{params['start_time']:.3f}:{params['end_time']:.3f}]s, {params['interpolation']} interpolation"


# Convenience function for creating the widget
def create_data_aligner_widget(parent=None) -> DataAlignerWidget:
    """Create a DataAlignerWidget instance"""
    return DataAlignerWidget(parent) 