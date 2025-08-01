from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, 
    QPushButton, QTextEdit, QTabWidget, QWidget, QFrame,
    QScrollArea, QGroupBox, QDialogButtonBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette
import numpy as np
from typing import Optional
from datetime import datetime
import pandas as pd

from channel import Channel


class MetadataWizard(QDialog):
    """
    Comprehensive metadata viewer for channel data
    """
    
    def __init__(self, channel: Channel, parent=None, file_manager=None):
        super().__init__(parent)
        self.channel = channel
        self.file_manager = file_manager
        
        self.setWindowTitle(f"Channel Metadata - {channel.ylabel or 'Unnamed'}")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        self.init_ui()
        self.populate_metadata()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
             
        # Tab widget for organized information
        self.tab_widget = QTabWidget()
        
        # Tab 1: Basic Information
        self.basic_tab = self.create_basic_info_tab()
        self.tab_widget.addTab(self.basic_tab, "Basic Info")
        
        # Tab 2: Statistical Analysis
        self.stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(self.stats_tab, "Statistics")
        
        # Tab 3: Data Quality
        self.quality_tab = self.create_data_quality_tab()
        self.tab_widget.addTab(self.quality_tab, "Data Quality")
        

        
        layout.addWidget(self.tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.close)
        layout.addWidget(button_box)
    
    def create_basic_info_tab(self) -> QWidget:
        """Create the basic information tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create grid for basic info
        grid = QGridLayout()
        
        self.basic_labels = {}
        basic_info = [
            ("File Name", "file_name"),
            ("Channel Name", "channel_name"),
            ("Channel Stage", "data_type"),            
            ("Total Memory", "total_memory"),
            ("Data Format", "data_format"),
            ("Modified Time", "modified_time"),
        ]
        
        for i, (label_text, key) in enumerate(basic_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Loading...")
            value_label.setWordWrap(True)
            
            grid.addWidget(label, i, 0)
            grid.addWidget(value_label, i, 1)
            self.basic_labels[key] = value_label
        
        layout.addLayout(grid)
        
        # Categorical Mapping section (will be shown/hidden dynamically)
        self.categorical_group = QGroupBox("Categorical Mapping")
        categorical_layout = QVBoxLayout(self.categorical_group)
        
        self.categorical_text = QLabel("No categorical mapping available")
        self.categorical_text.setWordWrap(True)
        self.categorical_text.setStyleSheet("color: #7f8c8d; font-style: italic;")
        categorical_layout.addWidget(self.categorical_text)
        
        layout.addWidget(self.categorical_group)
        layout.addStretch()
        
        return widget
    
    def create_statistics_tab(self) -> QWidget:
        """Create the statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # X-axis statistics group
        x_group = QGroupBox("X-Axis Statistics")
        x_layout = QGridLayout(x_group)
        
        self.x_stats_labels = {}
        x_stats = [
            ("Count (non-NaN)", "x_count"),
            ("Min", "x_min"),
            ("Max", "x_max"),
            ("Mean Sampling Rate", "x_mean_interval"),
            ("Median Sampling Rate", "x_median_interval"),
            ("Std Sampling Rate", "x_std_interval"),
            ("Min Sampling Rate", "x_min_interval"),
            ("Max Sampling Rate", "x_max_interval")
        ]
        
        for i, (label_text, key) in enumerate(x_stats):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            
            x_layout.addWidget(label, i, 0)
            x_layout.addWidget(value_label, i, 1)
            self.x_stats_labels[key] = value_label
        
        layout.addWidget(x_group)
        
        # Y-axis statistics group
        y_group = QGroupBox("Y-Axis Statistics")
        y_layout = QGridLayout(y_group)
        
        self.y_stats_labels = {}
        y_stats = [
            ("Count (non-NaN)", "y_count"),
            ("Min", "y_min"),
            ("Max", "y_max"),
            ("Mean", "y_mean"),
            ("Median", "y_median"),
            ("Std Dev", "y_std"),            
            ("25th Percentile", "y_q25"),
            ("75th Percentile", "y_q75"),            
        ]
        
        for i, (label_text, key) in enumerate(y_stats):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            
            y_layout.addWidget(label, i, 0)
            y_layout.addWidget(value_label, i, 1)
            self.y_stats_labels[key] = value_label
        
        layout.addWidget(y_group)
        layout.addStretch()
        
        return widget
    
    def create_data_quality_tab(self) -> QWidget:
        """Create the data quality tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Data quality metrics
        quality_layout = QGridLayout()
        
        self.quality_labels = {}
        quality_metrics = [
            ("Total Data Points", "total_points"),
            ("Valid Data Points", "valid_points"),
            ("Missing Values (NaN)", "nan_count"),
            ("Infinite Values", "inf_count"),
            ("Zero Values", "zero_count"),
            ("Duplicate Points", "duplicate_count"),
            ("X-Axis Monotonic", "x_monotonic"),
            ("Data Type Consistency", "type_consistency")
        ]
        
        for i, (label_text, key) in enumerate(quality_metrics):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Analyzing...")
            value_label.setWordWrap(True)
            
            quality_layout.addWidget(label, i, 0)
            quality_layout.addWidget(value_label, i, 1)
            self.quality_labels[key] = value_label
        
        layout.addLayout(quality_layout)
        layout.addStretch()
        
        return widget
    

    
    def populate_metadata(self):
        """Populate all metadata information"""
        self.populate_basic_info()
        self.populate_statistics()
        self.populate_data_quality()
    
    def populate_basic_info(self):
        """Populate basic information tab"""
        # Basic channel information
        self.basic_labels["channel_name"].setText(self.channel.ylabel or "Unnamed")
        
        # Get file name from file_id
        file_name = "Unknown"
        if hasattr(self.channel, 'file_id') and self.channel.file_id:
            if self.file_manager:
                # Get the actual file object and filename
                file_obj = self.file_manager.get_file(self.channel.file_id)
                if file_obj:
                    file_name = file_obj.filename
                else:
                    file_name = f"File ID: {self.channel.file_id} (not found)"
            else:
                file_name = f"File ID: {self.channel.file_id}"
        self.basic_labels["file_name"].setText(file_name)
        
        self.basic_labels["data_type"].setText(
            self.channel.type.value if hasattr(self.channel.type, 'value') else str(self.channel.type)
        )
        

        
        # Modified Time
        self.basic_labels["modified_time"].setText(
            self.channel.modified_at.strftime("%Y-%m-%d %H:%M:%S") if self.channel.modified_at else "Unknown"
        )
        
        # Total Memory
        if self.channel.xdata is not None and self.channel.ydata is not None:
            x_memory = self.channel.xdata.nbytes
            y_memory = self.channel.ydata.nbytes
            total_memory = x_memory + y_memory
            self.basic_labels["total_memory"].setText(f"{total_memory / 1024:.1f} KB")
        else:
            self.basic_labels["total_memory"].setText("Unknown")
        
        # Data Format
        if self.channel.ydata is not None:
            self.basic_labels["data_format"].setText(f"NumPy {self.channel.ydata.dtype}")
        else:
            self.basic_labels["data_format"].setText("Unknown")
        
        # Handle categorical mapping section
        self._populate_categorical_mapping()
    
    def populate_statistics(self):
        """Populate statistics tab"""
        # X-axis statistics
        if self.channel.xdata is not None and len(self.channel.xdata) > 0:
            x_data = self.channel.xdata
            x_clean = x_data[~np.isnan(x_data)] if np.any(np.isnan(x_data)) else x_data
            
            # Count non-NaN values
            self.x_stats_labels["x_count"].setText(f"{len(x_clean):,}")
            
            # Basic min/max
            self.x_stats_labels["x_min"].setText(f"{np.min(x_clean):.6g}")
            self.x_stats_labels["x_max"].setText(f"{np.max(x_clean):.6g}")
            
            # Sampling rates (1/sampling_interval in Hz)
            if len(x_clean) > 1:
                intervals = np.diff(x_clean)
                # Convert intervals to sampling rates (Hz)
                mean_rate = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')
                median_rate = 1.0 / np.median(intervals) if np.median(intervals) > 0 else float('inf')
                min_rate = 1.0 / np.max(intervals) if np.max(intervals) > 0 else float('inf')  # Note: min rate = 1/max_interval
                max_rate = 1.0 / np.min(intervals) if np.min(intervals) > 0 else float('inf')  # Note: max rate = 1/min_interval
                
                # For std, we need to be more careful - convert each interval to rate first
                rates = 1.0 / intervals
                rates = rates[np.isfinite(rates)]  # Remove infinite values
                std_rate = np.std(rates) if len(rates) > 0 else 0
                
                self.x_stats_labels["x_mean_interval"].setText(f"{mean_rate:.6g} Hz")
                self.x_stats_labels["x_median_interval"].setText(f"{median_rate:.6g} Hz")
                self.x_stats_labels["x_std_interval"].setText(f"{std_rate:.6g} Hz")
                self.x_stats_labels["x_min_interval"].setText(f"{min_rate:.6g} Hz")
                self.x_stats_labels["x_max_interval"].setText(f"{max_rate:.6g} Hz")
            else:
                self.x_stats_labels["x_mean_interval"].setText("N/A")
                self.x_stats_labels["x_median_interval"].setText("N/A")
                self.x_stats_labels["x_std_interval"].setText("N/A")
                self.x_stats_labels["x_min_interval"].setText("N/A")
                self.x_stats_labels["x_max_interval"].setText("N/A")
        else:
            for key in self.x_stats_labels:
                self.x_stats_labels[key].setText("No data")
        
        # Y-axis statistics
        if self.channel.ydata is not None and len(self.channel.ydata) > 0:
            y_data = self.channel.ydata
            y_clean = y_data[~np.isnan(y_data)] if np.any(np.isnan(y_data)) else y_data
            
            # Count non-NaN values
            self.y_stats_labels["y_count"].setText(f"{len(y_clean):,}")
            
            # Basic statistics
            self.y_stats_labels["y_min"].setText(f"{np.min(y_clean):.6g}")
            self.y_stats_labels["y_max"].setText(f"{np.max(y_clean):.6g}")
            self.y_stats_labels["y_mean"].setText(f"{np.mean(y_clean):.6g}")
            self.y_stats_labels["y_median"].setText(f"{np.median(y_clean):.6g}")
            self.y_stats_labels["y_std"].setText(f"{np.std(y_clean):.6g}")            
            self.y_stats_labels["y_q25"].setText(f"{np.percentile(y_clean, 25):.6g}")
            self.y_stats_labels["y_q75"].setText(f"{np.percentile(y_clean, 75):.6g}")
        else:
            for key in self.y_stats_labels:
                self.y_stats_labels[key].setText("No data")
    
    def populate_data_quality(self):
        """Populate data quality tab"""
        issues = []
        
        # Analyze X data
        x_data = self.channel.xdata
        y_data = self.channel.ydata
        
        if x_data is not None and y_data is not None:
            total_points = len(y_data)
            self.quality_labels["total_points"].setText(f"{total_points:,}")
            
            # Check for NaN values
            x_nan_count = np.sum(np.isnan(x_data)) if x_data is not None else 0
            y_nan_count = np.sum(np.isnan(y_data)) if y_data is not None else 0
            total_nan = x_nan_count + y_nan_count
            self.quality_labels["nan_count"].setText(f"{total_nan} ({x_nan_count} X, {y_nan_count} Y)")
            
            # Check for infinite values
            x_inf_count = np.sum(np.isinf(x_data)) if x_data is not None else 0
            y_inf_count = np.sum(np.isinf(y_data)) if y_data is not None else 0
            total_inf = x_inf_count + y_inf_count
            self.quality_labels["inf_count"].setText(f"{total_inf} ({x_inf_count} X, {y_inf_count} Y)")
            
            # Count zeros
            x_zero_count = np.sum(x_data == 0) if x_data is not None else 0
            y_zero_count = np.sum(y_data == 0) if y_data is not None else 0
            self.quality_labels["zero_count"].setText(f"{y_zero_count} Y-values")
            
            # Valid points
            valid_points = total_points - y_nan_count - y_inf_count
            self.quality_labels["valid_points"].setText(f"{valid_points:,}")
            
            # Check for duplicates (X values)
            if x_data is not None:
                unique_x = len(np.unique(x_data))
                duplicate_count = len(x_data) - unique_x
                self.quality_labels["duplicate_count"].setText(f"{duplicate_count}")
                
                # Check if X is monotonic
                is_monotonic = np.all(np.diff(x_data) >= 0) or np.all(np.diff(x_data) <= 0)
                self.quality_labels["x_monotonic"].setText("Yes" if is_monotonic else "No")
                if not is_monotonic:
                    issues.append("X-axis values are not monotonic (may affect plotting)")
            
            # Outlier detection removed - not displaying in wizard
            
            # Data type consistency
            self.quality_labels["type_consistency"].setText("Consistent" if len(set(type(val) for val in y_data[:100])) <= 2 else "Mixed types")
            
            # Add quality issues
            if total_nan > 0:
                issues.append(f"Dataset contains {total_nan} missing values (NaN)")
            if total_inf > 0:
                issues.append(f"Dataset contains {total_inf} infinite values")
            if duplicate_count > 0:
                issues.append(f"{duplicate_count} duplicate X-values found")
        
        else:
            for key in self.quality_labels:
                self.quality_labels[key].setText("No data available")
            issues.append("No data available for quality analysis")
        
        # Note: Issues tracking removed - only metrics are displayed now
    


    def _determine_channel_type(self) -> str:
        """Determine if the channel is category, numeric, or datetime based on data and metadata"""
        # Check metadata first for explicit type information
        if self.channel.metadata:
            # Check for category mapping
            if 'category_mapping' in self.channel.metadata:
                return "Category"
            
            # Check for datetime information
            if self.channel.metadata.get('x_is_datetime', False):
                return "DateTime"
        
        # Analyze the data to determine type
        if self.channel.ydata is not None and len(self.channel.ydata) > 0:
            # Check if data looks categorical (small number of unique values)
            unique_values = np.unique(self.channel.ydata)
            unique_count = len(unique_values)
            total_count = len(self.channel.ydata)
            
            if total_count > 0:
                unique_ratio = unique_count / total_count
                
                # If very few unique values relative to total, likely categorical
                if unique_ratio < 0.1 and unique_count < 20:
                    return "Category"
                
                # Check if values are all integers (could be category codes)
                if np.all(np.mod(self.channel.ydata, 1) == 0):  # All integers
                    if unique_count < 50:  # Reasonable number of categories
                        return "Category"
        
        # Check X-axis for datetime patterns
        if self.channel.xdata is not None and len(self.channel.xdata) > 0:
            # Check if X-axis has datetime metadata
            if self.channel.metadata and self.channel.metadata.get('x_is_datetime', False):
                return "DateTime"
            
            # Check if X-axis values look like datetime (large numbers, increasing)
            if (np.all(self.channel.xdata > 1e6) and  # Large numbers (could be timestamps)
                np.all(np.diff(self.channel.xdata) > 0)):  # Strictly increasing
                return "DateTime"
        
        # Default to numeric
        return "Numeric"

    def _populate_categorical_mapping(self):
        """Populate the categorical mapping section with category information"""
        mapping_text = "No categorical mapping available"
        has_mapping = False
        
        # Check metadata for category mapping
        if self.channel.metadata and 'category_mapping' in self.channel.metadata:
            category_mapping = self.channel.metadata['category_mapping']
            if category_mapping:
                # Format the mapping nicely
                mapping_items = []
                for code, label in category_mapping.items():
                    mapping_items.append(f"{code} → '{label}'")
                mapping_text = "; ".join(mapping_items)
                has_mapping = True
        
        # If no explicit mapping, try to infer from data
        elif self.channel.ydata is not None and len(self.channel.ydata) > 0:
            unique_values = np.unique(self.channel.ydata)
            if len(unique_values) <= 20:  # Reasonable number to display
                mapping_items = [f"{val}" for val in unique_values]
                mapping_text = f"Unique values: {', '.join(mapping_items)}"
                has_mapping = True
        
        # Update the categorical mapping text
        self.categorical_text.setText(mapping_text)
        
        # Show/hide the categorical mapping section based on whether mapping exists
        if has_mapping:
            self.categorical_group.setVisible(True)
            self.categorical_text.setStyleSheet("color: #2c3e50;")
        else:
            self.categorical_group.setVisible(False)


# Convenience function for opening the metadata wizard
def show_channel_metadata(channel: Channel, parent=None, file_manager=None):
    """
    Show the metadata wizard for a channel
    
    Args:
        channel: Channel to analyze
        parent: Parent widget
        file_manager: File manager to get file information
    """
    wizard = MetadataWizard(channel, parent, file_manager)
    wizard.exec() 