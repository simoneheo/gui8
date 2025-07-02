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
    
    def __init__(self, channel: Channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        
        self.setWindowTitle(f"Channel Metadata - {channel.ylabel or 'Unnamed'}")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        self.init_ui()
        self.populate_metadata()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"📊 {self.channel.ylabel or 'Unnamed Channel'}")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin: 10px; color: #2c3e50;")
        layout.addWidget(title)
        
        # Tab widget for organized information
        self.tab_widget = QTabWidget()
        
        # Tab 1: Basic Information
        self.basic_tab = self.create_basic_info_tab()
        self.tab_widget.addTab(self.basic_tab, "📋 Basic Info")
        
        # Tab 2: Statistical Analysis
        self.stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(self.stats_tab, "📈 Statistics")
        
        # Tab 3: Data Quality
        self.quality_tab = self.create_data_quality_tab()
        self.tab_widget.addTab(self.quality_tab, "🔍 Data Quality")
        
        # Tab 4: Raw Metadata
        self.raw_tab = self.create_raw_metadata_tab()
        self.tab_widget.addTab(self.raw_tab, "🔧 Technical")
        
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
            ("Channel Name", "channel_name"),
            ("Legend Label", "legend_label"),
            ("Data Type", "data_type"),
            ("File ID", "file_id"),
            ("Y-Axis", "y_axis"),
            ("Dimension", "dimension"),
            ("Sampling Rate", "sampling_rate"),
            ("X Range", "x_range"),
            ("Y Range", "y_range"),
            ("Color", "color"),
            ("Line Style", "line_style"),
            ("Marker", "marker"),
            ("Visible", "visible")
        ]
        
        for i, (label_text, key) in enumerate(basic_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold; color: #34495e;")
            value_label = QLabel("Loading...")
            value_label.setWordWrap(True)
            
            grid.addWidget(label, i, 0)
            grid.addWidget(value_label, i, 1)
            self.basic_labels[key] = value_label
        
        layout.addLayout(grid)
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
            ("Count", "x_count"),
            ("Mean", "x_mean"),
            ("Median", "x_median"),
            ("Std Dev", "x_std"),
            ("Min", "x_min"),
            ("Max", "x_max"),
            ("Range", "x_range_stat"),
            ("25th Percentile", "x_q25"),
            ("75th Percentile", "x_q75")
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
            ("Count", "y_count"),
            ("Mean", "y_mean"),
            ("Median", "y_median"),
            ("Std Dev", "y_std"),
            ("Min", "y_min"),
            ("Max", "y_max"),
            ("Range", "y_range_stat"),
            ("25th Percentile", "y_q25"),
            ("75th Percentile", "y_q75")
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
        quality_group = QGroupBox("Data Quality Assessment")
        quality_layout = QGridLayout(quality_group)
        
        self.quality_labels = {}
        quality_metrics = [
            ("Total Data Points", "total_points"),
            ("Valid Data Points", "valid_points"),
            ("Missing Values (NaN)", "nan_count"),
            ("Infinite Values", "inf_count"),
            ("Zero Values", "zero_count"),
            ("Duplicate Points", "duplicate_count"),
            ("Data Completeness", "completeness"),
            ("X-Axis Monotonic", "x_monotonic"),
            ("Y-Axis Outliers", "y_outliers"),
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
        
        layout.addWidget(quality_group)
        
        # Data issues section
        issues_group = QGroupBox("Data Issues & Recommendations")
        issues_layout = QVBoxLayout(issues_group)
        
        self.issues_text = QTextEdit()
        self.issues_text.setMaximumHeight(150)
        self.issues_text.setReadOnly(True)
        issues_layout.addWidget(self.issues_text)
        
        layout.addWidget(issues_group)
        layout.addStretch()
        
        return widget
    
    def create_raw_metadata_tab(self) -> QWidget:
        """Create the raw metadata tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Technical information
        tech_group = QGroupBox("Technical Metadata")
        tech_layout = QGridLayout(tech_group)
        
        self.tech_labels = {}
        tech_info = [
            ("Channel ID", "channel_id"),
            ("Creation Time", "creation_time"),
            ("Modified Time", "modified_time"),
            ("Processing Step", "processing_step"),
            ("Parent Channels", "parent_channels"),
            ("Memory Usage (X)", "memory_x"),
            ("Memory Usage (Y)", "memory_y"),
            ("Total Memory", "total_memory"),
            ("Data Format", "data_format"),
            ("Encoding", "encoding")
        ]
        
        for i, (label_text, key) in enumerate(tech_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Loading...")
            value_label.setWordWrap(True)
            
            tech_layout.addWidget(label, i, 0)
            tech_layout.addWidget(value_label, i, 1)
            self.tech_labels[key] = value_label
        
        layout.addWidget(tech_group)
        
        # Raw channel properties
        raw_group = QGroupBox("Raw Channel Properties")
        raw_layout = QVBoxLayout(raw_group)
        
        self.raw_text = QTextEdit()
        self.raw_text.setMaximumHeight(200)
        self.raw_text.setReadOnly(True)
        self.raw_text.setFont(QFont("Courier", 9))
        raw_layout.addWidget(self.raw_text)
        
        layout.addWidget(raw_group)
        layout.addStretch()
        
        return widget
    
    def populate_metadata(self):
        """Populate all metadata information"""
        self.populate_basic_info()
        self.populate_statistics()
        self.populate_data_quality()
        self.populate_technical_info()
    
    def populate_basic_info(self):
        """Populate basic information tab"""
        # Basic channel information
        self.basic_labels["channel_name"].setText(self.channel.ylabel or "Unnamed")
        self.basic_labels["legend_label"].setText(self.channel.legend_label or "None")
        self.basic_labels["data_type"].setText(
            self.channel.type.value if hasattr(self.channel.type, 'value') else str(self.channel.type)
        )
        self.basic_labels["file_id"].setText(str(self.channel.file_id))
        self.basic_labels["y_axis"].setText(self.channel.yaxis or "y-left")
        self.basic_labels["color"].setText(self.channel.color or "Default")
        self.basic_labels["line_style"].setText(self.channel.style or "Default")
        self.basic_labels["marker"].setText(self.channel.marker or "None")
        self.basic_labels["visible"].setText("Yes" if self.channel.show else "No")
        
        # Data dimensions
        if self.channel.ydata is not None:
            dimension = len(self.channel.ydata)
            self.basic_labels["dimension"].setText(f"{dimension:,} points")
        else:
            self.basic_labels["dimension"].setText("No data")
        
        # Sampling rate
        sampling_rate = self.channel.get_sampling_rate_description()
        self.basic_labels["sampling_rate"].setText(sampling_rate)
        
        # Ranges
        if self.channel.xdata is not None and len(self.channel.xdata) > 0:
            x_min, x_max = np.min(self.channel.xdata), np.max(self.channel.xdata)
            self.basic_labels["x_range"].setText(f"{x_min:.6g} to {x_max:.6g}")
        else:
            self.basic_labels["x_range"].setText("No X data")
        
        if self.channel.ydata is not None and len(self.channel.ydata) > 0:
            y_min, y_max = np.min(self.channel.ydata), np.max(self.channel.ydata)
            self.basic_labels["y_range"].setText(f"{y_min:.6g} to {y_max:.6g}")
        else:
            self.basic_labels["y_range"].setText("No Y data")
    
    def populate_statistics(self):
        """Populate statistics tab"""
        # X-axis statistics
        if self.channel.xdata is not None and len(self.channel.xdata) > 0:
            x_data = self.channel.xdata
            x_clean = x_data[~np.isnan(x_data)] if np.any(np.isnan(x_data)) else x_data
            
            self.x_stats_labels["x_count"].setText(f"{len(x_clean):,}")
            self.x_stats_labels["x_mean"].setText(f"{np.mean(x_clean):.6g}")
            self.x_stats_labels["x_median"].setText(f"{np.median(x_clean):.6g}")
            self.x_stats_labels["x_std"].setText(f"{np.std(x_clean):.6g}")
            self.x_stats_labels["x_min"].setText(f"{np.min(x_clean):.6g}")
            self.x_stats_labels["x_max"].setText(f"{np.max(x_clean):.6g}")
            self.x_stats_labels["x_range_stat"].setText(f"{np.max(x_clean) - np.min(x_clean):.6g}")
            self.x_stats_labels["x_q25"].setText(f"{np.percentile(x_clean, 25):.6g}")
            self.x_stats_labels["x_q75"].setText(f"{np.percentile(x_clean, 75):.6g}")
        else:
            for key in self.x_stats_labels:
                self.x_stats_labels[key].setText("No data")
        
        # Y-axis statistics
        if self.channel.ydata is not None and len(self.channel.ydata) > 0:
            y_data = self.channel.ydata
            y_clean = y_data[~np.isnan(y_data)] if np.any(np.isnan(y_data)) else y_data
            
            self.y_stats_labels["y_count"].setText(f"{len(y_clean):,}")
            self.y_stats_labels["y_mean"].setText(f"{np.mean(y_clean):.6g}")
            self.y_stats_labels["y_median"].setText(f"{np.median(y_clean):.6g}")
            self.y_stats_labels["y_std"].setText(f"{np.std(y_clean):.6g}")
            self.y_stats_labels["y_min"].setText(f"{np.min(y_clean):.6g}")
            self.y_stats_labels["y_max"].setText(f"{np.max(y_clean):.6g}")
            self.y_stats_labels["y_range_stat"].setText(f"{np.max(y_clean) - np.min(y_clean):.6g}")
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
            
            # Completeness
            completeness = (valid_points / total_points * 100) if total_points > 0 else 0
            self.quality_labels["completeness"].setText(f"{completeness:.1f}%")
            
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
            
            # Check for outliers in Y data (using IQR method)
            if y_data is not None and len(y_data) > 4:
                y_clean = y_data[~np.isnan(y_data)]
                if len(y_clean) > 4:
                    q1, q3 = np.percentile(y_clean, [25, 75])
                    iqr = q3 - q1
                    outlier_threshold = 1.5 * iqr
                    outliers = np.sum((y_clean < q1 - outlier_threshold) | (y_clean > q3 + outlier_threshold))
                    outlier_pct = (outliers / len(y_clean) * 100) if len(y_clean) > 0 else 0
                    self.quality_labels["y_outliers"].setText(f"{outliers} ({outlier_pct:.1f}%)")
                    
                    if outlier_pct > 5:
                        issues.append(f"High percentage of outliers ({outlier_pct:.1f}%) detected in Y data")
            
            # Data type consistency
            self.quality_labels["type_consistency"].setText("Consistent" if len(set(type(val) for val in y_data[:100])) <= 2 else "Mixed types")
            
            # Add quality issues
            if total_nan > 0:
                issues.append(f"Dataset contains {total_nan} missing values (NaN)")
            if total_inf > 0:
                issues.append(f"Dataset contains {total_inf} infinite values")
            if completeness < 95:
                issues.append(f"Data completeness is only {completeness:.1f}%")
            if duplicate_count > 0:
                issues.append(f"{duplicate_count} duplicate X-values found")
        
        else:
            for key in self.quality_labels:
                self.quality_labels[key].setText("No data available")
            issues.append("No data available for quality analysis")
        
        # Display issues or give all-clear
        if issues:
            issues_text = "⚠️ Issues Found:\n\n" + "\n".join(f"• {issue}" for issue in issues)
            issues_text += "\n\n💡 Recommendations:\n• Consider data cleaning before analysis\n• Check data source for potential issues\n• Verify data collection parameters"
        else:
            issues_text = "✅ No significant data quality issues detected!\n\nThe dataset appears to be clean and ready for analysis."
        
        self.issues_text.setPlainText(issues_text)
    
    def populate_technical_info(self):
        """Populate technical information tab"""
        # Technical metadata
        self.tech_labels["channel_id"].setText(str(self.channel.channel_id))
        self.tech_labels["creation_time"].setText(
            self.channel.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.channel.created_at else "Unknown"
        )
        self.tech_labels["modified_time"].setText(
            self.channel.modified_at.strftime("%Y-%m-%d %H:%M:%S") if self.channel.modified_at else "Unknown"
        )
        self.tech_labels["processing_step"].setText(str(self.channel.step))
        
        # Parent channels
        if hasattr(self.channel, 'parent_ids') and self.channel.parent_ids:
            parent_info = f"{len(self.channel.parent_ids)} parent(s)"
        else:
            parent_info = "None (original data)"
        self.tech_labels["parent_channels"].setText(parent_info)
        
        # Memory usage
        if self.channel.xdata is not None:
            x_memory = self.channel.xdata.nbytes
            self.tech_labels["memory_x"].setText(f"{x_memory / 1024:.1f} KB")
        else:
            self.tech_labels["memory_x"].setText("No X data")
        
        if self.channel.ydata is not None:
            y_memory = self.channel.ydata.nbytes
            self.tech_labels["memory_y"].setText(f"{y_memory / 1024:.1f} KB")
            total_memory = (x_memory if self.channel.xdata is not None else 0) + y_memory
            self.tech_labels["total_memory"].setText(f"{total_memory / 1024:.1f} KB")
        else:
            self.tech_labels["memory_y"].setText("No Y data")
            self.tech_labels["total_memory"].setText("Unknown")
        
        # Data format info
        if self.channel.ydata is not None:
            self.tech_labels["data_format"].setText(f"NumPy {self.channel.ydata.dtype}")
        else:
            self.tech_labels["data_format"].setText("Unknown")
        
        self.tech_labels["encoding"].setText("UTF-8 (assumed)")
        
        # Raw properties in text area
        raw_properties = []
        for attr in dir(self.channel):
            if not attr.startswith('_') and not callable(getattr(self.channel, attr)):
                try:
                    value = getattr(self.channel, attr)
                    if value is not None and not isinstance(value, (np.ndarray)):
                        raw_properties.append(f"{attr}: {value}")
                except:
                    pass
        
        self.raw_text.setPlainText("\n".join(raw_properties))


# Convenience function for opening the metadata wizard
def show_channel_metadata(channel: Channel, parent=None):
    """
    Show the metadata wizard for a channel
    
    Args:
        channel: Channel to analyze
        parent: Parent widget
    """
    wizard = MetadataWizard(channel, parent)
    wizard.exec() 