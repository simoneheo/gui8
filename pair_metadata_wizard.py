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

from pair import Pair


class PairMetadataWizard(QDialog):
    """
    Comprehensive metadata viewer for comparison pair data
    """
    
    def __init__(self, pair: Pair, parent=None, file_manager=None, channel_manager=None):
        super().__init__(parent)
        self.pair = pair
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        
        self.setWindowTitle(f"Pair Metadata - {pair.name or 'Unnamed Pair'}")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        self.resize(700, 600)
        
        self.init_ui()
        self.populate_metadata()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"Pair Metadata - {self.pair.name or 'Unnamed Pair'}")
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
        self.tab_widget.addTab(self.basic_tab, "Basic Info")
        
        # Tab 2: Statistical Analysis
        self.stats_tab = self.create_statistics_tab()
        self.tab_widget.addTab(self.stats_tab, "Channel Statistics")
        
        # Tab 3: Comparison Statistics
        self.comp_tab = self.create_comparison_statistics_tab()
        self.tab_widget.addTab(self.comp_tab, "Comparison Statistics")
        
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
            ("Pair Name", "pair_name"),
            ("Pair ID", "pair_id"),
            ("Reference Channel", "ref_channel"),
            ("Reference File", "ref_file"),
            ("Test Channel", "test_channel"),
            ("Test File", "test_file"),
            ("Alignment Method", "alignment_method"),
            ("Created Date", "created_date"),
            ("Modified Date", "modified_date"),
            ("Total Memory", "total_memory"),
            ("Data Format", "data_format"),
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
        
        # Reference Channel statistics group
        ref_group = QGroupBox("Reference Channel Statistics")
        ref_layout = QGridLayout(ref_group)
        
        self.ref_stats_labels = {}
        ref_stats = [
            ("Data Points", "ref_count"),
            ("Min", "ref_min"),
            ("Max", "ref_max"),
            ("Mean", "ref_mean"),
            ("Median", "ref_median"),
            ("Std Dev", "ref_std"),
            ("25th Percentile", "ref_q25"),
            ("75th Percentile", "ref_q75")
        ]
        
        for i, (label_text, key) in enumerate(ref_stats):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            
            ref_layout.addWidget(label, i, 0)
            ref_layout.addWidget(value_label, i, 1)
            self.ref_stats_labels[key] = value_label
        
        layout.addWidget(ref_group)
        
        # Test Channel statistics group
        test_group = QGroupBox("Test Channel Statistics")
        test_layout = QGridLayout(test_group)
        
        self.test_stats_labels = {}
        test_stats = [
            ("Data Points", "test_count"),
            ("Min", "test_min"),
            ("Max", "test_max"),
            ("Mean", "test_mean"),
            ("Median", "test_median"),
            ("Std Dev", "test_std"),
            ("25th Percentile", "test_q25"),
            ("75th Percentile", "test_q75")
        ]
        
        for i, (label_text, key) in enumerate(test_stats):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            
            test_layout.addWidget(label, i, 0)
            test_layout.addWidget(value_label, i, 1)
            self.test_stats_labels[key] = value_label
        
        layout.addWidget(test_group)
        layout.addStretch()
        
        return widget
    
    def create_comparison_statistics_tab(self) -> QWidget:
        """Create the comparison statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Comparison statistics group
        comp_group = QGroupBox("Comparison Statistics")
        comp_layout = QGridLayout(comp_group)
        
        self.comp_stats_labels = {}
        comp_stats = [
            ("Aligned Data Points", "aligned_count"),
            ("Correlation Coefficient", "correlation"),
            ("R-squared", "r_squared"),
            ("Mean Absolute Error", "mae"),
            ("Root Mean Square Error", "rmse"),
            ("Bias (Mean Difference)", "bias"),
            ("Max Absolute Error", "max_error")
        ]
        
        for i, (label_text, key) in enumerate(comp_stats):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            
            comp_layout.addWidget(label, i, 0)
            comp_layout.addWidget(value_label, i, 1)
            self.comp_stats_labels[key] = value_label
        
        layout.addWidget(comp_group)
        layout.addStretch()
        
        return widget
    
    def populate_metadata(self):
        """Populate all metadata information"""
        self.populate_basic_info()
        self.populate_statistics()
        self.populate_comparison_statistics()
    
    def populate_basic_info(self):
        """Populate basic information tab"""
        # Pair information
        self.basic_labels["pair_name"].setText(self.pair.name or "Unnamed Pair")
        self.basic_labels["pair_id"].setText(self.pair.pair_id or "Unknown")
        
        # Reference channel information
        ref_channel_name = self.pair.ref_channel_name or "Unknown"
        self.basic_labels["ref_channel"].setText(ref_channel_name)
        
        # Get reference file name
        ref_file_name = "Unknown"
        if self.pair.ref_file_id and self.file_manager:
            ref_file = self.file_manager.get_file(self.pair.ref_file_id)
            if ref_file:
                ref_file_name = ref_file.filename
        self.basic_labels["ref_file"].setText(ref_file_name)
        
        # Test channel information
        test_channel_name = self.pair.test_channel_name or "Unknown"
        self.basic_labels["test_channel"].setText(test_channel_name)
        
        # Get test file name
        test_file_name = "Unknown"
        if self.pair.test_file_id and self.file_manager:
            test_file = self.file_manager.get_file(self.pair.test_file_id)
            if test_file:
                test_file_name = test_file.filename
        self.basic_labels["test_file"].setText(test_file_name)
        
        # Alignment information
        if self.pair.alignment_config:
            alignment_method = self.pair.alignment_config.method.value
            self.basic_labels["alignment_method"].setText(alignment_method)
        else:
            self.basic_labels["alignment_method"].setText("Unknown")
        
        # Dates
        self.basic_labels["created_date"].setText(
            self.pair.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.pair.created_at else "Unknown"
        )
        self.basic_labels["modified_date"].setText(
            self.pair.modified_at.strftime("%Y-%m-%d %H:%M:%S") if self.pair.modified_at else "Unknown"
        )
        
        # Memory and format
        total_memory = 0
        data_format = "Unknown"
        
        if self.pair.aligned_ref_data is not None and self.pair.aligned_test_data is not None:
            ref_memory = self.pair.aligned_ref_data.nbytes
            test_memory = self.pair.aligned_test_data.nbytes
            total_memory = ref_memory + test_memory
            data_format = f"NumPy {self.pair.aligned_ref_data.dtype}"
        
        self.basic_labels["total_memory"].setText(f"{total_memory / 1024:.1f} KB")
        self.basic_labels["data_format"].setText(data_format)
    
    def populate_statistics(self):
        """Populate statistics tab"""
        # Reference channel statistics
        if self.pair.aligned_ref_data is not None and len(self.pair.aligned_ref_data) > 0:
            ref_data = self.pair.aligned_ref_data
            ref_clean = ref_data[~np.isnan(ref_data)] if np.any(np.isnan(ref_data)) else ref_data
            
            if len(ref_clean) > 0:
                self.ref_stats_labels["ref_count"].setText(f"{len(ref_clean):,}")
                self.ref_stats_labels["ref_min"].setText(f"{np.min(ref_clean):.6g}")
                self.ref_stats_labels["ref_max"].setText(f"{np.max(ref_clean):.6g}")
                self.ref_stats_labels["ref_mean"].setText(f"{np.mean(ref_clean):.6g}")
                self.ref_stats_labels["ref_median"].setText(f"{np.median(ref_clean):.6g}")
                self.ref_stats_labels["ref_std"].setText(f"{np.std(ref_clean):.6g}")
                self.ref_stats_labels["ref_q25"].setText(f"{np.percentile(ref_clean, 25):.6g}")
                self.ref_stats_labels["ref_q75"].setText(f"{np.percentile(ref_clean, 75):.6g}")
            else:
                for key in self.ref_stats_labels:
                    self.ref_stats_labels[key].setText("No valid data")
        else:
            for key in self.ref_stats_labels:
                self.ref_stats_labels[key].setText("No data")
        
        # Test channel statistics
        if self.pair.aligned_test_data is not None and len(self.pair.aligned_test_data) > 0:
            test_data = self.pair.aligned_test_data
            test_clean = test_data[~np.isnan(test_data)] if np.any(np.isnan(test_data)) else test_data
            
            if len(test_clean) > 0:
                self.test_stats_labels["test_count"].setText(f"{len(test_clean):,}")
                self.test_stats_labels["test_min"].setText(f"{np.min(test_clean):.6g}")
                self.test_stats_labels["test_max"].setText(f"{np.max(test_clean):.6g}")
                self.test_stats_labels["test_mean"].setText(f"{np.mean(test_clean):.6g}")
                self.test_stats_labels["test_median"].setText(f"{np.median(test_clean):.6g}")
                self.test_stats_labels["test_std"].setText(f"{np.std(test_clean):.6g}")
                self.test_stats_labels["test_q25"].setText(f"{np.percentile(test_clean, 25):.6g}")
                self.test_stats_labels["test_q75"].setText(f"{np.percentile(test_clean, 75):.6g}")
            else:
                for key in self.test_stats_labels:
                    self.test_stats_labels[key].setText("No data")
        else:
            for key in self.test_stats_labels:
                self.test_stats_labels[key].setText("No data")
    
    def populate_comparison_statistics(self):
        """Populate comparison statistics tab"""
        # Comparison statistics
        if (self.pair.aligned_ref_data is not None and self.pair.aligned_test_data is not None and
            len(self.pair.aligned_ref_data) > 0 and len(self.pair.aligned_test_data) > 0):
            
            ref_data = self.pair.aligned_ref_data
            test_data = self.pair.aligned_test_data
            
            # Find valid pairs (both non-NaN)
            valid_mask = ~(np.isnan(ref_data) | np.isnan(test_data))
            ref_valid = ref_data[valid_mask]
            test_valid = test_data[valid_mask]
            
            if len(ref_valid) > 0:
                self.comp_stats_labels["aligned_count"].setText(f"{len(ref_valid):,}")
                
                # Correlation coefficient
                correlation = np.corrcoef(ref_valid, test_valid)[0, 1]
                if not np.isnan(correlation):
                    self.comp_stats_labels["correlation"].setText(f"{correlation:.6f}")
                    self.comp_stats_labels["r_squared"].setText(f"{correlation**2:.6f}")
                else:
                    self.comp_stats_labels["correlation"].setText("N/A")
                    self.comp_stats_labels["r_squared"].setText("N/A")
                
                # Error metrics
                differences = test_valid - ref_valid
                mae = np.mean(np.abs(differences))
                rmse = np.sqrt(np.mean(differences**2))
                bias = np.mean(differences)
                max_error = np.max(np.abs(differences))
                
                self.comp_stats_labels["mae"].setText(f"{mae:.6g}")
                self.comp_stats_labels["rmse"].setText(f"{rmse:.6g}")
                self.comp_stats_labels["bias"].setText(f"{bias:.6g}")
                self.comp_stats_labels["max_error"].setText(f"{max_error:.6g}")
            else:
                for key in self.comp_stats_labels:
                    self.comp_stats_labels[key].setText("No valid pairs")
        else:
            for key in self.comp_stats_labels:
                self.comp_stats_labels[key].setText("No aligned data")


# Convenience function for opening the pair metadata wizard
def show_pair_metadata(pair: Pair, parent=None, file_manager=None, channel_manager=None):
    """
    Show the pair metadata wizard for a comparison pair
    
    Args:
        pair: Pair to analyze
        parent: Parent widget
        file_manager: File manager to get file information
        channel_manager: Channel manager to get channel information
    """
    wizard = PairMetadataWizard(pair, parent, file_manager, channel_manager)
    wizard.exec()