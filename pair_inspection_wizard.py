from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QHeaderView, QLineEdit, QSpinBox, QCheckBox,
    QGroupBox, QGridLayout, QMessageBox, QProgressBar, QComboBox,
    QDialogButtonBox, QSplitter, QTextEdit, QFrame, QWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QColor, QDoubleValidator, QIntValidator
import numpy as np
from typing import Optional, Tuple
import traceback
from datetime import datetime

from pair import Pair


class PairDataTableWidget(QTableWidget):
    """
    Custom table widget for displaying aligned pair data (read-only)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configure table appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.ExtendedSelection)
        
        # Set up headers
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Index", "Reference", "Test"])
        
        # Configure column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Index column
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # Reference column
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Test column
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Make table read-only
        self.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Store information about loaded subset
        self.is_subset_loaded = False
        self.total_data_length = 0
        self.first_n_rows = 0
        self.last_m_rows = 0
    
    def load_data(self, ref_data: np.ndarray, test_data: np.ndarray, first_n: int = 100, last_m: int = 100, load_all: bool = False, ref_label: str = "Reference", test_label: str = "Test"):
        """Load aligned pair data into the table with selective loading for large datasets"""
        if ref_data is None or test_data is None:
            return
        
        # Clear previous data
        self.clear()
        
        # Determine data length
        data_length = min(len(ref_data), len(test_data))
        self.total_data_length = data_length
        
        # Check if we should load all data or just a subset
        if load_all or data_length <= (first_n + last_m):
            # Load all data
            self.is_subset_loaded = False
            self.first_n_rows = 0
            self.last_m_rows = 0
            
            self.setRowCount(data_length)
            self.setHorizontalHeaderLabels(["Index", ref_label, test_label])
            
            # Populate table with all data
            for i in range(data_length):
                self._add_row(i, ref_data[i], test_data[i])
        else:
            # Load subset - first N and last M rows
            self.is_subset_loaded = True
            self.first_n_rows = first_n
            self.last_m_rows = last_m
            
            # Calculate actual rows to load
            actual_first_n = min(first_n, data_length)
            actual_last_m = min(last_m, data_length)
            
            # If there's overlap, adjust
            if actual_first_n + actual_last_m > data_length:
                actual_first_n = data_length
                actual_last_m = 0
            
            # Set table size
            loaded_rows = actual_first_n + actual_last_m
            if actual_first_n > 0 and actual_last_m > 0 and data_length > (actual_first_n + actual_last_m):
                loaded_rows += 1  # Add separator row
            
            self.setRowCount(loaded_rows)
            self.setHorizontalHeaderLabels(["Index", ref_label, test_label])
            
            row_index = 0
            
            # Load first N rows
            for i in range(actual_first_n):
                self._add_row(i, ref_data[i], test_data[i], row_index)
                row_index += 1
            
            # Add separator row if needed
            if actual_first_n > 0 and actual_last_m > 0 and data_length > (actual_first_n + actual_last_m):
                self._add_separator_row(row_index, data_length - actual_first_n - actual_last_m)
                row_index += 1
            
            # Load last M rows
            start_index = data_length - actual_last_m
            for i in range(start_index, data_length):
                self._add_row(i, ref_data[i], test_data[i], row_index)
                row_index += 1
    
    def _add_row(self, data_index: int, ref_value, test_value, table_row: int = None):
        """Add a single row to the table"""
        if table_row is None:
            table_row = data_index
        
        # Index column (read-only)
        index_item = QTableWidgetItem(str(data_index))
        index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
        index_item.setBackground(QColor(240, 240, 240))
        self.setItem(table_row, 0, index_item)
        
        # Helper function to format values safely
        def format_value(value):
            """Format a value safely, handling both numeric and string types"""
            try:
                # Try to convert to float and format as numeric
                float_val = float(value)
                if isinstance(value, (int, float, np.number)):
                    return f"{float_val:.8g}"
                else:
                    # For string values that can be converted to float, show both
                    return f"{value} ({float_val:.8g})"
            except (ValueError, TypeError):
                # If conversion fails, just return the string representation
                return str(value)
        
        def format_tooltip(value):
            """Format a value for tooltip display"""
            try:
                float_val = float(value)
                if isinstance(value, (int, float, np.number)):
                    return f"Value: {float_val:.12g}"
                else:
                    return f"Value: {value} (numeric: {float_val:.12g})"
            except (ValueError, TypeError):
                return f"Value: {value}"
        
        # Reference value (read-only)
        ref_display = format_value(ref_value)
        ref_tooltip = format_tooltip(ref_value)
        ref_item = QTableWidgetItem(ref_display)
        ref_item.setToolTip(ref_tooltip)
        ref_item.setFlags(ref_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(table_row, 1, ref_item)
        
        # Test value (read-only)
        test_display = format_value(test_value)
        test_tooltip = format_tooltip(test_value)
        test_item = QTableWidgetItem(test_display)
        test_item.setToolTip(test_tooltip)
        test_item.setFlags(test_item.flags() & ~Qt.ItemIsEditable)
        self.setItem(table_row, 2, test_item)
    
    def _add_separator_row(self, table_row: int, skipped_rows: int):
        """Add a separator row indicating skipped data"""
        # Create separator items
        separator_text = f"... {skipped_rows:,} rows skipped ..."
        
        for col in range(3):
            item = QTableWidgetItem(separator_text)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setBackground(QColor(220, 220, 220))
            item.setTextAlignment(Qt.AlignCenter)
            item.setFont(QFont("Arial", 9, QFont.Italic))
            self.setItem(table_row, col, item)


class PairInspectionWizard(QDialog):
    """
    Data inspection wizard specifically for aligned pair data
    """
    
    def __init__(self, pair: Pair, parent=None, file_manager=None, channel_manager=None):
        super().__init__(parent)
        self.pair = pair
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        
        self.setWindowTitle(f"Pair Data Inspector - {pair.name or 'Unnamed Pair'}")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self.init_ui()
        self.load_data()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title and info
        header_layout = QHBoxLayout()
        
        title = QLabel(f"Pair Data Inspector")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Pair info
        ref_label = self.pair.ref_channel_name or 'Unknown'
        test_label = self.pair.test_channel_name or 'Unknown'
        info_label = QLabel(f"Pair: {ref_label} vs {test_label}")
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(info_label)
        
        layout.addLayout(header_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: Controls and statistics
        controls_widget = self.create_controls_section()
        splitter.addWidget(controls_widget)
        
        # Bottom section: Data table
        table_widget = self.create_table_section()
        splitter.addWidget(table_widget)
        
        # Set splitter proportions (1:3 ratio)
        splitter.setSizes([150, 450])
        layout.addWidget(splitter)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        # Left side buttons
        self.export_button = QPushButton("Export")
        self.export_button.setToolTip("Export aligned pair data to file")
        button_layout.addWidget(self.export_button)
        
        button_layout.addStretch()
        
        # Right side buttons
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
    
    def create_controls_section(self) -> QWidget:
        """Create the controls section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Data Loading group
        loading_group = QGroupBox("Data Loading")
        loading_layout = QVBoxLayout(loading_group)
        
        # Row selection controls
        row_controls_layout = QHBoxLayout()
        
        # First N rows control
        row_controls_layout.addWidget(QLabel("First:"))
        self.first_n_spin = QSpinBox()
        self.first_n_spin.setRange(1, 10000)
        self.first_n_spin.setValue(100)
        self.first_n_spin.setSuffix(" rows")
        self.first_n_spin.setToolTip("Number of rows to load from the beginning")
        row_controls_layout.addWidget(self.first_n_spin)
        
        # Last M rows control
        row_controls_layout.addWidget(QLabel("Last:"))
        self.last_m_spin = QSpinBox()
        self.last_m_spin.setRange(1, 10000)
        self.last_m_spin.setValue(100)
        self.last_m_spin.setSuffix(" rows")
        self.last_m_spin.setToolTip("Number of rows to load from the end")
        row_controls_layout.addWidget(self.last_m_spin)
        
        # Reload button
        self.reload_button = QPushButton("Reload")
        self.reload_button.setToolTip("Reload data with current settings")
        self.reload_button.clicked.connect(self.reload_data_with_settings)
        row_controls_layout.addWidget(self.reload_button)
        
        loading_layout.addLayout(row_controls_layout)
        
        # Load all checkbox with warning
        load_all_layout = QHBoxLayout()
        self.load_all_checkbox = QCheckBox("Load all rows (may be slow for large files)")
        self.load_all_checkbox.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.load_all_checkbox.setToolTip("WARNING: Loading all rows may take a long time for large datasets.\nRecommended only for files with < 10,000 rows.")
        self.load_all_checkbox.toggled.connect(self.on_load_all_toggled)
        load_all_layout.addWidget(self.load_all_checkbox)
        load_all_layout.addStretch()
        
        loading_layout.addLayout(load_all_layout)
        
        layout.addWidget(loading_group)
        
        # Search group
        search_group = QGroupBox("Search & Navigation")
        search_layout = QVBoxLayout(search_group)
        
        # Row navigation
        row_nav_layout = QHBoxLayout()
        
        self.goto_row_input = QSpinBox()
        self.goto_row_input.setMinimum(0)
        self.goto_row_input.setMaximum(0)  # Will be updated when data loads
        self.goto_row_input.setToolTip("Go to specific row")
        row_nav_layout.addWidget(QLabel("Go to row:"))
        row_nav_layout.addWidget(self.goto_row_input)
        
        self.goto_button = QPushButton("Go")
        self.goto_button.clicked.connect(self.goto_row)
        row_nav_layout.addWidget(self.goto_button)
        
        search_layout.addLayout(row_nav_layout)
        
        # Search functionality
        search_layout_controls = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search values...")
        self.search_input.setToolTip("Search for specific values in reference or test data")
        search_layout_controls.addWidget(QLabel("Search:"))
        search_layout_controls.addWidget(self.search_input)
        
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_data)
        search_layout_controls.addWidget(self.search_button)
        
        search_layout.addLayout(search_layout_controls)
        
        layout.addWidget(search_group)
        
        return widget
    
    def create_table_section(self) -> QWidget:
        """Create the data table section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table header with help text
        header_layout = QHBoxLayout()
        table_header = QLabel("Aligned Pair Data (Read-Only)")
        table_header.setStyleSheet("font-weight: bold; font-size: 12px; margin: 5px;")
        header_layout.addWidget(table_header)
        
        header_layout.addStretch()
        
        # Help text for large files
        help_label = QLabel("For large files, only first/last rows are shown by default")
        help_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        header_layout.addWidget(help_label)
        
        layout.addLayout(header_layout)
        
        # Data table
        self.data_table = PairDataTableWidget()
        layout.addWidget(self.data_table)
        
        # Table status
        self.table_status = QLabel("Ready")
        self.table_status.setStyleSheet("color: #666; font-size: 10px; margin: 2px; padding: 3px; background-color: #f5f5f5; border-radius: 3px;")
        layout.addWidget(self.table_status)
        
        return widget
    
    def connect_signals(self):
        """Connect all signals"""
        self.export_button.clicked.connect(self.export_data)
        self.close_button.clicked.connect(self.accept)
        
        # Search as you type
        self.search_input.textChanged.connect(self.search_data)
    
    def load_data(self):
        """Load aligned pair data into the table"""
        try:
            # Get aligned data from the pair
            ref_data = self.pair.aligned_ref_data
            test_data = self.pair.aligned_test_data
            
            if ref_data is None or test_data is None:
                self.table_status.setText("No aligned data available")
                QMessageBox.warning(self, "No Data", "This pair has no aligned data to inspect.")
                return
            
            # Check data size and determine loading strategy
            data_length = len(ref_data)
            
            if data_length > 1000:
                load_all = False
                first_n = self.first_n_spin.value()
                last_m = self.last_m_spin.value()
                
                if data_length > 10000:
                    self.table_status.setText(f"⚠️ Large aligned dataset ({data_length:,} rows). Loading first {first_n} and last {last_m} rows for performance.")
                else:
                    self.table_status.setText(f"Loading first {first_n} and last {last_m} rows of {data_length:,} aligned rows.")
            else:
                load_all = True
                first_n = data_length
                last_m = 0
                self.table_status.setText(f"Loaded {data_length:,} aligned data points")
            
            # Load data with appropriate strategy
            ref_label = self.pair.ref_channel_name or "Reference"
            test_label = self.pair.test_channel_name or "Test"
            
            self.data_table.load_data(
                ref_data, 
                test_data, 
                first_n=first_n, 
                last_m=last_m, 
                load_all=load_all,
                ref_label=ref_label,
                test_label=test_label
            )
            
            # Update controls
            self.goto_row_input.setMaximum(data_length - 1)
            
        except Exception as e:
            self.table_status.setText(f"Error loading pair data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load pair data:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def reload_data_with_settings(self):
        """Reload data with current UI settings"""
        self.load_data()
    
    def on_load_all_toggled(self, checked):
        """Handle load all checkbox toggle"""
        if checked:
            # Show confirmation dialog for large files
            data_length = 0
            if self.pair.aligned_ref_data is not None:
                data_length = len(self.pair.aligned_ref_data)
            
            if data_length > 10000:
                reply = QMessageBox.question(
                    self,
                    "Load All Data Warning",
                    f"You are about to load {data_length:,} rows.\n"
                    "This may take a long time and use significant memory.\n"
                    "Are you sure you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.No:
                    self.load_all_checkbox.setChecked(False)
                    return
            
            # Disable row controls when loading all
            self.first_n_spin.setEnabled(False)
            self.last_m_spin.setEnabled(False)
        else:
            # Re-enable row controls
            self.first_n_spin.setEnabled(True)
            self.last_m_spin.setEnabled(True)
    
    def goto_row(self):
        """Go to specific row in the table"""
        row = self.goto_row_input.value()
        if 0 <= row < self.data_table.rowCount():
            self.data_table.selectRow(row)
            self.data_table.scrollToItem(self.data_table.item(row, 0))
    
    def search_data(self):
        """Search for specific values in the data"""
        search_text = self.search_input.text().strip()
        if not search_text:
            # Clear selection if search is empty
            self.data_table.clearSelection()
            return
        
        # Find matching rows
        matching_rows = []
        for row in range(self.data_table.rowCount()):
            for col in [1, 2]:  # Reference and Test columns
                item = self.data_table.item(row, col)
                if item and search_text.lower() in item.text().lower():
                    matching_rows.append(row)
                    break
        
        # Highlight matching rows
        self.data_table.clearSelection()
        for row in matching_rows:
            self.data_table.selectRow(row)
        
        # Scroll to first match
        if matching_rows:
            self.data_table.scrollToItem(self.data_table.item(matching_rows[0], 0))
            self.table_status.setText(f"Found {len(matching_rows)} matches")
        else:
            self.table_status.setText("No matches found")
    
    def export_data(self):
        """Export aligned pair data to file"""
        try:
            from PySide6.QtWidgets import QFileDialog
            import pandas as pd
            
            # Get file path
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Pair Data",
                f"{self.pair.name or 'pair_data'}.csv",
                "CSV Files (*.csv);;Text Files (*.txt)"
            )
            
            if file_path:
                # Get aligned data
                ref_data = self.pair.aligned_ref_data
                test_data = self.pair.aligned_test_data
                
                if ref_data is not None and test_data is not None:
                    # Create DataFrame
                    df = pd.DataFrame({
                        'Index': range(len(ref_data)),
                        'Reference': ref_data,
                        'Test': test_data
                    })
                    
                    # Export
                    if file_path.endswith('.csv'):
                        df.to_csv(file_path, index=False)
                    else:
                        df.to_csv(file_path, index=False, sep='\t')
                    
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Successfully exported {len(ref_data):,} data points to:\n{file_path}"
                    )
                else:
                    QMessageBox.warning(self, "No Data", "No aligned data available for export.")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export data:\n{str(e)}"
            )


# Convenience function for opening the pair inspection wizard
def inspect_pair_data(pair: Pair, parent=None, file_manager=None, channel_manager=None):
    """
    Show the pair inspection wizard for aligned pair data
    
    Args:
        pair: Pair to inspect
        parent: Parent widget
        file_manager: File manager to get file information
        channel_manager: Channel manager to get channel information
    """
    wizard = PairInspectionWizard(pair, parent, file_manager, channel_manager)
    wizard.exec()