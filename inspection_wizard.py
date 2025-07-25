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

from channel import Channel


class DataTableWidget(QTableWidget):
    """
    Custom table widget for displaying and editing X,Y data with enhanced features
    """
    
    data_changed = Signal()  # Emitted when data is modified
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_data = None
        self.modified_rows = set()
        
        # Configure table appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.ExtendedSelection)
        
        # Set up headers
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Index", "X Value", "Y Value"])
        
        # Configure column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Index column
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # X column
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Y column
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Connect item changed signal
        self.itemChanged.connect(self._on_item_changed)
        
        # Store information about loaded subset
        self.is_subset_loaded = False
        self.total_data_length = 0
        self.first_n_rows = 0
        self.last_m_rows = 0
    
    def set_column_headers(self, headers):
        """Set the column headers dynamically"""
        if len(headers) == 3:
            self.setHorizontalHeaderLabels(headers)
    
    def load_data(self, x_data: np.ndarray, y_data: np.ndarray, first_n: int = 100, last_m: int = 100, load_all: bool = False, x_label: str = "X Value", y_label: str = "Y Value"):
        """Load X,Y data into the table with selective loading for large datasets"""
        if x_data is None or y_data is None:
            return
        
        # Store original data for comparison
        self.original_data = (x_data.copy(), y_data.copy())
        
        # Clear previous data
        self.clear()
        self.modified_rows.clear()
        
        # Determine data length
        data_length = min(len(x_data), len(y_data))
        self.total_data_length = data_length
        
        # Check if we should load all data or just a subset
        if load_all or data_length <= (first_n + last_m):
            # Load all data
            self.is_subset_loaded = False
            self.first_n_rows = 0
            self.last_m_rows = 0
            
            self.setRowCount(data_length)
            self.setHorizontalHeaderLabels(["Index", x_label, y_label])
            
            # Populate table with all data
            for i in range(data_length):
                self._add_row(i, x_data[i], y_data[i])
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
            self.setHorizontalHeaderLabels(["Index", x_label, y_label])
            
            row_index = 0
            
            # Load first N rows
            for i in range(actual_first_n):
                self._add_row(i, x_data[i], y_data[i], row_index)
                row_index += 1
            
            # Add separator row if needed
            if actual_first_n > 0 and actual_last_m > 0 and data_length > (actual_first_n + actual_last_m):
                self._add_separator_row(row_index, data_length - actual_first_n - actual_last_m)
                row_index += 1
            
            # Load last M rows
            start_index = data_length - actual_last_m
            for i in range(start_index, data_length):
                self._add_row(i, x_data[i], y_data[i], row_index)
                row_index += 1
    
    def _add_row(self, data_index: int, x_value, y_value, table_row: int = None):
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
                    return f"Original: {float_val:.12g}"
                else:
                    return f"Original: {value} (numeric: {float_val:.12g})"
            except (ValueError, TypeError):
                return f"Original: {value}"
        
        # X value (editable)
        x_display = format_value(x_value)
        x_tooltip = format_tooltip(x_value)
        x_item = QTableWidgetItem(x_display)
        x_item.setToolTip(x_tooltip)
        x_item.setData(Qt.UserRole, data_index)  # Store original index
        self.setItem(table_row, 1, x_item)
        
        # Y value (editable)
        y_display = format_value(y_value)
        y_tooltip = format_tooltip(y_value)
        y_item = QTableWidgetItem(y_display)
        y_item.setToolTip(y_tooltip)
        y_item.setData(Qt.UserRole, data_index)  # Store original index
        self.setItem(table_row, 2, y_item)
    
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
    
    def _on_item_changed(self, item):
        """Handle when an item in the table is changed"""
        if item.column() in [1, 2]:  # X or Y column
            # Skip if this is a separator row
            if "skipped" in item.text():
                return
            
            row = item.row()
            self.modified_rows.add(row)
            
            # Highlight modified row
            for col in range(self.columnCount()):
                cell_item = self.item(row, col)
                if cell_item:
                    cell_item.setBackground(QColor(255, 255, 200))  # Light yellow
            
            # Validate numeric input
            try:
                float(item.text())
                item.setBackground(QColor(255, 255, 200))  # Valid - light yellow
            except ValueError:
                item.setBackground(QColor(255, 200, 200))  # Invalid - light red
                item.setToolTip("Invalid numeric value")
            
            self.data_changed.emit()
    
    def get_modified_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current data from the table, applying modifications to full dataset"""
        if self.original_data is None:
            return None, None
        
        # Start with original data
        x_data = self.original_data[0].copy()
        y_data = self.original_data[1].copy()
        
        # Apply modifications from the table
        for row in range(self.rowCount()):
            x_item = self.item(row, 1)
            y_item = self.item(row, 2)
            
            if x_item and y_item and "skipped" not in x_item.text():
                try:
                    # Get the original data index
                    original_index = x_item.data(Qt.UserRole)
                    if original_index is not None:
                        x_val = float(x_item.text())
                        y_val = float(y_item.text())
                        x_data[original_index] = x_val
                        y_data[original_index] = y_val
                except (ValueError, IndexError, TypeError):
                    # Skip invalid data
                    continue
        
        return x_data, y_data
    
    def has_modifications(self) -> bool:
        """Check if any data has been modified"""
        return len(self.modified_rows) > 0
    
    def reset_modifications(self):
        """Reset all modifications and restore original data"""
        if self.original_data is not None:
            # Preserve current loading settings
            self.load_data(
                self.original_data[0], 
                self.original_data[1], 
                first_n=self.first_n_rows if self.is_subset_loaded else 100,
                last_m=self.last_m_rows if self.is_subset_loaded else 100,
                load_all=not self.is_subset_loaded
            )


class InspectionWizard(QDialog):
    """
    Data inspection and editing wizard for channel X,Y data or comparison pair data
    """
    
    data_updated = Signal(str)  # Emitted when channel data is updated (channel_id)
    
    def __init__(self, data_source, parent=None):
        super().__init__(parent)
        self.data_source = data_source
        self.has_unsaved_changes = False
        
        # Determine if this is a channel or pair
        self.is_pair_data = isinstance(data_source, dict) and 'ref_channel' in data_source
        self.channel = data_source if not self.is_pair_data else None
        self.pair_config = data_source if self.is_pair_data else None
        
        # Set window title based on data type
        if self.is_pair_data:
            pair_name = data_source.get('name', 'Unnamed Pair')
            self.setWindowTitle(f"Pair Data Inspector - {pair_name}")
        else:
            self.setWindowTitle(f"Data Inspector - {data_source.ylabel or 'Unnamed Channel'}")
        
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
        
        title = QLabel(f"🔍 Data Inspector")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Data source info
        if self.is_pair_data:
            ref_label = self.pair_config.get('ref_channel', 'Unknown')
            test_label = self.pair_config.get('test_channel', 'Unknown')
            info_label = QLabel(f"Pair: {ref_label} vs {test_label}")
        else:
            info_label = QLabel(f"Channel: {self.channel.ylabel or 'Unnamed'}")
        
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
        if not self.is_pair_data:
            self.reset_button = QPushButton("🔄 Reset")
            self.reset_button.setToolTip("Reset all changes to original data")
            button_layout.addWidget(self.reset_button)
        
        self.export_button = QPushButton("💾 Export")
        self.export_button.setToolTip("Export current data to file")
        button_layout.addWidget(self.export_button)
        
        button_layout.addStretch()
        
        # Right side buttons
        if not self.is_pair_data:
            self.apply_button = QPushButton("✅ Apply Changes")
            self.apply_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.apply_button.setEnabled(False)
            button_layout.addWidget(self.apply_button)
        
        self.cancel_button = QPushButton("❌ Cancel")
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def create_controls_section(self) -> QWidget:
        """Create the controls and statistics section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Data loading controls group
        loading_group = QGroupBox("Data Loading Options")
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
        self.reload_button = QPushButton("🔄 Reload")
        self.reload_button.setToolTip("Reload data with current settings")
        self.reload_button.clicked.connect(self.reload_data_with_settings)
        row_controls_layout.addWidget(self.reload_button)
        
        loading_layout.addLayout(row_controls_layout)
        
        # Load all checkbox with warning
        load_all_layout = QHBoxLayout()
        self.load_all_checkbox = QCheckBox("⚠️ Load all rows (may be slow for large files)")
        self.load_all_checkbox.setStyleSheet("color: #d32f2f; font-weight: bold;")
        self.load_all_checkbox.setToolTip("WARNING: Loading all rows may take a long time for large datasets.\nRecommended only for files with < 10,000 rows.")
        self.load_all_checkbox.toggled.connect(self.on_load_all_toggled)
        load_all_layout.addWidget(self.load_all_checkbox)
        load_all_layout.addStretch()
        
        loading_layout.addLayout(load_all_layout)
        
        layout.addWidget(loading_group)
        
        # Data statistics group
        stats_group = QGroupBox("Data Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.stats_labels = {}
        
        # Adapt statistics labels based on data type
        if self.is_pair_data:
            stats_info = [
                ("Total Points", "total_points"),
                ("Loaded Rows", "loaded_rows"),
                ("Reference Range", "ref_range"),
                ("Test Range", "test_range"),
                ("Invalid Values", "invalid_values")
            ]
        else:
            stats_info = [
                ("Total Points", "total_points"),
                ("Loaded Rows", "loaded_rows"),
                ("X Range", "x_range"),
                ("Y Range", "y_range"),
                ("Modified Points", "modified_points"),
                ("Invalid Values", "invalid_values")
            ]
        
        for i, (label_text, key) in enumerate(stats_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            value_label.setStyleSheet("color: #666;")
            
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value_label, i, 1)
            self.stats_labels[key] = value_label
        
        layout.addWidget(stats_group)
        
        # Data operations group
        ops_group = QGroupBox("Data Operations")
        ops_layout = QVBoxLayout(ops_group)
        
        # Row operations
        row_ops_layout = QHBoxLayout()
        
        self.goto_row_input = QSpinBox()
        self.goto_row_input.setMinimum(0)
        self.goto_row_input.setMaximum(0)  # Will be updated when data loads
        self.goto_row_input.setToolTip("Go to specific row")
        row_ops_layout.addWidget(QLabel("Go to row:"))
        row_ops_layout.addWidget(self.goto_row_input)
        
        self.goto_button = QPushButton("Go")
        self.goto_button.clicked.connect(self.goto_row)
        row_ops_layout.addWidget(self.goto_button)
        
        ops_layout.addLayout(row_ops_layout)
        
        # Search operations
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search values...")
        self.search_input.setToolTip("Search for specific values in X or Y data")
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("🔍")
        self.search_button.clicked.connect(self.search_data)
        search_layout.addWidget(self.search_button)
        
        ops_layout.addLayout(search_layout)
        
        # Filter operations
        filter_layout = QHBoxLayout()
        
        if not self.is_pair_data:
            self.show_modified_only = QCheckBox("Show modified only")
            self.show_modified_only.toggled.connect(self.filter_table)
            filter_layout.addWidget(self.show_modified_only)
        
        self.show_invalid_only = QCheckBox("Show invalid only")
        self.show_invalid_only.toggled.connect(self.filter_table)
        filter_layout.addWidget(self.show_invalid_only)
        
        ops_layout.addLayout(filter_layout)
        
        layout.addWidget(ops_group)
        
        return widget
    
    def create_table_section(self) -> QWidget:
        """Create the data table section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table header with help text
        header_layout = QHBoxLayout()
        if self.is_pair_data:
            table_header = QLabel("📊 Aligned Pair Data (Read-Only)")
        else:
            table_header = QLabel("📊 Channel Data (Editable)")
        table_header.setStyleSheet("font-weight: bold; font-size: 12px; margin: 5px;")
        header_layout.addWidget(table_header)
        
        header_layout.addStretch()
        
        # Help text for large files
        help_label = QLabel("💡 For large files, only first/last rows are shown by default")
        help_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic;")
        header_layout.addWidget(help_label)
        
        layout.addLayout(header_layout)
        
        # Data table
        self.data_table = DataTableWidget()
        layout.addWidget(self.data_table)
        
        # Table status
        self.table_status = QLabel("Ready")
        self.table_status.setStyleSheet("color: #666; font-size: 10px; margin: 2px; padding: 3px; background-color: #f5f5f5; border-radius: 3px;")
        layout.addWidget(self.table_status)
        
        return widget
    
    def connect_signals(self):
        """Connect all signals"""
        self.data_table.data_changed.connect(self.on_data_changed)
        if not self.is_pair_data:
            self.reset_button.clicked.connect(self.reset_data)
        self.export_button.clicked.connect(self.export_data)
        if not self.is_pair_data:
            self.apply_button.clicked.connect(self.apply_changes)
        self.cancel_button.clicked.connect(self.cancel_changes)
        
        # Search as you type
        self.search_input.textChanged.connect(self.search_data)
        
        # Loading controls signals (already connected in create_controls_section)
        # self.reload_button.clicked.connect(self.reload_data_with_settings) - connected in create_controls_section
        # self.load_all_checkbox.toggled.connect(self.on_load_all_toggled) - connected in create_controls_section
    
    def load_data(self):
        """Load data into the table - handles both channel and pair data"""
        if self.is_pair_data:
            self._load_pair_data()
        else:
            self._load_channel_data()
    
    def _load_channel_data(self):
        """Load the channel's X,Y data into the table with selective loading for large files"""
        if self.channel.xdata is not None and self.channel.ydata is not None:
            # Check file size and determine loading strategy
            data_length = len(self.channel.ydata)
            
            # For very large files, show warning and use selective loading
            if data_length > 1000:
                load_all = False
                first_n = self.first_n_spin.value()
                last_m = self.last_m_spin.value()
                
                # Display warning for large files
                if data_length > 10000:
                    self.table_status.setText(f"⚠️ Large file detected ({data_length:,} rows). Loading first {first_n} and last {last_m} rows for performance.")
                else:
                    self.table_status.setText(f"Loading first {first_n} and last {last_m} rows of {data_length:,} total rows.")
            else:
                # Small files - load all data
                load_all = True
                first_n = data_length
                last_m = 0
                self.table_status.setText(f"Loaded {data_length:,} data points")
            
            # Load data with appropriate strategy
            self.data_table.load_data(
                self.channel.xdata, 
                self.channel.ydata, 
                first_n=first_n, 
                last_m=last_m, 
                load_all=load_all,
                x_label="X Value",
                y_label="Y Value"
            )
            
            # Update controls
            self.goto_row_input.setMaximum(data_length - 1)
            
            # Update statistics
            self.update_statistics()
            
        else:
            self.table_status.setText("No data available")
            QMessageBox.warning(self, "No Data", "This channel has no data to inspect.")
    
    def _load_pair_data(self):
        """Load aligned pair data into the table"""
        print(f"[InspectionWizard] Loading pair data for: {self.pair_config.get('name', 'Unknown')}")
        try:
            # Get the aligned data from the comparison manager
            from comparison_wizard_manager import ComparisonWizardManager
            
            # We need to get the aligned data from the comparison manager
            # For now, we'll try to get it from the parent window if available
            parent_window = self.parent()
            aligned_data = None
            
            print(f"[InspectionWizard] Parent window has comparison_manager: {hasattr(parent_window, 'comparison_manager')}")
            
            if hasattr(parent_window, 'comparison_manager') and parent_window.comparison_manager:
                # Try to get aligned data from the manager
                ref_channel = parent_window.comparison_manager._get_channel(
                    self.pair_config['ref_file'], 
                    self.pair_config['ref_channel']
                )
                test_channel = parent_window.comparison_manager._get_channel(
                    self.pair_config['test_file'], 
                    self.pair_config['test_channel']
                )
                
                print(f"[InspectionWizard] Found ref_channel: {ref_channel is not None}, test_channel: {test_channel is not None}")
                
                if ref_channel and test_channel:
                    # Use DataAligner instead of legacy alignment method
                    from data_aligner import DataAligner
                    from pair import AlignmentConfig, AlignmentMethod
                    
                    data_aligner = DataAligner()
                    alignment_config = AlignmentConfig(
                        method=AlignmentMethod.INDEX,
                        start_index=0,
                        end_index=500
                    )
                    
                    alignment_result = data_aligner.align_channels(
                        ref_channel, test_channel, alignment_config
                    )
                    
                    if alignment_result.success:
                        aligned_data = {
                            'ref_data': alignment_result.ref_data,
                            'test_data': alignment_result.test_data,
                            'ref_label': ref_channel.legend_label or ref_channel.ylabel,
                            'test_label': test_channel.legend_label or test_channel.ylabel,
                            'alignment_method': alignment_config.method.value,
                            'n_points': len(alignment_result.ref_data)
                        }
                    else:
                        print(f"[InspectionWizard] Alignment failed: {alignment_result.error_message}")
                        return
                    print(f"[InspectionWizard] Aligned data created: {aligned_data is not None}")
            
            if aligned_data is None:
                # Fallback: try to get from cache or recreate
                if hasattr(parent_window, 'comparison_manager') and parent_window.comparison_manager:
                    pair_name = self.pair_config['name']
                    cached_result = parent_window.comparison_manager.pair_cache.get_pair_result(pair_name)
                    if cached_result:
                        aligned_data = cached_result.computation_result.get('aligned_data')
                        print(f"[InspectionWizard] Retrieved from cache: {aligned_data is not None}")
                
                if aligned_data is None:
                    # Last resort: show error
                    self.table_status.setText("Could not retrieve aligned data")
                    QMessageBox.warning(self, "Data Error", "Could not retrieve aligned data for this pair.")
                    return
            
            # Load the aligned data into the table
            ref_data = aligned_data.get('ref_data', [])
            test_data = aligned_data.get('test_data', [])
            
            print(f"[InspectionWizard] Data lengths - ref: {len(ref_data)}, test: {len(test_data)}")
            
            if len(ref_data) == 0 or len(test_data) == 0:
                self.table_status.setText("No aligned data available")
                QMessageBox.warning(self, "No Data", "No aligned data available for this pair.")
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
            ref_label = aligned_data.get('ref_label', 'Reference')
            test_label = aligned_data.get('test_label', 'Test')
            
            print(f"[InspectionWizard] Loading data with labels: {ref_label}, {test_label}")
            
            self.data_table.load_data(
                ref_data, 
                test_data, 
                first_n=first_n, 
                last_m=last_m, 
                load_all=load_all,
                x_label=ref_label,
                y_label=test_label
            )
            
            # Make table read-only for pair data
            self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
            
            # Update controls
            self.goto_row_input.setMaximum(data_length - 1)
            
            # Store aligned data for later use
            self.aligned_data = aligned_data
            
            print(f"[InspectionWizard] About to call update_statistics()")
            
            # Update statistics
            self.update_statistics()
            
            print(f"[InspectionWizard] Pair data loading complete")
            
        except Exception as e:
            self.table_status.setText(f"Error loading pair data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load pair data:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def reload_data_with_settings(self):
        """Reload data with current UI settings"""
        if self.is_pair_data:
            self._reload_pair_data_with_settings()
        else:
            self._reload_channel_data_with_settings()
    
    def _reload_channel_data_with_settings(self):
        """Reload channel data with current UI settings"""
        if self.channel.xdata is not None and self.channel.ydata is not None:
            first_n = self.first_n_spin.value()
            last_m = self.last_m_spin.value()
            load_all = self.load_all_checkbox.isChecked()
            
            self.data_table.load_data(
                self.channel.xdata, 
                self.channel.ydata, 
                first_n=first_n, 
                last_m=last_m, 
                load_all=load_all,
                x_label="X Value",
                y_label="Y Value"
            )
            
            # Update status
            if load_all:
                self.table_status.setText(f"Loaded all {len(self.channel.ydata):,} data points")
            else:
                self.table_status.setText(f"Loaded first {first_n} and last {last_m} rows of {len(self.channel.ydata):,} total rows")
            
            # Update statistics
            self.update_statistics()
    
    def _reload_pair_data_with_settings(self):
        """Reload pair data with current UI settings"""
        if hasattr(self, 'aligned_data') and self.aligned_data:
            first_n = self.first_n_spin.value()
            last_m = self.last_m_spin.value()
            load_all = self.load_all_checkbox.isChecked()
            
            ref_data = self.aligned_data.get('ref_data', [])
            test_data = self.aligned_data.get('test_data', [])
            
            ref_label = self.aligned_data.get('ref_label', 'Reference')
            test_label = self.aligned_data.get('test_label', 'Test')
            
            self.data_table.load_data(
                ref_data, 
                test_data, 
                first_n=first_n, 
                last_m=last_m, 
                load_all=load_all,
                x_label=ref_label,
                y_label=test_label
            )
            
            # Update status
            if load_all:
                self.table_status.setText(f"Loaded all {len(ref_data):,} aligned data points")
            else:
                self.table_status.setText(f"Loaded first {first_n} and last {last_m} rows of {len(ref_data):,} aligned rows")
            
            # Update statistics
            self.update_statistics()
        else:
            # If no aligned data available, try to reload from scratch
            self._load_pair_data()
    
    def on_load_all_toggled(self, checked):
        """Handle load all checkbox toggle"""
        if checked:
            # Show confirmation dialog for large files
            data_length = 0
            if self.is_pair_data and hasattr(self, 'aligned_data') and self.aligned_data:
                data_length = len(self.aligned_data.get('ref_data', []))
            elif not self.is_pair_data and self.channel.ydata is not None:
                data_length = len(self.channel.ydata)
            
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
    
    def update_statistics(self):
        """Update the statistics display"""
        print(f"[InspectionWizard] update_statistics called, is_pair_data: {self.is_pair_data}")
        if self.is_pair_data:
            self._update_pair_statistics()
        else:
            self._update_channel_statistics()
    
    def _update_channel_statistics(self):
        """Update statistics for channel data"""
        if self.channel.xdata is None or self.channel.ydata is None:
            return
        
        # Basic statistics
        total_points = len(self.channel.ydata)
        self.stats_labels["total_points"].setText(f"{total_points:,}")
        
        # Loaded rows
        loaded_rows = self.data_table.rowCount()
        if self.data_table.is_subset_loaded:
            # Subtract separator row if present
            separator_count = 0
            for row in range(self.data_table.rowCount()):
                item = self.data_table.item(row, 1)
                if item and "skipped" in item.text():
                    separator_count += 1
                    break
            actual_loaded = loaded_rows - separator_count
            self.stats_labels["loaded_rows"].setText(f"{actual_loaded:,} (subset)")
        else:
            self.stats_labels["loaded_rows"].setText(f"{loaded_rows:,} (all)")
        
        # Ranges (use full dataset for accurate ranges)
        x_min, x_max = np.min(self.channel.xdata), np.max(self.channel.xdata)
        y_min, y_max = np.min(self.channel.ydata), np.max(self.channel.ydata)
        
        self.stats_labels["x_range"].setText(f"{x_min:.6g} to {x_max:.6g}")
        self.stats_labels["y_range"].setText(f"{y_min:.6g} to {y_max:.6g}")
        
        # Modified points (only count visible modifications)
        modified_count = len(self.data_table.modified_rows)
        self.stats_labels["modified_points"].setText(f"{modified_count}")
        
        # Count invalid values (only in loaded data)
        invalid_count = 0
        for row in range(self.data_table.rowCount()):
            for col in [1, 2]:  # X and Y columns
                item = self.data_table.item(row, col)
                if item and "skipped" not in item.text():
                    try:
                        float(item.text())
                    except ValueError:
                        invalid_count += 1
        
        self.stats_labels["invalid_values"].setText(f"{invalid_count}")
    
    def _update_pair_statistics(self):
        """Update statistics for pair data"""
        print(f"[InspectionWizard] Updating pair statistics...")
        
        if not hasattr(self, 'aligned_data') or not self.aligned_data:
            print(f"[InspectionWizard] No aligned data available")
            return
        
        ref_data = self.aligned_data.get('ref_data', [])
        test_data = self.aligned_data.get('test_data', [])
        
        print(f"[InspectionWizard] Ref data length: {len(ref_data)}, Test data length: {len(test_data)}")
        
        if len(ref_data) == 0 or len(test_data) == 0:
            print(f"[InspectionWizard] Empty data arrays")
            return
        
        # Basic statistics
        total_points = len(ref_data)
        print(f"[InspectionWizard] Total points: {total_points}")
        
        if "total_points" in self.stats_labels:
            self.stats_labels["total_points"].setText(f"{total_points:,}")
        else:
            print(f"[InspectionWizard] Warning: total_points label not found")
        
        # Loaded rows
        loaded_rows = self.data_table.rowCount()
        if self.data_table.is_subset_loaded:
            # Subtract separator row if present
            separator_count = 0
            for row in range(self.data_table.rowCount()):
                item = self.data_table.item(row, 1)
                if item and "skipped" in item.text():
                    separator_count += 1
                    break
            actual_loaded = loaded_rows - separator_count
            if "loaded_rows" in self.stats_labels:
                self.stats_labels["loaded_rows"].setText(f"{actual_loaded:,} (subset)")
        else:
            if "loaded_rows" in self.stats_labels:
                self.stats_labels["loaded_rows"].setText(f"{loaded_rows:,} (all)")
        
        # Ranges for reference and test data
        ref_min, ref_max = np.min(ref_data), np.max(ref_data)
        test_min, test_max = np.min(test_data), np.max(test_data)
        
        print(f"[InspectionWizard] Ref range: {ref_min:.6g} to {ref_max:.6g}")
        print(f"[InspectionWizard] Test range: {test_min:.6g} to {test_max:.6g}")
        
        if "ref_range" in self.stats_labels:
            self.stats_labels["ref_range"].setText(f"{ref_min:.6g} to {ref_max:.6g}")
        else:
            print(f"[InspectionWizard] Warning: ref_range label not found")
            
        if "test_range" in self.stats_labels:
            self.stats_labels["test_range"].setText(f"{test_min:.6g} to {test_max:.6g}")
        else:
            print(f"[InspectionWizard] Warning: test_range label not found")
        
        # Count invalid values (only in loaded data)
        invalid_count = 0
        for row in range(self.data_table.rowCount()):
            for col in [1, 2]:  # Reference and Test columns
                item = self.data_table.item(row, col)
                if item and "skipped" not in item.text():
                    try:
                        float(item.text())
                    except ValueError:
                        invalid_count += 1
        
        print(f"[InspectionWizard] Invalid values: {invalid_count}")
        
        if "invalid_values" in self.stats_labels:
            self.stats_labels["invalid_values"].setText(f"{invalid_count}")
        else:
            print(f"[InspectionWizard] Warning: invalid_values label not found")
        
        print(f"[InspectionWizard] Available stats labels: {list(self.stats_labels.keys())}")
    
    def on_data_changed(self):
        """Handle when data in the table changes"""
        self.has_unsaved_changes = True
        if not self.is_pair_data:
            self.apply_button.setEnabled(True)
        self.update_statistics()
        
        # Update status
        modified_count = len(self.data_table.modified_rows)
        if modified_count > 0:
            self.table_status.setText(f"Modified: {modified_count} rows")
            if self.is_pair_data:
                pair_name = self.pair_config.get('name', 'Unnamed Pair')
                self.setWindowTitle(f"Pair Data Inspector - {pair_name} *")
            else:
                self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'} *")
        else:
            self.table_status.setText("No modifications")
            if self.is_pair_data:
                pair_name = self.pair_config.get('name', 'Unnamed Pair')
                self.setWindowTitle(f"Pair Data Inspector - {pair_name}")
            else:
                self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'}")
    
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
            for col in [1, 2]:  # X and Y columns
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
    
    def filter_table(self):
        """Filter table based on checkboxes"""
        show_modified = False
        if hasattr(self, 'show_modified_only'):
            show_modified = self.show_modified_only.isChecked()
        show_invalid = self.show_invalid_only.isChecked()
        
        if not show_modified and not show_invalid:
            # Show all rows
            for row in range(self.data_table.rowCount()):
                self.data_table.setRowHidden(row, False)
        else:
            # Filter rows
            for row in range(self.data_table.rowCount()):
                should_show = True
                
                if show_modified:
                    should_show = should_show and (row in self.data_table.modified_rows)
                
                if show_invalid:
                    is_invalid = False
                    for col in [1, 2]:
                        item = self.data_table.item(row, col)
                        if item:
                            try:
                                float(item.text())
                            except ValueError:
                                is_invalid = True
                                break
                    should_show = should_show and is_invalid
                
                self.data_table.setRowHidden(row, not should_show)
    
    def reset_data(self):
        """Reset all modifications"""
        if self.data_table.has_modifications():
            reply = QMessageBox.question(
                self, 
                "Reset Data", 
                "Reset all changes to original data?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.data_table.reset_modifications()
                self.has_unsaved_changes = False
                if not self.is_pair_data:
                    self.apply_button.setEnabled(False)
                self.update_statistics()
                self.table_status.setText("Data reset to original values")
                if self.is_pair_data:
                    pair_name = self.pair_config.get('name', 'Unnamed Pair')
                    self.setWindowTitle(f"Pair Data Inspector - {pair_name}")
                else:
                    self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'}")
    
    def export_data(self):
        """Export current data to file"""
        # For now, just show a message - could implement actual export
        QMessageBox.information(
            self, 
            "Export Data", 
            "Export functionality would save current data to CSV/TXT file.\n(Feature not yet implemented)"
        )
    
    def apply_changes(self):
        """Apply the changes to the data"""
        if not self.has_unsaved_changes:
            self.accept()
            return
        
        try:
            if self.is_pair_data:
                self._apply_pair_changes()
            else:
                self._apply_channel_changes()
                
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Applying Changes", 
                f"An error occurred while applying changes:\n{str(e)}"
            )
    
    def _apply_channel_changes(self):
        """Apply changes to channel data"""
        # Get modified data
        new_x_data, new_y_data = self.data_table.get_modified_data()
        
        if new_x_data is not None and new_y_data is not None:
            # Update the channel
            self.channel.xdata = new_x_data
            self.channel.ydata = new_y_data
            self.channel.modified_at = datetime.now()
            
            # Emit signal that data was updated
            self.data_updated.emit(self.channel.channel_id)
            
            QMessageBox.information(
                self, 
                "Changes Applied", 
                f"Successfully updated {len(self.data_table.modified_rows)} data points."
            )
            
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Failed to get modified data.")
    
    def _apply_pair_changes(self):
        """Apply changes to pair data"""
        # Get modified data
        new_ref_data, new_test_data = self.data_table.get_modified_data()
        
        if new_ref_data is not None and new_test_data is not None:
            # Update the aligned data
            self.aligned_data['ref_data'] = new_ref_data
            self.aligned_data['test_data'] = new_test_data
            
            # Try to update the comparison manager cache
            parent_window = self.parent()
            if hasattr(parent_window, 'comparison_manager') and parent_window.comparison_manager:
                pair_name = self.pair_config['name']
                
                # Update the cached result if it exists
                cached_result = parent_window.comparison_manager.pair_cache.get_pair_result(pair_name)
                if cached_result:
                    cached_result.computation_result['aligned_data'] = self.aligned_data
                    cached_result.touch()  # Update timestamp
                
                # Recalculate statistics with modified data
                try:
                    # Use basic fallback statistics since this is just for inspection
                    try:
                        ref_data = self.aligned_data['ref_data']
                        test_data = self.aligned_data['test_data']
                        
                        # Basic validation
                        if ref_data is not None and test_data is not None and len(ref_data) > 0:
                            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                            if np.sum(valid_mask) > 3:
                                ref_clean = ref_data[valid_mask]
                                test_clean = test_data[valid_mask]
                                
                                # Basic correlation calculation
                                if np.var(ref_clean) > 0 and np.var(test_clean) > 0:
                                    correlation = np.corrcoef(ref_clean, test_clean)[0, 1]
                                else:
                                    correlation = np.nan
                                
                                # Basic RMS calculation
                                differences = test_clean - ref_clean
                                rms = np.sqrt(np.mean(differences ** 2))
                                
                                stats = {
                                    'r': correlation,
                                    'rms': rms,
                                    'n': len(ref_clean),
                                    'valid_ratio': len(ref_clean) / len(ref_data)
                                }
                            else:
                                stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'Insufficient valid data'}
                        else:
                            stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'No data available'}
                    except Exception as e:
                        stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': str(e)}
                    parent_window.comparison_manager.pair_statistics[pair_name] = stats
                    parent_window.comparison_manager._update_pair_statistics(pair_name, stats)
                except Exception as e:
                    print(f"[InspectionWizard] Error recalculating statistics: {e}")
            
            QMessageBox.information(
                self, 
                "Changes Applied", 
                f"Successfully updated {len(self.data_table.modified_rows)} aligned data points.\n"
                "Note: Changes are applied to the aligned data only."
            )
            
            self.accept()
        else:
            QMessageBox.warning(self, "Error", "Failed to get modified data.")
    
    def cancel_changes(self):
        """Cancel and close the dialog"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.reject()
        else:
            self.reject()
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Convenience function for opening the inspection wizard
def inspect_channel_data(channel: Channel, parent=None):
    """
    Open the data inspection wizard for a channel
    
    Args:
        channel: Channel to inspect
        parent: Parent widget
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    wizard = InspectionWizard(channel, parent)
    return wizard.exec()

def inspect_pair_data(pair_config: dict, parent=None):
    """
    Open the data inspection wizard for a comparison pair
    
    Args:
        pair_config: Pair configuration dictionary
        parent: Parent widget
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    wizard = InspectionWizard(pair_config, parent)
    return wizard.exec() 