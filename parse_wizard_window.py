from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QSlider, QFileDialog, QHeaderView
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor, QIntValidator, QColor, QFont
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Dict, Optional, Tuple, Any
import time

class ParseWizardWindow(QMainWindow):
    """
    Manual Parse Wizard - For when autoparsing fails
    Left panel: parsing controls, Right panel: data preview table
    """
    
    file_parsed = Signal(str)  # file_id
    parsing_complete = Signal(dict)
    
    def __init__(self, file_path=None, file_manager=None, channel_manager=None, parent=None):
        super().__init__(parent)
        
        # Store managers and file path
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.parent_window = parent
        self.file_path = file_path
        
        # Raw file data
        self.raw_lines = []
        self.encoding = 'utf-8'
        self.preview_data = None
        self.original_preview_data = None  # Store original data before type conversions
        self._last_column_names = []  # Track column names for dropdown preservation
        self._settings_applied = False  # Track if settings have been applied by manager
        
        # Parse parameters
        self.parse_params = {
            'delimiter': ',',
            'header_row': 0,
            'delete_rows': '',
            'max_rows': 1000,  # For preview
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'delete_na_rows': False,
            'replace_na_value': '',
            'parse_dates': True,
            'date_formats': ['%H:%M:%S', '%m/%d/%Y','%Y-%m-%d %H:%M:%S'],
            'time_column': None,
            'column_types': {},
            'downsample_enabled': False,
            'downsample_method': 'Every Nth row',
            'downsample_factor': 10,
            'downsample_window': 10
        }
        
        # UI update timer
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._update_preview)
        self.update_delay = 500  # ms
        
        # Track last parameters to prevent infinite loops
        self._last_parse_params = {}
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        
        # Show welcome messages for parse wizard
        print("DEBUG: About to call _show_parse_wizard_welcome")
        self._show_parse_wizard_welcome()
        print("DEBUG: Finished calling _show_parse_wizard_welcome")
        
        # Load file if provided or get from file manager
        if self.file_path:
            self._load_file(self.file_path)
        elif self.file_manager:
            # Get selected file from file manager
            selected_file = self._get_selected_file_from_manager()
            if selected_file:
                self.file_path = selected_file
                self._load_file(selected_file)
        
    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Parse Wizard - Manual File Parsing")
        self.setMinimumSize(800, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Build left and right panels
        self._build_left_panel(main_splitter)
        self._build_right_panel(main_splitter)
        
        # Set splitter to 1:1 ratio (equal left and right panels)
        main_splitter.setSizes([1, 1])
        
    def _build_left_panel(self, main_splitter):
        """Build the left control panel with compact grouped controls"""
        # Create left panel container
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Create vertical layout for basic and advanced parsing
        parsing_layout = QVBoxLayout()
        parsing_layout.setSpacing(10)
        
        # Add basic and advanced parsing groups top/bottom
        self._create_basic_parsing_group(parsing_layout)
        self._create_advanced_parsing_group(parsing_layout)
        
        left_layout.addLayout(parsing_layout)
        
        # Add downsample group
        self._create_downsample_group(left_layout)
        
        # Column configuration group that will stretch to fill available space
        self._create_column_configuration_group(left_layout)
        
        # Add stretch to push action buttons to bottom
        left_layout.addStretch()
        
        # Action buttons at the bottom
        self._create_action_buttons_group(left_layout)
        
        main_splitter.addWidget(left_panel)
        

        
    def _create_basic_parsing_group(self, layout):
        """Create basic parsing parameters group"""
        group = QGroupBox("Basic Parsing")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group.setFixedHeight(140)
        group_layout = QFormLayout(group)
        group_layout.setLabelAlignment(Qt.AlignLeft)
        group_layout.setFormAlignment(Qt.AlignLeft)
        
        # Delimiter and Custom delimiter on same row
        delimiter_layout = QHBoxLayout()
        delimiter_layout.setSpacing(10)
        
        # Delimiter combo
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([
            "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Pipe (|)", 
            "Space", "None", "Custom..."
        ])
        self.delimiter_combo.currentTextChanged.connect(self._on_delimiter_changed)
        delimiter_layout.addWidget(self.delimiter_combo)
        
        # Custom delimiter input (hidden by default)
        self.custom_delimiter_input = QLineEdit()
        self.custom_delimiter_input.setPlaceholderText("Enter custom delimiter")
        self.custom_delimiter_input.setStyleSheet("QLineEdit::placeholder { color: #888888; }")
        self.custom_delimiter_input.setVisible(False)
        self.custom_delimiter_input.textChanged.connect(self._on_custom_delimiter_changed)
        delimiter_layout.addWidget(self.custom_delimiter_input)
        
        group_layout.addRow("Delimiter:", delimiter_layout)
        
        # Header row
        self.header_row_spin = QSpinBox()
        self.header_row_spin.setRange(-1, 100)
        self.header_row_spin.setValue(0)
        self.header_row_spin.setSpecialValueText("No header")
        self.header_row_spin.setToolTip("Row number containing column names (-1 for no header)")
        self.header_row_spin.valueChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Header Row:", self.header_row_spin)
        
        # Delete rows
        self.delete_rows_input = QLineEdit()
        self.delete_rows_input.setPlaceholderText("e.g., 1,2,4,5-10 or every:2 or every:3:offset:1")
        self.delete_rows_input.setStyleSheet("QLineEdit::placeholder { color: #888888; }")
        self.delete_rows_input.setToolTip(
            "Delete rows using:\n"
            "• Individual numbers: 1,2,4\n"
            "• Ranges: 5-10\n"
            "• Every Nth row: every:2 (deletes rows 0,2,4,6,...)\n"
            "• Every Nth with offset: every:3:offset:1 (deletes rows 1,4,7,10,...)\n"
            "• Mixed: 1,2,every:3:offset:5\n"
            "Row numbers are 0-based."
        )
        self.delete_rows_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Delete Rows:", self.delete_rows_input)
        
        # Encoding
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii'])
        self.encoding_combo.currentTextChanged.connect(self._on_encoding_changed)
        self.encoding_combo.setToolTip("Character encoding (try latin-1 or cp1252 for files with non-ASCII characters)")
        group_layout.addRow("Encoding:", self.encoding_combo)
        

        
        layout.addWidget(group)
        
    def _create_advanced_parsing_group(self, layout):
        """Create advanced parsing parameters group - more compact"""
        group = QGroupBox("Advanced Parsing")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group.setFixedHeight(140)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)  # Increase spacing between rows
        
        # NA values
        na_layout = QHBoxLayout()
        na_label = QLabel("NA Values:")
        self.na_values_input = QLineEdit('NA,N/A,null,NULL,nan,NaN')
        self.na_values_input.setToolTip("Comma-separated list of values to treat as NaN")
        self.na_values_input.textChanged.connect(self._trigger_preview_update)
        na_layout.addWidget(na_label)
        na_layout.addWidget(self.na_values_input)
        group_layout.addLayout(na_layout)
        
        # Delete rows with NA - enforced and disabled
        self.delete_na_rows_checkbox = QCheckBox("Delete rows containing any NA values (enforced)")
        self.delete_na_rows_checkbox.setChecked(True)
        self.delete_na_rows_checkbox.setEnabled(False)
        self.delete_na_rows_checkbox.setToolTip("Always removes entire rows that contain any NA/null values (cannot be disabled)")
        group_layout.addWidget(self.delete_na_rows_checkbox)
        
        # Replace NA values
        replace_layout = QHBoxLayout()
        replace_label = QLabel("Replace NA with:")
        self.replace_na_input = QLineEdit()
        self.replace_na_input.setPlaceholderText("e.g., 0, -999, Unknown, or leave empty")
        self.replace_na_input.setStyleSheet("QLineEdit::placeholder { color: #888888; }")
        self.replace_na_input.setToolTip("Replace all NA/null values with this value (leave empty to keep as NaN)")
        self.replace_na_input.textChanged.connect(self._trigger_preview_update)
        replace_layout.addWidget(replace_label)
        replace_layout.addWidget(self.replace_na_input)
        group_layout.addLayout(replace_layout)
        
        # Date formats
        date_layout = QHBoxLayout()
        date_label = QLabel("Date Formats:")
        self.date_formats_input = QLineEdit('%H:%M:%S,%m/%d/%Y,%Y-%m-%d %H:%M:%S')
        self.date_formats_input.setToolTip("Comma-separated list of datetime formats to attempt parsing. Add your own formats as needed.")
        self.date_formats_input.textChanged.connect(self._trigger_preview_update)
        date_layout.addWidget(date_label)
        date_layout.addWidget(self.date_formats_input)
        group_layout.addLayout(date_layout)
        
        layout.addWidget(group)
        
    def _create_downsample_group(self, layout):
        """Create downsample parameters group"""
        group = QGroupBox("Downsample")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group.setFixedHeight(120)
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)  # Increase spacing between rows
        
        # Downsampling checkbox - no indentation
        self.downsample_checkbox = QCheckBox("Enable downsampling (for large files)")
        self.downsample_checkbox.setChecked(False)
        self.downsample_checkbox.toggled.connect(self._on_downsample_toggled)
        self.downsample_checkbox.setToolTip("Downsample data to reduce file size and improve performance")
        group_layout.addWidget(self.downsample_checkbox)
        
        # Downsample method - no indentation
        method_layout = QHBoxLayout()
        method_label = QLabel("Method:")
        self.downsample_method_combo = QComboBox()
        self.downsample_method_combo.addItems([
            "Every Nth row", "Moving average", "Random sampling"
        ])
        self.downsample_method_combo.setEnabled(False)
        self.downsample_method_combo.currentTextChanged.connect(self._on_downsample_method_changed)
        self.downsample_method_combo.setToolTip("Method for downsampling data")
        method_layout.addWidget(method_label)
        method_layout.addWidget(self.downsample_method_combo)
        method_layout.addStretch()  # Push to the left
        group_layout.addLayout(method_layout)
        
        # Create a horizontal layout for factor and window controls - no indentation
        downsample_controls_layout = QHBoxLayout()
        downsample_controls_layout.setAlignment(Qt.AlignLeft)
        
        # Downsample factor
        self.downsample_factor_spin = QSpinBox()
        self.downsample_factor_spin.setRange(2, 100)
        self.downsample_factor_spin.setValue(10)
        self.downsample_factor_spin.setEnabled(False)
        self.downsample_factor_spin.valueChanged.connect(self._trigger_preview_update)
        self.downsample_factor_spin.setToolTip("Downsampling factor (e.g., 10 = keep every 10th row)")
        
        # Downsample window size (for moving average)
        self.downsample_window_spin = QSpinBox()
        self.downsample_window_spin.setRange(2, 1000)
        self.downsample_window_spin.setValue(10)
        self.downsample_window_spin.setEnabled(False)
        self.downsample_window_spin.valueChanged.connect(self._trigger_preview_update)
        self.downsample_window_spin.setToolTip("Window size for moving average downsampling")
        
        # Add controls to horizontal layout with labels
        downsample_controls_layout.addWidget(QLabel("Factor:"))
        downsample_controls_layout.addWidget(self.downsample_factor_spin)
        downsample_controls_layout.addWidget(QLabel("Window:"))
        downsample_controls_layout.addWidget(self.downsample_window_spin)
        downsample_controls_layout.addStretch()  # Add stretch to push controls to the left
        
        # Add the horizontal layout
        group_layout.addLayout(downsample_controls_layout)
        
        layout.addWidget(group)
        
    def _create_column_configuration_group(self, layout):
        """Create column configuration group for left panel"""
        group = QGroupBox("Column Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Use Index as X checkbox
        self.use_index_as_x_checkbox = QCheckBox("Use Index as X")
        self.use_index_as_x_checkbox.setChecked(False)
        self.use_index_as_x_checkbox.setToolTip("When checked, row index will be used as X-axis for all channels. This disables column X-axis selection.")
        self.use_index_as_x_checkbox.toggled.connect(self._on_use_index_as_x_toggled)
        group_layout.addWidget(self.use_index_as_x_checkbox)
        
        # Warning label for X column selection
        self.x_axis_warning_label = QLabel("No X column selected: index values will be used as X.")
        self.x_axis_warning_label.setStyleSheet("color: orange; font-weight: bold; padding: 3px; background: #fffbe6; border: 1px solid #ffe58f; border-radius: 3px;")
        self.x_axis_warning_label.setVisible(False)
        group_layout.addWidget(self.x_axis_warning_label)
        
        # Column types table - compact version for left panel
        # Removed "Column Types:" label as requested
        
        # Add info label about column name editing
        info_label = QLabel("Tip: Double-click column names to edit them.")
        info_label.setStyleSheet("color: #333333; font-size: 10px; font-style: italic; padding: 3px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;")
        group_layout.addWidget(info_label)
        
        self.column_types_table = QTableWidget()
        self.column_types_table.setColumnCount(4)
        self.column_types_table.setHorizontalHeaderLabels(["Column Name", "X-Axis", "Channel Type", "Data Type"])
        
        # Configure table to stretch to full width
        self.column_types_table.horizontalHeader().setStretchLastSection(True)
        self.column_types_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)  # Column Name stretches
        self.column_types_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)    # X-Axis fixed
        self.column_types_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)    # Channel Type fixed
        self.column_types_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)    # Data Type fixed
        
        # Set fixed column widths for non-stretching columns
        self.column_types_table.setColumnWidth(1, 50)   # X-Axis - narrowest
        self.column_types_table.setColumnWidth(2, 80)   # Channel Type  
        self.column_types_table.setColumnWidth(3, 80)   # Data Type
        
        # Compact height for left panel
        self.column_types_table.setMinimumHeight(150)
        
        # Enable editing only for column names (first column)
        self.column_types_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Add tooltip to explain column name editing
        self.column_types_table.setToolTip(
            "Column Configuration Table:\n"
            "• Column Name: Click to edit column names (double-click or press F2)\n"
            "• Data Type: Original pandas data type (read-only)\n"
            "• Channel Type: Auto-detected channel type (read-only)\n"
            "• X-Axis: Select which column to use as X-axis (radio button)\n\n"
            "Tip: Edit column names to customize channel names when created!"
        )
        
        # Connect signal to trigger preview update when column names are edited
        self.column_types_table.itemChanged.connect(self._on_column_name_changed)
        
        # Enable editing only for column names (first column) via mouse double-click
        self.column_types_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        
        # Create button group for X-axis radio buttons
        self.x_axis_button_group = QButtonGroup()
        self.x_axis_button_group.setExclusive(True)  # Only one radio button can be selected
        self.x_axis_button_group.buttonClicked.connect(self._on_x_axis_selection_changed)
        
        # Add the table with stretch to make it expand vertically
        group_layout.addWidget(self.column_types_table, 1)  # Stretch factor of 1
        
        layout.addWidget(group)
        
    def _create_action_buttons_group(self, layout):
        """Create action buttons group"""
        # Remove group box, just use layout directly
        group_layout = QHBoxLayout()
        
        # Refresh preview button
        self.refresh_button = QPushButton("Refresh Preview")
        self.refresh_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.refresh_button.clicked.connect(self._force_refresh_preview)
        self.refresh_button.setToolTip("Force immediate preview update, bypassing caches and timers")
        group_layout.addWidget(self.refresh_button)
        
        # Parse and create channels button
        self.parse_button = QPushButton("Parse and Create Channels")
        self.parse_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.parse_button.clicked.connect(self._parse_and_create_channels)
        group_layout.addWidget(self.parse_button)
        
        layout.addLayout(group_layout)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with data preview and column configuration"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Removed "Data Preview" title as requested
        
        # Data Preview Section
        self._create_data_preview_section(right_layout)
        
        main_splitter.addWidget(right_panel)
    
    def _create_data_preview_section(self, layout):
        """Create the data preview section"""
        # Preview info label
        self.preview_info_label = QLabel("Load a file to see preview")
        self.preview_info_label.setStyleSheet("color: #444444; font-size: 11px; padding: 5px;")
        layout.addWidget(self.preview_info_label)
        
        # Parse status (moved above table)
        self.parse_status_label = QLabel("Status: Ready")
        self.parse_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc; color: #333333;")
        layout.addWidget(self.parse_status_label)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        # Remove height limit to fill entire right panel
        layout.addWidget(self.data_table)
    

        

        
    def _connect_signals(self):
        """Connect all signal handlers"""
        # Ensure all parsing-related controls are connected to trigger preview updates
        self._ensure_all_controls_connected()
        
    def _ensure_all_controls_connected(self):
        """Ensure all controls that affect parsing are connected to _trigger_preview_update"""
        # Basic parsing controls
        if hasattr(self, 'header_row_spin'):
            self.header_row_spin.valueChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'delete_rows_input'):
            self.delete_rows_input.textChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'encoding_combo'):
            self.encoding_combo.currentTextChanged.connect(self._on_encoding_changed)
            
        # Advanced parsing controls
        if hasattr(self, 'na_values_input'):
            self.na_values_input.textChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'delete_na_rows_checkbox'):
            self.delete_na_rows_checkbox.toggled.connect(self._trigger_preview_update)
        if hasattr(self, 'replace_na_input'):
            self.replace_na_input.textChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'date_formats_input'):
            self.date_formats_input.textChanged.connect(self._trigger_preview_update)
            
        # Downsampling controls
        if hasattr(self, 'downsample_checkbox'):
            self.downsample_checkbox.toggled.connect(self._trigger_preview_update)
        if hasattr(self, 'downsample_method_combo'):
            self.downsample_method_combo.currentTextChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'downsample_factor_spin'):
            self.downsample_factor_spin.valueChanged.connect(self._trigger_preview_update)
        if hasattr(self, 'downsample_window_spin'):
            self.downsample_window_spin.valueChanged.connect(self._trigger_preview_update)
            
        print(f"[ParseWizard] All control signals connected to preview updates")
        

        

        

            
    def _load_file(self, file_path: Path):
        """Load file and show initial preview"""
        try:
            self.file_path = file_path
            
            print(f"[ParseWizard] Loading file: {file_path.name}")
            
            # Read file with encoding detection
            self.raw_lines, self.encoding = self._read_file_with_encoding(file_path)
            
            print(f"[ParseWizard] File loaded: {len(self.raw_lines)} lines, encoding: {self.encoding}")
            
            # Update encoding combo
            if self.encoding in [self.encoding_combo.itemText(i) for i in range(self.encoding_combo.count())]:
                self.encoding_combo.setCurrentText(self.encoding)
            
            # Auto-detect settings only if not already applied by manager
            if not self._settings_applied:
                print("[ParseWizard] Auto-detecting parsing settings...")
                self._auto_detect_settings()
            else:
                print("[ParseWizard] Skipping auto-detection - settings already applied by manager")
            
            # Update preview
            self._trigger_preview_update()
            
            # Update readiness status
            self._update_readiness_status()
            
        except Exception as e:
            print(f"[ParseWizard] Error loading file: {str(e)}")
            QMessageBox.critical(self, "Error Loading File", f"Could not load file:\n{str(e)}")
    
    def _get_selected_file_from_manager(self):
        """Get the selected file from the file manager"""
        if not self.file_manager:
            return None
            
        try:
            # Get selected file from file manager
            selected_files = self.file_manager.get_selected_files()
            if selected_files and len(selected_files) > 0:
                # Use the first selected file
                selected_file_path = selected_files[0]
                if isinstance(selected_file_path, str):
                    return Path(selected_file_path)
                elif hasattr(selected_file_path, 'path'):
                    return Path(selected_file_path.path)
                else:
                    return Path(str(selected_file_path))
            return None
        except Exception as e:
            print(f"[ParseWizard] Error getting selected file from manager: {str(e)}")
            return None
            
    def _read_file_with_encoding(self, file_path: Path) -> Tuple[List[str], str]:
        """Read file with automatic encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    lines = [line.rstrip('\r\n') for line in f.readlines()]
                    return [line for line in lines if line.strip()], encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Final fallback
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                decoded = content.decode('utf-8', errors='replace')
                lines = [line.strip() for line in decoded.split('\n') if line.strip()]
                return lines, 'utf-8-fallback'
        except Exception:
            return [], 'unknown'
            
    def _on_auto_detect_clicked(self):
        """Handle auto-detect button click"""
        print(f"[ParseWizard] Auto-detect button clicked")
        self._auto_detect_settings()
        print(f"[ParseWizard] Auto-detect complete")
        
    def _auto_detect_settings(self):
        """Auto-detect parsing settings from file"""
        if not self.raw_lines:
            return
            
        print(f"[ParseWizard] Auto-detecting settings for {len(self.raw_lines)} lines")
        
        # Try to auto-detect delimiter first
        detected_delimiter = self._detect_delimiter()
        if detected_delimiter:
            # Map detected delimiter to UI option
            delimiter_map = {
                ',': 'Comma (,)',
                '\t': 'Tab (\\t)',
                ';': 'Semicolon (;)',
                '|': 'Pipe (|)',
                ' ': 'Space'
            }
            ui_option = delimiter_map.get(detected_delimiter, 'Comma (,)')
            self.delimiter_combo.setCurrentText(ui_option)
            print(f"[ParseWizard] Auto-detected delimiter: {ui_option}")
        else:
            # Fallback to comma if detection fails
            self.delimiter_combo.setCurrentText('Comma (,)')
            print(f"[ParseWizard] Delimiter detection failed, defaulting to comma")
            
        # Detect header row
        header_row = self._detect_header_row()
        self.header_row_spin.setValue(header_row)
        print(f"[ParseWizard] Detected header row: {header_row}")
        
        # Show sample of detected header if available
        if header_row < len(self.raw_lines):
            header_line = self.raw_lines[header_row]
            print(f"[ParseWizard] Header line: '{header_line}'")
        else:
            print(f"[ParseWizard] Warning: Header row {header_row} exceeds file length")
        
    def _detect_delimiter(self) -> str:
        """Detect the most likely delimiter"""
        if not self.raw_lines:
            return ','
            
        print(f"[ParseWizard] Detecting delimiter from {len(self.raw_lines)} lines")
        
        # Test common delimiters
        delimiters = [',', '\t', ';', '|']
        delimiter_scores = {}
        
        # Sample first few non-empty lines
        sample_lines = [line for line in self.raw_lines[:20] if line.strip()][:10]
        print(f"[ParseWizard] Analyzing {len(sample_lines)} sample lines for delimiter detection")
        
        for delimiter in delimiters:
            delimiter_name = 'TAB' if delimiter == '\t' else f"'{delimiter}'"
            scores = []
            for line in sample_lines:
                if delimiter in line:
                    parts = line.split(delimiter)
                    # Score based on consistent number of parts and reasonable part lengths
                    if len(parts) > 1:
                        avg_length = sum(len(part.strip()) for part in parts) / len(parts)
                        score = len(parts) * (1 + min(avg_length / 10, 1))
                        scores.append(score)
                        print(f"[ParseWizard]   {delimiter_name}: '{line}' -> {len(parts)} parts, avg_length={avg_length:.1f}, score={score:.1f}")
            
            if scores:
                avg_score = sum(scores) / len(scores)
                delimiter_scores[delimiter] = avg_score
                print(f"[ParseWizard]   {delimiter_name}: Average score = {avg_score:.2f}")
            else:
                print(f"[ParseWizard]   {delimiter_name}: No valid lines found")
        
        # Return delimiter with highest score
        if delimiter_scores:
            best_delimiter = max(delimiter_scores, key=delimiter_scores.get)
            best_delimiter_name = 'TAB' if best_delimiter == '\t' else f"'{best_delimiter}'"
            print(f"[ParseWizard] Selected delimiter: {best_delimiter_name} (score: {delimiter_scores[best_delimiter]:.2f})")
            return best_delimiter
        
        print(f"[ParseWizard] No delimiter detected, defaulting to comma")
        return ','
        
    def _detect_header_row(self) -> int:
        """Detect header row (first non-metadata line)"""
        if not self.raw_lines:
            return 0
            
        print(f"[ParseWizard] Detecting header row from {len(self.raw_lines)} lines")
        
        # Look for metadata lines at the beginning
        metadata_patterns = [r'^#', r'^//', r'^%', r'^;', r'^--', r'^\s*$']
        print(f"[ParseWizard] Metadata patterns: {metadata_patterns}")
        
        for i, line in enumerate(self.raw_lines):
            is_metadata = any(re.match(pattern, line) for pattern in metadata_patterns)
            if is_metadata:
                print(f"[ParseWizard] Row {i}: '{line}' -> METADATA")
            else:
                print(f"[ParseWizard] Row {i}: '{line}' -> HEADER (first non-metadata)")
                return i
                
        # If all lines are metadata, default to row 0
        print(f"[ParseWizard] All lines appear to be metadata, defaulting to row 0")
        return 0
        
    def _on_delimiter_changed(self, text):
        """Handle delimiter combo change"""
        delimiter_map = {
            'Comma (,)': ',',
            'Tab (\\t)': '\t',
            'Semicolon (;)': ';',
            'Pipe (|)': '|',
            'Space': ' ',
            'None': None
        }
        
        if text == 'Custom...':
            self.custom_delimiter_input.setVisible(True)
            self.parse_params['delimiter'] = self.custom_delimiter_input.text() or ','
        else:
            self.custom_delimiter_input.setVisible(False)
            self.parse_params['delimiter'] = delimiter_map.get(text, ',')
            
        self._trigger_preview_update()
        
    def _on_custom_delimiter_changed(self, text):
        """Handle custom delimiter input change"""
        self.parse_params['delimiter'] = text or ','
        self._trigger_preview_update()
        
    def _on_encoding_changed(self, text):
        """Handle encoding change"""
        self.parse_params['encoding'] = text
        if self.file_path:
            self._load_file(self.file_path)
            

    
    def _on_downsample_toggled(self, checked):
        """Handle downsample checkbox toggle"""
        self.downsample_method_combo.setEnabled(checked)
        self.downsample_factor_spin.setEnabled(checked)
        
        # Enable window spin only for moving average method
        method = self.downsample_method_combo.currentText()
        self.downsample_window_spin.setEnabled(checked and method == "Moving average")
        
        self._trigger_preview_update()
    
    def _on_downsample_method_changed(self, method):
        """Handle downsample method change"""
        # Show/hide window size control based on method
        show_window = method == "Moving average"
        self.downsample_window_spin.setEnabled(self.downsample_checkbox.isChecked() and show_window)
        self._trigger_preview_update()
        
    def _on_cell_double_clicked(self, row, column):
        """Handle cell double-click to enable editing only for column names"""
        if column == 0:  # Only allow editing for the first column (column names)
            # Temporarily enable editing for this specific cell
            self.column_types_table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed)
            # Start editing the cell
            self.column_types_table.editItem(self.column_types_table.item(row, column))
            # Disable editing again after a short delay
            QTimer.singleShot(100, lambda: self.column_types_table.setEditTriggers(QTableWidget.NoEditTriggers))
    
    def _on_column_name_changed(self, item):
        """Handle column name editing in the table"""
        if item.column() == 0:  # Only handle changes to the first column (column names)
            new_name = item.text().strip()
            print(f"[ParseWizard] Column name changed to: '{new_name}'")
            
            # Update visual styling to show it's been edited
            if new_name:
                item.setBackground(QColor(255, 255, 200))  # Light yellow for edited names
                # Find the original column name for the tooltip
                if self.original_preview_data is not None and item.row() < len(self.original_preview_data.columns):
                    original_name = str(self.original_preview_data.columns[item.row()])
                    item.setToolTip(f"Edited from '{original_name}' to '{new_name}'")
                else:
                    item.setToolTip(f"Edited column name: '{new_name}'")
            else:
                # Empty name - show warning styling
                item.setBackground(QColor(255, 200, 200))  # Light red for invalid names
                item.setToolTip("Column name cannot be empty")
            
            self._trigger_preview_update()
    
    def _on_x_axis_selection_changed(self, button):
        """Handle X-axis column selection change"""
        if button and button.isChecked() and button.isEnabled():
            row = self.x_axis_button_group.id(button)
            if self.preview_data is not None and row < len(self.preview_data.columns):
                selected_column = str(self.preview_data.columns[row])
                print(f"[ParseWizard] X-axis column selected: '{selected_column}' (row {row})")
                self._trigger_preview_update()
                # Update readiness status
                self._update_readiness_status()
        elif button and button.isChecked() and not button.isEnabled():
            # If a disabled button was somehow checked, uncheck it
            button.setChecked(False)
            print(f"[ParseWizard] Attempted to select disabled X-axis column, unchecking")
            # Update readiness status
            self._update_readiness_status()
    
    def _on_use_index_as_x_toggled(self, checked):
        """Handle Use Index as X checkbox toggle"""
        print(f"[ParseWizard] Use Index as X checkbox toggled: {checked}")
        
        # Enable/disable all radio buttons in the X-Axis column
        for i in range(self.column_types_table.rowCount()):
            x_axis_radio = self.column_types_table.cellWidget(i, 1)
            if x_axis_radio and isinstance(x_axis_radio, QRadioButton):
                x_axis_radio.setEnabled(not checked)
                if checked:
                    # Uncheck the radio button when disabled
                    x_axis_radio.setChecked(False)
        
        # Update warning label visibility
        if checked:
            # Hide warning when using index
            self.x_axis_warning_label.setVisible(False)
        else:
            # Show warning if no radio button is selected
            selected_button = self.x_axis_button_group.checkedButton()
            self.x_axis_warning_label.setVisible(not selected_button)
        
        # Update readiness status
        self._update_readiness_status()
        
    def _trigger_preview_update(self):
        """Trigger delayed preview update"""
        # Check if parameters actually changed to prevent infinite loops
        current_params = {
            'delimiter': self.parse_params.get('delimiter'),
            'header_row': self.header_row_spin.value(),
            'delete_rows': self.delete_rows_input.text().strip(),
            'encoding': self.encoding_combo.currentText(),
            'delete_na_rows': self.delete_na_rows_checkbox.isChecked(),
            'replace_na_value': self.replace_na_input.text().strip(),
        }
        
        # Only skip if EXACTLY the same AND we have recent valid data
        if (current_params == self._last_parse_params and 
            hasattr(self, 'preview_data') and 
            self.preview_data is not None and 
            not self.preview_data.empty):
            print(f"[ParseWizard] Parameters unchanged and preview data exists, skipping update")
            return
            
        print(f"[ParseWizard] Parameters changed, triggering preview update")
        print(f"[ParseWizard] New params: delimiter='{current_params['delimiter']}', header={current_params['header_row']}")
        
        self._last_parse_params = current_params.copy()
        self.update_timer.stop()
        self.update_timer.start(self.update_delay)
        
    def _force_preview_update(self):
        """Force immediate preview update"""
        self.update_timer.stop()
        self._update_preview()
        
    def _force_refresh_preview(self):
        """Force immediate preview refresh, bypassing all caches and timers"""
        try:
            # Visual feedback - show we're working
            original_text = self.refresh_button.text()
            self.refresh_button.setText("Refreshing...")
            self.refresh_button.setEnabled(False)
            
            print(f"[ParseWizard] Force refresh requested by user")
            
            # Reset any cached state that might block updates
            self._last_parse_params = {}
            
            # Stop any pending timer updates
            self.update_timer.stop()
            
            # Force immediate update
            self._update_preview()
            
            # Update status
            if hasattr(self, 'parse_status_label'):
                self.parse_status_label.setText("Preview refreshed successfully")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #e8f5e8; border: 1px solid #4CAF50; color: #2d5a2d;")
            
            print(f"[ParseWizard] Force refresh completed successfully")
            
        except Exception as e:
            # Show error to user
            error_msg = str(e)
            if hasattr(self, 'parse_status_label'):
                self.parse_status_label.setText(f"Refresh failed: {error_msg}")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            print(f"[ParseWizard] Force refresh error: {error_msg}")
            
        finally:
            # Restore button state
            self.refresh_button.setText(original_text)
            self.refresh_button.setEnabled(True)
        
    def _update_preview(self):
        """Update the data preview table"""
        if not self.raw_lines:
            if hasattr(self, 'parse_status_label'):
                self.parse_status_label.setText("No file data loaded")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
            
        print(f"[ParseWizard] Starting preview update with {len(self.raw_lines)} raw lines")
        
        try:
            # Update parse parameters from UI
            self._update_parse_params_from_ui()
            print(f"[ParseWizard] Parse parameters: delimiter='{self.parse_params.get('delimiter')}', header_row={self.parse_params.get('header_row')}")
            
            # Parse preview data
            self.preview_data = self._parse_preview_data()
            
            if self.preview_data is not None and not self.preview_data.empty:
                print(f"[ParseWizard] Preview parsed successfully: {len(self.preview_data)} rows, {len(self.preview_data.columns)} columns")
                print(f"[ParseWizard] Preview columns: {list(self.preview_data.columns)}")
                
                # Store original data for column type detection
                self.original_preview_data = self.preview_data.copy()
                
                # Apply user-selected column type conversions to preview
                user_channel_types = self._get_user_channel_types()
                if user_channel_types:
                    print(f"[ParseWizard] Applying user column type conversions to preview: {user_channel_types}")
                    self.preview_data = self._convert_column_types(self.preview_data, user_channel_types)
                
                # Apply user-edited column names to preview
                user_column_names = self._get_user_column_names()
                if user_column_names:
                    print(f"[ParseWizard] Applying user column name changes to preview: {user_column_names}")
                    self.preview_data = self._apply_column_name_changes(self.preview_data, user_column_names)
                
                print(f"[ParseWizard] Calling _update_preview_table()")
                self._update_preview_table()
                print(f"[ParseWizard] Calling _update_column_types_table()")
                self._update_column_types_table()
                
                # Update readiness status
                self._update_readiness_status()
                print(f"[ParseWizard] Preview update completed successfully")
                
            else:
                error_msg = "Preview parse failed - no data returned"
                print(f"[ParseWizard] {error_msg}")
                if hasattr(self, 'parse_status_label'):
                    self.parse_status_label.setText(f"{error_msg}")
                    self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
                self._update_readiness_status()
                
        except Exception as e:
            error_details = f"Preview update failed: {str(e)}"
            print(f"[ParseWizard] {error_details}")
            if hasattr(self, 'parse_status_label'):
                self.parse_status_label.setText(f"{error_details}")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            self._update_readiness_status()
            
    def _update_parse_params_from_ui(self):
        """Update parse parameters from UI controls"""
        self.parse_params.update({
            'header_row': self.header_row_spin.value() if self.header_row_spin.value() >= 0 else None,
            'delete_rows': self.delete_rows_input.text().strip(),
            'na_values': [v.strip() for v in self.na_values_input.text().split(',') if v.strip()],
            'delete_na_rows': self.delete_na_rows_checkbox.isChecked(),
            'replace_na_value': self.replace_na_input.text().strip(),
            'parse_dates': True,  # Always parse dates
            'date_formats': [v.strip() for v in self.date_formats_input.text().split(',') if v.strip()],
            'downsample_enabled': self.downsample_checkbox.isChecked(),
            'downsample_method': self.downsample_method_combo.currentText(),
            'downsample_factor': self.downsample_factor_spin.value(),
            'downsample_window': self.downsample_window_spin.value()
        })
        
    def _parse_delete_rows(self, delete_spec: str) -> set:
        """Parse delete rows specification into a set of row indices to delete"""
        if not delete_spec:
            return set()
            
        rows_to_delete = set()
        
        try:
            # Split by comma and process each part
            for part in delete_spec.split(','):
                part = part.strip()
                if not part:
                    continue
                    
                if part.startswith('every:'):
                    # Handle pattern like "every:2" or "every:3:offset:1"
                    pattern_parts = part.split(':')
                    if len(pattern_parts) >= 2:
                        try:
                            step = int(pattern_parts[1])
                            offset = 0
                            
                            # Check for offset specification
                            if len(pattern_parts) >= 4 and pattern_parts[2] == 'offset':
                                offset = int(pattern_parts[3])
                            
                            # Generate row indices based on pattern
                            # We need to estimate how many rows we might have
                            # For preview, we use max_rows, but for full parsing we need to be more careful
                            max_estimated_rows = len(self.raw_lines) if self.raw_lines else 10000
                            
                            # Generate every Nth row starting from offset
                            for i in range(offset, max_estimated_rows, step):
                                rows_to_delete.add(i)
                                
                        except ValueError:
                            print(f"Error parsing pattern '{part}': invalid step or offset")
                            
                elif '-' in part:
                    # Handle range like "5-10"
                    start, end = part.split('-', 1)
                    start = int(start.strip())
                    end = int(end.strip())
                    rows_to_delete.update(range(start, end + 1))
                else:
                    # Handle single number
                    rows_to_delete.add(int(part))
                    
        except ValueError as e:
            print(f"Error parsing delete rows specification '{delete_spec}': {e}")
            return set()
            
        return rows_to_delete
    
    def _apply_downsampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply downsampling to DataFrame based on parse parameters"""
        if not self.parse_params.get('downsample_enabled', False):
            return df
        
        try:
            method = self.parse_params.get('downsample_method', 'Every Nth row')
            factor = self.parse_params.get('downsample_factor', 10)
            window = self.parse_params.get('downsample_window', 10)
            
            if method == "Every Nth row":
                # Keep every Nth row
                return df.iloc[::factor].copy()
            
            elif method == "Moving average":
                # Apply moving average downsampling
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Create downsampled dataframe
                downsampled_indices = np.arange(0, len(df), factor)
                result_df = df.iloc[downsampled_indices].copy()
                
                # Apply moving average to numeric columns
                for col in numeric_cols:
                    if col in df.columns:
                        # Calculate moving average for the original data
                        moving_avg = df[col].rolling(window=window, center=True).mean()
                        # Sample the moving average at downsampled indices
                        result_df[col] = moving_avg.iloc[downsampled_indices].values
                
                return result_df
            
            elif method == "Random sampling":
                # Random sampling
                n_samples = max(1, len(df) // factor)
                return df.sample(n=min(n_samples, len(df)), random_state=42).sort_index()
            
            else:
                # Default to every Nth row if method not recognized
                return df.iloc[::factor].copy()
                
        except Exception as e:
            print(f"Error applying downsampling: {e}")
            return df  # Return original data if downsampling fails
    
    def _apply_na_handling_to_raw_lines(self, data_lines: List[str]) -> List[str]:
        """Apply NA/null value handling to raw text lines BEFORE parsing"""
        if not data_lines:
            return data_lines
            
        try:
            original_count = len(data_lines)
            na_values = self.parse_params.get('na_values', [])
            replace_value = self.parse_params.get('replace_na_value', '').strip()
            delete_na_rows = self.parse_params.get('delete_na_rows', False)
            delimiter = self.parse_params.get('delimiter', ',')
            
            # Skip if no NA handling is requested
            if not replace_value and not delete_na_rows:
                return data_lines
            
            print(f"[ParseWizard] Applying NA handling to {len(data_lines)} raw lines")
            
            processed_lines = []
            header_processed = False
            
            for line_idx, line in enumerate(data_lines):
                # Skip header row processing for NA values (keep as-is)
                header_row = self.parse_params.get('header_row', 0)
                if header_row is not None and line_idx == header_row and not header_processed:
                    processed_lines.append(line)
                    header_processed = True
                    continue
                
                # Split line into fields based on delimiter
                if delimiter and delimiter in line:
                    fields = line.split(delimiter)
                else:
                    # Handle case where delimiter is not found or is None
                    fields = [line]
                
                # Check for NA values in this line
                has_na = False
                for field in fields:
                    field_stripped = field.strip().strip('"\'')  # Remove quotes and whitespace
                    if field_stripped in na_values or field_stripped == '':
                        has_na = True
                        break
                
                # Handle delete rows with NA
                if delete_na_rows and has_na:
                    # Skip this line (don't add to processed_lines)
                    continue
                
                # Handle replace NA values
                if replace_value and has_na:
                    # Replace NA values in the line
                    processed_fields = []
                    for field in fields:
                        field_stripped = field.strip().strip('"\'')
                        if field_stripped in na_values or field_stripped == '':
                            # Preserve any quotes around the field when replacing
                            if field.startswith('"') and field.endswith('"'):
                                processed_fields.append(f'"{replace_value}"')
                            elif field.startswith("'") and field.endswith("'"):
                                processed_fields.append(f"'{replace_value}'")
                            else:
                                processed_fields.append(replace_value)
                        else:
                            processed_fields.append(field)
                    
                    processed_line = delimiter.join(processed_fields) if delimiter else processed_fields[0]
                    processed_lines.append(processed_line)
                else:
                    # No NA handling needed for this line
                    processed_lines.append(line)
            
            removed_count = original_count - len(processed_lines)
            
            if delete_na_rows and removed_count > 0:
                print(f"[ParseWizard] Removed {removed_count} rows containing NA values ({original_count} → {len(processed_lines)} rows)")
            elif delete_na_rows:
                print(f"[ParseWizard] No rows contained NA values")
                
            if replace_value:
                print(f"[ParseWizard] Replaced NA values with '{replace_value}' in raw text")
            
            return processed_lines
            
        except Exception as e:
            print(f"[ParseWizard] Error applying NA handling to raw lines: {e}")
            return data_lines  # Return original data if NA handling fails
    
    def _parse_preview_data(self) -> Optional[pd.DataFrame]:
        """Parse preview data using current parameters"""
        try:
            # Update parameters from UI first
            self._update_parse_params_from_ui()
            
            # Get max_rows for preview
            max_rows = self.parse_params.get('max_rows', 1000) or 1000
            
            # Parse rows to delete
            rows_to_delete = self._parse_delete_rows(self.parse_params.get('delete_rows', ''))
            
            # Prepare data lines (remove specified rows)
            data_lines = []
            for i, line in enumerate(self.raw_lines[:max_rows + len(rows_to_delete)]):
                if i not in rows_to_delete:
                    data_lines.append(line)
                if len(data_lines) >= max_rows:
                    break
            
            if not data_lines:
                return None
            
            # Apply NA handling to raw text lines BEFORE parsing
            data_lines = self._apply_na_handling_to_raw_lines(data_lines)
            
            if not data_lines:
                return None
                
            # Create temporary file content
            import io
            temp_content = '\n'.join(data_lines)
            
            # Parse with pandas
            # Header row position (no adjustment needed since we already removed deleted rows)
            header_row = self.parse_params['header_row']
            
            # Handle None delimiter (fixed-width parsing)
            if self.parse_params['delimiter'] is None:
                df = pd.read_fwf(
                    io.StringIO(temp_content),
                    header=header_row,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            else:
                df = pd.read_csv(
                    io.StringIO(temp_content),
                    sep=self.parse_params['delimiter'],
                    header=header_row,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            
            # Apply downsampling if enabled (AFTER parsing)
            df = self._apply_downsampling(df)
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "unsupported operand type" in error_msg and "NoneType" in error_msg:
                print(f"Preview parse error: Header row configuration issue - check header row and delete rows settings")
            elif "codec can't decode" in error_msg:
                print(f"Preview parse error: Encoding issue - try different encoding (current: {self.parse_params.get('encoding', 'utf-8')})")
            elif "expected str" in error_msg:
                print(f"Preview parse error: Non-ASCII characters in headers - try different encoding or set custom column names")
            else:
                print(f"Preview parse error: {error_msg}")
            return None
            
    def _update_preview_table(self):
        """Update the preview table with parsed data"""
        if self.preview_data is None:
            return
            
        # Set table size
        self.data_table.setRowCount(len(self.preview_data))  # Show all rows
        self.data_table.setColumnCount(len(self.preview_data.columns))
        self.data_table.setHorizontalHeaderLabels([str(col) for col in self.preview_data.columns])
        
        # Determine which column is the X-axis
        selected_button = self.x_axis_button_group.checkedButton()
        x_axis_col_idx = None
        
        if selected_button:
            row = self.x_axis_button_group.id(selected_button)
            if row < len(self.preview_data.columns):
                x_axis_col_idx = row
        
        # Fill table with data
        for row in range(len(self.preview_data)):
            for col in range(len(self.preview_data.columns)):
                value = self.preview_data.iloc[row, col]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                    item.setBackground(QColor(255, 255, 200))  # Light yellow for NaN
                else:
                    item = QTableWidgetItem(str(value))
                    
                # No blue highlighting for X-axis column
                self.data_table.setItem(row, col, item)
        
        # No blue highlighting for X-axis column header
                
        # Update info label
        total_rows = len(self.preview_data)
        self.preview_info_label.setText(f"Showing all {total_rows} rows, {len(self.preview_data.columns)} columns")
        
        # Update readiness status
        self._update_readiness_status()
        
    def _update_readiness_status(self):
        """Update the status label to show if ready for channel creation"""
        if not hasattr(self, 'parse_status_label'):
            return
            
        # Check all required conditions
        conditions = []
        
        # 1. File loaded
        if not hasattr(self, 'file_path') or self.file_path is None:
            self.parse_status_label.setText("No file loaded")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("File loaded")
        
        # 2. Preview data available
        if self.preview_data is None or len(self.preview_data) == 0:
            self.parse_status_label.setText("No preview data available")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("Preview data available")
        
        # 3. At least one column detected
        if len(self.preview_data.columns) == 0:
            self.parse_status_label.setText("No columns detected")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append(f"{len(self.preview_data.columns)} columns detected")
        
        # 4. X-axis column selected OR Use Index as X is checked
        selected_button = self.x_axis_button_group.checkedButton()
        use_index_checked = hasattr(self, 'use_index_as_x_checkbox') and self.use_index_as_x_checkbox.isChecked()
        
        if not selected_button and not use_index_checked:
            self.parse_status_label.setText("No X-axis column selected")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        
        if use_index_checked:
            conditions.append("X-axis: Using index")
        else:
            conditions.append("X-axis column selected")
        
        # 5. Basic parsing configuration
        if not hasattr(self, 'delimiter_combo') or not self.delimiter_combo.currentText():
            self.parse_status_label.setText("Delimiter not configured")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("Parsing configured")
        
        # All conditions met - ready for channel creation
        total_rows = len(self.preview_data)
        status_text = f"Ready to create channels ({total_rows} rows, {len(self.preview_data.columns)} columns)"
        self.parse_status_label.setText(status_text)
        self.parse_status_label.setStyleSheet("padding: 5px; background-color: #e8f5e8; border: 1px solid #4CAF50; color: #2d5a2d;")
        
    def _is_likely_datetime(self, value: str) -> bool:
        """Check if a value looks like datetime (from auto parser logic)"""
        if not value or not value.strip():
            return False
            
        value = value.strip()
        
        try:
            # Try pandas datetime parsing first - this is the most reliable
            pd.to_datetime(value, errors='raise')
            return True
        except:
            # Check common datetime patterns
            datetime_patterns = [
                r'^\d{4}-\d{1,2}-\d{1,2}$',          # YYYY-MM-DD or YYYY-M-D
                r'^\d{1,2}/\d{1,2}/\d{4}$',          # MM/DD/YYYY or M/D/YYYY
                r'^\d{1,2}-\d{1,2}-\d{4}$',          # MM-DD-YYYY or M-D-YYYY
                r'^\d{4}/\d{1,2}/\d{1,2}$',          # YYYY/MM/DD or YYYY/M/D
                r'^\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2}',  # YYYY-MM-DD HH:MM
                r'^\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}',  # MM/DD/YYYY HH:MM
                r'^\d{4}-\d{3}$',                    # YYYY-DDD (Julian date)
                r'^\d{8}$',                          # YYYYMMDD
            ]
            
            for pattern in datetime_patterns:
                if re.match(pattern, value):
                    return True
        
        return False
    
    def _is_likely_datetime_with_formats(self, value: str, date_formats: List[str]) -> bool:
        """Check if a value looks like datetime using configured date formats"""
        if not value or not value.strip():
            return False
            
        value = value.strip()
        
        # Try pandas datetime parsing with configured formats
        for date_format in date_formats:
            try:
                pd.to_datetime(value, format=date_format, errors='raise')
                return True
            except:
                continue
        
        # Fallback to general pandas parsing
        try:
            pd.to_datetime(value, errors='raise')
            return True
        except:
            pass
        
        # Fallback to regex patterns for common formats
        return self._is_likely_datetime(value)

    def _detect_channel_type(self, series: pd.Series) -> str:
        """Detect channel type from pandas Series using configured date formats"""
        dtype_str = str(series.dtype).lower()
        
        # Check for datetime types first
        if 'datetime' in dtype_str or 'timestamp' in dtype_str:
            return 'datetime'
        
        # Check for numeric types
        if series.dtype.kind in 'biufc':  # bool, int, uint, float, complex
            # NEW CHECK: Try to convert all values to float first
            try:
                # If all non-null values can be converted to float, it's numerical
                test_series = pd.to_numeric(series.dropna(), errors='raise')
                return 'numeric'  # All values are convertible to numbers
            except (ValueError, TypeError):
                # If conversion fails, then apply the categorical heuristics
                pass
            
            # Check if this might be categorical data (small number of unique values)
            unique_count = series.nunique()
            total_count = len(series.dropna())
            
            if total_count > 0:
                unique_ratio = unique_count / total_count
                
                # If very few unique values relative to total, likely categorical
                if unique_ratio < 0.1 and unique_count < 20:
                    return 'category'
                
                # Check if values are all integers (could be category codes)
                if series.dropna().apply(lambda x: float(x).is_integer()).all():
                    if unique_count < 50:  # Reasonable number of categories
                        return 'category'
            
            return 'numeric'
        
        # For object/string types, check the content using configured date formats
        if dtype_str == 'object':
            # Sample some values to check for datetime patterns
            sample_values = series.dropna().head(5).astype(str).tolist()
            if sample_values:
                datetime_count = 0
                date_formats = self.parse_params.get('date_formats', [])
                
                for value in sample_values:
                    if self._is_likely_datetime_with_formats(value, date_formats):
                        datetime_count += 1
                
                # If majority look like datetime, classify as datetime
                if datetime_count / len(sample_values) > 0.5:
                    return 'datetime'
            
            # Default to category for object types
            return 'category'
        
        # Default fallback
        return 'numeric'

    def _update_column_types_table(self):
        """Update the column types configuration table"""
        print(f"[ParseWizard] _update_column_types_table called")
        
        if self.preview_data is None:
            print(f"[ParseWizard] preview_data is None, returning early")
            return
        
        print(f"[ParseWizard] preview_data has {len(self.preview_data.columns)} columns: {list(self.preview_data.columns)}")
        
        # Store current edited column names before updating
        current_edited_names = {}
        current_columns = [str(col) for col in self.preview_data.columns]
        
        # Use original data for column type detection to avoid circular dependency
        data_for_detection = self.original_preview_data if hasattr(self, 'original_preview_data') and self.original_preview_data is not None else self.preview_data
        
        # Check if we should preserve edited names (same columns, not just same count)
        should_preserve = False
        if hasattr(self, '_last_column_names') and self._last_column_names == current_columns:
            should_preserve = True
            # Same columns - preserve edited names
            for i in range(self.column_types_table.rowCount()):
                col_name_item = self.column_types_table.item(i, 0)
                
                if col_name_item:
                    current_edited_names[i] = col_name_item.text()
        
        print(f"[ParseWizard] Setting table row count to {len(self.preview_data.columns)}")
        self.column_types_table.setRowCount(len(self.preview_data.columns))
        
        print(f"[ParseWizard] Starting to populate {len(self.preview_data.columns)} rows")
        for i, col in enumerate(self.preview_data.columns):
            col_str = str(col)
            print(f"[ParseWizard] Processing column {i}: '{col_str}'")
            
            # Column name - preserve user edits if available
            name_item = self.column_types_table.item(i, 0)
            if should_preserve and i in current_edited_names:
                # Restore user-edited name
                edited_name = current_edited_names[i]
                if not name_item or name_item.text() != edited_name:
                    name_item = QTableWidgetItem(edited_name)
                    name_item.setBackground(QColor(255, 255, 200))  # Light yellow for edited names
                    name_item.setForeground(QColor(0, 0, 0))  # Black text for contrast
                    name_item.setToolTip(f"Edited from '{col_str}' to '{edited_name}'")
                    self.column_types_table.setItem(i, 0, name_item)
                elif name_item:
                    name_item.setBackground(QColor(255, 255, 200))  # Light yellow for edited names
                    name_item.setForeground(QColor(0, 0, 0))  # Black text for contrast
                    name_item.setToolTip(f"Edited from '{col_str}' to '{name_item.text()}'")
            elif not name_item or name_item.text() != col_str:
                # Use original column name
                name_item = QTableWidgetItem(col_str)
                name_item.setBackground(QColor(240, 248, 255))  # Light blue for editable names
                name_item.setForeground(QColor(0, 0, 0))  # Black text for contrast
                name_item.setToolTip("Double-click to edit column name")
                self.column_types_table.setItem(i, 0, name_item)
            elif name_item:
                # Ensure original names have the editable styling
                name_item.setBackground(QColor(240, 248, 255))  # Light blue for editable names
                name_item.setForeground(QColor(0, 0, 0))  # Black text for contrast
                name_item.setToolTip("Double-click to edit column name")
            
            # Auto-detect channel type using original data
            try:
                print(f"[ParseWizard] Detecting channel type for column '{col}'")
                auto_detected_channel_type = self._detect_channel_type(data_for_detection[col])
                print(f"[ParseWizard] Detected type: {auto_detected_channel_type}")
            except Exception as e:
                print(f"[ParseWizard] Error detecting channel type for column '{col}': {str(e)}")
                auto_detected_channel_type = 'numeric'  # Fallback to numeric
            
            # X-axis column radio button
            x_axis_radio = self.column_types_table.cellWidget(i, 1)
            if not x_axis_radio or not isinstance(x_axis_radio, QRadioButton):
                x_axis_radio = QRadioButton()
                x_axis_radio.setStyleSheet("QRadioButton { background-color: black; color: white; }")
                self.x_axis_button_group.addButton(x_axis_radio, i)
                self.column_types_table.setCellWidget(i, 1, x_axis_radio)
            
            # Ensure radio button has consistent styling
            x_axis_radio.setStyleSheet("QRadioButton { background-color: black; color: white; }")
            
            # Enable/disable radio button based on column type and checkbox state
            use_index_checked = hasattr(self, 'use_index_as_x_checkbox') and self.use_index_as_x_checkbox.isChecked()
            
            if auto_detected_channel_type == 'category':
                x_axis_radio.setEnabled(False)
                x_axis_radio.setToolTip(f"Cannot use '{col_str}' as X-axis (categorical data)")
                x_axis_radio.setChecked(False)  # Uncheck if it was previously selected
            elif use_index_checked:
                x_axis_radio.setEnabled(False)
                x_axis_radio.setToolTip(f"X-axis selection disabled (using index)")
                x_axis_radio.setChecked(False)  # Uncheck if it was previously selected
            else:
                x_axis_radio.setEnabled(True)
                x_axis_radio.setToolTip(f"Select '{col_str}' as X-axis column")
                
                # Auto-select first datetime column or first numeric column if no datetime found
                if auto_detected_channel_type == 'datetime' and not self.x_axis_button_group.checkedButton():
                    x_axis_radio.setChecked(True)
                elif auto_detected_channel_type == 'numeric' and not self.x_axis_button_group.checkedButton():
                    x_axis_radio.setChecked(True)
            
            # Auto-detected type text item (read-only)
            type_item = self.column_types_table.item(i, 2)
            if not type_item or type_item.text() != auto_detected_channel_type:
                type_item = QTableWidgetItem(auto_detected_channel_type)
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                type_item.setBackground(QColor(0, 0, 0))  # Black background
                type_item.setForeground(QColor(255, 255, 255))  # White text
                self.column_types_table.setItem(i, 2, type_item)
            
            # Detected pandas type (for debugging)
            try:
                detected_type = str(self.preview_data[col].dtype)
                print(f"[ParseWizard] Pandas dtype for '{col}': {detected_type}")
            except Exception as e:
                print(f"[ParseWizard] Error getting dtype for column '{col}': {str(e)}")
                detected_type = "unknown"
            type_item = self.column_types_table.item(i, 3)
            if not type_item or type_item.text() != detected_type:
                dtype_item = QTableWidgetItem(detected_type)
                dtype_item.setFlags(dtype_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                dtype_item.setBackground(QColor(0, 0, 0))  # Black background
                dtype_item.setForeground(QColor(255, 255, 255))  # White text
                self.column_types_table.setItem(i, 3, dtype_item)
        
        # Remember the column names for next update
        self._last_column_names = current_columns
        
        # Update warning label visibility
        if hasattr(self, 'x_axis_warning_label'):
            use_index_checked = hasattr(self, 'use_index_as_x_checkbox') and self.use_index_as_x_checkbox.isChecked()
            
            if use_index_checked:
                # Hide warning when using index
                self.x_axis_warning_label.setVisible(False)
            else:
                # Show warning if no radio button is selected
                any_x_selected = False
                for i in range(self.column_types_table.rowCount()):
                    x_axis_radio = self.column_types_table.cellWidget(i, 1)
                    if x_axis_radio and x_axis_radio.isChecked():
                        any_x_selected = True
                        break
                self.x_axis_warning_label.setVisible(not any_x_selected)
        
        print(f"[ParseWizard] Column types table updated with {len(self.preview_data.columns)} columns")
        print(f"[ParseWizard] Final table row count: {self.column_types_table.rowCount()}")
        print(f"[ParseWizard] Final table column count: {self.column_types_table.columnCount()}")
            
        # After updating radio buttons, check if any X is selected or checkbox is checked
        use_index_checked = hasattr(self, 'use_index_as_x_checkbox') and self.use_index_as_x_checkbox.isChecked()
        
        if use_index_checked:
            # Hide warning when using index
            self.x_axis_warning_label.setVisible(False)
        else:
            # Show warning if no radio button is selected
            any_x_selected = False
            for i in range(self.column_types_table.rowCount()):
                x_axis_radio = self.column_types_table.cellWidget(i, 1)
                if x_axis_radio and x_axis_radio.isChecked():
                    any_x_selected = True
                    break
            self.x_axis_warning_label.setVisible(not any_x_selected)

    
    def _get_x_axis_info(self) -> str:
        """Get information about the currently selected X-axis column"""
        if self.preview_data is None:
            return "No data"
            
        # Get the selected radio button
        selected_button = self.x_axis_button_group.checkedButton()
        if not selected_button:
            return "No X-axis column selected"
            
        row = self.x_axis_button_group.id(selected_button)
        if row < len(self.preview_data.columns):
            selected_column = str(self.preview_data.columns[row])
            col_type = str(self.preview_data[selected_column].dtype)
            return f"'{selected_column}' ({col_type})"
        else:
            return "Invalid X-axis selection"
            
    def _reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.parse_params = {
            'delimiter': ',',
            'header_row': 0,
            'delete_rows': '',
            'max_rows': 1000,
            'encoding': 'utf-8',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'delete_na_rows': False,
            'replace_na_value': '',
            'parse_dates': True,
            'date_formats': ['%H:%M:%S', '%m/%d/%Y','%Y-%m-%d %H:%M:%S'],
            'time_column': None,
            'column_types': {},
            'downsample_enabled': False,
            'downsample_method': 'Every Nth row',
            'downsample_factor': 10,
            'downsample_window': 10
        }
        
        # Update UI controls
        self.delimiter_combo.setCurrentText('Comma (,)')
        self.header_row_spin.setValue(0)
        self.delete_rows_input.setText('')
        self.na_values_input.setText('NA,N/A,null,NULL,nan,NaN')
        self.delete_na_rows_checkbox.setChecked(False)
        self.replace_na_input.setText('')
        self.date_formats_input.setText('%Y-%m-%d %H:%M:%S,%H:%M:%S,%m/%d/%Y')
        self.encoding_combo.setCurrentText('utf-8')
        
        # Reset downsampling controls
        self.downsample_checkbox.setChecked(False)
        self.downsample_method_combo.setCurrentText('Every Nth row')
        self.downsample_factor_spin.setValue(10)
        self.downsample_window_spin.setValue(10)
        
        self._trigger_preview_update()
    
    def _enable_large_file_warnings(self):
        """Re-enable large file warnings in the main window"""
        try:
            # Access the main window through parent
            if hasattr(self.parent_window, 'show_large_file_warnings'):
                self.parent_window.show_large_file_warnings = True
                QMessageBox.information(self, "Warnings Enabled", 
                                      "Large file warnings have been re-enabled.\n\n"
                                      "You will now receive warnings when loading files larger than "
                                      f"{getattr(self.parent_window, 'large_file_threshold_mb', 50)} MB.")
                self.log_message("Large file warnings re-enabled via Manual Parse")
            else:
                QMessageBox.warning(self, "Cannot Enable", 
                                  "Could not access main window settings to enable warnings.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to enable warnings: {str(e)}")
    
    def log_message(self, message):
        """Helper method to log messages to console"""
        from console import log_message
        log_message(message, "info", "PARSE")
    
    def _show_parse_wizard_welcome(self):
        """Show welcome and usage instructions for the parse wizard"""
        print("DEBUG: Parse wizard welcome method called")
        # Try multiple ways to log the message
        try:
            # Method 1: Through parent window manager
            if hasattr(self.parent_window, 'manager') and hasattr(self.parent_window.manager, 'log_message'):
                welcome_message = """Welcome to the Parse Wizard!  
Quick Start:  
1. Set delimiter and header row 
2. Configure NA handling and date formats (optional)  
3. Edit column names and set X-axis column  
5. Preview parsed data on the right  
6. Click 'Parse and Create Channels' to continue  

Tips:  
* Enable downsampling for large files if needed  
• Double-click column names to rename them  
• Only one column can be selected as the X-axis"""
                self.parent_window.manager.log_message(welcome_message, "info", "PARSE")
            # Method 2: Direct console manager access
            elif hasattr(self.parent_window, 'manager') and hasattr(self.parent_window.manager, 'console_manager'):
                console_manager = self.parent_window.manager.console_manager
                welcome_message = """Welcome to the Parse Wizard!  
Quick Start:  
1. Set delimiter and header row 
2. Configure NA handling and date formats (optional)  
3. Edit column names and set X-axis column  
5. Preview parsed data on the right  
6. Click 'Parse and Create Channels' to continue  

Tips:  
* Enable downsampling for large files if needed  
• Double-click column names to rename them  
• Only one column can be selected as the X-axis"""
                console_manager.log_message(welcome_message, "info", "PARSE")
            else:
                # Fallback: Use the existing log_message method
                welcome_message = """Welcome to the Parse Wizard!  
Quick Start:  
1. Set delimiter and header row 
2. Configure NA handling and date formats (optional)  
3. Edit column names and set X-axis column  
5. Preview parsed data on the right  
6. Click 'Parse and Create Channels' to continue  

Tips:  
* Enable downsampling for large files if needed  
• Double-click column names to rename them  
• Only one column can be selected as the X-axis"""
                self.log_message(welcome_message)
        except Exception as e:
            print(f"Error showing parse wizard welcome messages: {e}")
        
    def _parse_and_create_channels(self):
        """Parse the full file and create channels"""
        if not self.file_path or not self.raw_lines:
            QMessageBox.warning(self, "No File", "Please select a file first.")
            return
            
        try:
            # Check if file is already parsed
            existing_file = self._find_existing_file()
            if existing_file:
                # Show warning dialog for re-parsing
                reply = QMessageBox.question(
                    self, 
                    "Re-parse Existing File",
                    f"This file '{existing_file.filename}' has already been parsed and contains "
                    f"{len(self.channel_manager.get_channels_by_file(existing_file.file_id))} channels.\n\n"
                    "Re-parsing will replace all existing data from this file.\n\n"
                    "Are you sure you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply != QMessageBox.Yes:
                    return
            
            # Parse full file (not just preview)
            self.parse_params['max_rows'] = None  # Parse all rows
            full_data = self._parse_full_data()
            
            if full_data is None or full_data.empty:
                QMessageBox.critical(self, "Parse Error", "Failed to parse the file with current settings.")
                return
            
            # Apply user-selected column type conversions
            user_channel_types = self._get_user_channel_types()
            if user_channel_types:
                self.log_message(f"Applying user column type conversions for final parsing")
                full_data = self._convert_column_types(full_data, user_channel_types)
                
            # Apply user-edited column names
            user_column_names = self._get_user_column_names()
            if user_column_names:
                self.log_message(f"Applying user column name changes for final parsing")
                full_data = self._apply_column_name_changes(full_data, user_column_names)
            
            # Clean data - remove rows with NA, Inf, or invalid values
            original_rows = len(full_data)
            full_data = self._clean_data_comprehensive(full_data)
            cleaned_rows = len(full_data)
            rows_removed = original_rows - cleaned_rows
            
            if rows_removed > 0:
                self.log_message(f"Data cleaning removed {rows_removed} rows with invalid values")
                
            # Create file and channels using the managers
            if existing_file:
                # Re-parsing existing file - replace it
                file_obj = self._update_existing_file(existing_file, full_data)
                channels = self._create_channels_from_data(full_data, file_obj)
                
                if channels:
                    self.log_message(f"Creating {len(channels)} channels from parsed data")
                    
                    # Remove old channels first
                    old_channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
                    if old_channels:
                        self.log_message(f"Removing {len(old_channels)} existing channels")
                        for old_channel in old_channels:
                            self.channel_manager.remove_channel(old_channel.channel_id)
                    
                    # Assign colors to channels before adding to manager
                    self._assign_colors_to_channels(channels, file_obj.file_id)
                    
                    # Add new channels
                    self.channel_manager.add_channels(channels)
                    self.log_message(f"Successfully created {len(channels)} channels")
                    
                    # Emit completion signal
                    self.file_parsed.emit(file_obj.file_id)
                    self.parsing_complete.emit({
                        'file_id': file_obj.file_id,
                        'channels_created': len(channels),
                        'rows_parsed': len(full_data),
                        'columns': len(full_data.columns),
                        'reparsed': True
                    })
                    
                    self.close()
                else:
                    QMessageBox.warning(self, "No Channels", "No valid channels were created from the data.")
            else:
                # New file - create normally
                file_obj = self._create_file_object(full_data)
                channels = self._create_channels_from_data(full_data, file_obj)
                
                if channels:
                    self.log_message(f"Creating {len(channels)} channels from parsed data")
                    
                    # Assign colors to channels before adding to manager
                    self._assign_colors_to_channels(channels, file_obj.file_id)
                    
                    # Add to managers
                    self.file_manager.add_file(file_obj)
                    self.channel_manager.add_channels(channels)
                    self.log_message(f"Successfully created {len(channels)} channels")
                    
                    # Emit completion signal
                    self.file_parsed.emit(file_obj.file_id)
                    self.parsing_complete.emit({
                        'file_id': file_obj.file_id,
                        'channels_created': len(channels),
                        'rows_parsed': len(full_data),
                        'columns': len(full_data.columns),
                        'reparsed': False
                    })
                    
                    self.close()
                else:
                    QMessageBox.warning(self, "No Channels", "No valid channels were created from the data.")
                
        except Exception as e:
            QMessageBox.critical(self, "Parse Error", f"Error parsing file:\n{str(e)}")
            
    def _parse_full_data(self) -> Optional[pd.DataFrame]:
        """Parse the full file data"""
        try:
            # Update parameters from UI
            self._update_parse_params_from_ui()
            
            # Parse rows to delete
            rows_to_delete = self._parse_delete_rows(self.parse_params.get('delete_rows', ''))
            
            # If we have rows to delete, we need to filter the file content manually
            if rows_to_delete:
                # Filter out deleted rows from raw_lines
                filtered_lines = []
                for i, line in enumerate(self.raw_lines):
                    if i not in rows_to_delete:
                        filtered_lines.append(line)
                        
                # Apply NA handling to raw text lines BEFORE parsing
                filtered_lines = self._apply_na_handling_to_raw_lines(filtered_lines)
                
                if not filtered_lines:
                    return None
                        
                # Create temporary file content
                import io
                temp_content = '\n'.join(filtered_lines)
                
                # Use StringIO for parsing
                file_input = io.StringIO(temp_content)
            else:
                # No rows to delete, but still apply NA handling to all lines
                processed_lines = self._apply_na_handling_to_raw_lines(self.raw_lines.copy())
                
                if not processed_lines:
                    return None
                
                # Create temporary file content with NA handling applied
                import io
                temp_content = '\n'.join(processed_lines)
                file_input = io.StringIO(temp_content)
            
            # Header row position (no adjustment needed since we already removed deleted rows)
            header_row = self.parse_params['header_row'] if self.parse_params['header_row'] is not None else None
            
            # Parse full file
            # Handle None delimiter (fixed-width parsing)
            if self.parse_params['delimiter'] is None:
                df = pd.read_fwf(
                    file_input,
                    header=header_row,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            else:
                df = pd.read_csv(
                    file_input,
                    sep=self.parse_params['delimiter'],
                    header=header_row,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            
            # Apply downsampling if enabled (AFTER parsing)
            df = self._apply_downsampling(df)
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages for common issues
            if "unsupported operand type" in error_msg and "NoneType" in error_msg:
                print(f"Full parse error: Header row configuration issue - check header row and delete rows settings")
            elif "codec can't decode" in error_msg:
                print(f"Full parse error: Encoding issue - try different encoding (current: {self.parse_params.get('encoding', 'utf-8')})")
            elif "expected str" in error_msg:
                print(f"Full parse error: Non-ASCII characters in headers - try different encoding or set custom column names")
            else:
                print(f"Full parse error: {error_msg}")
            return None
            
    def _create_file_object(self, data: pd.DataFrame):
        """Create a File object from parsed data"""
        from file import File, FileStatus
        
        file_obj = File(self.file_path)
        file_obj.state.set_status(FileStatus.PARSED)
        file_obj.state.data_rows = len(data)
        file_obj.state.parse_method = "Manual Parse Wizard"
        file_obj.state.parsing_time = 0.0  # Not tracking time for manual parse
        
        return file_obj
        
    def _create_channels_from_data(self, data: pd.DataFrame, file_obj):
        """Create Channel objects from parsed data with user-selected X-axis column"""
        from channel import Channel
        
        channels = []
        
        # Determine X-axis column based on checkbox or radio button selection
        use_index_checked = hasattr(self, 'use_index_as_x_checkbox') and self.use_index_as_x_checkbox.isChecked()
        selected_button = self.x_axis_button_group.checkedButton()
        time_col = None
        use_index = False
        
        if use_index_checked:
            # Checkbox is checked - always use index
            use_index = True
            print(f"[ParseWizard] Using index as X-axis (checkbox checked)")
        elif selected_button:
            row = self.x_axis_button_group.id(selected_button)
            if row < len(data.columns):
                time_col = str(data.columns[row])
                print(f"[ParseWizard] Using column '{time_col}' as X-axis (radio button selected)")
            else:
                use_index = True
        else:
            # No selection, use row index
            use_index = True
            print(f"[ParseWizard] Using index as X-axis (no selection made)")
                
        # Create channels for each data column
        for col in data.columns:
            if col == time_col:
                continue  # Skip the X-axis column
                
            series = data[col]
            
            # Skip non-numeric columns unless they're categorical
            if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_categorical_dtype(series):
                continue
            
            # Determine X-axis data and label
            if use_index or time_col is None:
                xdata = None  # Will use row index
                xlabel = "Index"
            else:
                xdata = data[time_col].values
                xlabel = str(time_col)
                
            # Create channel using the proper factory method
            # Never zero the x-axis - use original data or preserve existing values
            if use_index or time_col is None:
                # Use original row indices without zeroing
                xdata = np.arange(len(series)) + 1  # Start from 1, not 0
                xlabel = "Row Number"
            else:
                # Use the selected time column data as-is
                xdata = data[time_col].values
                xlabel = str(time_col)
                
            channel = Channel.from_parsing(
                file_id=file_obj.file_id,
                filename=file_obj.filename,
                xdata=xdata,
                ydata=series.values,
                xlabel=xlabel,
                ylabel=str(col),
                legend_label=str(col)
            )
            
            channels.append(channel)
            
        return channels
        
    def _assign_colors_to_channels(self, channels: List, file_id: str):
        """Assign unique colors to channels based on existing channels in the file"""
        # Get existing channels from the same file to avoid color conflicts
        existing_channels = self.channel_manager.get_channels_by_file(file_id)
        existing_colors = set()
        for ch in existing_channels:
            if hasattr(ch, 'color') and ch.color is not None:
                existing_colors.add(ch.color)
        
        # Define color palette (same as used in ProcessWizardManager)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Assign colors to each new channel
        for i, channel in enumerate(channels):
            if not hasattr(channel, 'color') or channel.color is None:
                # Find the first available color not used by other channels in the file
                available_colors = [c for c in colors if c not in existing_colors]
                if available_colors:
                    channel.color = available_colors[0]
                    existing_colors.add(channel.color)  # Mark this color as used
                else:
                    # If all colors are used, cycle through them based on total channel count
                    color_index = (len(existing_channels) + i) % len(colors)
                    channel.color = colors[color_index]
                
                print(f"[ParseWizard] Assigned color {channel.color} to channel '{channel.legend_label}'")
        
    def _find_existing_file(self):
        """Find if the current file path already exists in the file manager"""
        if not self.file_path or not self.file_manager:
            return None
            
        current_path = Path(self.file_path).resolve()
        
        # Check all files in the file manager
        for file_obj in self.file_manager.get_all_files():
            if Path(file_obj.filepath).resolve() == current_path:
                return file_obj
        
        return None
        
    def _update_existing_file(self, existing_file, data):
        """Update an existing file object with new parsing data"""
        from file import FileStatus
        
        # Update the file object in place
        existing_file.state.set_status(FileStatus.PARSED)
        existing_file.state.data_rows = len(data)
        existing_file.state.parse_method = "Manual Parse Wizard (Re-parsed)"
        existing_file.state.parsing_time = 0.0  # Not tracking time for manual parse
        
        return existing_file
        
    def set_file_path(self, file_path):
        """Set the file path from external source"""
        if file_path:
            self._load_file(Path(file_path))
    
    def mark_settings_applied(self):
        """Mark that settings have been applied by the manager"""
        self._settings_applied = True 

    def _get_user_channel_types(self) -> Dict[str, str]:
        """Extract auto-detected channel types from the UI table"""
        channel_types = {}
        
        if not hasattr(self, 'column_types_table') or self.column_types_table is None:
            print("[ParseWizard] No column types table available")
            return channel_types
            
        for i in range(self.column_types_table.rowCount()):
            col_name_item = self.column_types_table.item(i, 0)
            type_item = self.column_types_table.item(i, 2)
            
            if col_name_item and type_item:
                # Get the original column name from the data (before any renaming)
                if self.original_preview_data is not None and i < len(self.original_preview_data.columns):
                    original_name = str(self.original_preview_data.columns[i])
                    auto_detected_type = type_item.text()
                    channel_types[original_name] = auto_detected_type
                    print(f"[ParseWizard] Auto-detected channel type for '{original_name}': {auto_detected_type}")
                else:
                    # Fallback to using the displayed name
                    col_name = col_name_item.text()
                    auto_detected_type = type_item.text()
                    channel_types[col_name] = auto_detected_type
                    print(f"[ParseWizard] Auto-detected channel type for '{col_name}': {auto_detected_type}")
            else:
                print(f"[ParseWizard] Warning: Missing column name item or type item for row {i}")
        
        print(f"[ParseWizard] Total auto-detected channel types: {len(channel_types)}")
        return channel_types

    def _get_user_column_names(self) -> Dict[str, str]:
        """Extract user-edited column names from the UI table"""
        column_names = {}
        
        if not hasattr(self, 'column_types_table') or self.column_types_table is None:
            print("[ParseWizard] No column types table available for column names")
            return column_names
            
        for i in range(self.column_types_table.rowCount()):
            col_name_item = self.column_types_table.item(i, 0)
            
            if col_name_item:
                # Get the original column name from the original data (before any renaming)
                if self.original_preview_data is not None and i < len(self.original_preview_data.columns):
                    original_name = str(self.original_preview_data.columns[i])
                    edited_name = col_name_item.text().strip()
                    
                    # Only include if the name was actually changed
                    if edited_name and edited_name != original_name:
                        column_names[original_name] = edited_name
                        print(f"[ParseWizard] User renamed column '{original_name}' to '{edited_name}'")
                else:
                    # Fallback to using preview data
                    if self.preview_data is not None and i < len(self.preview_data.columns):
                        original_name = str(self.preview_data.columns[i])
                        edited_name = col_name_item.text().strip()
                        
                        # Only include if the name was actually changed
                        if edited_name and edited_name != original_name:
                            column_names[original_name] = edited_name
                            print(f"[ParseWizard] User renamed column '{original_name}' to '{edited_name}'")
            else:
                print(f"[ParseWizard] Warning: Missing column name item for row {i}")
        
        print(f"[ParseWizard] Total user column name changes: {len(column_names)}")
        return column_names

    def _convert_column_types(self, df: pd.DataFrame, user_channel_types: Dict[str, str]) -> pd.DataFrame:
        """Convert DataFrame columns using auto parser logic"""
        print(f"[ParseWizard] Starting column type conversion for {len(df.columns)} columns")
        print(f"[ParseWizard] User selections: {user_channel_types}")
        
        # Initialize category mappings storage if not exists
        if not hasattr(df, '_category_mappings'):
            df._category_mappings = {}
        
        for col in df.columns:
            channel_type = user_channel_types.get(str(col), 'numeric')  # Default to numeric
            original_dtype = str(df[col].dtype)
            print(f"[ParseWizard] Converting '{col}' ({original_dtype}) → {channel_type}")
            
            if channel_type == 'datetime':
                try:
                    print(f"[ParseWizard] Attempting datetime conversion for '{col}'")
                    
                    # Convert to datetime and store both datetime and numeric versions
                    datetime_series = pd.to_datetime(df[col], errors='coerce')
                    
                    # Count conversion results
                    total_values = len(datetime_series)
                    valid_dates = datetime_series.notna().sum()
                    invalid_dates = total_values - valid_dates
                    
                    if not datetime_series.isnull().all():
                        # Store original datetime values as metadata
                        df[f'{col}_datetime'] = datetime_series
                        # Convert to numeric for plotting
                        # Use absolute time values (Unix timestamp in seconds)
                        df[col] = datetime_series.astype('int64') // 10**9  # Convert nanoseconds to seconds
                        
                        success_rate = (valid_dates / total_values) * 100
                        print(f"[ParseWizard] Datetime conversion successful: {valid_dates}/{total_values} values ({success_rate:.1f}%)")
                        
                        if invalid_dates > 0:
                            print(f"[ParseWizard] {invalid_dates} values converted to NaT (invalid dates)")
                    else:
                        # Fallback to numeric if datetime conversion fails
                        print(f"[ParseWizard] All datetime conversions failed, falling back to numeric")
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                except Exception as e:
                    # Fallback to numeric
                    print(f"[ParseWizard] Datetime conversion error: {str(e)}, falling back to numeric")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            elif channel_type == 'category':
                try:
                    print(f"[ParseWizard] Attempting category encoding for '{col}'")
                    
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    
                    # Ensure strings, fill NaNs to avoid crashing
                    filled = df[col].astype(str).fillna('__missing__')
                    
                    # Count unique values before encoding
                    unique_values = filled.unique()
                    print(f"[ParseWizard] Found {len(unique_values)} unique categories")
                    
                    df[col] = le.fit_transform(filled)
                    df[col] = df[col].astype(float)
                    
                    # Save the category mapping
                    category_mapping = {idx: label for idx, label in enumerate(le.classes_)}
                    df._category_mappings[str(col)] = category_mapping
                    
                    # Display category mapping in console
                    print(f"[ParseWizard] Category mapping for '{col}':")
                    for code, label in category_mapping.items():
                        print(f"[ParseWizard]   {code} → '{label}'")
                    
                    print(f"[ParseWizard] Category encoding successful: {len(category_mapping)} categories mapped")
                    
                except Exception as e:
                    print(f"[ParseWizard] Category encoding failed: {str(e)}, falling back to numeric")
                    # Fallback to numeric
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            elif channel_type == 'numeric':
                try:
                    print(f"[ParseWizard] Converting '{col}' to numeric")
                    
                    original_dtype = str(df[col].dtype)
                    original_non_null = df[col].notna().sum()
                    
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    new_dtype = str(df[col].dtype)
                    new_non_null = df[col].notna().sum()
                    lost_values = original_non_null - new_non_null
                    
                    if lost_values == 0:
                        print(f"[ParseWizard] Numeric conversion successful: {original_dtype} → {new_dtype}")
                    else:
                        print(f"[ParseWizard] Numeric conversion: {original_dtype} → {new_dtype}, {lost_values} values became NaN")
                        
                except Exception as e:
                    print(f"[ParseWizard] Numeric conversion error: {str(e)}")
        
        return df 

    def _apply_column_name_changes(self, df: pd.DataFrame, user_column_names: Dict[str, str]) -> pd.DataFrame:
        """Apply user-edited column names to the DataFrame"""
        if not user_column_names:
            return df
            
        self.log_message(f"Applying column name changes: {user_column_names}")
        
        # Create a mapping of old names to new names
        rename_mapping = {}
        for old_name, new_name in user_column_names.items():
            if old_name in df.columns:
                rename_mapping[old_name] = new_name
                self.log_message(f"Renaming '{old_name}' → '{new_name}'")
            else:
                self.log_message(f"Column '{old_name}' not found in DataFrame")
        
        # Apply the renaming
        if rename_mapping:
            df = df.rename(columns=rename_mapping)
            self.log_message(f"Applied {len(rename_mapping)} column name changes")
        
        return df 

    def _on_x_axis_selection_changed(self, button):
        """Update warning label when X selection changes"""
        if hasattr(self, 'x_axis_warning_label'):
            any_x_selected = False
            for i in range(self.column_types_table.rowCount()):
                x_axis_radio = self.column_types_table.cellWidget(i, 1)
                if x_axis_radio and x_axis_radio.isChecked():
                    any_x_selected = True
                    break
            self.x_axis_warning_label.setVisible(not any_x_selected)
    
    def _clean_data_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data cleaning to remove rows with NA, Inf, or invalid values.
        This ensures only clean, valid data is used for channel creation.
        """
        if df is None or df.empty:
            return df
        
        original_rows = len(df)
        self.log_message(f"Starting comprehensive data cleaning for {original_rows} rows")
        
        # Step 1: Replace infinite values with NaN
        df_cleaned = df.replace([np.inf, -np.inf], np.nan)
        
        # Step 2: Identify rows with any invalid values
        # Check for NaN values in all columns
        rows_with_nan = df_cleaned.isna().any(axis=1)
        
        # Check for string representations of invalid values
        invalid_strings = ['nan', 'NaN', 'NAN', 'inf', 'Inf', 'INF', '-inf', '-Inf', '-INF', 
                         'null', 'NULL', 'Null', 'none', 'None', 'NONE', 'NA', 'na', 'Na']
        
        # Check for invalid strings in object columns
        object_columns = df_cleaned.select_dtypes(include=['object']).columns
        rows_with_invalid_strings = pd.Series([False] * len(df_cleaned), index=df_cleaned.index)
        
        for col in object_columns:
            # Convert to string and check for invalid values
            col_series = df_cleaned[col].astype(str)
            invalid_mask = col_series.str.lower().isin([s.lower() for s in invalid_strings])
            rows_with_invalid_strings = rows_with_invalid_strings | invalid_mask
        
        # Step 3: Combine all invalid row conditions
        rows_to_remove = rows_with_nan | rows_with_invalid_strings
        
        # Step 4: Remove rows with any invalid values
        df_cleaned = df_cleaned[~rows_to_remove]
        
        # Step 5: Additional cleaning for numeric columns
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Remove rows where all numeric columns are NaN (completely empty numeric data)
            all_numeric_nan = df_cleaned[numeric_columns].isna().all(axis=1)
            df_cleaned = df_cleaned[~all_numeric_nan]
        
        # Step 6: Reset index after cleaning
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        final_rows = len(df_cleaned)
        removed_rows = original_rows - final_rows
        
        self.log_message(f"Data cleaning complete: {removed_rows} rows removed, {final_rows} rows remaining")
        
        if removed_rows > 0:
            self.log_message(f"Cleaning details:")
            self.log_message(f"  - Rows with NaN values: {rows_with_nan.sum()}")
            self.log_message(f"  - Rows with invalid strings: {rows_with_invalid_strings.sum()}")
            if len(numeric_columns) > 0:
                self.log_message(f"  - Rows with all numeric NaN: {all_numeric_nan.sum()}")
        
        return df_cleaned