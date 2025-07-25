from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QSlider, QFileDialog
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
        self._last_column_names = []  # Track column names for dropdown preservation
        
        # Parse parameters
        self.parse_params = {
            'delimiter': ',',
            'header_row': 0,
            'delete_rows': '',
            'max_rows': 1000,  # For preview
            'quote_char': '"',
            'escape_char': '\\',
            'encoding': 'utf-8',
            'decimal': '.',
            'thousands': '',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'parse_dates': True,
            'date_format': 'auto',
            'time_column': None,
            'column_names': [],
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
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        
        # Load file if provided
        if self.file_path:
            self._load_file(self.file_path)
        
    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("🔨 Parse Wizard - Manual File Parsing")
        self.setMinimumSize(1500, 950)
        
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
        
        # Set splitter proportions (40% left, 60% right) - more space for left panel controls
        main_splitter.setSizes([560, 840])
        
    def _build_left_panel(self, main_splitter):
        """Build the left control panel with compact grouped controls"""
        # Create left panel container
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("🔧 Parse Configuration")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        left_layout.addWidget(title_label)
        
        # Add control groups without scrolling - more compact layout
        self._create_file_selection_group(left_layout)
        self._create_basic_parsing_group(left_layout)
        self._create_advanced_parsing_group(left_layout)
        self._create_datetime_configuration_group(left_layout)
        self._create_action_buttons_group(left_layout)
        
        # Add stretch to push everything to top
        left_layout.addStretch()
        
        main_splitter.addWidget(left_panel)
        
    def _create_file_selection_group(self, layout):
        """Create file selection group"""
        group = QGroupBox("📁 File Selection")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # File path display
        file_layout = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        file_layout.addWidget(self.file_path_label)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_button)
        
        group_layout.addLayout(file_layout)
        
        # File info
        self.file_info_label = QLabel("File info will appear here")
        self.file_info_label.setStyleSheet("color: #666; font-size: 11px;")
        group_layout.addWidget(self.file_info_label)
        
        layout.addWidget(group)
        
    def _create_basic_parsing_group(self, layout):
        """Create basic parsing parameters group"""
        group = QGroupBox("⚙️ Basic Parsing")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QFormLayout(group)
        
        # Delimiter
        self.delimiter_combo = QComboBox()
        self.delimiter_combo.addItems([
            "Comma (,)", "Tab (\\t)", "Semicolon (;)", "Pipe (|)", 
            "Space", "None", "Custom..."
        ])
        self.delimiter_combo.currentTextChanged.connect(self._on_delimiter_changed)
        group_layout.addRow("Delimiter:", self.delimiter_combo)
        
        # Custom delimiter input (hidden by default)
        self.custom_delimiter_input = QLineEdit()
        self.custom_delimiter_input.setPlaceholderText("Enter custom delimiter")
        self.custom_delimiter_input.setVisible(False)
        self.custom_delimiter_input.textChanged.connect(self._on_custom_delimiter_changed)
        group_layout.addRow("Custom:", self.custom_delimiter_input)
        
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
        group = QGroupBox("🔧 Advanced Parsing")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QFormLayout(group)
        
        # Quote character
        self.quote_char_input = QLineEdit('"')
        self.quote_char_input.setMaxLength(1)
        self.quote_char_input.setToolTip("Character used to quote fields")
        self.quote_char_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Quote Char:", self.quote_char_input)
        
        # Escape character
        self.escape_char_input = QLineEdit('\\')
        self.escape_char_input.setMaxLength(1)
        self.escape_char_input.setToolTip("Character used to escape quotes")
        self.escape_char_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Escape Char:", self.escape_char_input)
        
        # Decimal separator
        self.decimal_combo = QComboBox()
        self.decimal_combo.addItems(['.', ','])
        self.decimal_combo.currentTextChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Decimal:", self.decimal_combo)
        
        # Thousands separator
        self.thousands_combo = QComboBox()
        self.thousands_combo.addItems(['', ',', '.', ' ', "'"])
        self.thousands_combo.currentTextChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Thousands:", self.thousands_combo)
        
        # Add tooltip to explain thousands separator options
        self.thousands_combo.setToolTip("Thousands separator character (leave empty for none)")
        
        # NA values
        self.na_values_input = QLineEdit('NA,N/A,null,NULL,nan,NaN')
        self.na_values_input.setToolTip("Comma-separated list of values to treat as NaN")
        self.na_values_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("NA Values:", self.na_values_input)
        
        # Downsampling section - more compact
        downsample_frame = QFrame()
        downsample_layout = QVBoxLayout(downsample_frame)
        downsample_layout.setContentsMargins(0, 5, 0, 5)
        
        # Downsampling checkbox
        self.downsample_checkbox = QCheckBox("Enable downsampling (for large files)")
        self.downsample_checkbox.setChecked(False)
        self.downsample_checkbox.toggled.connect(self._on_downsample_toggled)
        self.downsample_checkbox.setToolTip("Downsample data to reduce file size and improve performance")
        downsample_layout.addWidget(self.downsample_checkbox)
        
        # Compact row for downsample controls
        downsample_controls_layout = QHBoxLayout()
        
        # Downsample method
        self.downsample_method_combo = QComboBox()
        self.downsample_method_combo.addItems([
            "Every Nth row", "Moving average", "Random sampling"
        ])
        self.downsample_method_combo.setEnabled(False)
        self.downsample_method_combo.currentTextChanged.connect(self._on_downsample_method_changed)
        self.downsample_method_combo.setToolTip("Method for downsampling data")
        downsample_controls_layout.addWidget(QLabel("Method:"))
        downsample_controls_layout.addWidget(self.downsample_method_combo)
        
        # Downsample factor
        self.downsample_factor_spin = QSpinBox()
        self.downsample_factor_spin.setRange(2, 100)
        self.downsample_factor_spin.setValue(10)
        self.downsample_factor_spin.setEnabled(False)
        self.downsample_factor_spin.valueChanged.connect(self._trigger_preview_update)
        self.downsample_factor_spin.setToolTip("Downsampling factor (e.g., 10 = keep every 10th row)")
        downsample_controls_layout.addWidget(QLabel("Factor:"))
        downsample_controls_layout.addWidget(self.downsample_factor_spin)
        
        # Downsample window size (for moving average)
        self.downsample_window_spin = QSpinBox()
        self.downsample_window_spin.setRange(2, 1000)
        self.downsample_window_spin.setValue(10)
        self.downsample_window_spin.setEnabled(False)
        self.downsample_window_spin.valueChanged.connect(self._trigger_preview_update)
        self.downsample_window_spin.setToolTip("Window size for moving average downsampling")
        downsample_controls_layout.addWidget(QLabel("Window:"))
        downsample_controls_layout.addWidget(self.downsample_window_spin)
        
        downsample_layout.addLayout(downsample_controls_layout)
        group_layout.addRow("Downsampling:", downsample_frame)
        
        layout.addWidget(group)
        
    def _create_datetime_configuration_group(self, layout):
        """Create datetime and X-axis configuration group - more compact"""
        group = QGroupBox("📅 Date/Time & X-Axis Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QFormLayout(group)
        
        # Parse dates checkbox
        self.parse_dates_checkbox = QCheckBox("Attempt to parse dates")
        self.parse_dates_checkbox.setChecked(True)
        self.parse_dates_checkbox.toggled.connect(self._trigger_preview_update)
        group_layout.addRow("", self.parse_dates_checkbox)
        
        # Date format
        self.date_format_combo = QComboBox()
        self.date_format_combo.addItems([
            'auto', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S', 'custom...'
        ])
        self.date_format_combo.currentTextChanged.connect(self._on_date_format_changed)
        group_layout.addRow("Date Format:", self.date_format_combo)
        
        # Custom date format input
        self.custom_date_format_input = QLineEdit()
        self.custom_date_format_input.setPlaceholderText("Enter custom date format")
        self.custom_date_format_input.setVisible(False)
        self.custom_date_format_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Custom Format:", self.custom_date_format_input)
        
        # X-axis / Time column selection
        self.time_column_combo = QComboBox()
        self.time_column_combo.addItem("Auto-detect")
        self.time_column_combo.addItem("Use row index")
        self.time_column_combo.currentTextChanged.connect(self._trigger_preview_update)
        self.time_column_combo.setToolTip("Select which column to use as X-axis (time/index). Auto-detect will find time-like columns automatically.")
        group_layout.addRow("X-Axis Column:", self.time_column_combo)
        
        layout.addWidget(group)
        
    def _create_action_buttons_group(self, layout):
        """Create action buttons group"""
        group = QGroupBox("🎯 Actions")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Preview button
        self.preview_button = QPushButton("🔍 Preview Data")
        self.preview_button.clicked.connect(self._force_preview_update)
        group_layout.addWidget(self.preview_button)
        
        # Auto-detect button
        self.auto_detect_button = QPushButton("🤖 Auto-Detect Settings")
        self.auto_detect_button.clicked.connect(self._auto_detect_settings)
        group_layout.addWidget(self.auto_detect_button)
        
        # Reset button
        self.reset_button = QPushButton("🔄 Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_to_defaults)
        group_layout.addWidget(self.reset_button)
        
        # Re-enable large file warnings button
        self.enable_warnings_button = QPushButton("⚠️ Re-enable Large File Warnings")
        self.enable_warnings_button.clicked.connect(self._enable_large_file_warnings)
        self.enable_warnings_button.setToolTip("Re-enable large file size warnings that were previously disabled")
        group_layout.addWidget(self.enable_warnings_button)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        group_layout.addWidget(separator)
        
        # Parse and create channels button
        self.parse_button = QPushButton("✅ Parse and Create Channels")
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
        
        # Cancel button
        self.cancel_button = QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.close)
        group_layout.addWidget(self.cancel_button)
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with data preview and column configuration"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("📊 Data Preview & Column Configuration")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        right_layout.addWidget(title_label)
        
        # Data Preview Section
        self._create_data_preview_section(right_layout)
        
        # Column Configuration Section
        self._create_column_configuration_section(right_layout)
        
        main_splitter.addWidget(right_panel)
    
    def _create_data_preview_section(self, layout):
        """Create the data preview section"""
        # Preview info label
        self.preview_info_label = QLabel("Load a file to see preview")
        self.preview_info_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        layout.addWidget(self.preview_info_label)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setStretchLastSection(True)
        self.data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.data_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.data_table.setMaximumHeight(200)  # Limit preview table height
        layout.addWidget(self.data_table)
        
        # Parse status
        self.parse_status_label = QLabel("Status: Ready")
        self.parse_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.parse_status_label)
    
    def _create_column_configuration_section(self, layout):
        """Create the column configuration section with longer table"""
        # Column Configuration Group
        group = QGroupBox("📊 Column Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Column names input
        names_layout = QHBoxLayout()
        names_layout.addWidget(QLabel("Column Names:"))
        self.column_names_input = QLineEdit()
        self.column_names_input.setPlaceholderText("Leave blank to use detected headers")
        self.column_names_input.setToolTip("Comma-separated list of column names (useful for non-ASCII headers or when no header row exists)")
        self.column_names_input.textChanged.connect(self._trigger_preview_update)
        names_layout.addWidget(self.column_names_input)
        group_layout.addLayout(names_layout)
        
        # Column types table - make it much longer
        types_label = QLabel("Column Types:")
        types_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        group_layout.addWidget(types_label)
        
        self.column_types_table = QTableWidget()
        self.column_types_table.setColumnCount(3)
        self.column_types_table.setHorizontalHeaderLabels(["Column", "Detected Type", "Override Type"])
        self.column_types_table.horizontalHeader().setStretchLastSection(True)
        # Remove maximum height restriction to allow table to expand
        self.column_types_table.setMinimumHeight(300)  # Set minimum height instead
        group_layout.addWidget(self.column_types_table)
        
        layout.addWidget(group)
        
    def _connect_signals(self):
        """Connect all signal handlers"""
        # Signals are connected in the individual control creation methods
        pass
        
    def _browse_file(self):
        """Browse for file to parse"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select File to Parse",
            "",
            "Data Files (*.csv *.tsv *.txt *.dat);;All Files (*)"
        )
        if file_path:
            self._load_file(Path(file_path))
            
    def _load_file(self, file_path: Path):
        """Load file and show initial preview"""
        try:
            self.file_path = file_path
            self.file_path_label.setText(str(file_path))
            
            # Read file with encoding detection
            self.raw_lines, self.encoding = self._read_file_with_encoding(file_path)
            
            # Update encoding combo
            if self.encoding in [self.encoding_combo.itemText(i) for i in range(self.encoding_combo.count())]:
                self.encoding_combo.setCurrentText(self.encoding)
            
            # Update file info
            file_size = file_path.stat().st_size
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            self.file_info_label.setText(f"Lines: {len(self.raw_lines)}, Size: {size_str}, Encoding: {self.encoding}")
            
            # Auto-detect settings
            self._auto_detect_settings()
            
            # Update preview
            self._trigger_preview_update()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", f"Could not load file:\n{str(e)}")
            
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
            
    def _auto_detect_settings(self):
        """Auto-detect parsing settings from file"""
        if not self.raw_lines:
            return
            
        print(f"[ParseWizard] Auto-detecting settings for {len(self.raw_lines)} lines")
        
        # Since autoparse failed, default delimiter to None instead of auto-detecting
        # This encourages user to manually choose the appropriate delimiter
        self.delimiter_combo.setCurrentText('None')
        print(f"[ParseWizard] Defaulting delimiter to 'None' (encouraging manual selection)")
            
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
        delimiters = [',', '\t', ';', '|', ' ']
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
            
        # Look for metadata lines at the beginning
        metadata_patterns = [r'^#', r'^//', r'^%', r'^;', r'^--', r'^\s*$']
        
        for i, line in enumerate(self.raw_lines):
            if not any(re.match(pattern, line) for pattern in metadata_patterns):
                # Found first non-metadata line - this is likely the header
                return i
                
        # If all lines are metadata, default to row 0
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
            
    def _on_date_format_changed(self, text):
        """Handle date format change"""
        if text == 'custom...':
            self.custom_date_format_input.setVisible(True)
        else:
            self.custom_date_format_input.setVisible(False)
            self.parse_params['date_format'] = text
            
        self._trigger_preview_update()
    
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
        
    def _trigger_preview_update(self):
        """Trigger delayed preview update"""
        self.update_timer.stop()
        self.update_timer.start(self.update_delay)
        
    def _force_preview_update(self):
        """Force immediate preview update"""
        self.update_timer.stop()
        self._update_preview()
        
    def _update_preview(self):
        """Update the data preview table"""
        if not self.raw_lines:
            return
            
        try:
            # Update parse parameters from UI
            self._update_parse_params_from_ui()
            
            # Parse preview data
            self.preview_data = self._parse_preview_data()
            
            if self.preview_data is not None and not self.preview_data.empty:
                self._update_preview_table()
                self._update_column_types_table()
                self._update_time_column_combo()
                
                # Show X-axis column info
                x_col_info = self._get_x_axis_info()
                self.parse_status_label.setText(f"Status: Preview ready ({len(self.preview_data)} rows, {len(self.preview_data.columns)} columns) | X-axis: {x_col_info}")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #e8f5e8; border: 1px solid #4CAF50;")
            else:
                self.parse_status_label.setText("Status: Parse error - check settings")
                self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
                
        except Exception as e:
            self.parse_status_label.setText(f"Status: Error - {str(e)}")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            
    def _update_parse_params_from_ui(self):
        """Update parse parameters from UI controls"""
        self.parse_params.update({
            'header_row': self.header_row_spin.value() if self.header_row_spin.value() >= 0 else None,
            'delete_rows': self.delete_rows_input.text().strip(),
            'quote_char': self.quote_char_input.text() or '"',
            'escape_char': self.escape_char_input.text() or '\\',
            'decimal': self.decimal_combo.currentText(),
            'thousands': self.thousands_combo.currentText(),
            'na_values': [v.strip() for v in self.na_values_input.text().split(',') if v.strip()],
            'parse_dates': self.parse_dates_checkbox.isChecked(),
            'date_format': self.custom_date_format_input.text() if self.custom_date_format_input.isVisible() else self.date_format_combo.currentText(),
            'column_names': [v.strip() for v in self.column_names_input.text().split(',') if v.strip()] if self.column_names_input.text().strip() else [],
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
                    names=self.parse_params['column_names'] if self.parse_params['column_names'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    date_format=self.parse_params['date_format'] if self.parse_params['date_format'] != 'auto' else None,
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            else:
                df = pd.read_csv(
                    io.StringIO(temp_content),
                    sep=self.parse_params['delimiter'],
                    header=header_row,
                    names=self.parse_params['column_names'] if self.parse_params['column_names'] else None,
                    quotechar=self.parse_params['quote_char'],
                    escapechar=self.parse_params['escape_char'],
                    decimal=self.parse_params['decimal'],
                    thousands=self.parse_params['thousands'] if self.parse_params['thousands'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    date_format=self.parse_params['date_format'] if self.parse_params['date_format'] != 'auto' else None,
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            
            # Apply downsampling if enabled
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
        self.data_table.setRowCount(min(len(self.preview_data), 100))  # Limit to 100 rows for preview
        self.data_table.setColumnCount(len(self.preview_data.columns))
        self.data_table.setHorizontalHeaderLabels([str(col) for col in self.preview_data.columns])
        
        # Determine which column is the X-axis
        x_col_selection = self.time_column_combo.currentText()
        x_axis_col_idx = None
        
        if x_col_selection not in ["Auto-detect", "Use row index"]:
            # User manually selected a column
            try:
                x_axis_col_idx = list(self.preview_data.columns).index(x_col_selection)
            except ValueError:
                pass
        elif x_col_selection == "Auto-detect":
            # Find what auto-detect would choose
            auto_detected = None
            for col in self.preview_data.columns:
                if self.preview_data[col].dtype.name.startswith('datetime'):
                    auto_detected = col
                    break
            if not auto_detected:
                time_indicators = ['time', 'timestamp', 'datetime', 'date', 'seconds', 'ms', 't', 'x']
                for col in self.preview_data.columns:
                    col_lower = str(col).lower().strip()
                    if any(indicator in col_lower for indicator in time_indicators):
                        auto_detected = col
                        break
            if auto_detected:
                try:
                    x_axis_col_idx = list(self.preview_data.columns).index(auto_detected)
                except ValueError:
                    pass
        
        # Fill table with data
        for row in range(min(len(self.preview_data), 100)):
            for col in range(len(self.preview_data.columns)):
                value = self.preview_data.iloc[row, col]
                if pd.isna(value):
                    item = QTableWidgetItem("NaN")
                    item.setBackground(QColor(255, 255, 200))  # Light yellow for NaN
                else:
                    item = QTableWidgetItem(str(value))
                    
                # Highlight X-axis column
                if col == x_axis_col_idx:
                    item.setBackground(QColor(200, 230, 255))  # Light blue for X-axis column
                    
                self.data_table.setItem(row, col, item)
        
        # Highlight X-axis column header
        if x_axis_col_idx is not None:
            header_item = self.data_table.horizontalHeaderItem(x_axis_col_idx)
            if header_item:
                header_item.setBackground(QColor(150, 200, 255))  # Darker blue for header
                
        # Update info label
        total_rows = len(self.preview_data)
        preview_rows = min(total_rows, 100)
        x_axis_note = " | Blue highlighting shows X-axis column" if x_axis_col_idx is not None else ""
        self.preview_info_label.setText(f"Showing {preview_rows} of {total_rows} rows, {len(self.preview_data.columns)} columns{x_axis_note}")
        
    def _update_column_types_table(self):
        """Update the column types configuration table"""
        if self.preview_data is None:
            return
        
        # Store current dropdown values before updating
        current_dropdown_values = {}
        current_columns = [str(col) for col in self.preview_data.columns]
        
        # Check if we should preserve dropdown states (same columns, not just same count)
        should_preserve = False
        if hasattr(self, '_last_column_names') and self._last_column_names == current_columns:
            should_preserve = True
            # Same columns - preserve dropdown states
            for i in range(self.column_types_table.rowCount()):
                combo = self.column_types_table.cellWidget(i, 2)
                if combo and isinstance(combo, QComboBox):
                    col_name = self.column_types_table.item(i, 0)
                    if col_name:
                        current_dropdown_values[col_name.text()] = combo.currentText()
        
        self.column_types_table.setRowCount(len(self.preview_data.columns))
        
        for i, col in enumerate(self.preview_data.columns):
            col_str = str(col)
            
            # Column name
            name_item = self.column_types_table.item(i, 0)
            if not name_item or name_item.text() != col_str:
                self.column_types_table.setItem(i, 0, QTableWidgetItem(col_str))
            
            # Detected type
            detected_type = str(self.preview_data[col].dtype)
            type_item = self.column_types_table.item(i, 1)
            if not type_item or type_item.text() != detected_type:
                self.column_types_table.setItem(i, 1, QTableWidgetItem(detected_type))
            
            # Override type combo - only create if doesn't exist or columns changed
            existing_combo = self.column_types_table.cellWidget(i, 2)
            if not existing_combo or not isinstance(existing_combo, QComboBox) or not should_preserve:
                type_combo = QComboBox()
                type_combo.addItems(['auto', 'string', 'float', 'int', 'datetime', 'boolean', 'category'])
                
                # Restore previous value if available
                if should_preserve and col_str in current_dropdown_values:
                    type_combo.setCurrentText(current_dropdown_values[col_str])
                else:
                    type_combo.setCurrentText('auto')
                
                # Set focus policy to avoid interfering with user interaction
                type_combo.setFocusPolicy(Qt.StrongFocus)
                self.column_types_table.setCellWidget(i, 2, type_combo)
        
        # Remember the column names for next update
        self._last_column_names = current_columns
            
    def _update_time_column_combo(self):
        """Update the X-axis column combo with available columns"""
        if self.preview_data is None:
            return
            
        # Store current selection to preserve it
        current_selection = self.time_column_combo.currentText()
        
        self.time_column_combo.clear()
        self.time_column_combo.addItem("Auto-detect")
        self.time_column_combo.addItem("Use row index")
        
        # Add all data columns
        for col in self.preview_data.columns:
            self.time_column_combo.addItem(str(col))
        
        # Restore previous selection if it still exists
        index = self.time_column_combo.findText(current_selection)
        if index >= 0:
            self.time_column_combo.setCurrentIndex(index)
    
    def _get_x_axis_info(self) -> str:
        """Get information about the currently selected X-axis column"""
        if self.preview_data is None:
            return "No data"
            
        x_col_selection = self.time_column_combo.currentText()
        
        if x_col_selection == "Auto-detect":
            # Determine what auto-detect would choose
            auto_detected = None
            
            # Check for datetime columns first
            for col in self.preview_data.columns:
                if self.preview_data[col].dtype.name.startswith('datetime'):
                    auto_detected = col
                    break
            
            # If no datetime, check for time-like names
            if not auto_detected:
                time_indicators = ['time', 'timestamp', 'datetime', 'date', 'seconds', 'ms', 't', 'x']
                for col in self.preview_data.columns:
                    col_lower = str(col).lower().strip()
                    if any(indicator in col_lower for indicator in time_indicators):
                        auto_detected = col
                        break
            
            if auto_detected:
                return f"Auto-detected '{auto_detected}'"
            else:
                return "Auto-detected 'Row Index'"
                
        elif x_col_selection == "Use row index":
            return "Row Index (0, 1, 2, ...)"
            
        else:
            # User manually selected a column
            if x_col_selection in self.preview_data.columns:
                col_type = str(self.preview_data[x_col_selection].dtype)
                return f"'{x_col_selection}' ({col_type})"
            else:
                return f"'{x_col_selection}' (not found)"
            
    def _reset_to_defaults(self):
        """Reset all settings to defaults"""
        self.parse_params = {
            'delimiter': ',',
            'header_row': 0,
            'delete_rows': '',
            'max_rows': 1000,
            'quote_char': '"',
            'escape_char': '\\',
            'encoding': 'utf-8',
            'decimal': '.',
            'thousands': '',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'parse_dates': True,
            'date_format': 'auto',
            'time_column': None,
            'column_names': [],
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
        self.quote_char_input.setText('"')
        self.escape_char_input.setText('\\')
        self.decimal_combo.setCurrentText('.')
        self.thousands_combo.setCurrentText('')
        self.na_values_input.setText('NA,N/A,null,NULL,nan,NaN')
        self.parse_dates_checkbox.setChecked(True)
        self.date_format_combo.setCurrentText('auto')
        self.column_names_input.setText('')
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
        """Helper method to log messages to parent window if available"""
        if hasattr(self.parent_window, 'log_message'):
            self.parent_window.log_message(message, "info")
        
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
                
            # Create file and channels using the managers
            if existing_file:
                # Re-parsing existing file - replace it
                file_obj = self._update_existing_file(existing_file, full_data)
                channels = self._create_channels_from_data(full_data, file_obj)
                
                if channels:
                    # Remove old channels first
                    old_channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
                    for old_channel in old_channels:
                        self.channel_manager.remove_channel(old_channel.channel_id)
                    
                    # Add new channels
                    self.channel_manager.add_channels(channels)
                    
                    # Emit completion signal
                    self.file_parsed.emit(file_obj.file_id)
                    self.parsing_complete.emit({
                        'file_id': file_obj.file_id,
                        'channels_created': len(channels),
                        'rows_parsed': len(full_data),
                        'columns': len(full_data.columns),
                        'reparsed': True
                    })
                    
                    QMessageBox.information(self, "Success", f"Successfully re-parsed file and created {len(channels)} channels!")
                    self.close()
                else:
                    QMessageBox.warning(self, "No Channels", "No valid channels were created from the data.")
            else:
                # New file - create normally
                file_obj = self._create_file_object(full_data)
                channels = self._create_channels_from_data(full_data, file_obj)
                
                if channels:
                    # Add to managers
                    self.file_manager.add_file(file_obj)
                    self.channel_manager.add_channels(channels)
                    
                    # Emit completion signal
                    self.file_parsed.emit(file_obj.file_id)
                    self.parsing_complete.emit({
                        'file_id': file_obj.file_id,
                        'channels_created': len(channels),
                        'rows_parsed': len(full_data),
                        'columns': len(full_data.columns),
                        'reparsed': False
                    })
                    
                    QMessageBox.information(self, "Success", f"Successfully parsed file and created {len(channels)} channels!")
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
                        
                # Create temporary file content
                import io
                temp_content = '\n'.join(filtered_lines)
                
                # Use StringIO for parsing
                file_input = io.StringIO(temp_content)
            else:
                # No rows to delete, use original file
                file_input = self.file_path
            
            # Header row position (no adjustment needed since we already removed deleted rows)
            header_row = self.parse_params['header_row'] if self.parse_params['header_row'] is not None else None
            
            # Parse full file
            # Handle None delimiter (fixed-width parsing)
            if self.parse_params['delimiter'] is None:
                df = pd.read_fwf(
                    file_input,
                    header=header_row,
                    names=self.parse_params['column_names'] if self.parse_params['column_names'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    date_format=self.parse_params['date_format'] if self.parse_params['date_format'] != 'auto' else None,
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            else:
                df = pd.read_csv(
                    file_input,
                    sep=self.parse_params['delimiter'],
                    header=header_row,
                    names=self.parse_params['column_names'] if self.parse_params['column_names'] else None,
                    quotechar=self.parse_params['quote_char'],
                    escapechar=self.parse_params['escape_char'],
                    decimal=self.parse_params['decimal'],
                    thousands=self.parse_params['thousands'] if self.parse_params['thousands'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
                    date_format=self.parse_params['date_format'] if self.parse_params['date_format'] != 'auto' else None,
                    encoding=self.parse_params['encoding'],
                    on_bad_lines='skip'
                )
            
            # Apply downsampling if enabled
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
        
        # Determine X-axis column based on user selection
        x_col_selection = self.time_column_combo.currentText()
        time_col = None
        use_index = False
        
        if x_col_selection == "Auto-detect":
            # Auto-detect time column
            for col in data.columns:
                if data[col].dtype.name.startswith('datetime'):
                    time_col = col
                    break
            # If no datetime column found, check for time-like column names
            if not time_col:
                time_indicators = ['time', 'timestamp', 'datetime', 'date', 'seconds', 'ms', 't', 'x']
                for col in data.columns:
                    col_lower = str(col).lower().strip()
                    if any(indicator in col_lower for indicator in time_indicators):
                        time_col = col
                        break
        elif x_col_selection == "Use row index":
            use_index = True
            time_col = None
        else:
            # User manually selected a specific column
            if x_col_selection in data.columns:
                time_col = x_col_selection
            else:
                # Fallback to auto-detect if selected column doesn't exist
                use_index = True
                
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
            channel = Channel.from_parsing(
                file_id=file_obj.file_id,
                filename=file_obj.filename,
                xdata=xdata if xdata is not None else np.arange(len(series)),
                ydata=series.values,
                xlabel=xlabel,
                ylabel=str(col),
                legend_label=str(col)
            )
            
            channels.append(channel)
            
        return channels
        
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