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
        self.original_preview_data = None  # Store original data before type conversions
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
            'date_formats': ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'],
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
        
        # Load file if provided
        if self.file_path:
            self._load_file(self.file_path)
        
    def _init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Parse Wizard - Manual File Parsing")
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
        
        # Set splitter proportions (1:2 ratio) - left panel smaller, right panel larger
        main_splitter.setSizes([500, 1000])
        
    def _build_left_panel(self, main_splitter):
        """Build the left control panel with compact grouped controls"""
        # Create left panel container
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("Parse Configuration")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        left_layout.addWidget(title_label)
        
        # Add control groups without scrolling - more compact layout
        self._create_file_selection_group(left_layout)
        self._create_basic_parsing_group(left_layout)
        self._create_advanced_parsing_group(left_layout)
        self._create_column_configuration_group(left_layout)
        self._create_action_buttons_group(left_layout)
        
        # Add stretch to push everything to top
        left_layout.addStretch()
        
        main_splitter.addWidget(left_panel)
        
    def _create_file_selection_group(self, layout):
        """Create file selection group"""
        group = QGroupBox("File Selection")
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
        group = QGroupBox("Basic Parsing")
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
        group = QGroupBox("Advanced Parsing")
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
        
        # Date formats
        self.date_formats_input = QLineEdit('%Y-%m-%d,%m/%d/%Y,%d/%m/%Y,%Y-%m-%d %H:%M:%S,%m/%d/%Y %H:%M:%S,%d/%m/%Y %H:%M:%S')
        self.date_formats_input.setToolTip("Comma-separated list of datetime formats to attempt parsing. Add your own formats as needed.")
        self.date_formats_input.textChanged.connect(self._trigger_preview_update)
        group_layout.addRow("Date Formats:", self.date_formats_input)
        
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
        
    def _create_column_configuration_group(self, layout):
        """Create column configuration group for left panel"""
        group = QGroupBox("Column Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Column types table - compact version for left panel
        types_label = QLabel("Column Types:")
        types_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        group_layout.addWidget(types_label)
        
        # Add info label about column name editing
        info_label = QLabel("Tip: Double-click column names to edit them. Channel types are auto-detected and read-only.")
        info_label.setStyleSheet("color: #666; font-size: 10px; font-style: italic; padding: 3px; background-color: #f9f9f9; border: 1px solid #ddd; border-radius: 3px;")
        group_layout.addWidget(info_label)
        
        self.column_types_table = QTableWidget()
        self.column_types_table.setColumnCount(4)
        self.column_types_table.setHorizontalHeaderLabels(["Column", "Detected Type", "Auto-Detected Type", "X-Axis Column"])
        self.column_types_table.horizontalHeader().setStretchLastSection(True)
        # Compact height for left panel
        self.column_types_table.setMinimumHeight(200)
        
        # Enable editing only for column names (first column)
        self.column_types_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Add tooltip to explain column name editing
        self.column_types_table.setToolTip(
            "Column Configuration Table:\n"
            "• Column: Click to edit column names (double-click or press F2)\n"
            "• Detected Type: Original pandas data type (read-only)\n"
            "• Auto-Detected Type: Auto-detected channel type (read-only)\n"
            "• X-Axis Column: Select which column to use as X-axis (radio button)\n\n"
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
        
        group_layout.addWidget(self.column_types_table)
        
        layout.addWidget(group)
        
    def _create_action_buttons_group(self, layout):
        """Create action buttons group"""
        group = QGroupBox("Actions")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        

        
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
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with data preview and column configuration"""
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Data Preview")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        right_layout.addWidget(title_label)
        
        # Data Preview Section
        self._create_data_preview_section(right_layout)
        
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
        # Remove height limit to fill entire right panel
        layout.addWidget(self.data_table)
        
        # Parse status
        self.parse_status_label = QLabel("Status: Ready")
        self.parse_status_label.setStyleSheet("padding: 5px; background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.parse_status_label)
    

        

        
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
            
            print(f"[ParseWizard] Loading file: {file_path.name}")
            
            # Read file with encoding detection
            self.raw_lines, self.encoding = self._read_file_with_encoding(file_path)
            
            print(f"[ParseWizard] File loaded: {len(self.raw_lines)} lines, encoding: {self.encoding}")
            
            # Update encoding combo
            if self.encoding in [self.encoding_combo.itemText(i) for i in range(self.encoding_combo.count())]:
                self.encoding_combo.setCurrentText(self.encoding)
            
            # Update file info
            file_size = file_path.stat().st_size
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            self.file_info_label.setText(f"Lines: {len(self.raw_lines)}, Size: {size_str}, Encoding: {self.encoding}")
            
            # Auto-detect settings
            print("[ParseWizard] Auto-detecting parsing settings...")
            self._auto_detect_settings()
            
            # Update preview
            self._trigger_preview_update()
            
            # Update readiness status
            self._update_readiness_status()
            
        except Exception as e:
            print(f"[ParseWizard] Error loading file: {str(e)}")
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
        
    def _trigger_preview_update(self):
        """Trigger delayed preview update"""
        # Check if parameters actually changed to prevent infinite loops
        current_params = {
            'delimiter': self.parse_params.get('delimiter'),
            'header_row': self.parse_params.get('header_row'),
            'delete_rows': self.parse_params.get('delete_rows', ''),
            'encoding': self.parse_params.get('encoding'),
            'column_names': self.parse_params.get('column_names', [])
        }
        
        if current_params == self._last_parse_params:
            print(f"[ParseWizard] Parameters unchanged, skipping preview update")
            return
            
        self._last_parse_params = current_params.copy()
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
            
        print(f"[ParseWizard] Updating preview with {len(self.raw_lines)} raw lines")
        
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
                
                self._update_preview_table()
                self._update_column_types_table()
                
                # Update readiness status
                self._update_readiness_status()
            else:
                print(f"[ParseWizard] Preview parse failed - no data returned")
                self._update_readiness_status()
                
        except Exception as e:
            print(f"[ParseWizard] Preview update error: {str(e)}")
            self._update_readiness_status()
            
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
                    quotechar=self.parse_params['quote_char'],
                    escapechar=self.parse_params['escape_char'],
                    decimal=self.parse_params['decimal'],
                    thousands=self.parse_params['thousands'] if self.parse_params['thousands'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
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
        x_axis_note = " | Blue highlighting shows X-axis column" if x_axis_col_idx is not None else ""
        self.preview_info_label.setText(f"Showing all {total_rows} rows, {len(self.preview_data.columns)} columns{x_axis_note}")
        
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
            self.parse_status_label.setText("❌ No file loaded")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("✅ File loaded")
        
        # 2. Preview data available
        if self.preview_data is None or len(self.preview_data) == 0:
            self.parse_status_label.setText("❌ No preview data available")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("✅ Preview data available")
        
        # 3. At least one column detected
        if len(self.preview_data.columns) == 0:
            self.parse_status_label.setText("❌ No columns detected")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append(f"✅ {len(self.preview_data.columns)} columns detected")
        
        # 4. X-axis column selected
        selected_button = self.x_axis_button_group.checkedButton()
        if not selected_button:
            self.parse_status_label.setText("❌ No X-axis column selected")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("✅ X-axis column selected")
        
        # 5. Basic parsing configuration
        if not hasattr(self, 'delimiter_combo') or not self.delimiter_combo.currentText():
            self.parse_status_label.setText("❌ Delimiter not configured")
            self.parse_status_label.setStyleSheet("padding: 5px; background-color: #ffe8e8; border: 1px solid #f44336;")
            return
        conditions.append("✅ Parsing configured")
        
        # All conditions met - ready for channel creation
        total_rows = len(self.preview_data)
        status_text = f"✅ Ready to create channels ({total_rows} rows, {len(self.preview_data.columns)} columns)"
        self.parse_status_label.setText(status_text)
        self.parse_status_label.setStyleSheet("padding: 5px; background-color: #e8f5e8; border: 1px solid #4CAF50;")
        
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
        if self.preview_data is None:
            return
        
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
        
        self.column_types_table.setRowCount(len(self.preview_data.columns))
        
        for i, col in enumerate(self.preview_data.columns):
            col_str = str(col)
            
            # Column name - preserve user edits if available
            name_item = self.column_types_table.item(i, 0)
            if should_preserve and i in current_edited_names:
                # Restore user-edited name
                edited_name = current_edited_names[i]
                if not name_item or name_item.text() != edited_name:
                    name_item = QTableWidgetItem(edited_name)
                    name_item.setBackground(QColor(255, 255, 200))  # Light yellow for edited names
                    name_item.setToolTip(f"Edited from '{col_str}' to '{edited_name}'")
                    self.column_types_table.setItem(i, 0, name_item)
                elif name_item:
                    name_item.setBackground(QColor(255, 255, 200))  # Light yellow for edited names
                    name_item.setToolTip(f"Edited from '{col_str}' to '{name_item.text()}'")
            elif not name_item or name_item.text() != col_str:
                # Use original column name
                name_item = QTableWidgetItem(col_str)
                name_item.setBackground(QColor(240, 248, 255))  # Light blue for editable names
                name_item.setToolTip("Double-click to edit column name")
                self.column_types_table.setItem(i, 0, name_item)
            elif name_item:
                # Ensure original names have the editable styling
                name_item.setBackground(QColor(240, 248, 255))  # Light blue for editable names
                name_item.setToolTip("Double-click to edit column name")
            
            # Detected pandas type (for debugging)
            detected_type = str(self.preview_data[col].dtype)
            type_item = self.column_types_table.item(i, 1)
            if not type_item or type_item.text() != detected_type:
                self.column_types_table.setItem(i, 1, QTableWidgetItem(detected_type))
            
            # Auto-detect channel type using original data
            auto_detected_channel_type = self._detect_channel_type(data_for_detection[col])
            
            # Auto-detected type text item (read-only)
            type_item = self.column_types_table.item(i, 2)
            if not type_item or type_item.text() != auto_detected_channel_type:
                type_item = QTableWidgetItem(auto_detected_channel_type)
                type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)  # Make read-only
                type_item.setBackground(QColor(240, 240, 240))  # Light gray background for read-only
                self.column_types_table.setItem(i, 2, type_item)
            
            # X-axis column radio button
            x_axis_radio = self.column_types_table.cellWidget(i, 3)
            if not x_axis_radio or not isinstance(x_axis_radio, QRadioButton):
                x_axis_radio = QRadioButton()
                self.x_axis_button_group.addButton(x_axis_radio, i)
                self.column_types_table.setCellWidget(i, 3, x_axis_radio)
            
            # Enable/disable radio button based on column type
            if auto_detected_channel_type == 'category':
                x_axis_radio.setEnabled(False)
                x_axis_radio.setToolTip(f"Cannot use '{col_str}' as X-axis (categorical data)")
                x_axis_radio.setChecked(False)  # Uncheck if it was previously selected
            else:
                x_axis_radio.setEnabled(True)
                x_axis_radio.setToolTip(f"Select '{col_str}' as X-axis column")
                
                # Auto-select first datetime column or first numeric column if no datetime found
                if auto_detected_channel_type == 'datetime' and not self.x_axis_button_group.checkedButton():
                    x_axis_radio.setChecked(True)
                elif auto_detected_channel_type == 'numeric' and not self.x_axis_button_group.checkedButton():
                    x_axis_radio.setChecked(True)
        
        # Remember the column names for next update
        self._last_column_names = current_columns
            

    
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
            'quote_char': '"',
            'escape_char': '\\',
            'encoding': 'utf-8',
            'decimal': '.',
            'thousands': '',
            'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'nan', 'NaN'],
            'parse_dates': True,
            'date_formats': ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%d/%m/%Y %H:%M:%S'],
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
        self.quote_char_input.setText('"')
        self.escape_char_input.setText('\\')
        self.decimal_combo.setCurrentText('.')
        self.thousands_combo.setCurrentText('')
        self.na_values_input.setText('NA,N/A,null,NULL,nan,NaN')
        self.date_formats_input.setText('%Y-%m-%d,%m/%d/%Y,%d/%m/%Y,%Y-%m-%d %H:%M:%S,%m/%d/%Y %H:%M:%S,%d/%m/%Y %H:%M:%S')
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
            
            # Apply user-selected column type conversions
            user_channel_types = self._get_user_channel_types()
            if user_channel_types:
                print(f"[ParseWizard] Applying user column type conversions for final parsing")
                full_data = self._convert_column_types(full_data, user_channel_types)
                
            # Apply user-edited column names
            user_column_names = self._get_user_column_names()
            if user_column_names:
                print(f"[ParseWizard] Applying user column name changes for final parsing")
                full_data = self._apply_column_name_changes(full_data, user_column_names)
                
            # Create file and channels using the managers
            if existing_file:
                # Re-parsing existing file - replace it
                file_obj = self._update_existing_file(existing_file, full_data)
                channels = self._create_channels_from_data(full_data, file_obj)
                
                if channels:
                    print(f"[ParseWizard] Creating {len(channels)} channels from parsed data")
                    
                    # Remove old channels first
                    old_channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
                    if old_channels:
                        print(f"[ParseWizard] Removing {len(old_channels)} existing channels")
                        for old_channel in old_channels:
                            self.channel_manager.remove_channel(old_channel.channel_id)
                    
                    # Assign colors to channels before adding to manager
                    self._assign_colors_to_channels(channels, file_obj.file_id)
                    
                    # Add new channels
                    self.channel_manager.add_channels(channels)
                    print(f"[ParseWizard] Successfully created {len(channels)} channels")
                    
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
                    print(f"[ParseWizard] Creating {len(channels)} channels from parsed data")
                    
                    # Assign colors to channels before adding to manager
                    self._assign_colors_to_channels(channels, file_obj.file_id)
                    
                    # Add to managers
                    self.file_manager.add_file(file_obj)
                    self.channel_manager.add_channels(channels)
                    print(f"[ParseWizard] Successfully created {len(channels)} channels")
                    
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
                    quotechar=self.parse_params['quote_char'],
                    escapechar=self.parse_params['escape_char'],
                    decimal=self.parse_params['decimal'],
                    thousands=self.parse_params['thousands'] if self.parse_params['thousands'] else None,
                    na_values=self.parse_params['na_values'],
                    parse_dates=self.parse_params['parse_dates'],
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
        
        # Determine X-axis column based on radio button selection
        selected_button = self.x_axis_button_group.checkedButton()
        time_col = None
        use_index = False
        
        if selected_button:
            row = self.x_axis_button_group.id(selected_button)
            if row < len(data.columns):
                time_col = str(data.columns[row])
            else:
                use_index = True
        else:
            # No selection, use row index
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
                        # Convert to numeric for plotting (days since first date)
                        first_date = datetime_series.dropna().iloc[0]
                        df[col] = (datetime_series - first_date).dt.total_seconds() / 86400  # Days
                        
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
            
        print(f"[ParseWizard] Applying column name changes: {user_column_names}")
        
        # Create a mapping of old names to new names
        rename_mapping = {}
        for old_name, new_name in user_column_names.items():
            if old_name in df.columns:
                rename_mapping[old_name] = new_name
                print(f"[ParseWizard] Renaming '{old_name}' → '{new_name}'")
            else:
                print(f"[ParseWizard] Column '{old_name}' not found in DataFrame")
        
        # Apply the renaming
        if rename_mapping:
            df = df.rename(columns=rename_mapping)
            print(f"[ParseWizard] Applied {len(rename_mapping)} column name changes")
        
        return df 