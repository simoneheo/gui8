from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLabel, QLineEdit, QPushButton, QListWidget, QTableWidget, QTableWidgetItem,
    QSplitter, QTextEdit, QCheckBox, QFrame, QTabWidget, QRadioButton, QButtonGroup,
    QGroupBox, QSpinBox, QDoubleSpinBox, QHeaderView
)
from PySide6.QtCore import Qt, QEvent, Signal
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import sys
import time
from typing import Optional, Dict, List
import hashlib
from datetime import datetime

from steps.process_registry import load_all_steps
from steps.process_registry import ProcessRegistry
from process_wizard_manager import ProcessWizardManager
from file_manager import FileManager
from channel_manager import ChannelManager
from channel import SourceType
from steps.base_step import BaseStep
import numpy as np
from scipy.signal import find_peaks

class ScriptChangeTracker:
    """Track changes to process scripts"""
    
    def __init__(self):
        self.original_script = None
        self.script_hash = None
        self.modification_time = None
    
    def initialize_script(self, script):
        """Initialize original script and its hash"""
        self.original_script = script
        self.script_hash = hashlib.md5(script.encode()).hexdigest()
        self.modification_time = None
    
    def is_script_modified(self, current_script):
        """Check if script has been modified"""
        if self.script_hash is None:
            return False
        current_hash = hashlib.md5(current_script.encode()).hexdigest()
        return current_hash != self.script_hash
    
    def mark_script_modified(self):
        """Mark script as modified with timestamp"""
        self.modification_time = datetime.now()
    
    def reset_script(self, new_original):
        """Reset script tracking with new original"""
        self.original_script = new_original
        self.script_hash = hashlib.md5(new_original.encode()).hexdigest()
        self.modification_time = None

class ProcessWizardWindow(QMainWindow):
    """
    Process Wizard window for applying signal processing steps to channels
    """
    
    wizard_closed = Signal()
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None, default_file_id=None):
        super().__init__(parent)
        
        # Store managers with consistent naming
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Store the default file ID to select when opening
        self.default_file_id = default_file_id
        
        # Initialize script change tracker
        self.script_tracker = ScriptChangeTracker()
        
        self.setWindowTitle("Process File")
        self.setMinimumSize(1200, 800)
        
        # Initialize state tracking
        self._stats = {
            'total_processes': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'last_process_time': None,
            'session_start': time.time()
        }
        
        # Initialize processing state
        self.input_ch = None  # Track which channel is selected as input for next step
        self._adding_filter = False  # Flag to prevent dropdown interference during filter operations
        self.radio_button_group = None  # Button group for step table radio buttons
        self._initializing = True  # Flag to prevent filter selection during initialization
        
        # Validate initialization
        if not self._validate_initialization():
            return
            
        # Setup UI components
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Build panels
        self._build_left_panel(main_splitter)
        # self._build_right_panel(main_splitter)
        
        # Initialize manager with proper error handling
        self._initialize_manager()
        
        # Update UI state
        self._update_file_selector()
        self._update_channel_selector()
        
        # Log initialization
        if self.default_file_id:
            selected_file = self.file_manager.get_file(self.default_file_id) if self.file_manager else None
            if selected_file:
                self._log_state_change(f"Process wizard initialized with default file: {selected_file.filename}")
            else:
                self._log_state_change("Process wizard initialized with invalid default file ID")
        else:
            self._log_state_change("Process wizard initialized successfully")

    def _validate_initialization(self) -> bool:
        """Validate that required managers are available"""
        if not self.file_manager:
            self._show_error("File manager not available")
            return False
            
        if not self.channel_manager:
            self._show_error("Channel manager not available")
            return False
            
        return True
        
    def _show_error(self, message: str):
        """Show error message to user"""
        if hasattr(self, 'console_output'):
            self.console_output.append(f"ERROR: {message}")
            
    def _initialize_manager(self):
        """Initialize the process manager with error handling"""
        try:
            # UI initialization moved to _build_left_panel where UI elements exist
            pass
            
        except Exception as e:
            self._show_error(f"Failed to initialize processing manager: {e}")
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        # Debug logging disabled
        
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            **self._stats,
            'available_files': len(self.file_manager.get_all_files()) if self.file_manager else 0,
            'total_channels': self.channel_manager.get_channel_count() if self.channel_manager else 0,
            'session_duration': time.time() - self._stats['session_start']
        }
        
    def _build_left_panel(self, main_splitter):
        """Build the left panel with two vertical columns"""
        # Left Panel with two columns
        self.left_panel = QWidget()
        left_layout = QHBoxLayout(self.left_panel)
        
        # Left Column - Filters Section only
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        
        # Transformations Group
        transformations_group = QGroupBox("Transformations")
        transformations_layout = QVBoxLayout(transformations_group)
        
        # Transformation search
        self.filter_search = QLineEdit()
        self.filter_search.setPlaceholderText("Search transformations...")
        self.filter_search.textChanged.connect(self._on_filter_search)
        
        # Category filter
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.currentTextChanged.connect(self._on_category_filter_changed)
        
        # Transformation list (scrollable)
        self.filter_list = QListWidget()
        
        transformations_layout.addWidget(self.filter_search)
        transformations_layout.addWidget(self.category_filter)
        transformations_layout.addWidget(self.filter_list)
        
        left_column_layout.addWidget(transformations_group)
        
        # Right Column - Control Section stacked top-down
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        
        # File/Channel dropdowns (moved from right panel)
        file_channel_group = QGroupBox("File Selection")
        file_channel_layout = QVBoxLayout(file_channel_group)
        
        # File selector
        file_layout = QHBoxLayout()
        file_label = QLabel("File:")
        self.file_selector = QComboBox()
        self.file_selector.setMinimumWidth(200)  # Set consistent width
        self.file_selector.currentIndexChanged.connect(self._on_file_selected)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_selector)
        
        # Channel selector
        channel_layout = QHBoxLayout()
        channel_label = QLabel("Channel:")
        self.channel_selector = QComboBox()
        self.channel_selector.setMinimumWidth(200)  # Set consistent width
        self.channel_selector.currentIndexChanged.connect(self._on_channel_selected)
        channel_layout.addWidget(channel_label)
        channel_layout.addWidget(self.channel_selector)
        
        file_channel_layout.addLayout(file_layout)
        file_channel_layout.addLayout(channel_layout)
        
        # Input Channel Display
        input_channel_group = QGroupBox("Input Channel")
        input_channel_layout = QVBoxLayout(input_channel_group)
        
        # Replace spinbox with combobox to show channel names directly
        self.input_channel_combobox = QComboBox()
        self.input_channel_combobox.currentIndexChanged.connect(self._on_input_channel_changed)
        
        # Add the combobox directly to the vertical layout to make it span full width
        input_channel_layout.addWidget(self.input_channel_combobox)
        
        # Console Group - only around console output
        console_group = QGroupBox("Console")
        console_layout = QVBoxLayout(console_group)
        
        self.console_output = QTextEdit()
        self.console_output.setPlaceholderText("Output will appear here...")
        self.console_output.setReadOnly(True)
        
        console_layout.addWidget(self.console_output)
        
        # Parameters Group - for parameter table and script
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QVBoxLayout(parameters_group)
        
        # Parameter table
        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.param_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.param_table.setFixedHeight(250)  # Adjust as needed
        
        # Set column resize modes
        param_header = self.param_table.horizontalHeader()
        param_header.setSectionResizeMode(0, QHeaderView.Fixed)  # Parameter column - fixed width
        param_header.setSectionResizeMode(1, QHeaderView.Stretch)  # Value column - stretches
        self.param_table.setColumnWidth(0, 120)  # Parameter column width
        
        # Create tab widget for parameter table and script
        self.param_tab_widget = QTabWidget()
        
        # Parameters tab
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)
        params_layout.addWidget(self.param_table)
        self.param_tab_widget.addTab(params_tab, "Parameters")
        
        # Script tab
        script_tab = QWidget()
        script_layout = QVBoxLayout(script_tab)
        
        # Header with modification status
        header_layout = QHBoxLayout()
        header_label = QLabel("Script:")
        header_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(header_label)
        
        # Modification status label
        self.script_status_label = QLabel("Default")
        self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
        header_layout.addWidget(self.script_status_label)
        header_layout.addStretch()
        
        # Reset button
        self.reset_script_btn = QPushButton("Reset to Default")
        self.reset_script_btn.setMaximumWidth(120)
        self.reset_script_btn.clicked.connect(self._reset_script)
        header_layout.addWidget(self.reset_script_btn)
        
        script_layout.addLayout(header_layout)
        
        # Script editor
        self.script_editor = QTextEdit()
        self.script_editor.setPlaceholderText("Python script will appear here when a transformation is selected...")
        self.script_editor.setFont(QFont("Consolas", 10))
        self.script_editor.textChanged.connect(self._on_script_changed)
        
        # Script controls - simplified
        script_controls_layout = QHBoxLayout()
        self.script_readonly_checkbox = QCheckBox("Read-only")
        self.script_readonly_checkbox.setChecked(False)  # Default to editable
        self.script_readonly_checkbox.stateChanged.connect(self._on_script_readonly_changed)
        
        script_controls_layout.addWidget(self.script_readonly_checkbox)
        script_controls_layout.addStretch()
        
        script_layout.addWidget(self.script_editor)
        script_layout.addLayout(script_controls_layout)
        self.param_tab_widget.addTab(script_tab, "Script")
        
        # Connect tab change to update script when switching to script tab
        self.param_tab_widget.currentChanged.connect(self._on_param_tab_changed)
        
        parameters_layout.addWidget(self.param_tab_widget)
        
        # Channel Name Entry (outside any section, above Apply Operation button)
        channel_name_layout = QHBoxLayout()
        channel_name_label = QLabel("Channel Name:")
        self.channel_name_entry = QLineEdit()
        self.channel_name_entry.setPlaceholderText("Enter custom channel name...")
        self.channel_name_entry.setToolTip("Custom name for the new channel that will be created")
        channel_name_layout.addWidget(channel_name_label)
        channel_name_layout.addWidget(self.channel_name_entry)
        
        # Apply Operation button
        self.add_filter_btn = QPushButton("Apply Operation")
        self.add_filter_btn.clicked.connect(self._on_add_filter)
        self.add_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #228B22;
                color: white;
                border: 2px solid #1E7B1E;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #32CD32;
                border-color: #228B22;
            }
            QPushButton:pressed {
                background-color: #1E7B1E;
            }
            QPushButton:disabled {
                background-color: #9E9E9E;
                color: #666666;
                border-color: #CCCCCC;
            }
        """)
        
        # Add groups to right column
        right_column_layout.addWidget(file_channel_group)
        right_column_layout.addWidget(input_channel_group)
        right_column_layout.addWidget(console_group)
        right_column_layout.addWidget(parameters_group)
        
        # Channel Name Entry (outside any section, above Apply Operation button)
        right_column_layout.addLayout(channel_name_layout)
        
        # Apply Operation button
        right_column_layout.addWidget(self.add_filter_btn)
        
        # Add columns to left panel using splitter for better size control
        left_splitter = QSplitter(Qt.Horizontal)
        left_splitter.addWidget(left_column)
        left_splitter.addWidget(right_column)
        left_splitter.setSizes([350, 500])  # left column smaller, right column wider
        left_layout.addWidget(left_splitter)

        # Right Panel with vertical layout
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # Step Table - styled like main window channel table
        self.step_table = QTableWidget(0, 6)  # 6 columns like main window
        self.step_table.setHorizontalHeaderLabels(["Show", "Style", "Channel Name", "Shape", "fs (Hz)", "Actions"])
        self.step_table.verticalHeader().setVisible(False)
        
        # Set column resize modes for better layout (same as main window)
        header = self.step_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Channel Name - stretches
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Shape - fixed width
        header.setSectionResizeMode(4, QHeaderView.Fixed)     # fs (Hz) - fixed width
        header.setSectionResizeMode(5, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set specific column widths (same as main window)
        self.step_table.setColumnWidth(0, 60)   # Show checkbox
        self.step_table.setColumnWidth(1, 80)   # Style preview
        self.step_table.setColumnWidth(3, 80)   # Shape column
        self.step_table.setColumnWidth(4, 100)  # fs (Hz) column
        self.step_table.setColumnWidth(5, 180)  # Actions buttons
        
        self.step_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.step_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.step_table.setMaximumHeight(150)  # Small height
        self.step_table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_layout.addWidget(self.step_table)

        # Create tab widget for plot area
        self.tab_widget = QTabWidget()

        # Create Time Series tab
        time_series_tab = QWidget()
        time_series_layout = QVBoxLayout(time_series_tab)

        # Plot area
        self.figure = plt.figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, time_series_tab)
        time_series_layout.addWidget(self.toolbar)
        time_series_layout.addWidget(self.canvas)

        # Add tabs
        self.tab_widget.addTab(time_series_tab, "Time Series")
        
        # Connect tab change event
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Add plot area to right panel with stretch priority
        right_layout.addWidget(self.tab_widget, 1)  # Stretch factor 1 = takes remaining space
        
        # Add panels to main splitter
        main_splitter.addWidget(self.left_panel)
        main_splitter.addWidget(self.right_panel)
        
        # Set splitter proportions (left panel smaller, right panel larger)
        main_splitter.setSizes([500, 800])  # Left: 300px, Right: 800px

        # Load steps and setup wizard manager
        self._refresh_steps()
        
        # Clear any default selection that might have been set
        self.filter_list.setCurrentRow(-1)  # Clear selection
        # Create wizard manager
        try:
            print(f"[ProcessWizard] Creating wizard manager...")
            self.wizard_manager = ProcessWizardManager(
                ui=self,
                registry=self.process_registry,
                channel_lookup=self.get_active_channel_info
            )
            print(f"[ProcessWizard] Wizard manager created successfully")
        except Exception as e:
            print(f"[ProcessWizard] Error creating wizard manager: {e}")
            import traceback
            traceback.print_exc()
            # Create a dummy manager to prevent attribute errors
            self.wizard_manager = None
        
        # Set default selection to "resample" filter (after manager is initialized)
        # Do this BEFORE connecting the signal to avoid triggering filter selection
        self._set_default_filter_selection()

        # Connect filter list to manager logic AFTER setting default selection
        if self.wizard_manager:
            self.filter_list.itemClicked.connect(self.wizard_manager._on_filter_selected)
            print(f"[ProcessWizard] Filter list connected to wizard manager")
        else:
            print(f"[ProcessWizard] WARNING: Wizard manager is None, filter list not connected")

        # Update selectors in order: file -> channel -> plots
        self._update_file_selector()
        self._update_channel_selector()

        # Initialize input_ch will be set when dropdown is populated

        # Initial update of tables and plots
        self._update_step_table()
        self._update_plot()
        
        # Set initial helpful console message
        self._set_initial_console_message()
        
        # Mark initialization as complete
        self._initializing = False
        print(f"[ProcessWizard] Initialization completed, filter selection now enabled")
        
    def _update_file_selector(self):
        """Update the file selector with all successfully parsed files."""
        if not self.file_manager:
            return

        # Store previously selected file
        current_file_id = self.file_selector.currentData()
        prev_file_id = current_file_id if current_file_id else None

        self.file_selector.clear()

        # Get all files from file manager
        all_files = self.file_manager.get_all_files()
        if not all_files:
            return

        # Filter for files with successful parse status
        from file import FileStatus
        successfully_parsed_files = [
            file for file in all_files 
            if file.state.status == FileStatus.PARSED or file.state.status == FileStatus.PROCESSED
        ]

        if not successfully_parsed_files:
            return

        # Add successfully parsed files to selector
        for file_info in successfully_parsed_files:
            # Add status indicator in the display name for clarity
            status_text = file_info.state.status.value
            display_name = f"{file_info.filename} ({status_text})"
            self.file_selector.addItem(display_name, file_info.file_id)

        # Priority order for file selection:
        # 1. Previously selected file (if exists)
        # 2. Default file from main window (if provided and valid)
        # 3. First file in the list
        selection_restored = False
        
        # Try to restore previously selected file first
        if prev_file_id:
            for i in range(self.file_selector.count()):
                if self.file_selector.itemData(i) == prev_file_id:
                    self.file_selector.setCurrentIndex(i)
                    selection_restored = True
                    break
        
        # If no previous selection and we have a default file ID, try to select it
        if not selection_restored and self.default_file_id:
            for i in range(self.file_selector.count()):
                if self.file_selector.itemData(i) == self.default_file_id:
                    self.file_selector.setCurrentIndex(i)
                    selection_restored = True
                    self._log_state_change(f"Selected default file from main window")
                    break
        
        # Fall back to first file if nothing else worked
        if not selection_restored and self.file_selector.count() > 0:
            self.file_selector.setCurrentIndex(0)

    def _update_channel_selector(self):
        """Update the channel selector based on selected file and type."""
        if not self.channel_manager:
            return

        # Store previously selected channel
        current_channel = self.get_active_channel_info()
        prev_channel_id = current_channel.channel_id if current_channel else None

        self.channel_selector.clear()

        # Get selected file
        selected_file_id = self.file_selector.currentData()
        if not selected_file_id:
            return

        # Get channels for selected file
        file_channels = self.channel_manager.get_channels_by_file(selected_file_id)
        
        # Filter for RAW channels only
        filtered_channels = [ch for ch in file_channels if ch.type == SourceType.RAW]

        # Add channels to selector
        for ch in filtered_channels:
            self.channel_selector.addItem(ch.legend_label, ch)

        # Restore selection if possible, otherwise select first channel
        selection_restored = False
        if prev_channel_id:
            for i in range(self.channel_selector.count()):
                channel = self.channel_selector.itemData(i)
                if channel.channel_id == prev_channel_id:
                    self.channel_selector.setCurrentIndex(i)
                    selection_restored = True
                    break
        
        if not selection_restored and self.channel_selector.count() > 0:
            self.channel_selector.setCurrentIndex(0)
            selected_channel = self.channel_selector.currentData()

    def get_active_channel_info(self):
        """Get the currently selected channel info."""
        # During filter addition, preserve the explicitly set input_ch
        if (hasattr(self, '_adding_filter') and self._adding_filter and 
            hasattr(self, 'input_ch') and self.input_ch):
            return self.input_ch
        
        # Use the explicitly set input_ch if available
        if hasattr(self, 'input_ch') and self.input_ch:
            return self.input_ch
        
        # Use the input channel combobox selection if cached lineage is available
        if (hasattr(self, '_cached_lineage') and self._cached_lineage and 
            hasattr(self, 'input_channel_combobox') and
            self.input_channel_combobox.currentIndex() >= 0 and 
            self.input_channel_combobox.currentIndex() < len(self._cached_lineage)):
            channel = self._cached_lineage[self.input_channel_combobox.currentIndex()]
            return channel
        
        # Fallback to main channel selector
        if (hasattr(self, 'channel_selector') and 
            self.channel_selector.currentIndex() >= 0):
            channel = self.channel_selector.currentData()
            if channel is not None:
                return channel
        
        return None

    def _build_lineage_for_channel(self, channel):
        """Build lineage for a given channel and cache it."""
        if not channel:
            self._cached_lineage = []
            return
        
        # Get all channels in the lineage
        lineage_dict = self.channel_manager.get_channels_by_lineage(channel.lineage_id)
        
        # Collect all channels from the lineage (parents, children, siblings)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Filter by file_id and sort by step
        lineage = [ch for ch in all_lineage_channels if ch.file_id == channel.file_id]
        lineage.sort(key=lambda ch: ch.step)
        
        # If no lineage found, get all channels from the same file and lineage_id
        if not lineage:
            file_channels = self.channel_manager.get_channels_by_file(channel.file_id)
            lineage = [ch for ch in file_channels if ch.lineage_id == channel.lineage_id]
            # Remove duplicates based on channel_id
            seen_ids = set()
            unique_lineage = []
            for ch in lineage:
                if ch.channel_id not in seen_ids:
                    seen_ids.add(ch.channel_id)
                    unique_lineage.append(ch)
            lineage = unique_lineage
            lineage.sort(key=lambda ch: ch.step)
        
        # Filter lineage based on current tab
        current_tab_index = self.tab_widget.currentIndex()
        filtered_lineage = self._filter_lineage_by_tab(lineage, current_tab_index)
        
        # Cache lineage
        self._cached_lineage = filtered_lineage

    def _update_step_table(self):
        """Update the unified step table with the current channel lineage."""
        # Use cached lineage if available, otherwise build it
        if not hasattr(self, '_cached_lineage') or not self._cached_lineage:
            active_channel = self.get_active_channel_info()
            if not active_channel:
                return
            self._build_lineage_for_channel(active_channel)
        
        filtered_lineage = self._cached_lineage
        
        self.step_table.setRowCount(len(filtered_lineage))

        for i, channel in enumerate(filtered_lineage):
            # Column 0: Show (checkbox for visibility toggle)
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(channel.show)
            show_checkbox.stateChanged.connect(lambda state, ch_id=channel.channel_id: self._toggle_channel_visibility(ch_id))
            
            # Center the checkbox in the cell
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(show_checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.step_table.setCellWidget(i, 0, checkbox_widget)
            
            # Column 1: Style (visual preview widget)
            from plot_manager import StylePreviewWidget
            style_widget = StylePreviewWidget(
                color=channel.color or '#1f77b4',
                style=channel.style or '-',
                marker=channel.marker if channel.marker != "None" else None
            )
            self.step_table.setCellWidget(i, 1, style_widget)
            
            # Column 2: Channel Name (legend label)
            channel_name = channel.legend_label or channel.ylabel or f"Step {channel.step}"
            self.step_table.setItem(i, 2, QTableWidgetItem(channel_name))
            
            # Column 3: Shape (data shape/length)
            if channel.xdata is not None and channel.ydata is not None:
                shape_str = f"({len(channel.xdata)}, 2)"
            elif channel.ydata is not None:
                shape_str = f"({len(channel.ydata)},)"
            else:
                shape_str = "No data"
            self.step_table.setItem(i, 3, QTableWidgetItem(shape_str))
            
            # Column 4: fs (Hz) - sampling frequency (median ± std)
            if hasattr(channel, 'fs_median') and channel.fs_median:
                if hasattr(channel, 'fs_std') and channel.fs_std:
                    fs_str = f"{channel.fs_median:.1f}±{channel.fs_std:.1f}"
                else:
                    fs_str = f"{channel.fs_median:.1f}"
            else:
                fs_str = "N/A"
            self.step_table.setItem(i, 4, QTableWidgetItem(fs_str))
            
            # Column 5: Actions (info, inspect, styling, transform, delete)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Info button (channel information)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(25)
            info_button.setMaximumHeight(25)
            info_button.setToolTip("Channel information and metadata")
            info_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._show_channel_info(ch_id))
            actions_layout.addWidget(info_button)
            
            # Magnifying glass button (inspect data)
            zoom_button = QPushButton("🔍")
            zoom_button.setMaximumWidth(25)
            zoom_button.setMaximumHeight(25)
            zoom_button.setToolTip("Inspect and edit channel data")
            zoom_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._inspect_channel_data(ch_id))
            actions_layout.addWidget(zoom_button)
            
            # Paint brush button (styling)
            style_button = QPushButton("🎨")
            style_button.setMaximumWidth(25)
            style_button.setMaximumHeight(25)
            style_button.setToolTip("Channel styling and appearance settings")
            style_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._handle_gear_button_clicked(ch_id))
            actions_layout.addWidget(style_button)
            
            # Tool button (transform data)
            tool_button = QPushButton("🔨")
            tool_button.setMaximumWidth(25)
            tool_button.setMaximumHeight(25)
            tool_button.setToolTip("Transform channel data with math expressions")
            tool_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._transform_channel_data(ch_id))
            actions_layout.addWidget(tool_button)
            
            # Trash button (delete) - always last
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(25)
            delete_button.setMaximumHeight(25)
            
            # Disable delete button for RAW channels (step = 0)
            if hasattr(channel, 'step') and channel.step == 0:
                delete_button.setEnabled(False)
                delete_button.setToolTip("Cannot delete RAW channel")
            else:
                delete_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._delete_channel(ch_id))
                delete_button.setToolTip("Delete channel")
            
            actions_layout.addWidget(delete_button)
            
            self.step_table.setCellWidget(i, 5, actions_widget)

            # Enhanced tooltip with filename, parameters and parent step information
            tooltip_parts = []
            
            # Add filename information
            if hasattr(channel, "file_id") and channel.file_id and self.file_manager:
                file_obj = self.file_manager.get_file(channel.file_id)
                if file_obj:
                    tooltip_parts.append(f"📁 File: {file_obj.filename}")
                else:
                    tooltip_parts.append(f"📁 File: Unknown (ID: {channel.file_id})")
            else:
                tooltip_parts.append("📁 File: Unknown")
            
            # Add channel details
            tooltip_parts.append(f"🏷️ Channel: {channel.legend_label or channel.ylabel or channel.channel_id}")
            tooltip_parts.append(f"📊 Step: {channel.step}")
            
            # Add data shape information
            if hasattr(channel, 'ydata') and channel.ydata is not None:
                tooltip_parts.append(f"📈 Data points: {len(channel.ydata):,}")
            
            # Add sampling frequency if available
            if hasattr(channel, 'fs_median') and channel.fs_median:
                if hasattr(channel, 'fs_std') and channel.fs_std:
                    tooltip_parts.append(f"⏱️ Sampling rate: {channel.fs_median:.1f}±{channel.fs_std:.1f} Hz")
                else:
                    tooltip_parts.append(f"⏱️ Sampling rate: {channel.fs_median:.1f} Hz")
            
            # Add parameters information
            if hasattr(channel, "params") and channel.params:
                param_str = ", ".join(f"{k}={v}" for k, v in channel.params.items() if k != "fs")
                if param_str:
                    tooltip_parts.append(f"⚙️ Params: {param_str}")
                else:
                    tooltip_parts.append("⚙️ Params: None")
            else:
                tooltip_parts.append("⚙️ Params: None")
            
            # Add parent step information
            if hasattr(channel, "parent_ids") and channel.parent_ids:
                parent_steps = []
                for parent_id in channel.parent_ids:
                    parent_channel = self.channel_manager.get_channel(parent_id)
                    if parent_channel:
                        parent_steps.append(str(parent_channel.step))
                if parent_steps:
                    tooltip_parts.append(f"⬆️ Parent step(s): {', '.join(parent_steps)}")
                else:
                    tooltip_parts.append("⬆️ Parent step: unknown")
            else:
                if channel.step == 0:
                    tooltip_parts.append("⬆️ Parent step: none (RAW)")
                else:
                    tooltip_parts.append("⬆️ Parent step: unknown")
            
            # Add processing description if available
            if hasattr(channel, 'description') and channel.description:
                tooltip_parts.append(f"📝 Description: {channel.description}")
            
            tooltip = "\n".join(tooltip_parts)
            
            for col in range(self.step_table.columnCount()):
                item = self.step_table.item(i, col)
                if item:
                    item.setToolTip(tooltip)
                # Also set tooltip on widgets
                widget = self.step_table.cellWidget(i, col)
                if widget and col != 1:  # Don't override the style widget tooltip
                    widget.setToolTip(tooltip)
        
        # Ensure column widths are maintained after table refresh (same as main window)
        self.step_table.setColumnWidth(0, 60)   # Show checkbox
        self.step_table.setColumnWidth(1, 80)   # Style preview
        self.step_table.setColumnWidth(3, 80)   # Shape column
        self.step_table.setColumnWidth(4, 100)  # fs (Hz) column
        self.step_table.setColumnWidth(5, 180)  # Actions buttons

    def _filter_lineage_by_tab(self, lineage, tab_index):
        """Filter lineage channels based on the current tab."""
        # Since we only have time series tab now, return all channels
        return lineage

    def _update_plot(self):
        """Update the plot with the current channel data."""
        active_channel = self.get_active_channel_info()
        if not active_channel:
            return

        # Get all channels in the lineage
        lineage_dict = self.channel_manager.get_channels_by_lineage(active_channel.lineage_id)
        
        # Collect all channels from the lineage (parents, children, siblings)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Filter by file_id and sort by step
        lineage = [ch for ch in all_lineage_channels if ch.file_id == active_channel.file_id]
        lineage.sort(key=lambda ch: ch.step)
        
        # If no lineage found, get all channels from the same file and lineage_id
        if not lineage:
            file_channels = self.channel_manager.get_channels_by_file(active_channel.file_id)
            lineage = [ch for ch in file_channels if ch.lineage_id == active_channel.lineage_id]
            lineage.sort(key=lambda ch: ch.step)
        
        # Update time series plot
        self._update_time_series_plot(lineage, active_channel)

    def _update_time_series_plot(self, lineage, active_channel):
        """Update the time series plot."""
        # Include all channels since we only have time series now
        channels = lineage
        
        # Clear the plot
        self.ax.clear()
        
        # Plot each channel using stored style properties
        for ch in channels:
            if ch.show:  # Only plot visible channels
                # Use stored style properties from channel
                color = getattr(ch, 'color', '#1f77b4')
                linestyle = getattr(ch, 'style', '-')
                marker = getattr(ch, 'marker', 'None')
                
                # Convert "None" strings to None for matplotlib
                linestyle = None if linestyle == "None" else linestyle
                marker = None if marker == "None" else marker
                
                # Check if data exists
                if hasattr(ch, 'xdata') and hasattr(ch, 'ydata') and ch.xdata is not None and ch.ydata is not None:
                    self.ax.plot(ch.xdata, ch.ydata, 
                               color=color, 
                               linestyle=linestyle, 
                               marker=marker, 
                               label=ch.legend_label)

        # Set title and update
        self.ax.set_title(f"File: {active_channel.filename}")
        self.ax.legend().set_visible(False)
        self.canvas.draw()

    def _on_tab_changed(self, index):
        """Handle tab change event."""
        # Since we only have one tab now, just update the plot
        self._update_plot()

    def _on_file_selected(self, index):
        """Handle file selection change."""
        if index < 0:
            return
            
        # Don't override input_ch if we're in the middle of adding a filter
        if self._adding_filter:
            return
            
        # Get selected file info
        selected_file_id = self.file_selector.currentData()
        selected_file_name = self.file_selector.currentText()
        
        # Clear input_ch so we auto-select the most recent channel in the new file
        self.input_ch = None
        
        # Clear cached lineage to ensure we don't use old file's channels
        self._cached_lineage = []
        
        # Update channel selector when file changes
        self._update_channel_selector()
        
        # Force immediate update of lineage and Input Channel dropdown
        # Get the newly selected channel from the channel selector
        if self.channel_selector.count() > 0:
            selected_channel = self.channel_selector.currentData()
            if selected_channel:
                # Build lineage for the new channel
                self._build_lineage_for_channel(selected_channel)
                # Update Input Channel dropdown with the new lineage
                self._update_input_channel_combobox()
        
        # Update table and plot (this will now use the updated Input Channel dropdown)
        self._update_step_table()
        self._update_plot()

    def _on_channel_selected(self, index):
        """Handle channel selection change."""
        if index < 0:
            return
            
        # Get the selected channel
        selected_channel = self.channel_selector.currentData()
        if not selected_channel:
            return
        
        # Don't override input_ch if we're in the middle of adding a filter
        if self._adding_filter:
            return
        
        # Clear input_ch so we auto-select the most recent channel in the new lineage
        # This ensures we show the most recent channel in the selected lineage
        self.input_ch = None
        
        # Build lineage for the newly selected channel
        self._build_lineage_for_channel(selected_channel)
        
        # Update Input Channel dropdown with the new lineage
        self._update_input_channel_combobox()
        
        # Update both table and plot (this will now use the updated Input Channel dropdown)
        self._update_step_table()
        self._update_plot()

    def _on_show_changed(self, channel, state):
        """Handle show checkbox state change."""
        channel.show = bool(state)
        self._update_plot()

    def _on_input_step_selected(self, channel):
        """Handle radio input selection for processing."""
        # When user selects a step as input, update the plots
        self._update_plot()

    def _on_add_filter(self):
        # Set flag to prevent dropdown interference
        self._adding_filter = True

        # Ensure we use the radio-button-selected channel as the parent
        parent_channel = self.get_active_channel_info()
        if not parent_channel:
            self._adding_filter = False
            return

        parent_id = parent_channel.channel_id
        lineage_id = parent_channel.lineage_id

        # Try to use custom script first, then fallback to hardcoded script
        new_channel = None
        used_custom_script = False
        
        try:
            # Check if we have a custom script in the script editor
            if hasattr(self, 'script_editor') and self.script_editor:
                script_text = self.script_editor.toPlainText().strip()
                
                # Check if script has been modified from default (contains user changes)
                if script_text and self._is_script_customized(script_text):
                    print(f"[ProcessWizard] Attempting to use custom script...")
                    print(f"[ProcessWizard] Script text: {script_text}")
                    
                    # Try to execute the custom script
                    try:
                        # Collect current parameters for fallback
                        fallback_params = self._get_params_from_table()
                        
                        # Execute the custom script
                        new_channel = self.wizard_manager._execute_script_safely(script_text, fallback_params)
                        used_custom_script = True
                        print(f"[ProcessWizard] Custom script executed successfully!")
                        
                        # Update console with success message
                        self.console_output.setPlainText(
                            f"Custom script executed successfully!\n"
                            f"Created channel: {new_channel.channel_id if new_channel else 'Unknown'}"
                        )
                        
                    except Exception as script_e:
                        print(f"[ProcessWizard] Custom script failed: {script_e}")
                        # Don't return here - fall through to use hardcoded script
                        self.console_output.setPlainText(
                            f"Custom script failed: {script_e}\n"
                            f"Falling back to hardcoded script..."
                        )
                        
        except Exception as e:
            print(f"[ProcessWizard] Error checking custom script: {e}")
        
        # If custom script didn't work or wasn't used, use hardcoded script
        if not new_channel:
            print(f"[ProcessWizard] Using hardcoded script...")
            try:
                new_channel = self.wizard_manager.apply_pending_step()
                if new_channel:
                    print(f"[ProcessWizard] Hardcoded script executed successfully!")
                    if not used_custom_script:  # Only update console if we didn't already show a custom script message
                        # Get the channel name for display
                        channel_name = new_channel.legend_label or new_channel.ylabel or f"Channel {new_channel.channel_id}"
                        console_message = f"Operation applied successfully!\nCreated channel: {channel_name}"
                        
                        # Check for repair information in the created channel
                        if (hasattr(new_channel, 'metadata') and new_channel.metadata is not None and 
                            'data_repair_info' in new_channel.metadata):
                            repair_info = new_channel.metadata['data_repair_info']
                            if repair_info and repair_info != "No repairs needed":
                                console_message += f"\n\nData Repair Applied:\n{repair_info}"
                        
                        self.console_output.setPlainText(console_message)
            except Exception as fallback_e:
                print(f"[ProcessWizard] Hardcoded script also failed: {fallback_e}")
                self.console_output.setPlainText(f"Both custom and hardcoded scripts failed: {fallback_e}")
                self._adding_filter = False
                return

        if not new_channel:
            self._adding_filter = False
            return

        # CRITICAL: Set the new channel as the input for next step BEFORE any UI updates
        self.input_ch = new_channel

        # Update selectors to include new channels (but dropdown won't override input_ch due to flag)
        self._update_file_selector()
        self._update_channel_selector()

        # CRITICAL: Rebuild the lineage to include the new channel
        # Build lineage for the new channel to include it in the cached lineage
        self._build_lineage_for_channel(new_channel)
        
        # Update Input Channel dropdown with the new lineage
        self._update_input_channel_combobox()

        # Switch to appropriate tab based on created channel type
        # Since we only have time series tab now, no need to switch tabs

        # Update step table and plot with the new channel properly selected
        self._update_step_table()  # This should now show the new channel in the updated lineage
        
        # Clear the flag after all updates are complete
        self._adding_filter = False
        
        self._update_plot()
        
        # Force redraw of canvas
        self.canvas.draw()

    def _is_script_customized(self, script_text):
        """Check if the script has been customized by the user (not just the default generated script)"""
        try:
            # Use the wizard manager's script customization detection if available
            if hasattr(self, 'wizard_manager') and self.wizard_manager:
                return self.wizard_manager._is_script_customized()
            
            # Fallback: Basic check for customization signs
            if not script_text or script_text.strip().startswith("# No filter selected"):
                return False
            
            # Look for actual code changes (presence of return statements, different variables, etc.)
            if ('return' in script_text and 
                ('y_new' in script_text or 'result_channel' in script_text or 'result_channels' in script_text)):
                return True
            
            # Check for actual y_new assignments (not just comments)
            lines = script_text.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('y_new') and '=' in stripped and not stripped.startswith('#'):
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ProcessWizard] Error checking if script is customized: {e}")
            # If we can't determine, assume it might be customized if it contains return statements
            return 'return' in script_text.lower()


            


    def _on_console_input(self):
        """Handle console input submission."""
        # Use the wizard manager's unified parameter collection method
        if hasattr(self, 'wizard_manager') and self.wizard_manager:
            params = self.wizard_manager._collect_parameters_from_table()
        else:
            # Fallback to local collection if wizard manager is not available
            params = self._get_params_from_table()
        
        # Debug output
        print(f"[ProcessWizardWindow] Console input parameters: {params}")
        
        self.wizard_manager.on_input_submitted(params)
        self._update_step_table()
        self._update_plot()

    def _refresh_steps(self):
        """Refresh the step list by reloading all steps from the steps directory."""
        print("[ProcessWizard] Refreshing steps...")
        
        # Initialize or clear the registry
        if not hasattr(self, 'process_registry') or self.process_registry is None:
            self.process_registry = ProcessRegistry
        else:
            # Clear the current registry (this is important to avoid duplicates)
            self.process_registry._registry.clear()
        
        # Reload all steps
        load_all_steps("steps")
        self.all_filters = self.process_registry.all_steps()
        
        # Update the filter list
        self.filter_list.clear()
        self.filter_list.addItems(self.all_filters)
        
        # Update the category filter
        self._populate_category_filter()
        
        print(f"[ProcessWizard] Refreshed {len(self.all_filters)} steps with categories: {[self.category_filter.itemText(i) for i in range(1, self.category_filter.count())]}")

    def _populate_category_filter(self):
        """Populate the category dropdown with unique categories from all steps."""
        categories = set()
        
        for step_name in self.all_filters:
            step_cls = self.process_registry.get(step_name)
            if step_cls and hasattr(step_cls, 'category'):
                categories.add(step_cls.category)
        
        # Clear existing items and add "All Categories" option
        self.category_filter.clear()
        self.category_filter.addItem("All Categories")
        
        # Sort categories alphabetically and add to dropdown
        sorted_categories = sorted(categories)
        for category in sorted_categories:
            self.category_filter.addItem(category)

    def _on_category_filter_changed(self, category_text):
        """Handle category filter change."""
        self._apply_filters()

    def _on_filter_search(self, text):
        """Handle filter search functionality."""
        self._apply_filters()

    def _apply_filters(self):
        """Apply both search and category filters to the filter list."""
        search_text = self.filter_search.text().strip().lower()
        selected_category = self.category_filter.currentText()
        
        # Clear the current list
        self.filter_list.clear()
        
        # Start with all filters
        filtered_steps = []
        
        for step_name in self.all_filters:
            # Get the step class to access its metadata
            step_cls = self.process_registry.get(step_name)
            if not step_cls:
                continue
            
            # Apply category filter
            if selected_category != "All Categories":
                step_category = getattr(step_cls, 'category', '')
                if step_category != selected_category:
                    continue  # Skip this step if it doesn't match the selected category
            
            # Apply search filter
            if search_text:
                # Search in multiple fields for better matches
                searchable_text = " ".join([
                    step_name.lower(),
                    step_cls.category.lower() if hasattr(step_cls, 'category') else "",
                    step_cls.description.lower() if hasattr(step_cls, 'description') else "",
                    " ".join(step_cls.tags).lower() if hasattr(step_cls, 'tags') else ""
                ])
                
                # Check if search text matches any part
                if search_text not in searchable_text:
                    continue  # Skip this step if it doesn't match the search
            
            # If we get here, the step passed both filters
            filtered_steps.append(step_name)
        
        # Add filtered results to the list
        self.filter_list.addItems(filtered_steps)
        
        # If only one result, auto-select it
        if len(filtered_steps) == 1:
            self.filter_list.setCurrentRow(0)



    # Action methods for the Actions column
    def _toggle_channel_visibility(self, channel_id: str):
        """Toggle visibility of a channel and update UI"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Toggle visibility
            new_state = not channel.show
            self.channel_manager.set_channel_visibility(channel_id, new_state)

            
            # Refresh the step table to update button appearance
            self._update_step_table()
            self._update_plot()

    def _handle_gear_button_clicked(self, channel_id: str):
        """Handle gear button click for channel settings"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the line wizard for channel styling
            from line_wizard import LineWizard
            wizard = LineWizard(channel, self)
            wizard.exec()

            self._update_step_table()
            self._update_plot()

    def _show_channel_info(self, channel_id: str):
        """Show detailed information about a channel using the metadata wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the comprehensive metadata wizard
            from metadata_wizard import MetadataWizard
            wizard = MetadataWizard(channel, self, self.file_manager)
            wizard.exec()


    def _inspect_channel_data(self, channel_id: str):
        """Open the data inspection wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data inspection wizard
            from inspection_wizard import InspectionWizard
            wizard = InspectionWizard(channel, self)
            wizard.data_updated.connect(self._handle_channel_data_updated)
            wizard.exec()


    def _transform_channel_data(self, channel_id: str):
        """Open the data transformation wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data transformation wizard
            from transform_wizard import TransformWizard
            wizard = TransformWizard(channel, self)
            wizard.data_updated.connect(self._handle_channel_data_updated)
            wizard.exec()


    def _delete_channel(self, channel_id: str):
        """Delete a channel with confirmation"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, 
                "Delete Channel", 
                f"Delete channel '{channel.ylabel}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # Check if this was the currently selected input channel
                was_current_input = (hasattr(self, 'input_ch') and 
                                   self.input_ch and 
                                   self.input_ch.channel_id == channel_id)
                
                # Store parent info before deletion for fallback selection
                parent_channel_id = None
                if hasattr(channel, 'parent_ids') and channel.parent_ids:
                    parent_channel_id = channel.parent_ids[0]
                
                # Remove the channel
                self.channel_manager.remove_channel(channel_id)
                
                # Clear cached lineage to force rebuild with updated channel list
                self._cached_lineage = []
                
                # Update UI components
                self._update_step_table()
                self._update_input_channel_combobox()
                self._update_plot()
                
                # If we deleted the current input channel, select a new one
                if was_current_input:
                    self._select_appropriate_input_channel(parent_channel_id)


    def _handle_channel_data_updated(self, channel_id: str):
        """Handle when channel data is updated via inspection/transform wizards"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Refresh all UI components immediately
            self._update_step_table()  # Update step table with new statistics
            self._update_plot()  # Update plot canvas
            
            # Force plot canvas to redraw immediately
            self.canvas.draw()
            




    def _update_input_channel_combobox(self):
        """Update the input channel combobox with all channels in the steps table."""
        
        if not hasattr(self, '_cached_lineage') or not self._cached_lineage:
            self.input_channel_combobox.clear()
            self.input_channel_combobox.addItem("No channels available")
            return
            
        # Priority 1: Use the explicitly set input_ch (most recently created/selected channel)
        target_channel_id = None
        if hasattr(self, 'input_ch') and self.input_ch:
            target_channel_id = self.input_ch.channel_id
        
        # Priority 2: Use current combobox selection if input_ch is not set
        if not target_channel_id and self.input_channel_combobox.currentIndex() >= 0 and self.input_channel_combobox.currentIndex() < len(self._cached_lineage):
            target_channel_id = self._cached_lineage[self.input_channel_combobox.currentIndex()].channel_id
        
        # Clear and populate combobox with channel names
        self.input_channel_combobox.clear()
        for channel in self._cached_lineage:
            channel_name = channel.legend_label or channel.ylabel or f"Step {channel.step}"
            self.input_channel_combobox.addItem(channel_name)
        
        # Set selection based on target channel
        selection_made = False
        if target_channel_id:
            for i, channel in enumerate(self._cached_lineage):
                if channel.channel_id == target_channel_id:
                    self.input_channel_combobox.setCurrentIndex(i)
                    self.input_ch = channel
                    selection_made = True
                    break
        
        # Fallback: Select the most recent channel (highest step number)
        if not selection_made and len(self._cached_lineage) > 0:
            # Find the channel with the highest step number (most recently created)
            most_recent_channel = max(self._cached_lineage, key=lambda ch: ch.step)
            for i, channel in enumerate(self._cached_lineage):
                if channel.channel_id == most_recent_channel.channel_id:
                    self.input_channel_combobox.setCurrentIndex(i)
                    self.input_ch = channel
                    break

    def _on_input_channel_changed(self, index):
        """Handle input channel combobox selection change."""
        if (index < 0 or not hasattr(self, '_cached_lineage') or 
            not self._cached_lineage or index >= len(self._cached_lineage)):
            return
            
        selected_channel = self._cached_lineage[index]
        if selected_channel:
            self.input_ch = selected_channel
            self._update_plot()

    def _select_appropriate_input_channel(self, deleted_parent_id: str = None):
        """Select an appropriate input channel after deletion using priority order:
        1. Most recent channel (highest step number)
        2. Parent channel (if available)
        3. First available channel
        """
        if not hasattr(self, '_cached_lineage') or not self._cached_lineage:
            return
        
        # Priority 1: Most recent channel (highest step number)
        most_recent = max(self._cached_lineage, key=lambda ch: ch.step)
        if most_recent:
            self._set_input_channel(most_recent)
            return
        
        # Priority 2: Parent channel (if available and still exists)
        if deleted_parent_id:
            parent_channel = self.channel_manager.get_channel(deleted_parent_id)
            if parent_channel and parent_channel in self._cached_lineage:
                self._set_input_channel(parent_channel)
                return
        
        # Priority 3: First available channel
        if self._cached_lineage:
            self._set_input_channel(self._cached_lineage[0])

    def _set_input_channel(self, channel):
        """Set the input channel and update the combobox selection"""
        self.input_ch = channel
        
        # Update combobox selection
        if hasattr(self, 'input_channel_combobox'):
            for i, cached_channel in enumerate(self._cached_lineage):
                if cached_channel.channel_id == channel.channel_id:
                    self.input_channel_combobox.setCurrentIndex(i)
                    break

    def showEvent(self, event):
        """Handle when the wizard window is shown."""
        super().showEvent(event)

    def _set_initial_console_message(self):
        """Set initial helpful console message when the Process Wizard opens"""
        try:
            welcome_msg = """Welcome to the Process Wizard!

Quick Start:
1. Select a Transformation from the left panel
2. (Optional) Edit the Channel Name
3. Click "Apply Operation" to plot

Tips:
• Transformations apply to the selected Input Channel
• Adjust parameters in the tab below
• The Scripts tab shows editable Python code — use with caution"""

            
            self.console_output.setPlainText(welcome_msg)
            
        except Exception as e:
            print(f"[ProcessWizard] Error setting initial console message: {e}")
            # Fallback to basic message
            try:
                self.console_output.setPlainText("Welcome to the Process Wizard!\n\nSelect a transformation, configure parameters, and click Apply Operation.")
            except:
                pass

    def _set_default_filter_selection(self):
        """Set default selection to "resample" filter."""
        try:
            print(f"[ProcessWizard] Setting default filter selection...")
            
            # Block signals during default selection to prevent crashes
            self.filter_list.blockSignals(True)
            
            # Find "resample" in the filter list
            for i in range(self.filter_list.count()):
                if self.filter_list.item(i).text() == "resample":
                    self.filter_list.setCurrentRow(i)
                    print(f"[ProcessWizard] Default filter 'resample' selected at index {i}")
                    break
                    
            # Re-enable signals
            self.filter_list.blockSignals(False)
            print(f"[ProcessWizard] Default filter selection completed")
            
        except Exception as e:
            print(f"[ProcessWizard] Error setting default filter selection: {e}")
            # Make sure to re-enable signals even if there's an error
            try:
                self.filter_list.blockSignals(False)
            except:
                pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.wizard_closed.emit()
        event.accept()
    
    def _on_param_tab_changed(self, index):
        """Handle parameter tab change to update script when switching to script tab"""
        try:
            print(f"[ProcessWizard] Parameter tab changed to index: {index}")
            if index == 1:  # Script tab selected
                print(f"[ProcessWizard] Script tab selected, syncing script...")
                self._sync_script_from_params()
                print(f"[ProcessWizard] Script sync completed")
        except Exception as e:
            print(f"[ProcessWizard] Error in _on_param_tab_changed: {e}")
            import traceback
            traceback.print_exc()
            # Try to set an error message in the script editor
            try:
                if hasattr(self, 'script_editor'):
                    self.script_editor.setPlainText(f"# Error updating script: {e}\n# Please try selecting the filter again")
            except:
                pass
    
    def _on_script_readonly_changed(self, state):
        """Handle script read-only checkbox change"""
        self.script_editor.setReadOnly(state == Qt.Checked)
        # self.sync_to_params_btn.setEnabled(state != Qt.Checked) # Removed as per edit hint
    
    def _sync_script_from_params(self):
        """Generate script from current parameters or expose step's script method"""
        try:
            print(f"[ProcessWizard] _sync_script_from_params() called")
            
            if not hasattr(self.wizard_manager, 'pending_step') or not self.wizard_manager.pending_step:
                no_filter_script = "# No filter selected\n# Select a filter from the list to generate script"
                self.script_editor.setPlainText(no_filter_script)
                # Initialize script tracker with no filter message
                self.script_tracker.initialize_script(no_filter_script)
                self.script_status_label.setText("Default")
                self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
                return
            
            step_cls = self.wizard_manager.pending_step
            step_name = step_cls.name
            print(f"[ProcessWizard] Generating script for step: {step_name}")
            
            # Get current parameters from table safely
            try:
                params = self._get_params_from_table()
                print(f"[ProcessWizard] Parameters extracted: {params}")
            except Exception as param_e:
                print(f"[ProcessWizard] Error getting parameters: {param_e}")
                error_script = f"# Error getting parameters: {param_e}\n# Please check the parameter table"
                self.script_editor.setPlainText(error_script)
                # Initialize script tracker with error message
                self.script_tracker.initialize_script(error_script)
                self.script_status_label.setText("Default")
                self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
                return
            
            # Try to get the step's actual script method
            script = self._generate_script_from_step_method(step_cls, params)
            if script:
                self.script_editor.setPlainText(script)
                # Initialize script tracker with the generated script
                self.script_tracker.initialize_script(script)
                self.script_status_label.setText("Default")
                self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
                print(f"[ProcessWizard] Step script method exposed successfully")
                return
            
            # Fallback to generated script if step doesn't have script method
            try:
                script = self._generate_script(step_cls, params)
                print(f"[ProcessWizard] Generated script successfully")
            except Exception as gen_e:
                print(f"[ProcessWizard] Error generating script: {gen_e}")
                self.script_editor.setPlainText(f"# Error generating script: {gen_e}\n# Using basic template instead")
                script = self._generate_basic_script(step_name, params)
            
            self.script_editor.setPlainText(script)
            # Initialize script tracker with the generated script
            self.script_tracker.initialize_script(script)
            self.script_status_label.setText("Default")
            self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
            
            # Set script text safely
            try:
                self.script_editor.setPlainText(script)
                print(f"[ProcessWizard] Script set in editor successfully")
            except Exception as set_e:
                print(f"[ProcessWizard] Error setting script text: {set_e}")
                # Try with basic text
                try:
                    self.script_editor.setPlainText(f"# Script generation failed\n# Step: {step_name}\n# Parameters: {params}")
                except:
                    pass
                    
        except Exception as e:
            print(f"[ProcessWizard] Critical error in _sync_script_from_params: {e}")
            fallback_script = f"# Script generation failed: {e}"
            self.script_editor.setPlainText(fallback_script)
            # Initialize script tracker even with fallback script
            self.script_tracker.initialize_script(fallback_script)
            self.script_status_label.setText("Default")
            self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
    
    def _sync_script_to_params(self):
        """Parse script and update parameter table (basic implementation)"""
        script_text = self.script_editor.toPlainText()
        
        # This is a basic implementation - could be enhanced with actual Python AST parsing
        # For now, just show a message that this feature is under development
        self.wizard_manager.ui.console_output.setPlainText(
            "Script-to-parameters synchronization is under development.\n"
            "For now, please use the Parameters tab to modify values."
        )
    
    def _get_params_from_table(self):
        """Extract parameters from the parameter table"""
        params = {}
        for row in range(self.param_table.rowCount()):
            key_item = self.param_table.item(row, 0)
            if not key_item:
                continue
                
            key = key_item.text().strip()
            if not key:
                continue
            
            # Check if the value cell contains a widget or text item
            widget = self.param_table.cellWidget(row, 1)
            
            if widget:
                # Handle different widget types in correct order
                if isinstance(widget, QComboBox):
                    val = widget.currentText().strip()
                elif isinstance(widget, QTextEdit):
                    val = widget.toPlainText().strip()
                elif isinstance(widget, QSpinBox):
                    val = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    val = widget.value()
                elif hasattr(widget, 'findChild'):
                    # This is a container widget (like for checkboxes)
                    from PySide6.QtWidgets import QCheckBox
                    checkbox = widget.findChild(QCheckBox)
                    if checkbox:
                        val = checkbox.isChecked()
                    else:
                        val = ""
                else:
                    val = ""
            else:
                # It's a regular text item
                val_item = self.param_table.item(row, 1)
                val = val_item.text().strip() if val_item else ""
            
            # Store the value with appropriate type conversion
            if isinstance(val, bool):
                params[key] = val
            elif isinstance(val, (int, float)):
                params[key] = val
            elif val:
                # Try to convert string values to appropriate types
                try:
                    if '.' in str(val) and str(val).replace('.', '').replace('-', '').isdigit():
                        params[key] = float(val)
                    elif str(val).replace('-', '').isdigit():
                        params[key] = int(val)
                    else:
                        params[key] = str(val)  # Keep as string
                except ValueError:
                    params[key] = str(val)  # Keep as string if conversion fails
            else:
                # Handle empty values - don't add them to the dict unless they're boolean False
                if isinstance(val, bool):
                    params[key] = val
        
        return params
    
    def _generate_script(self, step_cls, params):
        """Generate Python script for the processing step"""
        try:
            step_name = step_cls.name
            
            # Get current channel info safely
            current_channel = None
            try:
                current_channel = self.get_active_channel_info()
            except:
                pass
            
            channel_name = "input_channel"
            if current_channel:
                try:
                    channel_name = f"'{current_channel.channel_id}'"
                except:
                    pass
            
            # Format parameters for script safely
            param_lines = []
            try:
                for key, value in params.items():
                    if isinstance(value, str):
                        # Escape quotes in string values
                        escaped_value = value.replace("'", "\\'")
                        param_lines.append(f"    '{key}': '{escaped_value}'")
                    else:
                        param_lines.append(f"    '{key}': {value}")
            except Exception as param_e:
                print(f"[ProcessWizard] Error formatting parameters: {param_e}")
                param_lines = [f"    # Error formatting parameters: {param_e}"]
            
            params_str = "{\n" + ",\n".join(param_lines) + "\n}" if param_lines else "{}"
            
            # Generate the script without dangerous f-string nesting
            script_parts = [
                f"# Generated script for {step_name} processing step",
                "# Edit this script to customize the processing behavior",
                "# The script will be executed when you click 'Apply Operation'",
                "",
                "import numpy as np",
                "import scipy.signal",
                "import copy",
                "",
                "# Input channel is available as 'parent_channel'",
                "# Parameters:",
                f"params = {params_str}",
                "",
                f"# Method 1: Use the built-in processing step",
                f"step_instance = registry.get_step('{step_name}')",
                "if step_instance:",
                "    # Parse parameters using the step's parameter parser",
                "    parsed_params = step_instance.parse_input(params)",
                "    # Apply the step to get the result",
                "    result = step_instance.apply(parent_channel, parsed_params)",
                "    ",
                "    # Handle single channel or multiple channels",
                "    if isinstance(result, list):",
                "        result_channels = result",
                "    else:",
                "        result_channel = result",
                "else:",
                "    # Method 2: Custom processing (edit this section)",
                "    # Access input data",
                "    input_data = parent_channel.ydata",
                "    input_time = parent_channel.xdata",
                "    fs = getattr(parent_channel, 'fs_median', 1.0)",
                "    ",
                "    # Example: Apply your custom processing here",
                "    processed_data = input_data.copy()",
                "    # processed_data = your_custom_function(input_data, **params)",
                "    ",
                "    # NEW FORMAT: Return list of channel dictionaries",
                "    # Each dictionary must have 'tags', 'x', 'y' fields",
                "    result_channels_data = [",
                "        {",
                "            'tags': ['time-series'],",
                "            'x': input_time,",
                "            'y': processed_data",
                "        }",
                "    ]",
                "    ",

                "",
                "# IMPORTANT: The script must define 'result_channels_data' (new format)",
                "# or 'result_channel'/'result_channels' (legacy format)",
                "# result_channel = your_single_processed_channel",
                "# result_channels = [channel1, channel2, ...]  # for multiple outputs"
            ]
            
            script = "\n".join(script_parts)
            return script
            
        except Exception as e:
            print(f"[ProcessWizard] Error in _generate_script: {e}")
            return self._generate_basic_script(step_cls.name if step_cls else "unknown", params)
    
    def _generate_basic_script(self, step_name, params):
        """Generate a basic fallback script when main generation fails"""
        try:
            script = f"""# Basic script template for {step_name}
# Script generation encountered an error, using simplified template

import numpy as np
import copy

# Input channel is available as 'parent_channel'
# Process the channel data here

# Example: Copy input to output (no processing)
result_channel = copy.deepcopy(parent_channel)
result_channel.description = parent_channel.description + ' -> {step_name}'

# Parameters that were extracted:
# {params}
"""
            return script
        except:
            return "# Basic script generation failed\n# Please select a filter again"

    def _generate_script_from_step_method(self, step_cls, params):
        """Extract and format the step's actual script method for user editing"""
        try:
            # Check if the step has a script method
            if not hasattr(step_cls, 'script'):
                return None
            
            # Get the source code of the script method
            import inspect
            try:
                source_lines = inspect.getsource(step_cls.script).split('\n')
            except (OSError, TypeError):
                # If we can't get the source, create a simple template
                return self._generate_simple_script_template(step_cls, params)
            
            # Find the method definition
            start_line = None
            for i, line in enumerate(source_lines):
                if line.strip().startswith('def script('):
                    start_line = i
                    break
            
            if start_line is None:
                return self._generate_simple_script_template(step_cls, params)
            
            # Extract method body (skip the def line)
            method_body = source_lines[start_line + 1:]
            
            # Find minimum indentation and adjust
            min_indent = float('inf')
            non_empty_lines = [line for line in method_body if line.strip()]
            
            if not non_empty_lines:
                return self._generate_simple_script_template(step_cls, params)
            
            for line in non_empty_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Remove common indentation
            if min_indent < float('inf'):
                adjusted_lines = []
                for line in method_body:
                    if line.strip():
                        adjusted_lines.append(line[min_indent:])
                    else:
                        adjusted_lines.append(line)
            else:
                adjusted_lines = method_body
            
            # Remove empty lines at the end
            while adjusted_lines and not adjusted_lines[-1].strip():
                adjusted_lines.pop()
            
            # Format parameters for display
            param_info = []
            if params:
                for key, value in params.items():
                    param_info.append(f"# {key}: {value}")
            else:
                param_info.append("# No parameters")
            
            # Create the user-friendly script
            script_parts = [
                f"# {step_cls.name} step script - edit to customize",
                "# Available: x (time), y (signal), fs (frequency), params",
                "",
                "import numpy as np",
                "import scipy.signal",
                "",
                "# Parameters:",
                *param_info,
                "",
                "# Original step logic:",
            ]
            
            # Add the actual method body with comments, converting return statements
            for line in adjusted_lines:
                if line.strip():
                    # Convert return statements to result_channels_data assignment
                    if line.strip().startswith('return '):
                        # Convert "return expression" to "result_channels_data = expression"
                        return_expr = line.strip()[7:]  # Remove "return "
                        script_parts.append(f"result_channels_data = {return_expr}")
                    else:
                        script_parts.append(line)
                else:
                    script_parts.append("")
            
            # Add comment about the new format
            script_parts.append("")
            script_parts.append("# The result_channels_data contains structured channel data")
            script_parts.append("# Each dictionary has 'tags', 'x', 'y' and optional fields like 't', 'f', 'z'")
            
            return "\n".join(script_parts)
            
        except Exception as e:
            print(f"[ProcessWizard] Error extracting step script: {e}")
            return self._generate_simple_script_template(step_cls, params)
    
    def _generate_simple_script_template(self, step_cls, params):
        """Generate a simple script template based on step information"""
        try:
            step_name = step_cls.name
            description = getattr(step_cls, 'description', '')
            
            # Format parameters
            param_info = []
            if params:
                for key, value in params.items():
                    param_info.append(f"# {key}: {value}")
            else:
                param_info.append("# No parameters")
            
            # Create basic template based on common step patterns
            processing_logic = self._guess_processing_logic(step_name, params)
            
            script_parts = [
                f"# {step_name} step script - edit to customize",
                "# Available: x (time), y (signal), fs (frequency), params",
                "",
                "import numpy as np",
                "import scipy.signal",
                "",
                "# Parameters:",
                *param_info,
                "",
                f"# Step description: {description}" if description else "# Custom processing step",
                "",
                "# Processing logic:",
                processing_logic,
                "",
                "# Result must be stored in result_channels_data variable",
                "result_channels_data = [",
                "    {",
                "        'tags': ['time-series'],",
                "        'x': x,",
                "        'y': y_new",
                "    }",
                "]"
            ]
            
            return "\n".join(script_parts)
            
        except Exception as e:
            print(f"[ProcessWizard] Error generating simple template: {e}")
            return f"# Error generating script for {step_cls.name}\n# Please check the step implementation"
    
    def _guess_processing_logic(self, step_name, params):
        """Guess the processing logic based on step name and parameters"""
        
        # Common processing patterns
        if 'abs' in step_name.lower():
            return "y_new = np.abs(y)"
        elif 'normalize' in step_name.lower():
            return "y_new = (y - np.min(y)) / (np.max(y) - np.min(y))"
        elif 'smooth' in step_name.lower() or 'average' in step_name.lower():
            window = params.get('window', 5)
            return f"# Apply smoothing with window size {window}\ny_new = np.convolve(y, np.ones({window})/{window}, mode='same')"
        elif 'filter' in step_name.lower():
            return "# Apply filtering operation\ny_new = scipy.signal.filtfilt(b, a, y)  # Define b, a coefficients"
        elif 'threshold' in step_name.lower():
            threshold = params.get('threshold', 0.0)
            return f"# Apply threshold operation\ny_new = np.where(y > {threshold}, y, 0)"
        elif 'power' in step_name.lower():
            exponent = params.get('exponent', 2.0)
            return f"y_new = np.power(y, {exponent})"
        elif 'log' in step_name.lower():
            return "y_new = np.log(np.abs(y) + 1e-10)  # Add small value to avoid log(0)"
        elif 'exp' in step_name.lower():
            return "y_new = np.exp(y)"
        elif 'derivative' in step_name.lower():
            return "y_new = np.gradient(y)"
        elif 'cumulative' in step_name.lower():
            return "y_new = np.cumsum(y)"
        elif 'clip' in step_name.lower():
            return "y_new = np.clip(y, a_min, a_max)  # Define a_min, a_max"
        elif 'multiply' in step_name.lower():
            constant = params.get('constant', 1.0)
            return f"y_new = y * {constant}"
        elif 'add' in step_name.lower():
            constant = params.get('constant', 0.0)
            return f"y_new = y + {constant}"
        else:
            return "# Add your custom processing here\ny_new = y.copy()  # Replace with your processing logic"

    def _on_script_changed(self):
        """Handle script text changes"""
        try:
            current_script = self.script_editor.toPlainText()
            is_modified = self.script_tracker.is_script_modified(current_script)
            
            if is_modified:
                self.script_status_label.setText("Modified")
                self.script_status_label.setStyleSheet("color: orange; font-weight: bold;")
                self.script_tracker.mark_script_modified()
            else:
                self.script_status_label.setText("Default")
                self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
                
        except Exception as e:
            print(f"[ProcessWizard] Error handling script change: {e}")

    def _reset_script(self):
        """Reset script to default"""
        try:
            if self.script_tracker.original_script:
                self.script_editor.setPlainText(self.script_tracker.original_script)
                self.script_status_label.setText("Default")
                self.script_status_label.setStyleSheet("color: gray; font-style: italic;")
                print("[ProcessWizard] Script reset to default")
        except Exception as e:
            print(f"[ProcessWizard] Error resetting script: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProcessWizardWindow(FileManager(), ChannelManager(), default_file_id=None)
    window.show()
    sys.exit(app.exec())
