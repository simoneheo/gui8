# signal_mixer_wizard_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QListWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QCheckBox, QSplitter, QFrame, QSpinBox, QGroupBox, QGridLayout,
    QFormLayout, QMessageBox, QDoubleSpinBox, QSlider, QAbstractItemView, QHeaderView
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Optional, Dict, List

# Import the reusable DataAlignerWidget
from data_aligner_window import DataAlignerWidget

# Handle mixer imports gracefully
try:
    from mixer.mixer_registry import MixerRegistry, load_all_mixers
    MIXER_AVAILABLE = True
except ImportError as e:
    MIXER_AVAILABLE = False
    
    def load_all_mixers(directory):
        pass
    
    class MixerRegistry:
        @staticmethod
        def all_mixers():
            return ["add", "subtract", "multiply", "divide"]
        
        @staticmethod
        def get(name):
            return None

from signal_mixer_wizard_manager import SignalMixerWizardManager
from plot_manager import StylePreviewWidget


class SignalMixerWizardWindow(QMainWindow):
    """
    Signal Mixer Wizard window - Single step process similar to Process Wizard format
    """
    
    wizard_closed = Signal()
    mixer_applied = Signal(dict)
    state_changed = Signal(str)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        self.setWindowTitle("Signal Mixer Wizard")
        self.setMinimumSize(1200, 800)
        
        # Initialize state tracking
        self._stats = {
            'total_mixes': 0,
            'successful_mixes': 0,
            'failed_mixes': 0,
            'last_mix_time': None,
            'session_start': time.time()
        }
        
        # Track channel visibility states to preserve checkbox states across table updates
        self._channel_visibility_states = {}
        
        # Initialize alignment method tracking
        self._previous_alignment_method = 'time'
        
        # Validate initialization
        if not self._validate_initialization():
            return
            
        # Load mixer plugins
        self._load_mixer_plugins()

        # Initialize manager
        self.manager = SignalMixerWizardManager(
            file_manager=self.file_manager,
            channel_manager=self.channel_manager,
            signal_bus=self.signal_bus,
            parent=self
        )
        self._setup_manager_callbacks()

        # Build UI
        self._build_ui()
        
        # Initialize with data
        self._initialize_ui()
        
        self._log_state_change("Mixer wizard initialized successfully")

    def _validate_initialization(self) -> bool:
        """Validate that required managers are available"""
        if not self.file_manager or not self.channel_manager:
            return False
        return True
        
    def _load_mixer_plugins(self):
        """Load mixer plugins safely"""
        try:
            if MIXER_AVAILABLE:
                load_all_mixers("mixer")
        except Exception as e:
            pass
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        self.state_changed.emit(message)

    def _setup_manager_callbacks(self):
        """Setup callbacks for manager events"""
        self.manager.register_ui_callback('add_channel_to_table', self._add_channel_to_table)
        self.manager.register_ui_callback('replace_channel_in_table', self._replace_channel_in_table)
        self.manager.register_ui_callback('refresh_after_undo', self._refresh_after_undo)

    def _build_ui(self):
        """Build the main UI using process wizard format"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)
        
        # Build left and right panels
        self._build_left_panel(main_splitter)
        self._build_right_panel(main_splitter)
        
    def _build_left_panel(self, main_splitter):
        """Build the left panel with mixing controls"""
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        
        # Create horizontal splitter for two columns
        left_splitter = QSplitter(Qt.Horizontal)
        left_layout.addWidget(left_splitter)
        
        # Create left and right column widgets
        left_col_widget = QWidget()
        left_col_layout = QVBoxLayout(left_col_widget)
        
        right_col_widget = QWidget()
        right_col_layout = QVBoxLayout(right_col_widget)
        
        # Mixing Operations Group → left column
        self._build_mixing_operations_group(left_col_layout)
        
        # All other groups → right column
        self._build_channel_selection_group(right_col_layout)
        self._build_alignment_controls_group(right_col_layout)
        self._build_console_group(right_col_layout)
        
        # Add columns to splitter
        left_splitter.addWidget(left_col_widget)
        left_splitter.addWidget(right_col_widget)
        
        main_splitter.addWidget(self.left_panel)
    
    def _build_channel_selection_group(self, layout):
        """Build channel selection controls"""
        group = QGroupBox("Channel Selection")
        group_layout = QVBoxLayout(group)
        
        # Channel A
        a_layout = QVBoxLayout()
        a_layout.addWidget(QLabel("Channel A:"))
        
        # File dropdown for Channel A
        a_file_layout = QHBoxLayout()
        a_file_layout.addWidget(QLabel("File:"))
        self.channel_a_file_combo = QComboBox()
        self.channel_a_file_combo.setMinimumWidth(200)
        self.channel_a_file_combo.currentTextChanged.connect(self._on_channel_a_file_changed)
        a_file_layout.addWidget(self.channel_a_file_combo)
        a_layout.addLayout(a_file_layout)
        
        # Channel dropdown for Channel A
        a_channel_layout = QHBoxLayout()
        a_channel_layout.addWidget(QLabel("Channel:"))
        self.channel_a_combo = QComboBox()
        self.channel_a_combo.setMinimumWidth(200)
        self.channel_a_combo.currentTextChanged.connect(self._on_channel_a_changed)
        a_channel_layout.addWidget(self.channel_a_combo)
        a_layout.addLayout(a_channel_layout)
        
        # Channel A stats
        self.channel_a_stats = QLabel("No channel selected")
        self.channel_a_stats.setStyleSheet("color: #666; font-size: 10px;")
        a_layout.addWidget(self.channel_a_stats)
        
        group_layout.addLayout(a_layout)
        
        # Channel B
        b_layout = QVBoxLayout()
        b_layout.addWidget(QLabel("Channel B:"))
        
        # File dropdown for Channel B
        b_file_layout = QHBoxLayout()
        b_file_layout.addWidget(QLabel("File:"))
        self.channel_b_file_combo = QComboBox()
        self.channel_b_file_combo.setMinimumWidth(200)
        self.channel_b_file_combo.currentTextChanged.connect(self._on_channel_b_file_changed)
        b_file_layout.addWidget(self.channel_b_file_combo)
        b_layout.addLayout(b_file_layout)
        
        # Channel dropdown for Channel B
        b_channel_layout = QHBoxLayout()
        b_channel_layout.addWidget(QLabel("Channel:"))
        self.channel_b_combo = QComboBox()
        self.channel_b_combo.setMinimumWidth(200)
        self.channel_b_combo.currentTextChanged.connect(self._on_channel_b_changed)
        b_channel_layout.addWidget(self.channel_b_combo)
        b_layout.addLayout(b_channel_layout)
        
        # Channel B stats
        self.channel_b_stats = QLabel("No channel selected")
        self.channel_b_stats.setStyleSheet("color: #666; font-size: 10px;")
        b_layout.addWidget(self.channel_b_stats)
        
        group_layout.addLayout(b_layout)
        
        layout.addWidget(group)
    
    def _build_alignment_controls_group(self, layout):
        """Build alignment controls using the reusable DataAlignerWidget"""
        self.data_aligner_widget = DataAlignerWidget(self)
        
        # Connect to parameter changes
        self.data_aligner_widget.parameters_changed.connect(self._on_alignment_parameter_changed)
        
        layout.addWidget(self.data_aligner_widget)
    
    def _build_mixing_operations_group(self, layout):
        """Build mixing operations controls"""
        group = QGroupBox("Mixing Operations")
        group_layout = QVBoxLayout(group)
        
        # Operation templates list
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Templates:"))
        
        # Filter dropdown
        self.operation_filter = QComboBox()
        self.operation_filter.addItems(["All", "Arithmetic", "Expression", "Logic", "Threshold", "Masking", "Unary"])
        self.operation_filter.currentTextChanged.connect(self._on_operation_filter_changed)
        op_layout.addWidget(self.operation_filter)
        group_layout.addLayout(op_layout)
        
        # Operations list
        self.operations_list = QListWidget()
        self.operations_list.itemClicked.connect(self._on_operation_selected)
        group_layout.addWidget(self.operations_list)
        
        layout.addWidget(group)
    

    
    def _build_console_group(self, layout):
        """Build console controls"""
        group = QGroupBox("Console")
        group_layout = QVBoxLayout(group)
        
        # Expression input
        expr_layout = QHBoxLayout()
        expr_layout.addWidget(QLabel("Expression:"))
        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("e.g., C = A + B")
        self.expression_input.setToolTip("Enter expression with label (e.g., C = A + B)\nLabel (C) is used for referencing in other expressions")
        self.expression_input.returnPressed.connect(self._on_expression_submitted)
        self.expression_input.textChanged.connect(self._on_input_changed)
        expr_layout.addWidget(self.expression_input)
        group_layout.addLayout(expr_layout)
        
        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Logs and messages will appear here")
        self.console_output.setMaximumHeight(150)
        group_layout.addWidget(self.console_output)
        
        # Channel Name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Channel Name:"))
        self.channel_name_input = QLineEdit()
        self.channel_name_input.setPlaceholderText("e.g., Sum of Signals, Combined Data")
        self.channel_name_input.setToolTip("Descriptive name for display in plots and channel lists\n(separate from expression label)")
        self.channel_name_input.textChanged.connect(self._on_input_changed)
        name_layout.addWidget(self.channel_name_input)
        group_layout.addLayout(name_layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.create_btn = QPushButton("Apply Mixer")
        self.create_btn.setEnabled(False)
        self.create_btn.clicked.connect(self._on_create_mixed_channel)
        self.create_btn.setStyleSheet("""
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
        button_layout.addWidget(self.create_btn)
        
        group_layout.addLayout(button_layout)
        
        layout.addWidget(group)
    
    def _build_right_panel(self, main_splitter):
        """Build the right panel with results table and plots"""
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # Results table
        self._build_results_table(right_layout)
        
        # Plot area
        self._build_plot_area(right_layout)
        
        main_splitter.addWidget(self.right_panel)
    
    def _build_results_table(self, layout):
        """Build the results table showing channels A, B, and mixed channels"""
        layout.addWidget(QLabel("Channels:"))
        
        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels(["Show", "Style", "Label", "Expression", "Shape", "Actions"])
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setMaximumHeight(200)
        
        # Set column resize modes for better layout (matching process wizard)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width
        header.setSectionResizeMode(2, QHeaderView.Fixed)     # Label - fixed width
        header.setSectionResizeMode(3, QHeaderView.Stretch)   # Expression - stretches
        header.setSectionResizeMode(4, QHeaderView.Fixed)     # Shape - fixed width
        header.setSectionResizeMode(5, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set specific column widths (matching process wizard)
        self.results_table.setColumnWidth(0, 60)   # Show checkbox
        self.results_table.setColumnWidth(1, 80)   # Style preview
        self.results_table.setColumnWidth(2, 60)   # Label column
        self.results_table.setColumnWidth(4, 80)   # Shape column
        self.results_table.setColumnWidth(5, 200)  # Actions column (wider for 5 buttons)
        
        layout.addWidget(self.results_table)
    
    def _build_plot_area(self, layout):
        """Build the plot area"""
        self.figure, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.right_panel)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def _initialize_ui(self):
        """Initialize UI with data from manager"""
        
        # Populate channel dropdowns
        self._populate_channel_dropdowns()
        
        # Populate operations list
        self._populate_operations_list()
        
        # Set default operation to "A + B" and default filter to "Arithmetic"
        self._set_default_operation_selection()
        
        # Auto-select best channel pair 
        self._autopopulate_best_channels()
        
        # Update compatibility
        self._update_compatibility()

    def _populate_channel_dropdowns(self):
        """Populate file dropdowns with available files"""
        files = self.manager.get_available_files()
        
        # Clear existing items
        self.channel_a_file_combo.clear()
        self.channel_b_file_combo.clear()
        self.channel_a_combo.clear()
        self.channel_b_combo.clear()
        
        if not files:
            self.channel_a_file_combo.addItem("No files available")
            self.channel_b_file_combo.addItem("No files available")
            return
        
        # Add files to dropdowns
        for file_info in files:
            self.channel_a_file_combo.addItem(file_info.filename, file_info)
            self.channel_b_file_combo.addItem(file_info.filename, file_info)

    def _populate_operations_list(self):
        """Populate the operations list with available mixing operations from mixer folder"""
        try:
            # Try to get templates from the mixer registry
            if MIXER_AVAILABLE:
                from mixer.mixer_registry import MixerRegistry
                
                # Get all templates organized by category
                all_templates = MixerRegistry.get_all_templates()
                categories = MixerRegistry.get_all_categories()
                
                # Organize templates by category for filtering
                self.all_operations = {}
                for category in categories:
                    category_templates = MixerRegistry.get_templates_by_category(category)
                    self.all_operations[category] = category_templates
                
                pass
                
            else:
                # Fallback to basic templates if mixer folder not available
                pass
                self.all_operations = {
            "Arithmetic": [
                ("A + B", "add"),
                ("A - B", "subtract"), 
                ("A * B", "multiply"),
                ("A / B", "divide"),
            ],
            "Expression": [
                ("A**2 + B**2", "expression"),
                ("sqrt(A**2 + B**2)", "expression"),
                    ]
                }
                
        except Exception as e:
            # Fallback to basic templates
            self.all_operations = {
                "Arithmetic": [
                    ("A + B", "add"),
                    ("A - B", "subtract"), 
                    ("A * B", "multiply"),
                    ("A / B", "divide"),
            ]
        }
        
        # Initially show all operations
        self._filter_operations("All")

    def _filter_operations(self, category):
        """Filter operations based on selected category"""
        self.operations_list.clear()
        
        if category == "All":
            # Show all operations from all categories
            for cat, operations in self.all_operations.items():
                for op_display, op_code in operations:
                    item = self.operations_list.addItem(op_display)
        else:
            # Show only operations from selected category
            if category in self.all_operations:
                operations = self.all_operations[category]
                for op_display, op_code in operations:
                    item = self.operations_list.addItem(op_display)
                    
    def _set_default_operation_selection(self):
        """Set default operation selection to 'A + B' and default filter to 'All'."""
        # Set filter to "All" to show all available operations
        all_index = self.operation_filter.findText("All")
        if all_index >= 0:
            self.operation_filter.setCurrentIndex(all_index)
            self._filter_operations("All")  # Apply the filter
        
        # Find and select "A + B" in the operations list
        for i in range(self.operations_list.count()):
            item = self.operations_list.item(i)
            if item and item.text() == "A + B":
                self.operations_list.setCurrentRow(i)
                # Trigger the operation selection to populate expression and channel name
                self._on_operation_selected(item)
                break

    def _autopopulate_best_channels(self):
        """Auto-populate the dropdowns with the best channel pair"""
        
        channel_a, channel_b = self.manager.find_best_channel_pair()
        
        if channel_a:
            # Find and select file for channel A
            for i in range(self.channel_a_file_combo.count()):
                file_info = self.channel_a_file_combo.itemData(i)
                if file_info and file_info.file_id == channel_a.file_id:
                    self.channel_a_file_combo.setCurrentIndex(i)
                    # This will trigger _on_channel_a_file_changed which populates channels
                    # Then find and select the specific channel
                    for j in range(self.channel_a_combo.count()):
                        if self.channel_a_combo.itemData(j) == channel_a:
                            self.channel_a_combo.setCurrentIndex(j)
                            break
                    break
            
        if channel_b:
            # Find and select file for channel B
            for i in range(self.channel_b_file_combo.count()):
                file_info = self.channel_b_file_combo.itemData(i)
                if file_info and file_info.file_id == channel_b.file_id:
                    self.channel_b_file_combo.setCurrentIndex(i)
                    # This will trigger _on_channel_b_file_changed which populates channels
                    # Then find and select the specific channel
                    for j in range(self.channel_b_combo.count()):
                        if self.channel_b_combo.itemData(j) == channel_b:
                            self.channel_b_combo.setCurrentIndex(j)
                            break
                    break

    # Event Handlers
    def _on_channel_a_file_changed(self):
        """Handle channel A file selection change"""
        file_info = self.channel_a_file_combo.currentData()
        self.channel_a_combo.clear()
        
        if file_info:
            # Get channels for this file
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            for channel in channels:
                if channel.show and channel.ydata is not None:
                    display_name = channel.legend_label or channel.channel_id
                    self.channel_a_combo.addItem(display_name, channel)
        
        self._update_compatibility()
        self._update_plot()

    def _on_channel_b_file_changed(self):
        """Handle channel B file selection change"""
        file_info = self.channel_b_file_combo.currentData()
        self.channel_b_combo.clear()
        
        if file_info:
            # Get channels for this file
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            for channel in channels:
                if channel.show and channel.ydata is not None:
                    display_name = channel.legend_label or channel.channel_id
                    self.channel_b_combo.addItem(display_name, channel)
        
        self._update_compatibility()
        self._update_plot()

    def _on_channel_a_changed(self):
        """Handle channel A selection change"""
        channel = self.channel_a_combo.currentData()
        if channel:
            stats = self.manager.get_channel_stats(channel)
            self.channel_a_stats.setText(
                f"Length: {stats.get('length', 0)} | "
                f"Range: {stats.get('min', 0):.3f} to {stats.get('max', 0):.3f}"
            )
        else:
            self.channel_a_stats.setText("No channel selected")
        
        # Auto-configure alignment parameters when channel changes
        channel_b = self.channel_b_combo.currentData()
        if channel and channel_b:
            self.data_aligner_widget.auto_configure_for_channels(channel, channel_b)
            # Initialize previous alignment method tracking
            params = self.data_aligner_widget.get_alignment_parameters()
            self._previous_alignment_method = params.get('alignment_method', 'time')
        
        self._update_compatibility()
        self._update_plot()

    def _on_channel_b_changed(self):
        """Handle channel B selection change"""
        channel = self.channel_b_combo.currentData()
        if channel:
            stats = self.manager.get_channel_stats(channel)
            self.channel_b_stats.setText(
                f"Length: {stats.get('length', 0)} | "
                f"Range: {stats.get('min', 0):.3f} to {stats.get('max', 0):.3f}"
            )
        else:
            self.channel_b_stats.setText("No channel selected")
        
        # Auto-configure alignment parameters when channel changes
        channel_a = self.channel_a_combo.currentData()
        if channel and channel_a:
            self.data_aligner_widget.auto_configure_for_channels(channel_a, channel)
            # Initialize previous alignment method tracking
            params = self.data_aligner_widget.get_alignment_parameters()
            self._previous_alignment_method = params.get('alignment_method', 'time')
        
        self._update_compatibility()
        self._update_plot()

    def _on_suggest_pair(self):
        """Handle auto-suggest pair button click"""
        suggestions = self.manager.suggest_channel_pairs()
        
        if not suggestions:
            QMessageBox.information(self, "No Suggestions", 
                                  "No compatible channel pairs found.")
            return
        
        # Cycle through suggestions
        current_a = self.channel_a_combo.currentData()
        current_b = self.channel_b_combo.currentData()
        
        # Find current suggestion index
        current_index = -1
        for i, (ch_a, ch_b) in enumerate(suggestions):
            if ch_a == current_a and ch_b == current_b:
                current_index = i
                break
        
        # Get next suggestion
        next_index = (current_index + 1) % len(suggestions)
        channel_a, channel_b = suggestions[next_index]
        
        # Update selections
        for i in range(self.channel_a_combo.count()):
            if self.channel_a_combo.itemData(i) == channel_a:
                self.channel_a_combo.setCurrentIndex(i)
                break
        
        for i in range(self.channel_b_combo.count()):
            if self.channel_b_combo.itemData(i) == channel_b:
                self.channel_b_combo.setCurrentIndex(i)
                break
        
        self._log_message(f"Suggested pair {next_index + 1}/{len(suggestions)}")

    def _on_alignment_parameter_changed(self, parameters=None):
        """Handle alignment parameter value changes"""
        try:
            # Check if alignment method has changed
            if parameters:
                current_method = parameters.get('alignment_method', 'time')
                previous_method = getattr(self, '_previous_alignment_method', None)
                
                if current_method != previous_method:
                    # Alignment method changed - trigger auto-configuration
                    channel_a = self.channel_a_combo.currentData()
                    channel_b = self.channel_b_combo.currentData()
                    
                    if channel_a and channel_b:
                        print(f"[SignalMixerWizard] Alignment method changed from {previous_method} to {current_method}, auto-configuring...")
                        self.data_aligner_widget.auto_configure_for_channels(channel_a, channel_b)
                    
                    # Store the new method for next comparison
                    self._previous_alignment_method = current_method
            
            # Update compatibility check and validation display
            self._update_compatibility()
            
        except Exception as e:
            print(f"[SignalMixerWizard] Error handling alignment parameter change: {e}")
            # Still update compatibility even if there's an error
            self._update_compatibility()

    def _on_input_changed(self):
        """Handle expression or channel name input changes"""
        # Update button states when inputs change
        self._update_button_states()

    def _on_operation_filter_changed(self, category):
        """Handle operation filter change"""
        self._filter_operations(category)

    def _on_operation_selected(self, item):
        """Handle operation template selection"""
        # Clear console for new template selection
        self.console_output.clear()
        
        template = item.text()
        
        # Auto-generate next available label (C, D, E, etc.)
        next_label = self._get_next_available_label()
        expression = f"{next_label} = {template}"
        self.expression_input.setText(expression)
        
        # Generate a descriptive channel name based on the template
        descriptive_name = self._generate_descriptive_name(template, next_label)
        self.channel_name_input.setText(descriptive_name)
        
        # Log clean template information
        self._log_clean_template_info(template, next_label)

    def _log_clean_template_info(self, template, next_label):
        """Log clean, concise template information"""
        self._log_message(f"Template: {template}")
        self._log_message(f"Expression Label: {next_label} (for referencing in other expressions)")
        
        # Get available channels with shape validation
        available_channels = self._get_available_channels_with_validation()
        
        if available_channels:
            self._log_message(f"Available Channels: {', '.join(available_channels)}")
        else:
            self._log_message("Available Channels: None (select channels A and B first)")
        
        # Provide simple usage tip based on template type
        self._log_simple_template_tip(template)

    def _log_simple_template_tip(self, template):
        """Log simple, focused tip for the template"""
        try:
            # Try to get description from template registry
            if MIXER_AVAILABLE:
                from mixer.template_registry import TemplateRegistry
                guidance = TemplateRegistry.get_template_guidance(template)
                if guidance.get('description'):
                    self._log_message(guidance['description'])
                    if guidance.get('tip'):
                        self._log_message(f"Tip: {guidance['tip']}")
                    return
        except Exception as e:
            pass
        
        # Fallback to pattern-based tips
        template_lower = template.lower()
        
        if any(op in template_lower for op in ['+', '-', '*', '/']):
            self._log_message("Basic arithmetic operation")
        elif '%' in template:
            self._log_message("Modulo operation (great for cyclic signals)")
        elif "np.where" in template:
            if "np.nan" in template:
                self._log_message("Conditional selection with NaN masking")
            else:
                self._log_message("Conditional selection (like max/min but flexible)")
        elif "np.angle" in template:
            self._log_message("Phase angle calculation from complex vector")
        elif "np.arctan2" in template:
            self._log_message("Quadrant-aware angle (e.g., rotation vector)")
        elif "np.logical_and" in template:
            self._log_message("Logical AND operation for range detection")
        elif "np.isnan" in template:
            self._log_message("NaN detection and masking")
        elif any(func in template_lower for func in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs']):
            func_name = next((f for f in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs'] if f in template_lower), 'function')
            self._log_message(f"Applies {func_name}() function")
        elif '**' in template:
            self._log_message("Power operation (e.g., squaring)")
        elif any(op in template for op in ['>', '<', '>=', '<=', '==', '!=']):
            if '*' in template:
                self._log_message("Threshold masking operation")
            else:
                self._log_message("Logical comparison operation")
        elif 'max' in template_lower and 'abs' in template_lower:
            self._log_message("Normalization operation")
        else:
            self._log_message("Custom expression")

    def _log_template_info(self, template, next_label):
        """Log comprehensive template information including available channels and suggestions"""
        self._log_message(f"Template selected: {template}")
        self._log_message(f"Auto-generated label: '{next_label}' (you can change it to any name)")
        
        # Get available channels
        available_channels = []
        
        # Add input channels A and B
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        if channel_a:
            available_channels.append("A")
        if channel_b:
            available_channels.append("B")
        
        # Add existing mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        for channel in mixed_channels:
            label = getattr(channel, 'step_table_label', None)
            if label:
                available_channels.append(label)
        
        if available_channels:
            self._log_message(f"Available channels: {', '.join(available_channels)}")
        else:
            self._log_message("No channels available - please select channels A and B first")
        
        # Provide template-specific suggestions
        self._log_template_suggestions(template, available_channels)

    def _log_template_suggestions(self, template, available_channels):
        """Log template-specific suggestions and usage examples"""
        template_lower = template.lower()
        
        # Arithmetic operations
        if any(op in template_lower for op in ['+', '-', '*', '/']):
            if len(available_channels) >= 2:
                # Safely extract operator from template
                template_parts = template.split()
                if len(template_parts) >= 3:  # e.g., "A + B"
                    operator = template_parts[1]
                    self._log_message(f"Example: {available_channels[0]} {operator} {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
                else:
                    # Fallback for templates that don't follow A op B pattern
                    self._log_message(f"Example: Apply operation to {available_channels[0]} and {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
            self._log_message("Tip: Basic arithmetic operations work element-wise on arrays")
        
        # Element-wise maximum/minimum operations using conditional logic
        elif ('>=') in template and ('<=') in template:
            if len(available_channels) >= 2:
                if '>=' in template and 'A *' in template and 'B *' in template:
                    op_name = "maximum" if '(A >= B)' in template else "minimum"
                    self._log_message(f"Example: Find element-wise {op_name} between {available_channels[0]} and {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
            self._log_message("Tip: Uses conditional logic to find element-wise max/min between signals")
        
        # Mathematical functions
        elif any(func in template_lower for func in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs']):
            func_name = next((f for f in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs'] if f in template_lower), 'function')
            self._log_message(f"Applies {func_name}() function to signal data")
            if available_channels:
                self._log_message(f"Example: Apply {func_name} to channel {available_channels[0]}")
        
        # Power operations
        elif '**' in template:
            self._log_message("Tip: Raises signal values to specified power (e.g., A**2 for squaring)")
            if available_channels:
                self._log_message(f"Example: Square all values in channel {available_channels[0]}")
        
        # Logical operations
        elif any(op in template for op in ['>', '<', '>=', '<=', '==', '!=']):
            self._log_message("Tip: Creates binary signal (True/False) based on comparison")
            if len(available_channels) >= 2:
                self._log_message(f"Example: Compare {available_channels[0]} with {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
        
        # Threshold masking operations
        elif any(op in template for op in ['>', '<', '>=', '<=']) and '*' in template:
            self._log_message("Tip: Creates masked signal - zeros out values that don't meet condition")
            if len(available_channels) >= 2:
                self._log_message(f"Example: Keep {available_channels[0]} values only where condition is true")
        
        # Normalization
        elif 'max' in template_lower and 'abs' in template_lower:
            self._log_message("Tip: Normalizes signal to range [-1, 1] by dividing by maximum absolute value")
            if available_channels:
                self._log_message(f"Example: Normalize channel {available_channels[0]} amplitude")
        
        # General suggestions
        if len(available_channels) > 2:
            other_channels = available_channels[2:]
            self._log_message(f"You can also use mixed channels: {', '.join(other_channels)}")
        
        # Usage tips
        self._log_message("Tips:")
        self._log_message("  • Change the label to any name (e.g., RESULT, DOG, Z)")
        self._log_message("  • Use parentheses for complex expressions: (A + B) * C")
        self._log_message("  • Available functions: abs, sqrt, sin, cos, tan, log, exp, max, min, mean, std, sum")
        self._log_message("  • Mix channels with constants: A + 3 * B - 2")

    def _get_next_available_label(self):
        """Get the next available label (C, D, E, etc.) for mixed channels"""
        # Get existing labels from mixed channels
        existing_labels = set()
        for channel in self.manager.get_mixed_channels():
            label = getattr(channel, 'step_table_label', None)
            if label:
                existing_labels.add(label)
        
        # Generate next available label starting from C
        for i in range(26):  # A-Z
            candidate = chr(ord('C') + i)  # Start from C since A and B are input channels
            if candidate not in existing_labels:
                return candidate
        
        # Fallback to numbered labels if we run out of letters
        return f"Mixed_{len(self.manager.get_mixed_channels()) + 1}"

    def _is_channel_compatible_for_expression(self, mixed_channel, channel_a, channel_b):
        """Check if a mixed channel has compatible dimensions with channels A and B for expressions"""
        if not mixed_channel or mixed_channel.ydata is None or len(mixed_channel.ydata) == 0:
            return False
        
        # If neither A nor B is selected, can't determine compatibility
        if not channel_a and not channel_b:
            return True  # Default to compatible if no reference channels
        
        mixed_length = len(mixed_channel.ydata)
        
        # Check compatibility with channel A
        if channel_a and channel_a.ydata is not None and len(channel_a.ydata) > 0:
            a_length = len(channel_a.ydata)
            if mixed_length == a_length:
                return True
        
        # Check compatibility with channel B
        if channel_b and channel_b.ydata is not None and len(channel_b.ydata) > 0:
            b_length = len(channel_b.ydata)
            if mixed_length == b_length:
                return True
        
        # If alignment is configured, check if the mixed channel could be aligned
        alignment_config = self._get_alignment_config()
        if alignment_config:
            # For index-based alignment
            if alignment_config.get('alignment_method') == 'index':
                mode = alignment_config.get('mode', 'truncate')
                if mode == 'truncate':
                    # With truncation, any channel with data can be made compatible
                    return True
                else:  # custom range
                    start_idx = alignment_config.get('start_index', 0)
                    end_idx = alignment_config.get('end_index', 500)
                    # Check if the mixed channel has enough data for the custom range
                    return mixed_length > end_idx
            
            # For time-based alignment, assume compatibility if the channel has time data
            elif alignment_config.get('alignment_method') == 'time':
                # If mixed channel has time data or can have it created, it's potentially compatible
                return hasattr(mixed_channel, 'xdata') and mixed_channel.xdata is not None
        
        return False

    def _get_available_channels_with_validation(self):
        """Get available channels with shape validation - only channels with compatible dimensions"""
        available_channels = []
        
        # Get input channels A and B
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        # Determine the reference shape for validation
        reference_shape = None
        reference_channel = None
        
        # Use channel A as reference if available
        if channel_a and channel_a.ydata is not None and len(channel_a.ydata) > 0:
            reference_shape = len(channel_a.ydata)
            reference_channel = channel_a
            available_channels.append("A")
        
        # Check channel B compatibility
        if channel_b and channel_b.ydata is not None and len(channel_b.ydata) > 0:
            if reference_shape is None:
                # B becomes the reference if A is not available
                reference_shape = len(channel_b.ydata)
                reference_channel = channel_b
                available_channels.append("B")
            elif len(channel_b.ydata) == reference_shape:
                # B is compatible with A
                available_channels.append("B")
            else:
                # B has different shape - check if alignment can make them compatible
                if self._can_channels_be_aligned(channel_a, channel_b):
                    available_channels.append("B")
        
        # Add mixed channels that have compatible dimensions
        mixed_channels = self.manager.get_mixed_channels()
        for channel in mixed_channels:
            label = getattr(channel, 'step_table_label', None)
            if label and channel.ydata is not None and len(channel.ydata) > 0:
                if reference_shape is None:
                    # First valid channel becomes reference
                    reference_shape = len(channel.ydata)
                    available_channels.append(label)
                elif len(channel.ydata) == reference_shape:
                    # Compatible with reference shape
                    available_channels.append(label)
                elif reference_channel and self._can_channels_be_aligned(reference_channel, channel):
                    # Can be aligned with reference
                    available_channels.append(label)
        
        return available_channels

    def _can_channels_be_aligned(self, channel_a, channel_b):
        """Check if two channels can be aligned using current alignment settings"""
        if not channel_a or not channel_b:
            return False
        
        # Get current alignment configuration
        alignment_config = self._get_alignment_config()
        if not alignment_config:
            return False
        
        # Use the manager's validation method to check if alignment is possible
        try:
            is_valid, _, _ = self.manager.validate_channels_for_mixing_with_alignment(
                channel_a, channel_b, alignment_config
            )
            return is_valid
        except Exception:
            return False

    def _generate_descriptive_name(self, template, label):
        """Generate a descriptive channel name based on the template"""
        template_lower = template.lower()
        
        # Map common templates to descriptive names
        if template == "A + B":
            return "Sum of Channels"
        elif template == "A - B":
            return "Difference (A-B)"
        elif template == "A * B":
            return "Product of Channels"
        elif template == "A / B":
            return "Ratio (A/B)"
        elif template == "(A + B) / 2":
            return "Average of Channels"
        elif template == "abs(A - B)":
            return "Absolute Difference"
        elif template == "A % B":
            return "Modulo Operation"
        elif "np.where" in template:
            if "np.nan" in template:
                return "Conditional Mask (NaN)"
            else:
                return "Conditional Selection"
        elif "np.angle" in template:
            return "Phase Angle"
        elif "np.arctan2" in template:
            return "Quadrant-Aware Angle"
        elif "np.logical_and" in template:
            return "Logical AND Mask"
        elif "np.isnan" in template:
            return "NaN Mask"
        elif "max" in template_lower:
            return "Element-wise Maximum"
        elif "min" in template_lower:
            return "Element-wise Minimum"
        elif "sqrt" in template_lower:
            return "Square Root Transform"
        elif "**2" in template:
            return "Squared Signal"
        elif "sin" in template_lower:
            return "Sine Transform"
        elif "cos" in template_lower:
            return "Cosine Transform"
        elif "exp" in template_lower:
            return "Exponential Transform"
        elif "log" in template_lower:
            return "Logarithmic Transform"
        elif "abs" in template_lower:
            return "Absolute Value"
        elif any(op in template for op in ['>', '<', '>=', '<=', '==', '!=']):
            if '*' in template:
                return "Threshold Mask"
            else:
                return "Logical Comparison"
        elif "normalize" in template_lower or ("max" in template_lower and "abs" in template_lower):
            return "Normalized Signal"
        else:
            # Fallback to a generic name with the label
            return f"Mixed Signal {label}"





    def _on_expression_submitted(self):
        """Handle expression input submission"""
        self._on_create_mixed_channel()

    def _on_create_mixed_channel(self):
        """Handle create mixed channel button click"""
        expression = self.expression_input.text().strip()
        channel_name = self.channel_name_input.text().strip()
        
        if not expression:
            self._log_message("Please enter an expression (e.g., C = A + B)")
            return
        
        if not channel_name:
            self._log_message("Please enter a channel name for display")
            return
        
        # Get selected channels
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        if not channel_a or not channel_b:
            self._log_message("Please select both channels A and B")
            return
        
        # Validate channels for mixing (only when button is pressed)
        alignment_config = self._get_alignment_config()
        is_valid, validation_message, alignment_info = self.manager.validate_channels_for_mixing_with_alignment(
            channel_a, channel_b, alignment_config
        )
        
        if not is_valid:
            self._log_message(f"Channel incompatibility: {validation_message}")
            return
        
        # Create channel context
        channel_context = {'A': channel_a, 'B': channel_b}
        
        # Add existing mixed channels to context
        mixed_channels = self.manager.get_mixed_channels()
        for i, mixed_channel in enumerate(mixed_channels):
            label = chr(ord('C') + i)
            channel_context[label] = mixed_channel
        
        # Extract the label from the expression (left side of =)
        if '=' in expression:
            expression_label = expression.split('=')[0].strip()
        else:
            expression_label = self._get_next_available_label()
        
        # Create the full expression with the extracted label
        full_expression = f"{expression_label} = {expression.split('=', 1)[1].strip() if '=' in expression else expression}"
        
        # Process expression with alignment, passing both label and name
        new_channel, message = self.manager.process_mixer_expression_with_alignment(
            full_expression, channel_context, alignment_config, channel_name
        )
        
        if new_channel:
            self.expression_input.clear()
            self.channel_name_input.clear()
            self._log_message(f"{message}")
            self._update_button_states()
            self._update_plot()
        else:
            self._log_message(f"{message}")

    def _on_undo_last(self):
        """Handle undo last button click"""
        success, message = self.manager.undo_last_step()
        self._log_message(message)
        
        if success:
            self._update_button_states()
            self._update_plot()

    def _on_clear_all(self):
        """Handle clear all button click"""
        reply = QMessageBox.question(
            self, "Clear All", 
            "Are you sure you want to clear all mixed channels?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.manager.clear_mixed_channels()
            
            # Clear visibility states for mixed channels (keep A and B)
            keys_to_remove = [key for key in self._channel_visibility_states.keys() if key not in ["A", "B"]]
            for key in keys_to_remove:
                del self._channel_visibility_states[key]
            
            self._refresh_after_undo()
            self._log_message("All mixed channels cleared")

    def _update_compatibility(self):
        """Update alignment parameters and show validation information"""
        # Check if required UI elements exist (defensive programming for initialization)
        if not hasattr(self, 'channel_a_combo') or not hasattr(self, 'channel_b_combo'):
            return
        if not hasattr(self, 'data_aligner_widget'):
            return
            
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        if not channel_a or not channel_b:
            self.data_aligner_widget.set_status_message("Select channels to configure alignment")
            return
            
        # Auto-configure alignment parameters based on selected channels
        self.data_aligner_widget.auto_configure_for_channels(channel_a, channel_b)
        
        # Get alignment configuration and validate
        alignment_config = self._get_alignment_config()
        is_valid, validation_message, alignment_info = self.manager.validate_channels_for_mixing_with_alignment(
            channel_a, channel_b, alignment_config
        )
        
        # Create informative status message
        if is_valid:
            len_a = len(channel_a.ydata) if channel_a.ydata is not None else 0
            len_b = len(channel_b.ydata) if channel_b.ydata is not None else 0
            
            if len_a == len_b:
                status_msg = f"Channels compatible: {len_a} samples each"
            else:
                status_msg = f"Channels compatible: A({len_a}) + B({len_b}) samples"
            
            if alignment_info.get('needs_alignment', False):
                align_msg = alignment_info.get('alignment_message', 'Alignment will be applied')
                status_msg += f" - {align_msg}"
                self.data_aligner_widget.set_status_message(status_msg, "warning")
            else:
                status_msg += " - No alignment needed"
                self.data_aligner_widget.set_status_message(status_msg, "success")
        else:
            status_msg = f"Validation failed: {validation_message}"
            self.data_aligner_widget.set_status_message(status_msg, "error")
        
        # Keep create button enabled based only on having expression and channel name
        if hasattr(self, 'create_btn') and hasattr(self, 'expression_input') and hasattr(self, 'channel_name_input'):
            has_expression = bool(self.expression_input.text().strip())
            has_channel_name = bool(self.channel_name_input.text().strip())
            self.create_btn.setEnabled(has_expression and has_channel_name)
    

    def _get_alignment_config(self):
        """Get current alignment configuration from DataAlignerWidget"""
        return self.data_aligner_widget.get_alignment_parameters()

    def _update_button_states(self):
        """Update button enabled/disabled states"""
        # Create button enabled if both expression and name are provided (no validation)
        has_expression = bool(self.expression_input.text().strip())
        has_channel_name = bool(self.channel_name_input.text().strip())
        
        self.create_btn.setEnabled(has_expression and has_channel_name)

    def _update_plot(self):
        """Update the plot with selected and mixed channels"""
        self.ax.clear()
        
        # Clear any existing top axis
        if hasattr(self, 'ax_top') and self.ax_top is not None:
            self.ax_top.remove()
            self.ax_top = None
        
        # Plot selected channels A and B
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        plotted_any = False
        max_data_length = 0
        
        # Check if we should use x-axis data or indices for index-based alignment
        use_x_axis_data = self._should_use_x_axis_for_index_mode(channel_a, channel_b)
        
        # Track lines for legend
        channel_a_line = None
        channel_b_line = None
        
        if channel_a and channel_a.ydata is not None and self._is_channel_visible_in_table("A"):
            if use_x_axis_data:
                x_data = channel_a.xdata if channel_a.xdata is not None else np.arange(len(channel_a.ydata))
            else:
                x_data = np.arange(len(channel_a.ydata))
            channel_a_line, = self.ax.plot(x_data, channel_a.ydata, 'b-', alpha=0.7, linewidth=1,
                        label=f"A: {channel_a.legend_label or channel_a.channel_id}")
            plotted_any = True
            max_data_length = max(max_data_length, len(channel_a.ydata))
        
        if channel_b and channel_b.ydata is not None and self._is_channel_visible_in_table("B"):
            if use_x_axis_data:
                x_data = channel_b.xdata if channel_b.xdata is not None else np.arange(len(channel_b.ydata))
            else:
                x_data = np.arange(len(channel_b.ydata))
            channel_b_line, = self.ax.plot(x_data, channel_b.ydata, 'r-', alpha=0.7, linewidth=1,
                        label=f"B: {channel_b.legend_label or channel_b.channel_id}")
            plotted_any = True
            max_data_length = max(max_data_length, len(channel_b.ydata))
        
        # Plot mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        colors = self._get_mixed_channel_colors()
        
        for i, channel in enumerate(mixed_channels):
            if channel.ydata is not None:
                # Check if this channel is visible (from results table)
                channel_label = getattr(channel, 'step_table_label', f"Mixed_{i + 1}")
                is_visible = self._is_channel_visible_in_table(channel_label)
                
                if is_visible:
                    if use_x_axis_data:
                        x_data = channel.xdata if channel.xdata is not None else np.arange(len(channel.ydata))
                    else:
                        x_data = np.arange(len(channel.ydata))
                    color = colors[i % len(colors)]
                    self.ax.plot(x_data, channel.ydata, color=color, linewidth=2,
                                label=f"{channel_label}: {channel.legend_label or channel.channel_id}")
                    plotted_any = True
                    max_data_length = max(max_data_length, len(channel.ydata))

        # Set up plot
        if plotted_any:
            # Add top x-axis when using indices instead of inherited x-data
            # This helps users understand that the x-axis represents sample indices
            if not use_x_axis_data:
                self._add_index_axis()
        
        # Remove all labels and title
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.set_title("")
        
        self.ax.grid(True)
        
        self.canvas.draw()
        
        # Update the results table with all channels (A, B, and mixed)
        self._update_results_table(channel_a, channel_b)
    
    def _update_results_table(self, channel_a, channel_b):
        """Update the results table with channels A, B, and all mixed channels"""
        # Clear the table
        self.results_table.setRowCount(0)
        
        # Add channel A if available
        if channel_a and channel_a.ydata is not None:
            self._add_input_channel_to_table(channel_a, "A", 'b-')
        
        # Add channel B if available
        if channel_b and channel_b.ydata is not None:
            self._add_input_channel_to_table(channel_b, "B", 'r-')
        
        # Add all mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        for i, channel in enumerate(mixed_channels):
            if channel.ydata is not None:
                self._add_mixed_channel_to_table(channel, i)
    
    def _add_input_channel_to_table(self, channel, label, color_style):
        """Add an input channel (A or B) to the results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Create tooltip for input channel
        tooltip = self._create_input_channel_tooltip(channel, label)
        
        # Column 0: Show checkbox
        checkbox = QCheckBox()
        # Use stored visibility state, default to True if not stored
        is_visible = self._channel_visibility_states.get(label, True)
        checkbox.setChecked(is_visible)
        checkbox.stateChanged.connect(lambda state, lbl=label: self._on_channel_visibility_changed(lbl, state))
        
        # Center the checkbox in the cell
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_widget.setToolTip(tooltip)
        self.results_table.setCellWidget(row, 0, checkbox_widget)
        
        # Column 1: Style (visual preview widget)
        color = 'blue' if label == 'A' else 'red'
        style_widget = StylePreviewWidget(
            color=color,
            style='-',
            marker=None
        )
        style_widget.setToolTip(tooltip)
        self.results_table.setCellWidget(row, 1, style_widget)
        
        # Column 2: Label (for input channels, show "A" or "B")
        label_item = QTableWidgetItem(label)
        label_item.setToolTip(tooltip)
        self.results_table.setItem(row, 2, label_item)
        
        # Column 3: Expression (for input channels, show "Input Channel A/B")
        expr_item = QTableWidgetItem(f"Input Channel {label}")
        expr_item.setToolTip(tooltip)
        self.results_table.setItem(row, 3, expr_item)
        
        # Column 4: Shape
        if channel.xdata is not None and channel.ydata is not None:
            shape_str = f"({len(channel.xdata)}, 2)"
        elif channel.ydata is not None:
            shape_str = f"({len(channel.ydata)},)"
        else:
            shape_str = "No data"
        shape_item = QTableWidgetItem(shape_str)
        shape_item.setToolTip(tooltip)
        self.results_table.setItem(row, 4, shape_item)
        
        # Column 5: Actions
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        
        # Info button
        info_button = QPushButton("❗")
        info_button.setMaximumWidth(25)
        info_button.setMaximumHeight(25)
        info_button.setToolTip("Channel information")
        info_button.clicked.connect(lambda checked=False, ch=channel: self._show_channel_info(ch))
        actions_layout.addWidget(info_button)
        
        # Inspect button
        inspect_button = QPushButton("🔍")
        inspect_button.setMaximumWidth(25)
        inspect_button.setMaximumHeight(25)
        inspect_button.setToolTip("Inspect channel data")
        inspect_button.clicked.connect(lambda checked=False, ch=channel: self._inspect_channel_data(ch))
        actions_layout.addWidget(inspect_button)
        
        # Style button
        style_button = QPushButton("🎨")
        style_button.setMaximumWidth(25)
        style_button.setMaximumHeight(25)
        style_button.setToolTip("Channel styling")
        style_button.clicked.connect(lambda checked=False, ch=channel: self._style_channel(ch))
        actions_layout.addWidget(style_button)
        
        # Transform button
        transform_button = QPushButton("🔨")
        transform_button.setMaximumWidth(25)
        transform_button.setMaximumHeight(25)
        transform_button.setToolTip("Transform channel data")
        transform_button.clicked.connect(lambda checked=False, ch=channel: self._transform_channel_data(ch))
        actions_layout.addWidget(transform_button)
        
        # Delete button (disabled for input channels A and B)
        delete_button = QPushButton("🗑️")
        delete_button.setMaximumWidth(25)
        delete_button.setMaximumHeight(25)
        delete_button.setEnabled(False)  # Always disabled for input channels
        delete_button.setToolTip("Cannot delete input channel")
        actions_layout.addWidget(delete_button)
        
        self.results_table.setCellWidget(row, 5, actions_widget)
    
    def _add_mixed_channel_to_table(self, channel, index):
        """Add a mixed channel to the results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Get label and color info
        label = getattr(channel, 'step_table_label', f"Mixed_{index + 1}")
        color_info = self._get_mixed_channel_color_info(index)
        
        # Create tooltip with parent channel information
        tooltip = self._create_mixed_channel_tooltip(channel)
        
        # Column 0: Show checkbox
        checkbox = QCheckBox()
        # Use stored visibility state, default to True if not stored
        is_visible = self._channel_visibility_states.get(label, True)
        checkbox.setChecked(is_visible)
        checkbox.stateChanged.connect(lambda state, lbl=label: self._on_channel_visibility_changed(lbl, state))
        
        # Center the checkbox in the cell
        checkbox_widget = QWidget()
        checkbox_layout = QHBoxLayout(checkbox_widget)
        checkbox_layout.addWidget(checkbox)
        checkbox_layout.setAlignment(Qt.AlignCenter)
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_widget.setToolTip(tooltip)
        self.results_table.setCellWidget(row, 0, checkbox_widget)
        
        # Column 1: Style (visual preview widget)
        style_widget = StylePreviewWidget(
            color=color_info['plot_color'],
            style=getattr(channel, 'style', '-'),
            marker=getattr(channel, 'marker', None) if getattr(channel, 'marker', None) != "None" else None
        )
        style_widget.setToolTip(tooltip)
        self.results_table.setCellWidget(row, 1, style_widget)
        
        # Column 2: Label (mixed channel label, e.g., "C", "D", "E")
        label_item = QTableWidgetItem(label)
        label_item.setToolTip(tooltip)
        self.results_table.setItem(row, 2, label_item)
        
        # Column 3: Expression (full expression with label, e.g., "C = A + B")
        full_expression = getattr(channel, 'description', f"{label} = mixed signal")
        expr_item = QTableWidgetItem(full_expression)
        expr_item.setToolTip(tooltip)
        self.results_table.setItem(row, 3, expr_item)
        
        # Column 4: Shape
        if channel.xdata is not None and channel.ydata is not None:
            shape_str = f"({len(channel.xdata)}, 2)"
        elif channel.ydata is not None:
            shape_str = f"({len(channel.ydata)},)"
        else:
            shape_str = "No data"
        shape_item = QTableWidgetItem(shape_str)
        shape_item.setToolTip(tooltip)
        self.results_table.setItem(row, 4, shape_item)
        
        # Column 5: Actions
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(2, 2, 2, 2)
        actions_layout.setSpacing(2)
        
        # Info button
        info_button = QPushButton("❗")
        info_button.setMaximumWidth(25)
        info_button.setMaximumHeight(25)
        info_button.setToolTip("Channel information")
        info_button.clicked.connect(lambda checked=False, ch=channel: self._show_channel_info(ch))
        actions_layout.addWidget(info_button)
        
        # Inspect button
        inspect_button = QPushButton("🔍")
        inspect_button.setMaximumWidth(25)
        inspect_button.setMaximumHeight(25)
        inspect_button.setToolTip("Inspect channel data")
        inspect_button.clicked.connect(lambda checked=False, ch=channel: self._inspect_channel_data(ch))
        actions_layout.addWidget(inspect_button)
        
        # Style button
        style_button = QPushButton("🎨")
        style_button.setMaximumWidth(25)
        style_button.setMaximumHeight(25)
        style_button.setToolTip("Channel styling")
        style_button.clicked.connect(lambda checked=False, ch=channel: self._style_channel(ch))
        actions_layout.addWidget(style_button)
        
        # Transform button
        transform_button = QPushButton("🔨")
        transform_button.setMaximumWidth(25)
        transform_button.setMaximumHeight(25)
        transform_button.setToolTip("Transform channel data")
        transform_button.clicked.connect(lambda checked=False, ch=channel: self._transform_channel_data(ch))
        actions_layout.addWidget(transform_button)
        
        # Delete button (only for mixed channels)
        delete_button = QPushButton("🗑️")
        delete_button.setMaximumWidth(25)
        delete_button.setMaximumHeight(25)
        delete_button.setToolTip("Delete mixed channel")
        delete_button.clicked.connect(lambda checked=False, ch=channel: self._delete_mixed_channel(ch))
        actions_layout.addWidget(delete_button)
        
        self.results_table.setCellWidget(row, 5, actions_widget)

    def _create_mixed_channel_tooltip(self, mixed_channel):
        """Create a tooltip showing parent channel information for a mixed channel"""
        tooltip_parts = []
        
        # Get current channels A and B
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        # Add parent channel information
        if channel_a:
            a_file = getattr(channel_a, 'filename', 'Unknown file')
            a_name = channel_a.legend_label or channel_a.channel_id
            tooltip_parts.append(f"Channel A: {a_name}")
            tooltip_parts.append(f"  File: {a_file}")
        
        if channel_b:
            b_file = getattr(channel_b, 'filename', 'Unknown file')
            b_name = channel_b.legend_label or channel_b.channel_id
            tooltip_parts.append(f"Channel B: {b_name}")
            tooltip_parts.append(f"  File: {b_file}")
        
        # Add mixed channel information
        mixed_name = mixed_channel.legend_label or mixed_channel.channel_id
        mixed_label = getattr(mixed_channel, 'step_table_label', 'Unknown')
        tooltip_parts.append(f"")
        tooltip_parts.append(f"Mixed Channel: {mixed_name}")
        tooltip_parts.append(f"Label: {mixed_label}")
        
        # Add expression information
        full_expression = getattr(mixed_channel, 'description', '')
        if full_expression:
            tooltip_parts.append(f"Expression: {full_expression}")
        
        # Add data information
        if mixed_channel.ydata is not None:
            tooltip_parts.append(f"Data points: {len(mixed_channel.ydata):,}")
        
        return "\n".join(tooltip_parts)

    def _create_input_channel_tooltip(self, channel, label):
        """Create a tooltip showing information for an input channel (A or B)"""
        tooltip_parts = []
        
        # Add channel information
        channel_name = channel.legend_label or channel.channel_id
        tooltip_parts.append(f"Input Channel {label}: {channel_name}")
        
        # Add file information
        file_name = getattr(channel, 'filename', 'Unknown file')
        tooltip_parts.append(f"File: {file_name}")
        
        # Add data information
        if channel.ydata is not None:
            tooltip_parts.append(f"Data points: {len(channel.ydata):,}")
            
            # Add data range
            y_min = np.min(channel.ydata)
            y_max = np.max(channel.ydata)
            tooltip_parts.append(f"Y range: {y_min:.3f} to {y_max:.3f}")
        
        # Add time information if available
        if channel.xdata is not None:
            x_min = np.min(channel.xdata)
            x_max = np.max(channel.xdata)
            tooltip_parts.append(f"X range: {x_min:.3f} to {x_max:.3f}")
        
        # Add step information
        step = getattr(channel, 'step', 0)
        tooltip_parts.append(f"Processing step: {step}")
        
        return "\n".join(tooltip_parts)

    def _show_channel_info(self, channel):
        """Show channel information"""
        from metadata_wizard import MetadataWizard
        wizard = MetadataWizard(channel, self)
        wizard.exec()


    def _inspect_channel_data(self, channel):
        """Inspect channel data"""
        from inspection_wizard import InspectionWizard
        wizard = InspectionWizard(channel, self)
        wizard.data_updated.connect(self._handle_channel_data_updated)
        wizard.exec()


    def _style_channel(self, channel):
        """Style channel"""
        from line_wizard import LineWizard
        wizard = LineWizard(channel, self)
        wizard.channel_updated.connect(self._handle_channel_updated)
        wizard.exec()


    def _transform_channel_data(self, channel):
        """Transform channel data"""
        from transform_wizard import TransformWizard
        wizard = TransformWizard(channel, self)
        wizard.data_updated.connect(self._handle_channel_data_updated)
        wizard.exec()


    def _delete_mixed_channel(self, channel):
        """Delete a mixed channel"""
        label = getattr(channel, 'step_table_label', 'Unknown')
        reply = QMessageBox.question(
            self, 
            "Delete Mixed Channel", 
            f"Delete mixed channel '{label}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Remove from manager
            mixed_channels = self.manager.get_mixed_channels()
            for i, mixed_channel in enumerate(mixed_channels):
                if mixed_channel == channel:
                    # Remove from channel manager
                    if hasattr(self.channel_manager, 'remove_channel'):
                        self.channel_manager.remove_channel(channel.channel_id)
                    
                    # Remove from mixed channels list
                    self.manager.mixed_channels.pop(i)
                    break
            
            # Remove visibility state
            if label in self._channel_visibility_states:
                del self._channel_visibility_states[label]
            
            # Update table and plot
            self._update_plot()


    def _handle_channel_data_updated(self, channel_id):
        """Handle when channel data is updated"""
        self._update_plot()


    def _handle_channel_updated(self, channel_id):
        """Handle when channel properties are updated"""
        self._update_plot()


    def _should_use_x_axis_for_index_mode(self, channel_a, channel_b):
        """
        Check if we should use x-axis data or indices when in index-based alignment mode.
        Returns True if x-axis data should be used (when x-data is identical), False for indices.
        """
        # If not in index-based mode, always use x-axis data
        if hasattr(self, 'data_aligner_widget'):
            alignment_params = self.data_aligner_widget.get_alignment_parameters()
            alignment_method = alignment_params.get('alignment_method', 'time')
            if alignment_method != 'index':
                return True
        else:
            # Fallback if widget not available
            return True
        
        # First, check if both channels A and B have identical x-data
        if channel_a and channel_b and channel_a.ydata is not None and channel_b.ydata is not None:
            # Check if both channels have x-data
            if channel_a.xdata is not None and channel_b.xdata is not None:
                try:
                    # Check if x-data lengths match
                    if len(channel_a.xdata) == len(channel_b.xdata):
                        # Check if x-data values are identical
                        if np.allclose(channel_a.xdata, channel_b.xdata, rtol=1e-10, atol=1e-10):
                            return True
                        else:
                            return False
                    else:
                        return False
                except Exception as e:
                    return False
            else:
                return False
        
        # If we don't have both channels A and B, fall back to checking all channels
        channels_to_check = []
        
        if channel_a and channel_a.ydata is not None:
            channels_to_check.append(channel_a)
        
        if channel_b and channel_b.ydata is not None:
            channels_to_check.append(channel_b)
        
        # Add visible mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        for channel in mixed_channels:
            if channel.ydata is not None:
                channel_label = getattr(channel, 'step_table_label', f"Mixed_{len(channels_to_check)}")
                is_visible = self._is_channel_visible_in_table(channel_label)
                if is_visible:
                    channels_to_check.append(channel)
        
        # If we don't have at least 2 channels, default to using x-axis data
        if len(channels_to_check) < 2:
            return True
        
        # Check if all channels have x-data
        for channel in channels_to_check:
            if channel.xdata is None:
                return False
        
        try:
            # Use the first channel as reference
            reference_xdata = channels_to_check[0].xdata
            
            # Check if all other channels have identical x-data
            for i, channel in enumerate(channels_to_check[1:], 1):
                if len(channel.xdata) != len(reference_xdata):
                    return False
                
                if not np.allclose(channel.xdata, reference_xdata, rtol=1e-10, atol=1e-10):
                    return False
            
            return True
                
        except Exception as e:
            return False
    
    def _add_index_axis(self):
        """Add a top x-axis showing sample indices"""
        try:
            # Create a secondary x-axis on top
            ax_top = self.ax.twiny()
            
            # Get the current x-axis limits from the main plot
            x_min, x_max = self.ax.get_xlim()
            
            # Determine if we're dealing with time data or index data
            channel_a = self.channel_a_combo.currentData()
            channel_b = self.channel_b_combo.currentData()
            use_x_axis_data = self._should_use_x_axis_for_index_mode(channel_a, channel_b)
            
            if use_x_axis_data:
                # We have time data on the main axis, show corresponding indices on top
                reference_channel = channel_a if channel_a and channel_a.ydata is not None else channel_b
                
                if reference_channel and reference_channel.xdata is not None:
                    time_data = reference_channel.xdata
                    
                    # Find the indices corresponding to the current x-axis limits
                    if len(time_data) > 1:
                        # Calculate indices based on time values
                        time_range = time_data[-1] - time_data[0]
                        if time_range > 0:
                            # Linear interpolation to find index positions
                            index_min = int((x_min - time_data[0]) / time_range * (len(time_data) - 1))
                            index_max = int((x_max - time_data[0]) / time_range * (len(time_data) - 1))
                            
                            # Ensure indices are within bounds
                            index_min = max(0, min(index_min, len(time_data) - 1))
                            index_max = max(0, min(index_max, len(time_data) - 1))
                        else:
                            index_min, index_max = 0, len(time_data) - 1
                    else:
                        index_min, index_max = 0, 0
                else:
                    # Fallback to using x-axis limits as indices
                    index_min, index_max = int(x_min), int(x_max)
            else:
                # We're using indices on the main axis, show time values on top if available
                reference_channel = channel_a if channel_a and channel_a.ydata is not None else channel_b
                
                if reference_channel and reference_channel.xdata is not None:
                    # Main axis shows indices, top axis shows corresponding time values
                    time_data = reference_channel.xdata
                    
                    # Ensure limits are within bounds
                    index_min = max(0, int(x_min))
                    index_max = min(len(time_data) - 1, int(x_max)) if len(time_data) > 0 else int(x_max)
                    
                    # Get corresponding time values
                    if index_min < len(time_data) and index_max < len(time_data):
                        time_min = float(time_data[index_min]) if index_min >= 0 else 0.0
                        time_max = float(time_data[index_max]) if index_max >= 0 else 1.0
                        
                        # Set the top axis to show time values
                        ax_top.set_xlim(time_min, time_max)
                        ax_top.set_xlabel("Time", fontsize=10, color='gray')
                        ax_top.tick_params(axis='x', labelsize=8, colors='gray')
                        
                        # Store reference to top axis for cleanup
                        self.ax_top = ax_top
                        return
                
                # Fallback: just show the same index values on top
                index_min, index_max = int(x_min), int(x_max)
            
            # Set the top axis limits to match the index range
            ax_top.set_xlim(index_min, index_max)
            
            # Set appropriate tick locations for the index axis
            if index_max - index_min > 0:
                # Calculate reasonable tick spacing
                index_range = index_max - index_min
                if index_range <= 10:
                    tick_spacing = 1
                elif index_range <= 50:
                    tick_spacing = 5
                elif index_range <= 100:
                    tick_spacing = 10
                elif index_range <= 500:
                    tick_spacing = 50
                elif index_range <= 1000:
                    tick_spacing = 100
                else:
                    tick_spacing = int(index_range / 10)
                
                # Generate tick positions
                tick_start = int(index_min / tick_spacing) * tick_spacing
                tick_positions = np.arange(tick_start, index_max + tick_spacing, tick_spacing)
                tick_positions = tick_positions[(tick_positions >= index_min) & (tick_positions <= index_max)]
                
                if len(tick_positions) > 0:
                    ax_top.set_xticks(tick_positions)
                    ax_top.set_xticklabels([str(int(pos)) for pos in tick_positions])
            
            # Set the top axis label
            ax_top.set_xlabel("Sample Index", fontsize=10)
            
            # Style the top axis
            ax_top.tick_params(axis='x', labelsize=9)
            
            # Store reference to the top axis for potential cleanup
            self.ax_top = ax_top
            
        except Exception as e:
            # Don't let axis creation errors break the plot
            pass

    def _get_mixed_channel_colors(self):
        """Get the list of colors used for mixed channels in plots"""
        return ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def _get_mixed_channel_color_info(self, index):
        """Get color information for a mixed channel at given index"""
        colors = self._get_mixed_channel_colors()
        plot_color = colors[index % len(colors)]
        
        # Map plot color names to Qt colors
        color_mapping = {
            'green': Qt.green,
            'orange': Qt.GlobalColor(16),  # Qt doesn't have orange, use custom
            'purple': Qt.magenta,
            'brown': Qt.darkYellow,
            'pink': Qt.GlobalColor(13),  # Qt doesn't have pink, use custom  
            'gray': Qt.gray,
            'olive': Qt.darkGreen,
            'cyan': Qt.cyan
        }
        
        # For colors Qt doesn't have, create custom QColor
        if plot_color == 'orange':
            from PySide6.QtGui import QColor
            qt_color = QColor(255, 165, 0)  # Orange RGB
        elif plot_color == 'pink':
            from PySide6.QtGui import QColor
            qt_color = QColor(255, 192, 203)  # Pink RGB
        else:
            qt_color = color_mapping.get(plot_color, Qt.green)
        
        return {
            'plot_color': plot_color,
            'qt_color': qt_color,
            'index': index
        }

    def _is_channel_visible_in_table(self, label):
        """Check if a channel is visible based on stored visibility state"""
        return self._channel_visibility_states.get(label, True)  # Default to visible if not stored

    def _log_message(self, message):
        """Add a message to the console output"""
        self.console_output.append(message)
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # Manager callback methods
    def _add_channel_to_table(self, channel):
        """Add a new mixed channel - now handled by _update_results_table"""
        # The table is now updated completely in _update_plot via _update_results_table
        # This method is kept for compatibility with the manager callbacks
        pass

    def _on_channel_visibility_changed(self, label, state):
        """Handle channel visibility change from results table"""
        is_checked = state == Qt.CheckState.Checked.value
        
        # Store the visibility state
        self._channel_visibility_states[label] = is_checked
        
        # Handle input channels A and B
        if label == "A":
            # For channel A, we can't actually hide it from the dropdown, but we can note the state
            # The plot update will handle showing/hiding based on checkbox state
            pass
        elif label == "B":
            # For channel B, we can't actually hide it from the dropdown, but we can note the state
            # The plot update will handle showing/hiding based on checkbox state
            pass
        else:
            # Handle mixed channels - find the channel by label and update visibility
            mixed_channels = self.manager.get_mixed_channels()
            for channel in mixed_channels:
                if getattr(channel, 'step_table_label', None) == label:
                    # Update the channel's show property if it exists
                    if hasattr(channel, 'show'):
                        channel.show = is_checked
                    break
        
        # Update plot when visibility changes
        self._update_plot()

    def _replace_channel_in_table(self, index, new_channel):
        """Replace an existing channel in the results table"""
        if index < self.results_table.rowCount():
            # Get label and color info
            label = getattr(new_channel, 'step_table_label', f"Mixed_{index + 1}")
            color_info = self._get_mixed_channel_color_info(index)
            
            # Update style widget (Column 1)
            style_widget = StylePreviewWidget(
                color=color_info['plot_color'],
                style=getattr(new_channel, 'style', '-'),
                marker=getattr(new_channel, 'marker', None) if getattr(new_channel, 'marker', None) != "None" else None
            )
            self.results_table.setCellWidget(index, 1, style_widget)
            
            # Update channel name (Column 2)
            channel_name = new_channel.legend_label or new_channel.channel_id
            self.results_table.setItem(index, 2, QTableWidgetItem(channel_name))
            
            # Update shape (Column 3)
            if new_channel.xdata is not None and new_channel.ydata is not None:
                shape_str = f"({len(new_channel.xdata)}, 2)"
            elif new_channel.ydata is not None:
                shape_str = f"({len(new_channel.ydata)},)"
            else:
                shape_str = "No data"
            self.results_table.setItem(index, 3, QTableWidgetItem(shape_str))
            
            # Update label (Column 4)
            self.results_table.setItem(index, 4, QTableWidgetItem(label))
            
            # Update expression (Column 5) - only the right side of the equal sign
            full_expression = getattr(new_channel, 'description', f"{label} = mixed signal")
            if '=' in full_expression:
                expression = full_expression.split('=', 1)[1].strip()
            else:
                expression = full_expression
            self.results_table.setItem(index, 5, QTableWidgetItem(expression))

    def _refresh_after_undo(self):
        """Refresh UI after undo operation"""
        # Clear results table
        self.results_table.setRowCount(0)
        
        # Re-add all remaining mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        for channel in mixed_channels:
            self._add_channel_to_table(channel)
        
        # Update plot and button states
        self._update_plot()
        self._update_button_states()

    def closeEvent(self, event):
        """Handle window close event"""
        self.wizard_closed.emit()
        event.accept()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    import sys
    from file_manager import FileManager
    from channel_manager import ChannelManager

    app = QApplication(sys.argv)
    fm = FileManager()
    cm = ChannelManager()
    win = SignalMixerWizardWindow(fm, cm)
    win.show()
    sys.exit(app.exec())