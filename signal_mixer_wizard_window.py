# signal_mixer_wizard_window.py

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QListWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QCheckBox, QSplitter, QFrame, QSpinBox, QGroupBox, QGridLayout,
    QFormLayout, QMessageBox, QDoubleSpinBox, QSlider
)
from PySide6.QtCore import Qt, Signal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Optional, Dict, List

# Handle mixer imports gracefully
try:
    from mixer.mixer_registry import MixerRegistry, load_all_mixers
    MIXER_AVAILABLE = True
    print("[SignalMixerWizard] Mixer registry imported successfully")
except ImportError as e:
    print(f"[SignalMixerWizard] Warning: Could not import mixer registry: {e}")
    MIXER_AVAILABLE = False
    
    def load_all_mixers(directory):
        print(f"[SignalMixerWizard] Warning: Mixer module not available")
    
    class MixerRegistry:
        @staticmethod
        def all_mixers():
            return ["add", "subtract", "multiply", "divide"]
        
        @staticmethod
        def get(name):
            return None

from signal_mixer_wizard_manager import SignalMixerWizardManager


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
            print("[SignalMixerWizard] ERROR: Required managers not available")
            return False
        return True
        
    def _load_mixer_plugins(self):
        """Load mixer plugins safely"""
        try:
            if MIXER_AVAILABLE:
                load_all_mixers("mixer")
                self._log_state_change("Mixer plugins loaded successfully")
            else:
                self._log_state_change("Mixer plugins not available - using basic operations only")
        except Exception as e:
            print(f"[SignalMixerWizard] Warning: Could not load mixer plugins: {e}")
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[SignalMixerWizard {timestamp}] {message}")
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
        
        # Channel Selection Group
        self._build_channel_selection_group(left_layout)
        
        # Alignment Controls Group
        self._build_alignment_controls_group(left_layout)
        
        # Mixing Operations Group
        self._build_mixing_operations_group(left_layout)
        
        # Parameters Group
        self._build_parameters_group(left_layout)
        
        # Console Group
        self._build_console_group(left_layout)
        
        main_splitter.addWidget(self.left_panel)
    
    def _build_channel_selection_group(self, layout):
        """Build channel selection controls"""
        group = QGroupBox("Channel Selection")
        group_layout = QVBoxLayout(group)
        
        # Channel A
        a_layout = QHBoxLayout()
        a_layout.addWidget(QLabel("Channel A:"))
        self.channel_a_combo = QComboBox()
        self.channel_a_combo.setMinimumWidth(200)
        self.channel_a_combo.currentTextChanged.connect(self._on_channel_a_changed)
        a_layout.addWidget(self.channel_a_combo)
        group_layout.addLayout(a_layout)
        
        # Channel A stats
        self.channel_a_stats = QLabel("No channel selected")
        self.channel_a_stats.setStyleSheet("color: #666; font-size: 10px;")
        group_layout.addWidget(self.channel_a_stats)
        
        # Channel B
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("Channel B:"))
        self.channel_b_combo = QComboBox()
        self.channel_b_combo.setMinimumWidth(200)
        self.channel_b_combo.currentTextChanged.connect(self._on_channel_b_changed)
        b_layout.addWidget(self.channel_b_combo)
        group_layout.addLayout(b_layout)
        
        # Channel B stats
        self.channel_b_stats = QLabel("No channel selected")
        self.channel_b_stats.setStyleSheet("color: #666; font-size: 10px;")
        group_layout.addWidget(self.channel_b_stats)
        
        # Compatibility status
        self.compatibility_label = QLabel("")
        self.compatibility_label.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(self.compatibility_label)
        
        # Auto-suggest button
        self.suggest_btn = QPushButton("🎯 Auto-Suggest Pair")
        self.suggest_btn.clicked.connect(self._on_suggest_pair)
        group_layout.addWidget(self.suggest_btn)
        
        layout.addWidget(group)
    
    def _build_alignment_controls_group(self, layout):
        """Build alignment controls for handling channels of different dimensions"""
        group = QGroupBox("Data Alignment")
        group_layout = QVBoxLayout(group)
        
        # Alignment mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Alignment Mode:"))
        self.alignment_mode_combo = QComboBox()
        self.alignment_mode_combo.addItems(["Index-Based", "Time-Based"])
        self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
        mode_layout.addWidget(self.alignment_mode_combo)
        group_layout.addLayout(mode_layout)
        
        # Index-based options
        self.index_group = QGroupBox("Index Options")
        index_layout = QFormLayout(self.index_group)
        
        # Index mode
        self.index_mode_combo = QComboBox()
        self.index_mode_combo.addItems(["Truncate to Shortest", "Custom Range"])
        self.index_mode_combo.currentTextChanged.connect(self._on_index_mode_changed)
        index_layout.addRow("Mode:", self.index_mode_combo)
        
        # Custom range controls
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(0, 999999)
        self.start_index_spin.setValue(0)
        index_layout.addRow("Start Index:", self.start_index_spin)
        
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(0, 999999)
        self.end_index_spin.setValue(500)
        index_layout.addRow("End Index:", self.end_index_spin)
        
        # Index offset
        self.index_offset_spin = QSpinBox()
        self.index_offset_spin.setRange(-999999, 999999)
        self.index_offset_spin.setValue(0)
        self.index_offset_spin.setToolTip("Positive: shift Channel B forward, Negative: shift Channel A forward")
        index_layout.addRow("Offset:", self.index_offset_spin)
        
        group_layout.addWidget(self.index_group)
        
        # Time-based options
        self.time_group = QGroupBox("Time Options")
        time_layout = QFormLayout(self.time_group)
        
        # Time mode
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems(["Overlap Region", "Custom Range"])
        self.time_mode_combo.currentTextChanged.connect(self._on_time_mode_changed)
        time_layout.addRow("Mode:", self.time_mode_combo)
        
        # Custom time range
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(-999999.0, 999999.0)
        self.start_time_spin.setValue(0.0)
        self.start_time_spin.setDecimals(3)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-999999.0, 999999.0)
        self.end_time_spin.setValue(10.0)
        self.end_time_spin.setDecimals(3)
        time_layout.addRow("End Time:", self.end_time_spin)
        
        # Time offset
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-999999.0, 999999.0)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setDecimals(3)
        self.time_offset_spin.setToolTip("Time offset to apply to Channel B")
        time_layout.addRow("Time Offset:", self.time_offset_spin)
        
        # Interpolation method
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["linear", "nearest", "cubic"])
        time_layout.addRow("Interpolation:", self.interpolation_combo)
        
        # Time grid resolution
        self.round_to_spin = QDoubleSpinBox()
        self.round_to_spin.setRange(0.0001, 1.0)
        self.round_to_spin.setValue(0.01)
        self.round_to_spin.setDecimals(4)
        self.round_to_spin.setToolTip("Time grid resolution (smaller = more points)")
        time_layout.addRow("Grid Resolution:", self.round_to_spin)
        
        group_layout.addWidget(self.time_group)
        
        # Alignment status
        self.alignment_status_label = QLabel("No alignment needed")
        self.alignment_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        group_layout.addWidget(self.alignment_status_label)
        
        # Show/hide appropriate groups (call after all widgets are created)
        self._on_alignment_mode_changed("Index-Based")
        
        layout.addWidget(group)
    
    def _build_mixing_operations_group(self, layout):
        """Build mixing operations controls"""
        group = QGroupBox("Mixing Operations")
        group_layout = QVBoxLayout(group)
        
        # Operation templates list
        op_layout = QHBoxLayout()
        op_layout.addWidget(QLabel("Templates:"))
        
        # Filter dropdown
        self.operation_filter = QComboBox()
        self.operation_filter.addItems(["All", "Arithmetic", "Expression", "Logic", "Threshold", "Unary"])
        self.operation_filter.currentTextChanged.connect(self._on_operation_filter_changed)
        op_layout.addWidget(self.operation_filter)
        group_layout.addLayout(op_layout)
        
        # Operations list
        self.operations_list = QListWidget()
        self.operations_list.itemClicked.connect(self._on_operation_selected)
        group_layout.addWidget(self.operations_list)
        
        layout.addWidget(group)
    
    def _build_parameters_group(self, layout):
        """Build parameters controls with editable table like process wizard"""
        group = QGroupBox("Parameters")
        group_layout = QVBoxLayout(group)
        
        # Parameter table (similar to process wizard)
        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.setEditTriggers(QTableWidget.AllEditTriggers)
        self.param_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.param_table.setFixedHeight(120)
        group_layout.addWidget(self.param_table)
        
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
        self.expression_input.returnPressed.connect(self._on_expression_submitted)
        expr_layout.addWidget(self.expression_input)
        group_layout.addLayout(expr_layout)
        
        # Console output
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Logs and messages will appear here")
        self.console_output.setMaximumHeight(150)
        group_layout.addWidget(self.console_output)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.create_btn = QPushButton("✅ Create Mixed Channel")
        self.create_btn.setEnabled(False)
        self.create_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.create_btn.clicked.connect(self._on_create_mixed_channel)
        button_layout.addWidget(self.create_btn)
        
        self.undo_btn = QPushButton("↶ Undo Last")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._on_undo_last)
        button_layout.addWidget(self.undo_btn)
        
        self.clear_btn = QPushButton("🗑 Clear All")
        self.clear_btn.clicked.connect(self._on_clear_all)
        button_layout.addWidget(self.clear_btn)
        
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
        """Build the results table showing mixed channels"""
        layout.addWidget(QLabel("Mixed Channels:"))
        
        self.results_table = QTableWidget(0, 5)
        self.results_table.setHorizontalHeaderLabels(["Label", "Show", "Line", "Operation", "Description"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)
        
        # Set column widths
        self.results_table.setColumnWidth(0, 60)   # Label
        self.results_table.setColumnWidth(1, 50)   # Show
        self.results_table.setColumnWidth(2, 60)   # Line
        self.results_table.setColumnWidth(3, 100)  # Operation
        
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
        print("[SignalMixerWizard] Initializing UI")
        
        # Populate channel dropdowns
        self._populate_channel_dropdowns()
        
        # Populate operations list
        self._populate_operations_list()
        
        # Auto-select best channel pair 
        self._autopopulate_best_channels()
        
        # Update compatibility
        self._update_compatibility()
        
        print("[SignalMixerWizard] UI initialization complete")

    def _populate_channel_dropdowns(self):
        """Populate channel dropdowns with available channels"""
        channels = self.manager.get_available_channels()
        
        # Clear existing items
        self.channel_a_combo.clear()
        self.channel_b_combo.clear()
        
        if not channels:
            self.channel_a_combo.addItem("No channels available")
            self.channel_b_combo.addItem("No channels available")
            return
        
        # Add channels to dropdowns
        for channel in channels:
            display_name = self.manager.get_channel_display_name(channel)
            self.channel_a_combo.addItem(display_name, channel)
            self.channel_b_combo.addItem(display_name, channel)
        
        print(f"[SignalMixerWizard] Loaded {len(channels)} channels for mixing")

    def _populate_operations_list(self):
        """Populate the operations list with available mixing operations"""
        # Store all operations with their categories for filtering
        self.all_operations = {
            "Arithmetic": [
                ("A + B", "add"),
                ("A - B", "subtract"), 
                ("A * B", "multiply"),
                ("A / B", "divide"),
                ("(A + B) / 2", "average"),
                ("abs(A - B)", "difference"),
                ("A * (A >= B) + B * (B > A)", "max"),
                ("A * (A <= B) + B * (B < A)", "min")
            ],
            "Expression": [
                ("A**2 + B**2", "expression"),
                ("sqrt(A**2 + B**2)", "expression"),
                ("A * sin(B)", "expression"),
                ("A * cos(B)", "expression"),
                ("exp(A / max(abs(A)))", "expression"),
                ("log(abs(A) + 1)", "expression")
            ],
            "Logic": [
                ("A > B", "logic"),
                ("A < B", "logic"),
                ("A >= B", "logic"),
                ("A <= B", "logic"),
                ("A == B", "logic"),
                ("A != B", "logic")
            ],
            "Threshold": [
                ("A * (A > 0.5)", "threshold"),
                ("A * (B > 0.5)", "threshold"),
                ("A * (A > B)", "threshold"),
                ("A * (abs(A) > 0.5)", "threshold")
            ],
            "Unary": [
                ("abs(A)", "unary"),
                ("sqrt(abs(A))", "unary"),
                ("-A", "unary"),
                ("A**2", "unary"),
                ("A / max(abs(A))", "unary")
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

    def _autopopulate_best_channels(self):
        """Auto-populate the dropdowns with the best channel pair"""
        print("[SignalMixerWizard] Starting autopopulation")
        
        channel_a, channel_b = self.manager.find_best_channel_pair()
        
        if channel_a:
            # Find and select channel A
            for i in range(self.channel_a_combo.count()):
                if self.channel_a_combo.itemData(i) == channel_a:
                    self.channel_a_combo.setCurrentIndex(i)
                    break
            print(f"[SignalMixerWizard] Populated channel A")
            
        if channel_b:
            # Find and select channel B
            for i in range(self.channel_b_combo.count()):
                if self.channel_b_combo.itemData(i) == channel_b:
                    self.channel_b_combo.setCurrentIndex(i)
                    break
            print(f"[SignalMixerWizard] Populated channel B")

    # Event Handlers
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

    def _on_alignment_mode_changed(self, mode):
        """Handle alignment mode change"""
        if mode == "Index-Based":
            self.index_group.setVisible(True)
            self.time_group.setVisible(False)
        else:  # Time-Based
            self.index_group.setVisible(False)
            self.time_group.setVisible(True)
        
        # Update compatibility check
        self._update_compatibility()
        
    def _on_index_mode_changed(self, mode):
        """Handle index mode change"""
        enable_custom = (mode == "Custom Range")
        self.start_index_spin.setEnabled(enable_custom)
        self.end_index_spin.setEnabled(enable_custom)
        
        # Update compatibility check
        self._update_compatibility()
        
    def _on_time_mode_changed(self, mode):
        """Handle time mode change"""
        enable_custom = (mode == "Custom Range")
        self.start_time_spin.setEnabled(enable_custom)
        self.end_time_spin.setEnabled(enable_custom)
        
        # Update compatibility check
        self._update_compatibility()

    def _on_operation_filter_changed(self, category):
        """Handle operation filter change"""
        self._filter_operations(category)

    def _on_operation_selected(self, item):
        """Handle operation template selection"""
        template = item.text()
        
        # Auto-generate next available label (C, D, E, etc.)
        next_label = self._get_next_available_label()
        expression = f"{next_label} = {template}"
        self.expression_input.setText(expression)
        
        # Update parameter table based on template
        self._update_parameter_table_for_template(template)
        
        # Log comprehensive template information
        self._log_template_info(template, next_label)

    def _log_template_info(self, template, next_label):
        """Log comprehensive template information including available channels and suggestions"""
        self._log_message(f"📋 Template selected: {template}")
        self._log_message(f"🏷️ Auto-generated label: '{next_label}' (you can change it to any name)")
        
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
            self._log_message(f"🔗 Available channels: {', '.join(available_channels)}")
        else:
            self._log_message("⚠️ No channels available - please select channels A and B first")
        
        # Provide template-specific suggestions
        self._log_template_suggestions(template, available_channels)

    def _log_template_suggestions(self, template, available_channels):
        """Log template-specific suggestions and usage examples"""
        template_lower = template.lower()
        
        # Arithmetic operations
        if any(op in template_lower for op in ['+', '-', '*', '/']):
            if len(available_channels) >= 2:
                self._log_message(f"💡 Example: {available_channels[0]} {template.split()[1]} {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
            self._log_message("💡 Tip: Basic arithmetic operations work element-wise on arrays")
        
        # Element-wise maximum/minimum operations using conditional logic
        elif ('>=') in template and ('<=') in template:
            if len(available_channels) >= 2:
                if '>=' in template and 'A *' in template and 'B *' in template:
                    op_name = "maximum" if '(A >= B)' in template else "minimum"
                    self._log_message(f"💡 Example: Find element-wise {op_name} between {available_channels[0]} and {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
            self._log_message("💡 Tip: Uses conditional logic to find element-wise max/min between signals")
        
        # Mathematical functions
        elif any(func in template_lower for func in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs']):
            func_name = next((f for f in ['sqrt', 'sin', 'cos', 'exp', 'log', 'abs'] if f in template_lower), 'function')
            self._log_message(f"💡 Applies {func_name}() function to signal data")
            if available_channels:
                self._log_message(f"💡 Example: Apply {func_name} to channel {available_channels[0]}")
        
        # Power operations
        elif '**' in template:
            self._log_message("💡 Tip: Raises signal values to specified power (e.g., A**2 for squaring)")
            if available_channels:
                self._log_message(f"💡 Example: Square all values in channel {available_channels[0]}")
        
        # Logical operations
        elif any(op in template for op in ['>', '<', '>=', '<=', '==', '!=']):
            self._log_message("💡 Tip: Creates binary signal (True/False) based on comparison")
            if len(available_channels) >= 2:
                self._log_message(f"💡 Example: Compare {available_channels[0]} with {available_channels[1] if len(available_channels) > 1 else available_channels[0]}")
        
        # Threshold masking operations
        elif any(op in template for op in ['>', '<', '>=', '<=']) and '*' in template:
            self._log_message("💡 Tip: Creates masked signal - zeros out values that don't meet condition")
            if len(available_channels) >= 2:
                self._log_message(f"💡 Example: Keep {available_channels[0]} values only where condition is true")
        
        # Normalization
        elif 'max' in template_lower and 'abs' in template_lower:
            self._log_message("💡 Tip: Normalizes signal to range [-1, 1] by dividing by maximum absolute value")
            if available_channels:
                self._log_message(f"💡 Example: Normalize channel {available_channels[0]} amplitude")
        
        # General suggestions
        if len(available_channels) > 2:
            other_channels = available_channels[2:]
            self._log_message(f"💡 You can also use mixed channels: {', '.join(other_channels)}")
        
        # Usage tips
        self._log_message("💡 Tips:")
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

    def _update_parameter_table_for_template(self, template):
        """Update parameter table based on selected template"""
        # Clear existing parameters
        self.param_table.setRowCount(0)
        
        # Add parameters based on template type
        template_lower = template.lower()
        
        # Add common parameters that might be needed
        if any(op in template_lower for op in ['blend', 'mix', 'ratio']):
            self._add_parameter_row("mix_ratio", "0.5")
        
        if any(op in template_lower for op in ['threshold', '>', '<', 'clip']):
            self._add_parameter_row("threshold", "0.5")
        
        if any(op in template_lower for op in ['tolerance', 'equal']):
            self._add_parameter_row("tolerance", "0.1")
        
        if any(op in template_lower for op in ['custom', 'expression']):
            self._add_parameter_row("custom_expr", "")
        
        # Always add output name parameter
        self._add_parameter_row("output_name", f"Mixed_{len(self.manager.get_mixed_channels()) + 1}")

    def _add_parameter_row(self, param_name, default_value):
        """Add a parameter row to the table"""
        row = self.param_table.rowCount()
        self.param_table.insertRow(row)
        
        # Parameter name (read-only)
        param_item = QTableWidgetItem(param_name)
        param_item.setFlags(param_item.flags() & ~Qt.ItemIsEditable)
        self.param_table.setItem(row, 0, param_item)
        
        # Parameter value (editable)
        value_item = QTableWidgetItem(str(default_value))
        self.param_table.setItem(row, 1, value_item)

    def _get_parameter_value(self, param_name):
        """Get parameter value from table"""
        for row in range(self.param_table.rowCount()):
            name_item = self.param_table.item(row, 0)
            if name_item and name_item.text() == param_name:
                value_item = self.param_table.item(row, 1)
                return value_item.text() if value_item else ""
        return ""



    def _on_expression_submitted(self):
        """Handle expression input submission"""
        self._on_create_mixed_channel()

    def _on_create_mixed_channel(self):
        """Handle create mixed channel button click"""
        expression = self.expression_input.text().strip()
        
        if not expression:
            self._log_message("❌ Please enter an expression")
            return
        
        # Get selected channels
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        if not channel_a or not channel_b:
            self._log_message("❌ Please select both channels A and B")
            return
        
        # Get alignment configuration
        alignment_config = self._get_alignment_config()
        
        # Create channel context
        channel_context = {'A': channel_a, 'B': channel_b}
        
        # Add existing mixed channels to context
        mixed_channels = self.manager.get_mixed_channels()
        for i, mixed_channel in enumerate(mixed_channels):
            label = chr(ord('C') + i)
            channel_context[label] = mixed_channel
        
        # Process expression with alignment
        new_channel, message = self.manager.process_mixer_expression_with_alignment(
            expression, channel_context, alignment_config
        )
        
        if new_channel:
            self.expression_input.clear()
            self._log_message(f"✅ {message}")
            self._update_button_states()
            self._update_plot()
        else:
            self._log_message(f"❌ {message}")

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
            self._refresh_after_undo()
            self._log_message("All mixed channels cleared")

    def _update_compatibility(self):
        """Update compatibility status between selected channels with alignment options"""
        # Check if required UI elements exist (defensive programming for initialization)
        if not hasattr(self, 'channel_a_combo') or not hasattr(self, 'channel_b_combo'):
            return
        if not hasattr(self, 'compatibility_label') or not hasattr(self, 'alignment_status_label'):
            return
            
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        if not channel_a or not channel_b:
            self.compatibility_label.setText("Select both channels")
            self.compatibility_label.setStyleSheet("color: #666; font-weight: normal;")
            self.alignment_status_label.setText("No alignment needed")
            if hasattr(self, 'create_btn'):
                self.create_btn.setEnabled(False)
            return
            
        # Get alignment configuration
        alignment_config = self._get_alignment_config()
        
        # Check compatibility with alignment
        is_valid, message, alignment_info = self.manager.validate_channels_for_mixing_with_alignment(
            channel_a, channel_b, alignment_config
        )
        
        if is_valid:
            self.compatibility_label.setText(f"✅ {message}")
            self.compatibility_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Update alignment status
            if alignment_info.get('needs_alignment', False):
                align_msg = alignment_info.get('alignment_message', 'Alignment will be applied')
                self.alignment_status_label.setText(f"🔧 {align_msg}")
                self.alignment_status_label.setStyleSheet("color: orange; font-size: 10px; padding: 5px;")
            else:
                self.alignment_status_label.setText("✅ Channels compatible as-is")
                self.alignment_status_label.setStyleSheet("color: green; font-size: 10px; padding: 5px;")
                
            # Enable/disable based on expression too (only if create button exists)
            if hasattr(self, 'create_btn') and hasattr(self, 'expression_input'):
                has_expression = bool(self.expression_input.text().strip())
                self.create_btn.setEnabled(has_expression)
        else:
            self.compatibility_label.setText(f"❌ {message}")
            self.compatibility_label.setStyleSheet("color: red; font-weight: bold;")
            self.alignment_status_label.setText("❌ Cannot align channels")
            self.alignment_status_label.setStyleSheet("color: red; font-size: 10px; padding: 5px;")
            if hasattr(self, 'create_btn'):
                self.create_btn.setEnabled(False)
    
    def _get_alignment_config(self):
        """Get current alignment configuration from UI controls"""
        mode = self.alignment_mode_combo.currentText()
        
        if mode == "Index-Based":
            index_mode = self.index_mode_combo.currentText()
            return {
                'alignment_method': 'index',
                'mode': 'truncate' if index_mode == "Truncate to Shortest" else 'custom',
                'start_index': self.start_index_spin.value() if index_mode == "Custom Range" else 0,
                'end_index': self.end_index_spin.value() if index_mode == "Custom Range" else 500,
                'offset': self.index_offset_spin.value()
            }
        else:  # Time-Based
            time_mode = self.time_mode_combo.currentText()
            return {
                'alignment_method': 'time',
                'mode': 'overlap' if time_mode == "Overlap Region" else 'custom',
                'start_time': self.start_time_spin.value() if time_mode == "Custom Range" else 0.0,
                'end_time': self.end_time_spin.value() if time_mode == "Custom Range" else 10.0,
                'offset': self.time_offset_spin.value(),
                'interpolation': self.interpolation_combo.currentText(),
                'round_to': self.round_to_spin.value()
            }

    def _update_button_states(self):
        """Update button enabled/disabled states"""
        can_undo = self.manager.can_undo()
        self.undo_btn.setEnabled(can_undo)
        
        # Create button enabled if channels are compatible
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        has_expression = bool(self.expression_input.text().strip())
        
        is_valid, _ = self.manager.validate_channels_for_mixing(channel_a, channel_b)
        self.create_btn.setEnabled(is_valid and has_expression)

    def _update_plot(self):
        """Update the plot with selected and mixed channels"""
        self.ax.clear()
        
        # Plot selected channels A and B
        channel_a = self.channel_a_combo.currentData()
        channel_b = self.channel_b_combo.currentData()
        
        plotted_any = False
        
        if channel_a and channel_a.ydata is not None:
            x_data = channel_a.xdata if channel_a.xdata is not None else np.arange(len(channel_a.ydata))
            self.ax.plot(x_data, channel_a.ydata, 'b-', alpha=0.7, linewidth=1,
                        label=f"A: {channel_a.legend_label or channel_a.channel_id}")
            plotted_any = True
        
        if channel_b and channel_b.ydata is not None:
            x_data = channel_b.xdata if channel_b.xdata is not None else np.arange(len(channel_b.ydata))
            self.ax.plot(x_data, channel_b.ydata, 'r-', alpha=0.7, linewidth=1,
                        label=f"B: {channel_b.legend_label or channel_b.channel_id}")
            plotted_any = True
        
        # Plot mixed channels
        mixed_channels = self.manager.get_mixed_channels()
        colors = self._get_mixed_channel_colors()
        
        for i, channel in enumerate(mixed_channels):
            if channel.ydata is not None:
                # Check if this channel is visible (from results table)
                channel_label = getattr(channel, 'step_table_label', f"Mixed_{i + 1}")
                is_visible = self._is_channel_visible_in_table(channel_label)
                
                if is_visible:
                    x_data = channel.xdata if channel.xdata is not None else np.arange(len(channel.ydata))
                    color = colors[i % len(colors)]
                    self.ax.plot(x_data, channel.ydata, color=color, linewidth=2,
                                label=f"{channel_label}: {channel.legend_label or channel.channel_id}")
                    plotted_any = True

        # Set up plot
        if plotted_any:
            self.ax.set_title("Signal Mixing")
            self.ax.legend()
        else:
            self.ax.set_title("No data to display")
        
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time")
        self.ax.grid(True)
        
        self.canvas.draw()

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
        """Check if a channel is visible based on results table checkbox"""
        for row in range(self.results_table.rowCount()):
            label_item = self.results_table.item(row, 0)
            if label_item and label_item.text() == label:
                checkbox = self.results_table.cellWidget(row, 1)
                return checkbox.isChecked() if checkbox else True
        return True  # Default to visible if not found

    def _log_message(self, message):
        """Add a message to the console output"""
        timestamp = time.strftime("%H:%M:%S")
        self.console_output.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        scrollbar = self.console_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    # Manager callback methods
    def _add_channel_to_table(self, channel):
        """Add a new mixed channel to the results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        # Label - use the actual label from the channel (e.g., Z, DOG, etc.)
        label = getattr(channel, 'step_table_label', f"Mixed_{row + 1}")
        self.results_table.setItem(row, 0, QTableWidgetItem(label))
        
        # Show checkbox
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(lambda state, lbl=label: self._on_channel_visibility_changed(lbl, state))
        self.results_table.setCellWidget(row, 1, checkbox)
        
        # Line color indicator - match the plot colors exactly
        color_info = self._get_mixed_channel_color_info(row)
        color_item = QTableWidgetItem()
        color_item.setBackground(color_info['qt_color'])
        color_item.setText(f"●")  # Add a colored circle
        color_item.setTextAlignment(Qt.AlignCenter)
        self.results_table.setItem(row, 2, color_item)
        
        # Operation
        operation = getattr(channel, 'operation', 'Mixed')
        self.results_table.setItem(row, 3, QTableWidgetItem(operation))
        
        # Description
        description = getattr(channel, 'description', channel.legend_label or channel.channel_id)
        self.results_table.setItem(row, 4, QTableWidgetItem(description))
        
        # Update plot to show the new channel
        self._update_plot()

    def _on_channel_visibility_changed(self, label, state):
        """Handle channel visibility change from results table"""
        # Update plot when visibility changes
        self._update_plot()

    def _replace_channel_in_table(self, index, new_channel):
        """Replace an existing channel in the results table"""
        if index < self.results_table.rowCount():
            # Update operation and description
            operation = getattr(new_channel, 'operation', 'Mixed')
            self.results_table.setItem(index, 3, QTableWidgetItem(operation))
            
            description = getattr(new_channel, 'description', new_channel.legend_label or new_channel.channel_id)
            self.results_table.setItem(index, 4, QTableWidgetItem(description))

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