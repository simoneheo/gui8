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

from base_plot_wizard import BasePlotWizard

# Handle mixer imports gracefully
try:
    from mixer.mixer_registry import MixerRegistry, load_all_mixers
    MIXER_AVAILABLE = True
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
from plot_manager import StylePreviewWidget
from channel import Channel


class SignalMixerWizardRefactored(BasePlotWizard):
    """
    Refactored Signal Mixer Wizard inheriting from BasePlotWizard
    Demonstrates consolidation of plotting functionality
    """
    
    # Additional signals specific to mixer
    mixer_applied = Signal(dict)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        # Initialize mixer-specific state
        self.manager = None
        self.all_operations = {}
        self._channel_visibility_states = {}
        
        # UI components specific to mixer wizard
        self.channel_a_file_combo = None
        self.channel_b_file_combo = None
        self.channel_a_combo = None
        self.channel_b_combo = None
        self.channel_a_stats = None
        self.channel_b_stats = None
        self.operations_list = None
        self.operations_filter = None
        self.create_btn = None
        self.undo_btn = None
        self.clear_btn = None
        self.expression_input = None
        self.mixed_channels_table = None
        
        # Alignment controls
        self.alignment_mode_combo = None
        self.index_mode_combo = None
        self.time_mode_combo = None
        self.alignment_parameter_spin = None
        
        # Initialize base class
        super().__init__(file_manager, channel_manager, signal_bus, parent)
        
        # Set window properties
        self.setWindowTitle("Signal Mixer Wizard")
        self.setMinimumSize(1200, 800)
        
        # Initialize mixer-specific components
        self._initialize_mixer_components()
        
        # Initialize UI with data
        self._initialize_ui()
        
        self._log_message("Mixer wizard initialized successfully")
    
    def _get_wizard_type(self) -> str:
        """Get the wizard type name"""
        return "Mixer"
    
    def _initialize_mixer_components(self):
        """Initialize mixer-specific components"""
        try:
            # Load mixer plugins
            if MIXER_AVAILABLE:
                load_all_mixers("mixer")
            
            # Create manager
            self.manager = SignalMixerWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=self.signal_bus,
                parent=self
            )
            
            # Setup callbacks
            self.manager.register_ui_callback('add_channel_to_table', self._add_channel_to_table)
            self.manager.register_ui_callback('replace_channel_in_table', self._replace_channel_in_table)
            self.manager.register_ui_callback('refresh_after_undo', self._refresh_after_undo)
            
        except Exception as e:
            self._log_message(f"Error initializing mixer components: {str(e)}")
    
    def _create_main_content(self, layout: QVBoxLayout):
        """Create the main content area specific to mixer wizard"""
        # Create horizontal splitter for left panel columns
        left_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(left_splitter)
        
        # Left column - Operations
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        self._create_operations_section(left_column_layout)
        
        # Right column - Controls
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        self._create_channel_selection_section(right_column_layout)
        self._create_alignment_controls_section(right_column_layout)
        self._create_console_section(right_column_layout)
        
        # Add columns to splitter
        left_splitter.addWidget(left_column)
        left_splitter.addWidget(right_column)
        left_splitter.setSizes([350, 500])
    
    def _create_operations_section(self, layout: QVBoxLayout):
        """Create the mixing operations section"""
        group = QGroupBox("Mixing Operations")
        group_layout = QVBoxLayout(group)
        
        # Operation filter
        self.operations_filter = QComboBox()
        self.operations_filter.addItems(["All", "Arithmetic", "Expression", "Logic"])
        self.operations_filter.currentTextChanged.connect(self._on_operation_filter_changed)
        group_layout.addWidget(QLabel("Category:"))
        group_layout.addWidget(self.operations_filter)
        
        # Operations list
        self.operations_list = QListWidget()
        self.operations_list.itemClicked.connect(self._on_operation_selected)
        group_layout.addWidget(self.operations_list)
        
        # Expression input
        group_layout.addWidget(QLabel("Custom Expression:"))
        self.expression_input = QLineEdit()
        self.expression_input.setPlaceholderText("e.g., A + B, A**2 + B**2")
        self.expression_input.returnPressed.connect(self._on_expression_submitted)
        group_layout.addWidget(self.expression_input)
        
        layout.addWidget(group)
    
    def _create_channel_selection_section(self, layout: QVBoxLayout):
        """Create the channel selection section"""
        group = QGroupBox("Channel Selection")
        group_layout = QVBoxLayout(group)
        
        # Channel A selection
        group_layout.addWidget(QLabel("Channel A:"))
        
        a_file_layout = QHBoxLayout()
        a_file_layout.addWidget(QLabel("File:"))
        self.channel_a_file_combo = QComboBox()
        self.channel_a_file_combo.currentTextChanged.connect(self._on_channel_a_file_changed)
        a_file_layout.addWidget(self.channel_a_file_combo)
        group_layout.addLayout(a_file_layout)
        
        a_channel_layout = QHBoxLayout()
        a_channel_layout.addWidget(QLabel("Channel:"))
        self.channel_a_combo = QComboBox()
        self.channel_a_combo.currentTextChanged.connect(self._on_channel_a_changed)
        a_channel_layout.addWidget(self.channel_a_combo)
        group_layout.addLayout(a_channel_layout)
        
        self.channel_a_stats = QLabel("No channel selected")
        group_layout.addWidget(self.channel_a_stats)
        
        # Channel B selection
        group_layout.addWidget(QLabel("Channel B:"))
        
        b_file_layout = QHBoxLayout()
        b_file_layout.addWidget(QLabel("File:"))
        self.channel_b_file_combo = QComboBox()
        self.channel_b_file_combo.currentTextChanged.connect(self._on_channel_b_file_changed)
        b_file_layout.addWidget(self.channel_b_file_combo)
        group_layout.addLayout(b_file_layout)
        
        b_channel_layout = QHBoxLayout()
        b_channel_layout.addWidget(QLabel("Channel:"))
        self.channel_b_combo = QComboBox()
        self.channel_b_combo.currentTextChanged.connect(self._on_channel_b_changed)
        b_channel_layout.addWidget(self.channel_b_combo)
        group_layout.addLayout(b_channel_layout)
        
        self.channel_b_stats = QLabel("No channel selected")
        group_layout.addWidget(self.channel_b_stats)
        
        layout.addWidget(group)
    
    def _create_alignment_controls_section(self, layout: QVBoxLayout):
        """Create the alignment controls section"""
        group = QGroupBox("Alignment Controls")
        group_layout = QFormLayout(group)
        
        # Alignment mode
        self.alignment_mode_combo = QComboBox()
        self.alignment_mode_combo.addItems(["None", "Index", "Time"])
        self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
        group_layout.addRow("Alignment Mode:", self.alignment_mode_combo)
        
        # Index mode
        self.index_mode_combo = QComboBox()
        self.index_mode_combo.addItems(["Start", "End", "Center"])
        self.index_mode_combo.currentTextChanged.connect(self._on_index_mode_changed)
        group_layout.addRow("Index Mode:", self.index_mode_combo)
        
        # Time mode
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems(["Start", "End", "Center"])
        self.time_mode_combo.currentTextChanged.connect(self._on_time_mode_changed)
        group_layout.addRow("Time Mode:", self.time_mode_combo)
        
        # Alignment parameter
        self.alignment_parameter_spin = QDoubleSpinBox()
        self.alignment_parameter_spin.setRange(-1000000, 1000000)
        self.alignment_parameter_spin.valueChanged.connect(self._on_alignment_parameter_changed)
        group_layout.addRow("Alignment Parameter:", self.alignment_parameter_spin)
        
        layout.addWidget(group)
    
    def _create_console_section(self, layout: QVBoxLayout):
        """Create the console section"""
        group = QGroupBox("Actions")
        group_layout = QVBoxLayout(group)
        
        # Create buttons
        self.create_btn = QPushButton("Create Mixed Channel")
        self.create_btn.clicked.connect(self._on_create_mixed_channel)
        
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.clicked.connect(self._on_undo_last)
        
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._on_clear_all)
        
        group_layout.addWidget(self.create_btn)
        group_layout.addWidget(self.undo_btn)
        group_layout.addWidget(self.clear_btn)
        
        layout.addWidget(group)
    
    def _create_results_tab(self) -> Optional[QWidget]:
        """Override to create mixed channels results table"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Mixed channels table
        self.mixed_channels_table = QTableWidget()
        self.mixed_channels_table.setColumnCount(6)
        self.mixed_channels_table.setHorizontalHeaderLabels([
            "Show", "Channel", "Style", "Operation", "Inputs", "Actions"
        ])
        self.mixed_channels_table.setAlternatingRowColors(True)
        self.mixed_channels_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Configure column widths
        header = self.mixed_channels_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Fixed)   # Show
        header.setSectionResizeMode(1, QHeaderView.Stretch) # Channel
        header.setSectionResizeMode(2, QHeaderView.Fixed)   # Style
        header.setSectionResizeMode(3, QHeaderView.Fixed)   # Operation
        header.setSectionResizeMode(4, QHeaderView.Fixed)   # Inputs
        header.setSectionResizeMode(5, QHeaderView.Fixed)   # Actions
        
        self.mixed_channels_table.setColumnWidth(0, 50)   # Show
        self.mixed_channels_table.setColumnWidth(2, 100)  # Style
        self.mixed_channels_table.setColumnWidth(3, 100)  # Operation
        self.mixed_channels_table.setColumnWidth(4, 150)  # Inputs
        self.mixed_channels_table.setColumnWidth(5, 200)  # Actions
        
        layout.addWidget(self.mixed_channels_table)
        
        return tab
    
    def _get_channels_to_plot(self) -> List[Channel]:
        """Get channels to plot - includes input channels and mixed channels"""
        channels = []
        
        # Add input channels
        channel_a = self.channel_a_combo.currentData() if self.channel_a_combo else None
        channel_b = self.channel_b_combo.currentData() if self.channel_b_combo else None
        
        if channel_a:
            channels.append(channel_a)
        if channel_b:
            channels.append(channel_b)
        
        # Add mixed channels from manager
        if self.manager:
            mixed_channels = self.manager.get_mixed_channels()
            channels.extend(mixed_channels)
        
        return channels
    
    def _initialize_ui(self):
        """Initialize UI with data"""
        # Populate channel dropdowns
        self._populate_channel_dropdowns()
        
        # Populate operations list
        self._populate_operations_list()
        
        # Auto-select best channels
        self._autopopulate_best_channels()
        
        # Update compatibility
        self._update_compatibility()
    
    def _populate_channel_dropdowns(self):
        """Populate file dropdowns with available files"""
        if not self.manager:
            return
        
        files = self.manager.get_available_files()
        
        # Clear existing items
        if self.channel_a_file_combo:
            self.channel_a_file_combo.clear()
        if self.channel_b_file_combo:
            self.channel_b_file_combo.clear()
        if self.channel_a_combo:
            self.channel_a_combo.clear()
        if self.channel_b_combo:
            self.channel_b_combo.clear()
        
        if not files:
            if self.channel_a_file_combo:
                self.channel_a_file_combo.addItem("No files available")
            if self.channel_b_file_combo:
                self.channel_b_file_combo.addItem("No files available")
            return
        
        # Add files to dropdowns
        for file_info in files:
            if self.channel_a_file_combo:
                self.channel_a_file_combo.addItem(file_info.filename, file_info)
            if self.channel_b_file_combo:
                self.channel_b_file_combo.addItem(file_info.filename, file_info)
        
        self._log_message(f"Loaded {len(files)} files for mixing")
    
    def _populate_operations_list(self):
        """Populate operations list with available mixing operations"""
        try:
            if MIXER_AVAILABLE:
                from mixer.mixer_registry import MixerRegistry
                all_templates = MixerRegistry.get_all_templates()
                categories = MixerRegistry.get_all_categories()
                
                self.all_operations = {}
                for category in categories:
                    category_templates = MixerRegistry.get_templates_by_category(category)
                    self.all_operations[category] = category_templates
                
                self._log_message(f"Loaded {len(all_templates)} templates from mixer folder")
            else:
                # Fallback templates
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
                self._log_message("Using fallback templates")
        
        except Exception as e:
            self._log_message(f"Error loading templates: {str(e)}")
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
        if not self.operations_list:
            return
        
        self.operations_list.clear()
        
        if category == "All":
            for cat, operations in self.all_operations.items():
                for op_display, op_code in operations:
                    self.operations_list.addItem(op_display)
        else:
            if category in self.all_operations:
                operations = self.all_operations[category]
                for op_display, op_code in operations:
                    self.operations_list.addItem(op_display)
    
    def _autopopulate_best_channels(self):
        """Auto-populate dropdowns with best channel pair"""
        if not self.manager:
            return
        
        channel_a, channel_b = self.manager.find_best_channel_pair()
        
        if channel_a and self.channel_a_file_combo and self.channel_a_combo:
            # Find and select file for channel A
            for i in range(self.channel_a_file_combo.count()):
                file_info = self.channel_a_file_combo.itemData(i)
                if file_info and file_info.file_id == channel_a.file_id:
                    self.channel_a_file_combo.setCurrentIndex(i)
                    # Select the specific channel
                    for j in range(self.channel_a_combo.count()):
                        if self.channel_a_combo.itemData(j) == channel_a:
                            self.channel_a_combo.setCurrentIndex(j)
                            break
                    break
        
        if channel_b and self.channel_b_file_combo and self.channel_b_combo:
            # Find and select file for channel B
            for i in range(self.channel_b_file_combo.count()):
                file_info = self.channel_b_file_combo.itemData(i)
                if file_info and file_info.file_id == channel_b.file_id:
                    self.channel_b_file_combo.setCurrentIndex(i)
                    # Select the specific channel
                    for j in range(self.channel_b_combo.count()):
                        if self.channel_b_combo.itemData(j) == channel_b:
                            self.channel_b_combo.setCurrentIndex(j)
                            break
                    break
    
    def _update_compatibility(self):
        """Update compatibility status and enable/disable controls"""
        if not self.manager:
            return
        
        channel_a = self.channel_a_combo.currentData() if self.channel_a_combo else None
        channel_b = self.channel_b_combo.currentData() if self.channel_b_combo else None
        
        compatible = self.manager.check_compatibility(channel_a, channel_b)
        
        if self.create_btn:
            self.create_btn.setEnabled(compatible)
        
        if compatible:
            self._log_message("Channels are compatible for mixing")
        else:
            self._log_message("Channels are not compatible for mixing")
    
    def _update_mixed_channels_table(self):
        """Update the mixed channels table"""
        if not self.mixed_channels_table or not self.manager:
            return
        
        mixed_channels = self.manager.get_mixed_channels()
        self.mixed_channels_table.setRowCount(len(mixed_channels))
        
        for row, channel in enumerate(mixed_channels):
            # Show checkbox
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(channel.channel_id in self.visible_channels)
            show_checkbox.toggled.connect(
                lambda checked, ch_id=channel.channel_id: self._on_channel_visibility_changed(ch_id, checked)
            )
            self.mixed_channels_table.setCellWidget(row, 0, show_checkbox)
            
            # Channel name
            name_item = QTableWidgetItem(channel.legend_label or channel.channel_id)
            self.mixed_channels_table.setItem(row, 1, name_item)
            
            # Style preview
            style_widget = StylePreviewWidget(
                color=getattr(channel, 'color', '#1f77b4'),
                style=getattr(channel, 'style', '-'),
                marker=getattr(channel, 'marker', None)
            )
            self.mixed_channels_table.setCellWidget(row, 2, style_widget)
            
            # Operation
            operation_item = QTableWidgetItem(getattr(channel, 'operation', 'Unknown'))
            self.mixed_channels_table.setItem(row, 3, operation_item)
            
            # Inputs
            inputs_text = getattr(channel, 'inputs_text', 'A, B')
            inputs_item = QTableWidgetItem(inputs_text)
            self.mixed_channels_table.setItem(row, 4, inputs_item)
            
            # Actions
            actions_btn = QPushButton("⚙")
            actions_btn.clicked.connect(lambda: self._open_channel_config(channel))
            self.mixed_channels_table.setCellWidget(row, 5, actions_btn)
    
    # Event handlers
    def _on_channel_a_file_changed(self):
        """Handle channel A file selection change"""
        if not self.channel_a_file_combo or not self.channel_a_combo:
            return
        
        file_info = self.channel_a_file_combo.currentData()
        self.channel_a_combo.clear()
        
        if file_info:
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            for channel in channels:
                if channel.show and channel.ydata is not None:
                    display_name = channel.legend_label or channel.channel_id
                    self.channel_a_combo.addItem(display_name, channel)
        
        self._update_compatibility()
        self._schedule_plot_update()
    
    def _on_channel_b_file_changed(self):
        """Handle channel B file selection change"""
        if not self.channel_b_file_combo or not self.channel_b_combo:
            return
        
        file_info = self.channel_b_file_combo.currentData()
        self.channel_b_combo.clear()
        
        if file_info:
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            for channel in channels:
                if channel.show and channel.ydata is not None:
                    display_name = channel.legend_label or channel.channel_id
                    self.channel_b_combo.addItem(display_name, channel)
        
        self._update_compatibility()
        self._schedule_plot_update()
    
    def _on_channel_a_changed(self):
        """Handle channel A selection change"""
        if not self.channel_a_combo or not self.channel_a_stats:
            return
        
        channel = self.channel_a_combo.currentData()
        if channel and self.manager:
            stats = self.manager.get_channel_stats(channel)
            self.channel_a_stats.setText(
                f"Length: {stats.get('length', 0)} | "
                f"Range: {stats.get('min', 0):.3f} to {stats.get('max', 0):.3f}"
            )
        else:
            self.channel_a_stats.setText("No channel selected")
        
        self._update_compatibility()
        self._schedule_plot_update()
    
    def _on_channel_b_changed(self):
        """Handle channel B selection change"""
        if not self.channel_b_combo or not self.channel_b_stats:
            return
        
        channel = self.channel_b_combo.currentData()
        if channel and self.manager:
            stats = self.manager.get_channel_stats(channel)
            self.channel_b_stats.setText(
                f"Length: {stats.get('length', 0)} | "
                f"Range: {stats.get('min', 0):.3f} to {stats.get('max', 0):.3f}"
            )
        else:
            self.channel_b_stats.setText("No channel selected")
        
        self._update_compatibility()
        self._schedule_plot_update()
    
    def _on_operation_filter_changed(self, category):
        """Handle operation filter change"""
        self._filter_operations(category)
    
    def _on_operation_selected(self, item):
        """Handle operation selection"""
        if self.expression_input:
            self.expression_input.setText(item.text())
    
    def _on_expression_submitted(self):
        """Handle expression submission"""
        self._on_create_mixed_channel()
    
    def _on_create_mixed_channel(self):
        """Handle create mixed channel button click"""
        if not self.manager:
            return
        
        try:
            channel_a = self.channel_a_combo.currentData() if self.channel_a_combo else None
            channel_b = self.channel_b_combo.currentData() if self.channel_b_combo else None
            expression = self.expression_input.text() if self.expression_input else ""
            
            if not channel_a or not channel_b:
                self._log_message("Please select both channels")
                return
            
            if not expression:
                self._log_message("Please enter an expression")
                return
            
            # Create mixed channel through manager
            mixed_channel = self.manager.create_mixed_channel(channel_a, channel_b, expression)
            
            if mixed_channel:
                self.add_channel(mixed_channel, visible=True)
                self._update_mixed_channels_table()
                self._log_message(f"Created mixed channel: {mixed_channel.legend_label}")
                self.mixer_applied.emit({'channel': mixed_channel, 'expression': expression})
            else:
                self._log_message("Failed to create mixed channel")
        
        except Exception as e:
            self._log_message(f"Error creating mixed channel: {str(e)}")
    
    def _on_undo_last(self):
        """Handle undo last button click"""
        if self.manager:
            self.manager.undo_last_operation()
            self._update_mixed_channels_table()
            self._schedule_plot_update()
            self._log_message("Undid last operation")
    
    def _on_clear_all(self):
        """Handle clear all button click"""
        if self.manager:
            self.manager.clear_all_mixed_channels()
            self.clear_all_channels()
            self._update_mixed_channels_table()
            self._log_message("Cleared all mixed channels")
    
    def _on_alignment_mode_changed(self, mode):
        """Handle alignment mode change"""
        self._log_message(f"Alignment mode changed to: {mode}")
    
    def _on_index_mode_changed(self, mode):
        """Handle index mode change"""
        self._log_message(f"Index mode changed to: {mode}")
    
    def _on_time_mode_changed(self, mode):
        """Handle time mode change"""
        self._log_message(f"Time mode changed to: {mode}")
    
    def _on_alignment_parameter_changed(self):
        """Handle alignment parameter change"""
        if self.alignment_parameter_spin:
            value = self.alignment_parameter_spin.value()
            self._log_message(f"Alignment parameter changed to: {value}")
    
    # Manager callback methods
    def _add_channel_to_table(self, channel):
        """Add channel to table (manager callback)"""
        self.add_channel(channel, visible=True)
        self._update_mixed_channels_table()
    
    def _replace_channel_in_table(self, index, new_channel):
        """Replace channel in table (manager callback)"""
        # Remove old channel and add new one
        old_channels = list(self.plotted_channels.keys())
        if index < len(old_channels):
            old_channel_id = old_channels[index]
            self.remove_channel(old_channel_id)
        
        self.add_channel(new_channel, visible=True)
        self._update_mixed_channels_table()
    
    def _refresh_after_undo(self):
        """Refresh after undo (manager callback)"""
        self._update_mixed_channels_table()
        self._schedule_plot_update() 