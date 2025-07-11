from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QTableWidgetItem, QDialog, QTabWidget, QDoubleSpinBox, QAbstractItemView, 
    QHeaderView, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QPixmap, QPainter, QPen, QBrush, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import re
import hashlib
from datetime import datetime


class ScriptChangeTracker:
    """Track changes to plot and stats scripts"""
    
    def __init__(self):
        self.original_plot_script = None
        self.original_stats_script = None
        self.plot_script_hash = None
        self.stats_script_hash = None
        self.plot_modification_time = None
        self.stats_modification_time = None
    
    def initialize_scripts(self, plot_script, stats_script):
        """Initialize original scripts and their hashes"""
        self.original_plot_script = plot_script
        self.original_stats_script = stats_script
        self.plot_script_hash = hashlib.md5(plot_script.encode()).hexdigest()
        self.stats_script_hash = hashlib.md5(stats_script.encode()).hexdigest()
        self.plot_modification_time = None
        self.stats_modification_time = None
    
    def is_plot_script_modified(self, current_script):
        """Check if plot script has been modified"""
        if self.plot_script_hash is None:
            return False
        current_hash = hashlib.md5(current_script.encode()).hexdigest()
        return current_hash != self.plot_script_hash
    
    def is_stats_script_modified(self, current_script):
        """Check if stats script has been modified"""
        if self.stats_script_hash is None:
            return False
        current_hash = hashlib.md5(current_script.encode()).hexdigest()
        return current_hash != self.stats_script_hash
    
    def mark_plot_script_modified(self):
        """Mark plot script as modified with timestamp"""
        self.plot_modification_time = datetime.now()
    
    def mark_stats_script_modified(self):
        """Mark stats script as modified with timestamp"""
        self.stats_modification_time = datetime.now()
    
    def reset_plot_script(self, new_original):
        """Reset plot script tracking with new original"""
        self.original_plot_script = new_original
        self.plot_script_hash = hashlib.md5(new_original.encode()).hexdigest()
        self.plot_modification_time = None
    
    def reset_stats_script(self, new_original):
        """Reset stats script tracking with new original"""
        self.original_stats_script = new_original
        self.stats_script_hash = hashlib.md5(new_original.encode()).hexdigest()
        self.stats_modification_time = None


class DynamicParameterCapture:
    """Dynamically capture all UI widget values for method configuration"""
    
    def __init__(self, parent_widget):
        self.parent_widget = parent_widget
        self.parameter_widgets = {}
        self.widget_types = {}
        
    def register_widget(self, param_name, widget, widget_type):
        """Register a widget for parameter capture"""
        self.parameter_widgets[param_name] = widget
        self.widget_types[param_name] = widget_type
        
    def capture_all_parameters(self):
        """Capture all registered parameter values"""
        parameters = {}
        
        for param_name, widget in self.parameter_widgets.items():
            try:
                # Check if widget still exists
                if widget is None:
                    parameters[param_name] = None
                    continue
                
                widget_type = self.widget_types[param_name]
                value = self._extract_widget_value(widget, widget_type)
                parameters[param_name] = value
            except Exception as e:
                print(f"[DynamicCapture] Error capturing {param_name}: {e}")
                parameters[param_name] = None
                
        return parameters
    
    def _extract_widget_value(self, widget, widget_type):
        """Extract value from widget based on its type"""
        if widget_type == 'spinbox':
            return widget.value()
        elif widget_type == 'doublespinbox':
            return widget.value()
        elif widget_type == 'combobox':
            return widget.currentText()
        elif widget_type == 'checkbox':
            return widget.isChecked()
        elif widget_type == 'lineedit':
            return widget.text()
        elif widget_type == 'textedit':
            return widget.toPlainText()
        elif widget_type == 'slider':
            return widget.value()
        elif widget_type == 'radiobutton':
            return widget.isChecked()
        elif widget_type == 'groupbox':
            # For group boxes, return True if checked, False otherwise
            return widget.isChecked()
        elif widget_type == 'buttongroup':
            # For button groups, return the checked button's text
            checked_button = widget.checkedButton()
            return checked_button.text() if checked_button else None
        else:
            # Generic fallback - try common value methods
            if hasattr(widget, 'value'):
                return widget.value()
            elif hasattr(widget, 'text'):
                return widget.text()
            elif hasattr(widget, 'isChecked'):
                return widget.isChecked()
            else:
                return str(widget)
    
    def set_parameter_values(self, parameters):
        """Set parameter values back to widgets"""
        for param_name, value in parameters.items():
            if param_name in self.parameter_widgets:
                try:
                    widget = self.parameter_widgets[param_name]
                    widget_type = self.widget_types[param_name]
                    self._set_widget_value(widget, widget_type, value)
                except Exception as e:
                    print(f"[DynamicCapture] Error setting {param_name}: {e}")
    
    def _set_widget_value(self, widget, widget_type, value):
        """Set value to widget based on its type"""
        if value is None:
            return
            
        if widget_type == 'spinbox':
            widget.setValue(int(value))
        elif widget_type == 'doublespinbox':
            widget.setValue(float(value))
        elif widget_type == 'combobox':
            index = widget.findText(str(value))
            if index >= 0:
                widget.setCurrentIndex(index)
        elif widget_type == 'checkbox':
            widget.setChecked(bool(value))
        elif widget_type == 'lineedit':
            widget.setText(str(value))
        elif widget_type == 'textedit':
            widget.setPlainText(str(value))
        elif widget_type == 'slider':
            widget.setValue(int(value))
        elif widget_type == 'radiobutton':
            widget.setChecked(bool(value))
        elif widget_type == 'groupbox':
            widget.setChecked(bool(value))
    
    def get_parameter_summary(self):
        """Get a summary of all captured parameters"""
        parameters = self.capture_all_parameters()
        summary = []
        
        for param_name, value in parameters.items():
            widget_type = self.widget_types.get(param_name, 'unknown')
            summary.append(f"{param_name} ({widget_type}): {value}")
            
        return "\n".join(summary)
    
    def clear_all_widgets(self):
        """Clear all registered widgets"""
        self.parameter_widgets.clear()
        self.widget_types.clear()


class ComparisonWizardWindow(QWidget):
    """
    Streamlined Data Comparison Wizard - UI Elements Only
    Focused on showing/controlling UI components without heavy data processing
    """
    
    # Signals for communication
    pair_added = Signal(dict)
    pair_deleted = Signal()
    plot_generated = Signal(dict)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store manager references
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.manager = None  # Will be set by the manager
        
        # Property to handle manager assignment
        self._manager = None
        
        # Initialize script change tracker
        self.script_tracker = ScriptChangeTracker()
        
        # Initialize dynamic parameter capture
        self.param_capture = DynamicParameterCapture(self)
        
        # Initialize UI
        self._init_ui()
        self._connect_signals()
        self._populate_initial_data()
        
        print("[ComparisonWizard] Initialized with script change tracking and dynamic parameter capture")
    
    @property
    def manager(self):
        """Get the manager"""
        return self._manager
    
    @manager.setter
    def manager(self, value):
        """Set the manager and connect to its signals"""
        self._manager = value
        
        # Connect to manager's comparison_completed signal if available
        if self._manager and hasattr(self._manager, 'comparison_completed'):
            try:
                self._manager.comparison_completed.connect(self._on_manager_comparison_completed)
                print(f"[ComparisonWizard] Connected to manager's comparison_completed signal")
            except Exception as e:
                print(f"[ComparisonWizard] Error connecting to manager signal: {e}")
        
    def refresh_comparison_methods(self, methods=None):
        """Refresh the comparison method list with new methods"""
        try:
            print(f"[DEBUG] refresh_comparison_methods called with methods={methods}, manager={self.manager}")
            
            if methods is None:
                # Try to get methods from manager
                if self.manager and hasattr(self.manager, 'get_comparison_methods'):
                    methods = self.manager.get_comparison_methods()
                    print(f"[DEBUG] Got {len(methods) if methods else 0} methods from manager in refresh: {methods}")
                else:
                    print(f"[DEBUG] Cannot get methods from manager in refresh (manager={self.manager})")
            
            if methods:
                self.method_list.clear()
                self.method_list.addItems(methods)
                
                # Select correlation method by default
                self._select_default_method("correlation")
                print(f"[ComparisonWizard] Refreshed with {len(methods)} comparison methods")
                return True
            else:
                print("[ComparisonWizard] No methods available for refresh")
                return False
                
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing comparison methods: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _populate_comparison_methods(self):
        """Populate the comparison method list from the registry"""
        try:
            print(f"[DEBUG] _populate_comparison_methods called, manager={self.manager}")
            
            # Try to get methods from manager
            if self.manager and hasattr(self.manager, 'get_comparison_methods'):
                methods = self.manager.get_comparison_methods()
                print(f"[DEBUG] Got {len(methods) if methods else 0} methods from manager: {methods}")
                
                if methods:
                    self.method_list.clear()
                    self.method_list.addItems(methods)
                    
                    # Select correlation method by default
                    self._select_default_method("correlation")
                    print(f"[ComparisonWizard] Loaded {len(methods)} comparison methods from registry")
                    return
                else:
                    print(f"[DEBUG] Manager returned empty methods list")
            else:
                if not self.manager:
                    print(f"[DEBUG] Manager is None")
                elif not hasattr(self.manager, 'get_comparison_methods'):
                    print(f"[DEBUG] Manager does not have get_comparison_methods method")
            
            # Fallback to sample methods if manager not available
            sample_methods = ["Correlation", "Bland-Altman", "Error Distribution", "Residual Analysis"]
            self.method_list.clear()
            self.method_list.addItems(sample_methods)
            print(f"[ComparisonWizard] Using fallback: Loaded {len(sample_methods)} sample comparison methods")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error populating comparison methods: {e}")
            import traceback
            traceback.print_exc()
    
    def _select_default_method(self, default_method_name):
        """Select the default method in the method list"""
        try:
            # Find the method in the list
            for i in range(self.method_list.count()):
                item_text = self.method_list.item(i).text()
                if default_method_name.lower() in item_text.lower():
                    self.method_list.setCurrentRow(i)
                    print(f"[ComparisonWizard] Selected default method: {item_text}")
                    
                    # Trigger method configuration update for the selected method
                    # Only do this if UI is ready
                    if hasattr(self, 'param_table'):
                        self._update_method_configuration(item_text)
                    return
            
            # If not found, select first item as fallback
            if self.method_list.count() > 0:
                self.method_list.setCurrentRow(0)
                first_item = self.method_list.item(0).text()
                print(f"[ComparisonWizard] Selected fallback method: {first_item}")
                
                # Trigger method configuration update for the selected method
                # Only do this if UI is ready
                if hasattr(self, 'param_table'):
                    self._update_method_configuration(first_item)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error selecting default method: {e}")
        
    def _init_ui(self):
        """Initialize the streamlined user interface"""
        self.setWindowTitle("Data Comparison Wizard - UI Demo")
        self.setMinimumSize(1200, 800)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Build panels
        self._build_left_panel(main_splitter)
        self._build_right_panel(main_splitter)
        
        # Set splitter proportions
        main_splitter.setSizes([220, 220, 760])
        
    def _build_left_panel(self, main_splitter):
        """Build the left control panel"""
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Create horizontal splitter for two columns
        left_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_layout.addWidget(left_splitter)
        
        # Left column
        left_col_widget = QWidget()
        left_col_layout = QVBoxLayout(left_col_widget)
        left_col_layout.setContentsMargins(5, 5, 5, 5)
        left_col_layout.setSpacing(10)
        
        # Right column
        right_col_widget = QWidget()
        right_col_layout = QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(5, 5, 5, 5)
        right_col_layout.setSpacing(10)
        
        # Left column components
        self._create_comparison_method_group(left_col_layout)
        self._create_method_controls_group(left_col_layout)
        self._create_performance_options_group(left_col_layout)
        self._create_action_buttons(left_col_layout)
        
        # Right column components
        self._create_channel_selection_group(right_col_layout)
        self._create_alignment_group(right_col_layout)
        self._create_console_group(right_col_layout)
        
        # Add columns to splitter
        left_splitter.addWidget(left_col_widget)
        left_splitter.addWidget(right_col_widget)
        left_splitter.setSizes([200, 200])
        
        main_splitter.addWidget(self.left_panel)
        
    def _create_comparison_method_group(self, layout):
        """Create comparison method selection group"""
        group = QGroupBox("Comparison Methods")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Method list
        self.method_list = QListWidget()
        self.method_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Populate methods from registry (will be updated by manager)
        self._populate_comparison_methods()
        
        group_layout.addWidget(self.method_list)
        layout.addWidget(group)
        
    def _create_method_controls_group(self, layout):
        """Create method configuration tabs"""
        group = QGroupBox("Method Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Tabbed interface
        self.method_tabs = QTabWidget()
        group_layout.addWidget(self.method_tabs)
        
        # Parameters tab
        self._create_parameters_tab()
        
        # Plot Script tab
        plot_tab = self._create_plot_script_tab()
        self.method_tabs.addTab(plot_tab, "📊 Plot Script")
        
        # Stat Script tab
        stat_tab = self._create_stat_script_tab()
        self.method_tabs.addTab(stat_tab, "📈 Stat Script")
        
        layout.addWidget(group)
        
    def _create_parameters_tab(self):
        """Create parameters table tab"""
        self.params_tab = QWidget()
        params_layout = QVBoxLayout(self.params_tab)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Parameter table
        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.setAlternatingRowColors(True)
        self.param_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Set column resize modes
        header = self.param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        
        # Set table styling
        self.param_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #e0e0e0;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                padding: 6px;
                border: 1px solid #ddd;
                font-weight: bold;
            }
        """)
        
        # Parameters will be populated dynamically when method is selected
        
        params_layout.addWidget(self.param_table)
        self.method_tabs.addTab(self.params_tab, "⚙️ Parameters")
        
    def _create_plot_script_tab(self):
        """Create the plot script tab with modification tracking"""
        plot_tab = QWidget()
        layout = QVBoxLayout(plot_tab)
        
        # Header with modification status
        header_layout = QHBoxLayout()
        header_label = QLabel("Plot Script:")
        header_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(header_label)
        
        # Modification status label
        self.plot_script_status_label = QLabel("Default")
        self.plot_script_status_label.setStyleSheet("color: gray; font-style: italic;")
        header_layout.addWidget(self.plot_script_status_label)
        header_layout.addStretch()
        
        # Reset button
        self.reset_plot_script_btn = QPushButton("Reset to Default")
        self.reset_plot_script_btn.setMaximumWidth(120)
        self.reset_plot_script_btn.clicked.connect(self._reset_plot_script)
        header_layout.addWidget(self.reset_plot_script_btn)
        
        layout.addLayout(header_layout)
        
        # Script text area
        self.plot_script_text = QTextEdit()
        self.plot_script_text.setFont(QFont("Consolas", 10))
        self.plot_script_text.setPlaceholderText("Plot script will be loaded when a comparison method is selected...")
        self.plot_script_text.textChanged.connect(self._on_plot_script_changed)
        layout.addWidget(self.plot_script_text)
        
        return plot_tab

    def _create_stat_script_tab(self):
        """Create the statistics script tab with modification tracking"""
        stat_tab = QWidget()
        layout = QVBoxLayout(stat_tab)
        
        # Header with modification status
        header_layout = QHBoxLayout()
        header_label = QLabel("Statistics Script:")
        header_label.setFont(QFont("Arial", 10, QFont.Bold))
        header_layout.addWidget(header_label)
        
        # Modification status label
        self.stats_script_status_label = QLabel("Default")
        self.stats_script_status_label.setStyleSheet("color: gray; font-style: italic;")
        header_layout.addWidget(self.stats_script_status_label)
        header_layout.addStretch()
        
        # Reset button
        self.reset_stats_script_btn = QPushButton("Reset to Default")
        self.reset_stats_script_btn.setMaximumWidth(120)
        self.reset_stats_script_btn.clicked.connect(self._reset_stats_script)
        header_layout.addWidget(self.reset_stats_script_btn)
        
        layout.addLayout(header_layout)
        
        # Script text area
        self.stats_script_text = QTextEdit()
        self.stats_script_text.setFont(QFont("Consolas", 10))
        self.stats_script_text.setPlaceholderText("Statistics script will be loaded when a comparison method is selected...")
        self.stats_script_text.textChanged.connect(self._on_stats_script_changed)
        layout.addWidget(self.stats_script_text)
        
        return stat_tab

    def _create_performance_options_group(self, layout):
        """Create performance options group"""
        group = QGroupBox("Performance Options")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Max points option
        max_points_layout = QHBoxLayout()
        self.max_points_checkbox = QCheckBox("Max Points:")
        self.max_points_input = QSpinBox()
        self.max_points_input.setRange(100, 50000)
        self.max_points_input.setValue(5000)
        self.max_points_input.setMaximumWidth(80)
        self.max_points_input.setEnabled(False)
        
        max_points_layout.addWidget(self.max_points_checkbox)
        max_points_layout.addWidget(self.max_points_input)
        max_points_layout.addStretch()
        group_layout.addLayout(max_points_layout)
        
        # Density display
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["Scatter", "Hexbin", "KDE"])
        density_layout.addWidget(self.density_combo)
        
        density_layout.addWidget(QLabel("Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(5, 200)
        self.bins_spinbox.setValue(50)
        self.bins_spinbox.setMaximumWidth(70)
        self.bins_spinbox.setEnabled(False)
        density_layout.addWidget(self.bins_spinbox)
        
        density_layout.addStretch()
        group_layout.addLayout(density_layout)
        
        layout.addWidget(group)
        
    def _create_action_buttons(self, layout):
        """Create action buttons"""
        buttons_layout = QHBoxLayout()
        
        self.refresh_plot_btn = QPushButton("Refresh Plot")
        self.refresh_plot_btn.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2471a3;
            }
        """)
        buttons_layout.addWidget(self.refresh_plot_btn)
        
        # Add configuration button to demonstrate dynamic parameter capture
        self.show_config_btn = QPushButton("Show Configuration")
        self.show_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.show_config_btn.clicked.connect(self._show_configuration_dialog)
        buttons_layout.addWidget(self.show_config_btn)
        
        layout.addLayout(buttons_layout)
        
    def _show_configuration_dialog(self):
        """Show current configuration in a dialog"""
        try:
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Current Configuration")
            dialog.setMinimumSize(600, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Configuration text area
            config_text = QTextEdit()
            config_text.setFont(QFont("Consolas", 10))
            config_text.setReadOnly(True)
            config_text.setPlainText(self.get_configuration_summary())
            layout.addWidget(config_text)
            
            # Buttons
            buttons_layout = QHBoxLayout()
            
            copy_btn = QPushButton("Copy to Clipboard")
            copy_btn.clicked.connect(lambda: self._copy_to_clipboard(config_text.toPlainText()))
            buttons_layout.addWidget(copy_btn)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            buttons_layout.addWidget(close_btn)
            
            layout.addLayout(buttons_layout)
            
            dialog.exec()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error showing configuration dialog: {e}")
    
    def _copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        try:
            from PySide6.QtGui import QClipboard
            from PySide6.QtWidgets import QApplication
            
            clipboard = QApplication.clipboard()
            clipboard.setText(text)
            print("[ComparisonWizard] Configuration copied to clipboard")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error copying to clipboard: {e}")
    
    def _create_channel_selection_group(self, layout):
        """Create channel selection group"""
        group = QGroupBox("Channel Selection")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QFormLayout(group)
        
        # Reference file and channel
        self.ref_file_combo = QComboBox()
        self.ref_file_combo.addItems(["File1.csv", "File2.csv", "Data.xlsx"])
        group_layout.addRow("Reference File:", self.ref_file_combo)
        
        self.ref_channel_combo = QComboBox()
        self.ref_channel_combo.addItems(["Channel_A", "Channel_B", "Sensor_1"])
        group_layout.addRow("Reference Channel:", self.ref_channel_combo)
        
        # Test file and channel
        self.test_file_combo = QComboBox()
        self.test_file_combo.addItems(["File1.csv", "File2.csv", "Data.xlsx"])
        self.test_file_combo.setCurrentIndex(1)
        group_layout.addRow("Test File:", self.test_file_combo)
        
        self.test_channel_combo = QComboBox()
        self.test_channel_combo.addItems(["Channel_A", "Channel_B", "Sensor_1"])
        self.test_channel_combo.setCurrentIndex(1)
        group_layout.addRow("Test Channel:", self.test_channel_combo)
        
        layout.addWidget(group)
        
    def _create_alignment_group(self, layout):
        """Create data alignment controls"""
        group = QGroupBox("Data Alignment")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Alignment mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Alignment Mode:"))
        self.alignment_mode_combo = QComboBox()
        self.alignment_mode_combo.addItems(["Index-Based", "Time-Based"])
        mode_layout.addWidget(self.alignment_mode_combo)
        group_layout.addLayout(mode_layout)
        
        # Index-based options
        self.index_group = QGroupBox("Index Options")
        index_layout = QFormLayout(self.index_group)
        
        self.index_mode_combo = QComboBox()
        self.index_mode_combo.addItems(["Truncate to Shortest", "Custom Range"])
        index_layout.addRow("Mode:", self.index_mode_combo)
        
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(0, 999999)
        index_layout.addRow("Start Index:", self.start_index_spin)
        
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(0, 999999)
        index_layout.addRow("End Index:", self.end_index_spin)
        
        self.index_offset_spin = QSpinBox()
        self.index_offset_spin.setRange(-999999, 999999)
        index_layout.addRow("Offset:", self.index_offset_spin)
        
        group_layout.addWidget(self.index_group)
        
        # Time-based options
        self.time_group = QGroupBox("Time Options")
        time_layout = QFormLayout(self.time_group)
        
        self.time_mode_combo = QComboBox()
        self.time_mode_combo.addItems(["Overlap Region", "Custom Range"])
        time_layout.addRow("Mode:", self.time_mode_combo)
        
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(-999999999.0, 999999999.0)
        self.start_time_spin.setDecimals(9)
        self.start_time_spin.setSingleStep(0.1)
        self.start_time_spin.setKeyboardTracking(True)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-999999999.0, 999999999.0)
        self.end_time_spin.setDecimals(9)
        self.end_time_spin.setSingleStep(0.1)
        self.end_time_spin.setValue(10.0)
        self.end_time_spin.setKeyboardTracking(True)
        time_layout.addRow("End Time:", self.end_time_spin)
        
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-999999999.0, 999999999.0)
        self.time_offset_spin.setDecimals(9)
        self.time_offset_spin.setSingleStep(0.1)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setKeyboardTracking(True)
        time_layout.addRow("Offset:", self.time_offset_spin)
        
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["linear", "nearest", "cubic"])
        time_layout.addRow("Interpolation:", self.interpolation_combo)
        
        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.000000001, 999999999.0)
        self.resolution_spin.setDecimals(9)
        self.resolution_spin.setSingleStep(0.01)
        self.resolution_spin.setValue(0.1)
        self.resolution_spin.setSuffix(" s")
        self.resolution_spin.setKeyboardTracking(True)
        time_layout.addRow("Resolution:", self.resolution_spin)
        
        group_layout.addWidget(self.time_group)
        
        # Status
        self.alignment_status_label = QLabel("Alignment configured")
        self.alignment_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        group_layout.addWidget(self.alignment_status_label)
        
        # Show/hide groups based on mode
        self._on_alignment_mode_changed("Index-Based")
        
        layout.addWidget(group)
        
    def _create_console_group(self, layout):
        """Create console output and pair management"""
        group = QGroupBox("Console Output")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Console output
        self.info_output = QTextEdit()
        self.info_output.setReadOnly(True)
        self.info_output.setPlaceholderText("Logs and messages will appear here")
        self.info_output.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
            }
        """)
        group_layout.addWidget(self.info_output)
        
        # Pair name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Pair Name:"))
        self.pair_name_input = QLineEdit()
        self.pair_name_input.setPlaceholderText("Enter pair name (optional)")
        name_layout.addWidget(self.pair_name_input)
        group_layout.addLayout(name_layout)
        
        # Add Pair button
        self.add_pair_btn = QPushButton("Add Comparison Pair")
        self.add_pair_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #219a52;
            }
        """)
        group_layout.addWidget(self.add_pair_btn)
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with results and plots"""
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(5)
        
        # Create vertical splitter
        right_splitter = QSplitter(Qt.Vertical)
        right_layout.addWidget(right_splitter)
        
        # Top: Channels table
        channels_widget = QWidget()
        channels_layout = QVBoxLayout(channels_widget)
        channels_layout.setContentsMargins(0, 0, 0, 0)
        channels_layout.addWidget(QLabel("Channels:"))
        self._build_channels_table(channels_layout)
        
        # Middle: Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(QLabel("Plot:"))
        self._build_plot_area(plot_layout)
        
        # Bottom: Overlays table
        overlays_widget = QWidget()
        overlays_layout = QVBoxLayout(overlays_widget)
        overlays_layout.setContentsMargins(0, 0, 0, 0)
        overlays_layout.addWidget(QLabel("Overlays:"))
        self._build_overlays_table(overlays_layout)
        
        # Add to splitter
        right_splitter.addWidget(channels_widget)
        right_splitter.addWidget(plot_widget)
        right_splitter.addWidget(overlays_widget)
        
        # Set proportions
        right_splitter.setSizes([80, 520, 100])
        
        main_splitter.addWidget(self.right_panel)
        
    def _build_channels_table(self, layout):
        """Build the channels/pairs table"""
        self.channels_table = QTableWidget(0, 5)
        self.channels_table.setHorizontalHeaderLabels(["Show", "Style", "Pair Name", "Shape", "Actions"])
        self.channels_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.channels_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        self.channels_table.setColumnWidth(0, 60)
        self.channels_table.setColumnWidth(1, 80)
        self.channels_table.setColumnWidth(2, 120)
        self.channels_table.setColumnWidth(3, 80)
        
        # Table will be populated dynamically when pairs are added
        
        layout.addWidget(self.channels_table)
        
    def _build_plot_area(self, layout):
        """Build the plot area with matplotlib"""
        # Create matplotlib figure
        self.figure = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        
        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create initial plot
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Select channels and add pairs to generate plots', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add to layout
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
    def _build_overlays_table(self, layout):
        """Build overlays table"""
        self.overlay_table = QTableWidget(0, 4)
        self.overlay_table.setHorizontalHeaderLabels(["Show", "Style", "Name", "Actions"])
        self.overlay_table.setAlternatingRowColors(True)
        
        # Set column widths
        header = self.overlay_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        
        self.overlay_table.setColumnWidth(0, 50)
        self.overlay_table.setColumnWidth(1, 80)
        self.overlay_table.setColumnWidth(3, 100)
        
        # Table will be populated dynamically when overlays are generated
        
        layout.addWidget(self.overlay_table)
        
    def _connect_signals(self):
        """Connect UI signals to their respective handlers"""
        try:
            # Method selection
            self.method_list.currentItemChanged.connect(self._on_method_changed)
            
            # Alignment mode
            self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
            
            # Performance options
            self.max_points_checkbox.toggled.connect(self.max_points_input.setEnabled)
            self.density_combo.currentTextChanged.connect(self._on_density_changed)
            
            # Action buttons
            self.refresh_plot_btn.clicked.connect(self._on_refresh_clicked)
            self.add_pair_btn.clicked.connect(self._on_add_pair_clicked)
            
            # Channel selection changes - hook to update alignment
            self.ref_channel_combo.currentIndexChanged.connect(self._on_channel_selection_changed)
            self.test_channel_combo.currentIndexChanged.connect(self._on_channel_selection_changed)
            # New: file selection changes update channel dropdowns
            self.ref_file_combo.currentIndexChanged.connect(self._on_ref_file_changed)
            self.test_file_combo.currentIndexChanged.connect(self._on_test_file_changed)
            

            
        except Exception as e:
            print(f"[ComparisonWizard] Error connecting signals: {e}")
    
    def _populate_initial_data(self):
        """Populate initial data for the wizard"""
        try:
            # Tables will be populated dynamically when pairs are added
            # and overlays are generated by the analysis methods
            print("[ComparisonWizard] Initial data setup complete - tables will be populated dynamically")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting up initial data: {e}")
            

            

            
    def _update_method_configuration(self, method_name):
        """Update the method configuration section based on selected method"""
        try:
            # Get the comparison class from the manager
            comparison_cls = None
            if self.manager and hasattr(self.manager, 'get_comparison_class'):
                # Try exact match first
                comparison_cls = self.manager.get_comparison_class(method_name)
                
                # If not found, try lowercase version
                if not comparison_cls:
                    comparison_cls = self.manager.get_comparison_class(method_name.lower())
                
                # If still not found, try common variations
                if not comparison_cls:
                    method_variations = [
                        method_name.replace(' ', '_').lower(),
                        method_name.replace('-', '_').lower(),
                        method_name.replace(' ', '').lower()
                    ]
                    for variation in method_variations:
                        comparison_cls = self.manager.get_comparison_class(variation)
                        if comparison_cls:
                            break
            
            if comparison_cls:
                print(f"[ComparisonWizard] Found comparison class: {comparison_cls.__name__}")
                
                # Update parameters table
                self._update_parameters_table(comparison_cls)
                
                # Update plot script tab
                self._update_plot_script_tab(comparison_cls)
                
                # Update stat script tab
                self._update_stat_script_tab(comparison_cls)
                
            else:
                print(f"[ComparisonWizard] No comparison class found for method: {method_name}")
                # Clear configuration sections
                self._clear_method_configuration()
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating method configuration: {e}")
            self._clear_method_configuration()
    
    def _on_parameter_changed(self):
        """Handle parameter value changes"""
        try:
            # Get current method and parameters
            method_name = self.get_current_method_name()
            params = self.get_current_parameters()
            
            if method_name:
                print(f"[ComparisonWizard] Parameters changed for {method_name}: {params}")
                
                # Optionally trigger plot update or other actions
                # self._update_sample_plot()
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling parameter change: {e}")
    
    def _connect_parameter_signals(self):
        """Connect signals for parameter widgets to handle value changes"""
        try:
            # Connect signals for all parameter widgets
            for i in range(self.param_table.rowCount()):
                value_widget = self.param_table.cellWidget(i, 1)
                if value_widget:
                    if isinstance(value_widget, QCheckBox):
                        value_widget.toggled.connect(self._on_parameter_changed)
                    elif isinstance(value_widget, QComboBox):
                        value_widget.currentTextChanged.connect(self._on_parameter_changed)
                    elif isinstance(value_widget, QDoubleSpinBox):
                        value_widget.valueChanged.connect(self._on_parameter_changed)
                    elif isinstance(value_widget, QSpinBox):
                        value_widget.valueChanged.connect(self._on_parameter_changed)
                    elif isinstance(value_widget, QLineEdit):
                        value_widget.textChanged.connect(self._on_parameter_changed)
                        
        except Exception as e:
            print(f"[ComparisonWizard] Error connecting parameter signals: {e}")
    
    def _update_parameters_table(self, comparison_cls):
        """Update the parameters table with the selected comparison method's parameters"""
        try:
            # Clear parameter capture for new method
            self.param_capture.clear_all_widgets()
            
            # Clear existing rows
            self.param_table.setRowCount(0)
            
            # Get parameters from the comparison class
            if hasattr(comparison_cls, 'get_parameters'):
                parameters = comparison_cls.get_parameters()
                
                # Add each parameter to the table
                for i, param in enumerate(parameters):
                    self.param_table.insertRow(i)
                    
                    # Parameter name
                    param_name = param.get('name', f'param_{i}')
                    name_item = QTableWidgetItem(param_name)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                    self.param_table.setItem(i, 0, name_item)
                    
                    # Parameter widget
                    widget = self._create_parameter_widget(param)
                    self.param_table.setCellWidget(i, 1, widget)
                    
                # Connect parameter change signals
                self._connect_parameter_signals()
                
                print(f"[ComparisonWizard] Updated parameters table with {len(parameters)} parameters")
                
            else:
                print(f"[ComparisonWizard] No parameters found for {comparison_cls.__name__}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating parameters table: {e}")

    def get_all_current_configuration(self):
        """Get complete current configuration including parameters and scripts"""
        try:
            config = {
                'method_name': self.get_current_method_name(),
                'parameters': self.param_capture.capture_all_parameters(),
                'plot_script': self.plot_script_text.toPlainText(),
                'stats_script': self.stats_script_text.toPlainText(),
                'plot_script_modified': self.script_tracker.is_plot_script_modified(
                    self.plot_script_text.toPlainText()
                ),
                'stats_script_modified': self.script_tracker.is_stats_script_modified(
                    self.stats_script_text.toPlainText()
                ),
                'alignment_mode': self.alignment_mode_combo.currentText(),
                'timestamp': datetime.now().isoformat()
            }
            
            return config
            
        except Exception as e:
            print(f"[ComparisonWizard] Error getting current configuration: {e}")
            return {}

    def set_configuration(self, config):
        """Set configuration from a saved state"""
        try:
            # Set method
            if 'method_name' in config:
                method_name = config['method_name']
                # Find and select the method in the list widget
                for i in range(self.method_list.count()):
                    item = self.method_list.item(i)
                    if item and item.text() == method_name:
                        self.method_list.setCurrentRow(i)
                        break
            
            # Set parameters
            if 'parameters' in config:
                self.param_capture.set_parameter_values(config['parameters'])
            
            # Set scripts
            if 'plot_script' in config:
                self.plot_script_text.setPlainText(config['plot_script'])
            
            if 'stats_script' in config:
                self.stats_script_text.setPlainText(config['stats_script'])
            
            # Set alignment mode
            if 'alignment_mode' in config:
                index = self.alignment_mode_combo.findText(config['alignment_mode'])
                if index >= 0:
                    self.alignment_mode_combo.setCurrentIndex(index)
            
            print(f"[ComparisonWizard] Configuration loaded for method: {config.get('method_name', 'unknown')}")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting configuration: {e}")

    def get_configuration_summary(self):
        """Get a human-readable summary of the current configuration"""
        try:
            config = self.get_all_current_configuration()
            
            summary = []
            summary.append(f"Method: {config.get('method_name', 'None')}")
            summary.append(f"Alignment: {config.get('alignment_mode', 'None')}")
            summary.append(f"Plot Script Modified: {config.get('plot_script_modified', False)}")
            summary.append(f"Stats Script Modified: {config.get('stats_script_modified', False)}")
            summary.append(f"Timestamp: {config.get('timestamp', 'None')}")
            summary.append("")
            summary.append("Parameters:")
            summary.append(self.param_capture.get_parameter_summary())
            
            return "\n".join(summary)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error getting configuration summary: {e}")
            return "Error getting configuration summary"
    
    def _create_parameter_widget(self, param):
        """Create appropriate widget for parameter based on its type and constraints"""
        param_name = param.get('name', 'unknown')
        param_type = param.get('type', 'str')
        param_value = param.get('default')  # Use 'default' for initial value
        param_min = param.get('min')
        param_max = param.get('max')
        param_options = param.get('options', [])
        param_description = param.get('description', '')
        
        widget = None
        widget_type = None
        
        try:
            # Handle different parameter types (support both string and Python types)
            if param_options:
                # Dropdown for options
                widget = QComboBox()
                widget.addItems([str(opt) for opt in param_options])
                if param_value is not None:
                    index = widget.findText(str(param_value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                widget_type = 'combobox'
                
            elif param_type == bool or param_type == "bool":
                # Checkbox for boolean
                widget = QCheckBox()
                if param_value is not None:
                    widget.setChecked(bool(param_value))
                widget_type = 'checkbox'
                
            elif param_type == int or param_type == "int":
                # SpinBox for integers
                widget = QSpinBox()
                if param_min is not None:
                    widget.setMinimum(int(param_min))
                if param_max is not None:
                    widget.setMaximum(int(param_max))
                if param_value is not None:
                    widget.setValue(int(param_value))
                widget_type = 'spinbox'
                
            elif param_type == float or param_type == "float":
                # DoubleSpinBox for floats
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                if param_min is not None:
                    widget.setMinimum(float(param_min))
                if param_max is not None:
                    widget.setMaximum(float(param_max))
                if param_value is not None:
                    widget.setValue(float(param_value))
                widget_type = 'doublespinbox'
                
            elif param_type == str or param_type == "str":
                # LineEdit for strings
                widget = QLineEdit()
                if param_value is not None:
                    widget.setText(str(param_value))
                widget_type = 'lineedit'
                
            else:
                # Default to LineEdit
                widget = QLineEdit()
                if param_value is not None:
                    widget.setText(str(param_value))
                widget_type = 'lineedit'
                
            # Set tooltip if description available
            if param_description:
                widget.setToolTip(param_description)
                
            # Register widget with dynamic parameter capture
            self.param_capture.register_widget(param_name, widget, widget_type)
            
            return widget
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating parameter widget for {param_name}: {e}")
            # Fallback to LineEdit
            widget = QLineEdit()
            if param_value is not None:
                widget.setText(str(param_value))
            self.param_capture.register_widget(param_name, widget, 'lineedit')
            return widget
    
    def _update_plot_script_tab(self, comparison_cls):
        """Update plot script tab with method-specific script"""
        try:
            if not hasattr(comparison_cls, 'plot_script'):
                print(f"[ComparisonWizard] No plot script found for {comparison_cls.__name__}")
                return
            
            # Get the method's source code as a string
            import inspect
            plot_script = inspect.getsource(comparison_cls.plot_script)
            
            # Initialize script tracker if not already done
            if self.script_tracker.original_plot_script is None:
                stats_script = inspect.getsource(comparison_cls.stats_script) if hasattr(comparison_cls, 'stats_script') else ''
                self.script_tracker.initialize_scripts(plot_script, stats_script)
            else:
                # Reset to new script
                self.script_tracker.reset_plot_script(plot_script)
            
            # Load into text editor
            self.plot_script_text.setPlainText(plot_script)
            
            # Reset status
            self.plot_script_status_label.setText("Default")
            self.plot_script_status_label.setStyleSheet("color: gray; font-style: italic;")
            
            print(f"[ComparisonWizard] Plot script loaded for {comparison_cls.__name__}")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error updating plot script tab: {e}")

    def _update_stat_script_tab(self, comparison_cls):
        """Update statistics script tab with method-specific script"""
        try:
            if not hasattr(comparison_cls, 'stats_script'):
                print(f"[ComparisonWizard] No stats script found for {comparison_cls.__name__}")
                return
            
            # Get the method's source code as a string
            import inspect
            stats_script = inspect.getsource(comparison_cls.stats_script)
            
            # Initialize script tracker if not already done
            if self.script_tracker.original_stats_script is None:
                plot_script = inspect.getsource(comparison_cls.plot_script) if hasattr(comparison_cls, 'plot_script') else ''
                self.script_tracker.initialize_scripts(plot_script, stats_script)
            else:
                # Reset to new script
                self.script_tracker.reset_stats_script(stats_script)
            
            # Load into text editor
            self.stats_script_text.setPlainText(stats_script)
            
            # Reset status
            self.stats_script_status_label.setText("Default")
            self.stats_script_status_label.setStyleSheet("color: gray; font-style: italic;")
            
            print(f"[ComparisonWizard] Stats script loaded for {comparison_cls.__name__}")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error updating stats script tab: {e}")
    
    
    def _clear_method_configuration(self):
        """Clear all method configuration sections"""
        try:
            # Clear parameters table
            self.param_table.setRowCount(0)
            
            # Clear script editors
            self.plot_script_text.setPlainText("")
            self.stats_script_text.setPlainText("")
            
            print("[ComparisonWizard] Method configuration cleared")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error clearing method configuration: {e}")
    
    def _on_method_changed(self, current, previous):
        """Handle method selection change"""
        try:
            if current:
                method_name = current.text()
                print(f"[ComparisonWizard] Method changed to: {method_name}")
                
                # Update method configuration based on selected method
                self._update_method_configuration(method_name)
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling method change: {e}")
            
    def _on_alignment_mode_changed(self, mode):
        """Handle alignment mode change"""
        try:
            if mode == "Index-Based":
                self.index_group.show()
                self.time_group.hide()
            else:
                self.index_group.hide()
                self.time_group.show()
                
                # Update alignment status
                self.alignment_status_label.setText(f"Alignment mode: {mode}")
                
                # Call the manager's suggest_alignment_parameters for the new mode
                if self.manager and hasattr(self.manager, 'suggest_alignment_parameters'):
                    params = self.manager.suggest_alignment_parameters(mode=mode)
                    if params:
                        self._apply_alignment_params(params)
                        print(f"[ComparisonWizard] Applied suggested alignment parameters for {mode} mode")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling alignment mode change: {e}")
    
    def _on_density_changed(self, density_type):
        """Handle density type change"""
        try:
            # Enable/disable bins spinbox based on density type
            if density_type in ["Hexbin", "KDE"]:
                self.bins_spinbox.setEnabled(True)
            else:
                self.bins_spinbox.setEnabled(False)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error handling density change: {e}")
        
    def _on_refresh_clicked(self):
        """Handle refresh plot button click - rerun analysis with current method configuration"""
        try:
            print("[ComparisonWizard] Refresh button clicked - triggering analysis refresh")
            
            if hasattr(self, 'info_output'):
                self.info_output.append("🔄 Refreshing plot with current settings...")
            
            # Check if we have a manager and pairs to analyze
            if not self.manager:
                print("[ComparisonWizard] No manager available for refresh")
                if hasattr(self, 'info_output'):
                    self.info_output.append("❌ No manager available for refresh")
                return
            
            # Check if we have any pairs to analyze
            if not hasattr(self.manager, 'pair_manager') or not self.manager.pair_manager:
                print("[ComparisonWizard] No pair manager available")
                if hasattr(self, 'info_output'):
                    self.info_output.append("❌ No pair manager available")
                return
            
            all_pairs = self.manager.pair_manager.get_all_pairs()
            if not all_pairs:
                print("[ComparisonWizard] No pairs available to refresh")
                if hasattr(self, 'info_output'):
                    self.info_output.append("ℹ️ No pairs available to refresh - add some pairs first")
                return
            
            # Get current method configuration
            method_name = self.get_current_method_name()
            parameters = self.get_current_parameters()
            
            if not method_name:
                print("[ComparisonWizard] No method selected for refresh")
                if hasattr(self, 'info_output'):
                    self.info_output.append("❌ No comparison method selected")
                return
            
            print(f"[ComparisonWizard] Refreshing {len(all_pairs)} pairs with method '{method_name}' and parameters: {parameters}")
            
            # Trigger analysis refresh through the manager
            if hasattr(self.manager, '_perform_analysis'):
                print("[ComparisonWizard] Triggering analysis refresh...")
                self.manager._perform_analysis()
                print("[ComparisonWizard] Analysis refresh triggered successfully")
            else:
                print("[ComparisonWizard] Manager does not have _perform_analysis method")
                if hasattr(self, 'info_output'):
                    self.info_output.append("❌ Manager does not support analysis refresh")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing plot: {e}")
            if hasattr(self, 'info_output'):
                self.info_output.append(f"❌ Error refreshing plot: {e}")
        
    def _on_add_pair_clicked(self):
        """Handle add pair button click"""
        try:
            # Get current selections - use currentData() to get Channel objects
            ref_file = self.ref_file_combo.currentText()
            ref_channel = self.ref_channel_combo.currentData()  # Get Channel object
            test_file = self.test_file_combo.currentText()
            test_channel = self.test_channel_combo.currentData()  # Get Channel object
            method_name = self.get_current_method_name()
            params = self.get_current_parameters()
            
            # Get alignment parameters
            alignment_params = self._get_alignment_params()
            
            # Perform alignment if we have Channel objects
            alignment_result = None
            if ref_channel and test_channel and self.manager:
                print(f"[DEBUG] _on_add_pair_clicked: Performing alignment for {ref_channel.legend_label} vs {test_channel.legend_label}")
                alignment_result = self.manager.perform_alignment(ref_channel, test_channel, alignment_params)
            else:
                print(f"[DEBUG] _on_add_pair_clicked: Cannot perform alignment - missing channels or manager")
            
            # Create pair data (pure data, no method configuration)
            pair_data = {
                'name': self.pair_name_input.text() or f"{getattr(ref_channel, 'legend_label', 'Unknown')}_vs_{getattr(test_channel, 'legend_label', 'Unknown')}",
                'ref_file': ref_file,
                'ref_channel': ref_channel,
                'test_file': test_file,
                'test_channel': test_channel,
                'alignment_params': alignment_params,
                'alignment_result': alignment_result
            }
            
            # Send to manager for processing (manager will emit comparison_completed signal with result)
            if self.manager:
                self.manager._on_pair_added(pair_data)
            else:
                # Fallback if no manager - add to table and show success message
                self._add_pair_to_table(pair_data['name'])
                if hasattr(self, 'info_output'):
                    self.info_output.append(f"✅ Added comparison pair: {pair_data['name']}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error adding pair: {e}")
    
    def _on_manager_comparison_completed(self, result):
        """Handle comparison completion results from the manager"""
        try:
            result_type = result.get('type', 'unknown')
            
            if result_type == 'pair_added':
                # Pair was successfully added
                pair_data = result.get('data', {})
                pair_name = pair_data.get('name', 'Unnamed')
                
                # Update the table
                self._add_pair_to_table(pair_name)
                
                # Show success message
                if hasattr(self, 'info_output'):
                    self.info_output.append(f"✅ Added comparison pair: {pair_name}")
                    
            elif result_type == 'pair_add_blocked':
                # Pair was blocked (duplicate or other issue)
                pair_data = result.get('data', {})
                error_msg = result.get('error', 'Unknown error')
                pair_name = pair_data.get('name', 'Unnamed')
                
                # Show error message (manager already logs this, but we can add context)
                if hasattr(self, 'info_output'):
                    self.info_output.append(f"❌ Blocked: {pair_name} - {error_msg}")
                    
            elif result_type == 'pair_add_failed':
                # Pair addition failed due to alignment or other error
                pair_data = result.get('data', {})
                error_msg = result.get('error', 'Unknown error')
                pair_name = pair_data.get('name', 'Unnamed')
                
                # Show error message
                if hasattr(self, 'info_output'):
                    self.info_output.append(f"❌ Failed to add {pair_name}: {error_msg}")
                    
            elif result_type == 'pair_deleted':
                # Pair was deleted
                if hasattr(self, 'info_output'):
                    self.info_output.append("🗑️ Comparison pair deleted")
                    
            elif result_type == 'plot_generated':
                # Plot was generated
                if hasattr(self, 'info_output'):
                    self.info_output.append("📊 Plot generated successfully")
                    
            elif result_type == 'analysis_refreshed':
                # Analysis refresh completed successfully
                n_pairs = result.get('n_pairs', 0)
                cache_stats = result.get('cache_stats', {})
                
                if hasattr(self, 'info_output'):
                    cache_info = f"Cache: {cache_stats.get('hits', 0)} hits, {cache_stats.get('misses', 0)} misses"
                    self.info_output.append(f"✅ Plot refreshed successfully - {n_pairs} pairs analyzed. {cache_info}")
                    
            elif result_type == 'analysis_refresh_failed':
                # Analysis refresh failed
                error_msg = result.get('error', 'Unknown error')
                
                if hasattr(self, 'info_output'):
                    self.info_output.append(f"❌ Plot refresh failed: {error_msg}")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error handling manager comparison result: {e}")
    
    def _get_alignment_params(self):
        """Get current alignment parameters"""
        try:
            mode = self.alignment_mode_combo.currentText()
            
            if mode == "Index-Based":
                return {
                    'mode': 'Index-Based',  # Match the format expected by DataAligner
                    'start_index': self.start_index_spin.value(),
                    'end_index': self.end_index_spin.value(),
                    'offset': self.index_offset_spin.value()
                }
            else:
                return {
                    'mode': 'Time-Based',  # Match the format expected by DataAligner
                    'start_time': self.start_time_spin.value(),
                    'end_time': self.end_time_spin.value(),
                    'offset': self.time_offset_spin.value(),
                    'interpolation': self.interpolation_combo.currentText(),
                    'resolution': self.resolution_spin.value()
                }
                
        except Exception as e:
            print(f"[ComparisonWizard] Error getting alignment params: {e}")
            return {}
        
    def _add_pair_to_table(self, pair_name):
        """Add a pair to the channels table"""
        try:
            # This method is called when a pair is successfully added
            # If the pair was blocked as a duplicate, this method won't be called
            print(f"[ComparisonWizard] Refreshing channels table after adding pair '{pair_name}'")
            self._refresh_channels_table()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing channels table: {e}")
    
    def _refresh_channels_table(self):
        """Refresh the entire channels table with current pair data"""
        try:
            print(f"[ComparisonWizard] Refreshing channels table...")
            
            # Clear the table
            self.channels_table.setRowCount(0)
            
            # Get all pairs from the manager
            if not hasattr(self, 'manager') or not self.manager or not hasattr(self.manager, 'pair_manager'):
                print(f"[ComparisonWizard] No manager or pair_manager available for table refresh")
                return
            
            all_pairs = self.manager.pair_manager.get_all_pairs()
            print(f"[ComparisonWizard] Found {len(all_pairs)} pairs to display")
            
            # Add each pair to the table
            for pair in all_pairs:
                self._add_single_pair_to_table(pair)
                
            print(f"[ComparisonWizard] Channels table refreshed with {len(all_pairs)} pairs")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing channels table: {e}")
    
    def _add_single_pair_to_table(self, pair):
        """Add a single pair object to the channels table"""
        try:
            row = self.channels_table.rowCount()
            self.channels_table.insertRow(row)
            
            # Debug: Show pair information
            print(f"[ComparisonWizard] Adding pair '{pair.name}' to table (row {row})")
            print(f"  - color: {getattr(pair, 'color', 'NOT SET')}")
            print(f"  - marker_type: {getattr(pair, 'marker_type', 'NOT SET')}")
            print(f"  - marker_color: {getattr(pair, 'marker_color', 'NOT SET')}")
            print(f"  - alpha: {getattr(pair, 'alpha', 'NOT SET')}")
            
            # Show checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(pair.show if hasattr(pair, 'show') else True)
            checkbox.stateChanged.connect(lambda state, pair_id=pair.pair_id: self._on_pair_visibility_changed(pair_id, state))
            self.channels_table.setCellWidget(row, 0, checkbox)
            
            # Style preview - create based on actual pair styling
            style_widget = self._create_style_preview_from_pair_object(pair)
            self.channels_table.setCellWidget(row, 1, style_widget)
            
            # Name and shape
            self.channels_table.setItem(row, 2, QTableWidgetItem(pair.name))
            
            # Get data dimensions from pair object
            shape_text = self._get_pair_data_shape_from_object(pair)
            self.channels_table.setItem(row, 3, QTableWidgetItem(shape_text))
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            info_btn = QPushButton("ℹ️")
            info_btn.setMaximumSize(25, 25)
            actions_layout.addWidget(info_btn)
            
            style_btn = QPushButton("🎨")
            style_btn.setMaximumSize(25, 25)
            actions_layout.addWidget(style_btn)
            
            delete_btn = QPushButton("🗑️")
            delete_btn.setMaximumSize(25, 25)
            delete_btn.clicked.connect(lambda: self._delete_pair(row))
            actions_layout.addWidget(delete_btn)
            
            self.channels_table.setCellWidget(row, 4, actions_widget)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error adding single pair to table: {e}")
    
    def _create_style_preview_from_pair_object(self, pair):
        """Create a style preview widget based on the pair's actual styling"""
        try:
            from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
            from PySide6.QtCore import Qt, QPoint
            
            # Get styling properties directly from pair object
            color = getattr(pair, 'color', '#1f77b4')
            marker_type = getattr(pair, 'marker_type', 'o')
            alpha = getattr(pair, 'alpha', 0.7)
            print(f"[ComparisonWizard] Creating style preview for '{pair.name}': color={color}, marker={marker_type}, alpha={alpha}")
            
            # Create the preview widget
            widget = QLabel()
            pixmap = QPixmap(20, 20)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Set up color with alpha
            qcolor = QColor(color)
            qcolor.setAlphaF(alpha)
            
            # Draw the marker
            pen = QPen(qcolor, 2)
            brush = QBrush(qcolor)
            painter.setPen(pen)
            painter.setBrush(brush)
            
            center_x, center_y = 10, 10
            size = 6
            
            # Draw different marker shapes based on marker_type
            if marker_type == 'o':  # Circle
                painter.drawEllipse(center_x - size, center_y - size, size * 2, size * 2)
            elif marker_type == 's':  # Square
                painter.drawRect(center_x - size, center_y - size, size * 2, size * 2)
            elif marker_type == '^':  # Triangle up
                points = [QPoint(center_x, center_y - size), 
                         QPoint(center_x - size, center_y + size),
                         QPoint(center_x + size, center_y + size)]
                painter.drawPolygon(points)
            elif marker_type == 'D':  # Diamond
                points = [QPoint(center_x, center_y - size),
                         QPoint(center_x + size, center_y),
                         QPoint(center_x, center_y + size),
                         QPoint(center_x - size, center_y)]
                painter.drawPolygon(points)
            elif marker_type == 'v':  # Triangle down
                points = [QPoint(center_x, center_y + size),
                         QPoint(center_x - size, center_y - size),
                         QPoint(center_x + size, center_y - size)]
                painter.drawPolygon(points)
            elif marker_type == '<':  # Triangle left
                points = [QPoint(center_x - size, center_y),
                         QPoint(center_x + size, center_y - size),
                         QPoint(center_x + size, center_y + size)]
                painter.drawPolygon(points)
            elif marker_type == '>':  # Triangle right
                points = [QPoint(center_x + size, center_y),
                         QPoint(center_x - size, center_y - size),
                         QPoint(center_x - size, center_y + size)]
                painter.drawPolygon(points)
            elif marker_type == 'p':  # Pentagon
                import math
                points = []
                for i in range(5):
                    angle = i * 2 * math.pi / 5 - math.pi / 2
                    x = center_x + size * math.cos(angle)
                    y = center_y + size * math.sin(angle)
                    points.append(QPoint(int(x), int(y)))
                painter.drawPolygon(points)
            elif marker_type == '*':  # Star
                # Draw a simple star (simplified as a cross)
                painter.drawLine(center_x - size, center_y, center_x + size, center_y)
                painter.drawLine(center_x, center_y - size, center_x, center_y + size)
            elif marker_type == 'h':  # Hexagon
                import math
                points = []
                for i in range(6):
                    angle = i * 2 * math.pi / 6
                    x = center_x + size * math.cos(angle)
                    y = center_y + size * math.sin(angle)
                    points.append(QPoint(int(x), int(y)))
                painter.drawPolygon(points)
            elif marker_type == '+':  # Plus
                painter.drawLine(center_x - size, center_y, center_x + size, center_y)
                painter.drawLine(center_x, center_y - size, center_x, center_y + size)
            elif marker_type == 'x':  # Cross
                painter.drawLine(center_x - size, center_y - size, center_x + size, center_y + size)
                painter.drawLine(center_x - size, center_y + size, center_x + size, center_y - size)
            else:  # Default to circle
                painter.drawEllipse(center_x - size, center_y - size, size * 2, size * 2)
            
            painter.end()
            widget.setPixmap(pixmap)
            widget.setAlignment(Qt.AlignCenter)
            
            return widget
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating style preview for '{pair.name}': {e}")
            # Fallback to simple colored dot
            fallback_widget = QLabel("●")
            fallback_widget.setStyleSheet("color: #1f77b4; font-size: 16px;")
            fallback_widget.setAlignment(Qt.AlignCenter)
            return fallback_widget
    
    def _delete_pair(self, row):
        """Delete a pair from the channels table"""
        try:
            self.channels_table.removeRow(row)
            self.pair_deleted.emit()
            
            if hasattr(self, 'info_output'):
                self.info_output.append("🗑️ Pair deleted")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error deleting pair: {e}")
    
    def _on_pair_visibility_changed(self, pair_id: str, state: int):
        """Handle pair visibility checkbox state change"""
        try:
            print(f"[ComparisonWizard] Pair visibility changed: {pair_id}, state: {state}")
            
            # Convert Qt checkbox state to boolean
            is_visible = state == Qt.CheckState.Checked.value
            
            # Update pair visibility in PairManager via manager
            if self.manager and hasattr(self.manager, 'pair_manager'):
                pair_manager = self.manager.pair_manager
                if hasattr(pair_manager, 'set_pair_visibility'):
                    pair_manager.set_pair_visibility(pair_id, is_visible)
                    print(f"[ComparisonWizard] Updated pair {pair_id} visibility to {is_visible}")
                else:
                    print(f"[ComparisonWizard] PairManager missing set_pair_visibility method")
            
            # Trigger hybrid visibility update (fast visual + lightweight stats)
            if self.manager and hasattr(self.manager, '_trigger_hybrid_visibility_update'):
                self.manager._trigger_hybrid_visibility_update(pair_id, is_visible)
                print(f"[ComparisonWizard] Triggered hybrid visibility update for pair {pair_id}")
            else:
                # Fallback to full analysis if hybrid method not available
                if self.manager and hasattr(self.manager, '_trigger_analysis_update'):
                    self.manager._trigger_analysis_update()
                    print(f"[ComparisonWizard] Triggered full analysis update (hybrid not available)")
                else:
                    print(f"[ComparisonWizard] Manager missing update methods")
            
            # Update info output
            if hasattr(self, 'info_output'):
                visibility_text = "shown" if is_visible else "hidden"
                self.info_output.append(f"👁️ Pair {pair_id} {visibility_text}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling pair visibility change: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_channel_selection_changed(self):
        """Handle channel selection changes"""
        try:
            # Update pair name
            ref_channel = self.ref_channel_combo.currentData()
            test_channel = self.test_channel_combo.currentData()
            
            if ref_channel and test_channel:
                pair_name = f"{getattr(ref_channel, 'legend_label', 'Unknown')}_vs_{getattr(test_channel, 'legend_label', 'Unknown')}"
                self.pair_name_input.setText(pair_name)
                
                # Call the manager's suggest_alignment_parameters if available
                if self.manager and hasattr(self.manager, 'suggest_alignment_parameters'):
                    mode = self.alignment_mode_combo.currentText()
                    params = self.manager.suggest_alignment_parameters(mode=mode)
                    if params:
                        self._apply_alignment_params(params)
                        print(f"[ComparisonWizard] Applied suggested alignment parameters for {getattr(ref_channel, 'legend_label', 'Unknown')} vs {getattr(test_channel, 'legend_label', 'Unknown')}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling channel selection change: {e}")
    
    def _on_ref_file_changed(self):
        """Handle reference file selection change"""
        try:
            # Update reference channel dropdown based on selected file
            # This would typically populate channels from the selected file
            pass
        except Exception as e:
            print(f"[ComparisonWizard] Error handling ref file change: {e}")
    
    def _on_test_file_changed(self):
        """Handle test file selection change"""
        try:
            # Update test channel dropdown based on selected file
            # This would typically populate channels from the selected file
            pass
        except Exception as e:
            print(f"[ComparisonWizard] Error handling test file change: {e}")
    
    def _apply_alignment_params(self, params):
        """Apply alignment parameters to the UI"""
        try:
            if not params:
                print("[DEBUG] _apply_alignment_params: No params provided")
                return

            mode = params.get('mode', 'index')
            print(f"[DEBUG] _apply_alignment_params: mode={mode}, params={params}")
            
            # Apply parameter values based on the current mode (don't change the mode)
            if mode == 'index' or mode == 'Index-Based':
                start_index = params.get('start_index', 0)
                end_index = params.get('end_index', 0)
                offset = params.get('offset', 0)
                print(f"[DEBUG] _apply_alignment_params: Setting index params - start={start_index}, end={end_index}, offset={offset}")
                self.start_index_spin.setValue(start_index)
                self.end_index_spin.setValue(end_index)
                self.index_offset_spin.setValue(offset)
            else:
                self.start_time_spin.setValue(params.get('start_time', 0.0))
                self.end_time_spin.setValue(params.get('end_time', 0.0))
                self.time_offset_spin.setValue(params.get('offset', 0.0))
                interpolation = params.get('interpolation', 'nearest')
                interp_index = self.interpolation_combo.findText(interpolation)
                if interp_index >= 0:
                    self.interpolation_combo.setCurrentIndex(interp_index)
                self.resolution_spin.setValue(params.get('resolution', 0.1))
        except Exception as e:
            print(f"[ComparisonWizard] Error applying alignment params: {e}")
    
    def _generate_pair_name(self, ref_channel=None, test_channel=None):
        """Generate a descriptive pair name from channel names"""
        try:
            if ref_channel is None or test_channel is None:
                return "New_Comparison_Pair"
            
            # Get channel names, preferring legend_label over channel_id
            ref_name = getattr(ref_channel, 'legend_label', None) or getattr(ref_channel, 'channel_id', 'Unknown')
            test_name = getattr(test_channel, 'legend_label', None) or getattr(test_channel, 'channel_id', 'Unknown')
            
            # Clean up names (remove special characters, replace spaces with underscores)
            import re
            def clean_name(name):
                if name is None:
                    return "Unknown"
                # Replace spaces and special characters with underscores
                cleaned = re.sub(r'[^\w\s-]', '_', str(name))
                cleaned = re.sub(r'[\s-]+', '_', cleaned)
                return cleaned.strip('_')
            
            ref_clean = clean_name(ref_name)
            test_clean = clean_name(test_name)
            
            # Generate pair name
            pair_name = f"{ref_clean}_vs_{test_clean}"
            
            # Limit length to avoid UI issues
            if len(pair_name) > 50:
                pair_name = f"{ref_clean[:20]}_vs_{test_clean[:20]}"
            
            return pair_name
            
        except Exception as e:
            print(f"Error generating pair name: {e}")
            return "New_Comparison_Pair"
        
    def get_current_parameters(self):
        """Get current parameter values from the parameter table"""
        try:
            # Use the dynamic parameter capture system which properly handles all widget types
            params = self.param_capture.capture_all_parameters()
            return params
        except Exception as e:
            print(f"[ComparisonWizard] Error getting current parameters: {e}")
            return {}
    
    def get_current_method_name(self):
        """Get the currently selected method name"""
        try:
            current_item = self.method_list.currentItem()
            if current_item:
                ui_name = current_item.text()
                
                # Get all available methods from registry
                if self.manager and hasattr(self.manager, 'get_comparison_methods'):
                    available_methods = self.manager.get_comparison_methods()
                    
                    # Find best match using fuzzy matching
                    best_match = self._find_best_method_match(ui_name, available_methods)
                    print(f"[DEBUG] Method name mapping: '{ui_name}' -> '{best_match}'")
                    return best_match
                
                # Fallback if manager not available
                return ui_name.lower()
            return None
        except Exception as e:
            print(f"[ComparisonWizard] Error getting current method name: {e}")
            return None
    
    def _find_best_method_match(self, ui_name, available_methods):
        """Find the best matching method name using fuzzy matching"""
        ui_name_lower = ui_name.lower()
        
        print(f"[DEBUG] Finding match for '{ui_name}' in {available_methods}")
        
        # Strategy 1: Try exact match first
        for method in available_methods:
            if method.lower() == ui_name_lower:
                print(f"[DEBUG] Exact match found: {method}")
                return method
        
        # Strategy 2: Try exact match with space-to-underscore conversion
        ui_name_normalized = ui_name_lower.replace(' ', '_').replace('-', '_')
        for method in available_methods:
            if method.lower() == ui_name_normalized:
                print(f"[DEBUG] Normalized match found: {method}")
                return method
        
        # Strategy 3: Try partial matches (UI name contains method name)
        for method in available_methods:
            method_words = method.lower().replace('_', ' ').split()
            ui_words = ui_name_lower.replace('-', ' ').split()
            
            # Check if all method words are in UI name
            if all(word in ui_words for word in method_words):
                print(f"[DEBUG] Partial match found (method words in UI): {method}")
                return method
        
        # Strategy 4: Try reverse partial matches (method name contains UI name)
        for method in available_methods:
            method_clean = method.lower().replace('_', ' ')
            ui_clean = ui_name_lower.replace('-', ' ')
            
            if ui_clean in method_clean:
                print(f"[DEBUG] Partial match found (UI in method): {method}")
                return method
        
        # Strategy 5: Try keyword matching for common terms
        keyword_mappings = {
            'error': ['error_distribution_histogram'],
            'distribution': ['error_distribution_histogram'],
            'bland': ['bland_altman'],
            'altman': ['bland_altman'],
            'agreement': ['agreement_breakdown', 'bland_altman'],
            'breakdown': ['agreement_breakdown'],
            'correlation': ['correlation', 'time_lag_cross_correlation'],
            'residual': ['residual'],
            'stacked': ['stacked_error_time_band'],
            'time': ['stacked_error_time_band', 'time_lag_cross_correlation'],
            'band': ['stacked_error_time_band'],
            'lag': ['time_lag_cross_correlation'],
            'cross': ['time_lag_cross_correlation']
        }
        
        ui_words = ui_name_lower.replace('-', ' ').replace('_', ' ').split()
        for word in ui_words:
            if word in keyword_mappings:
                candidates = keyword_mappings[word]
                # Find the first candidate that exists in available methods
                for candidate in candidates:
                    if candidate in available_methods:
                        print(f"[DEBUG] Keyword match found: '{word}' -> {candidate}")
                        return candidate
        
        # Strategy 6: Fallback to normalized UI name
        fallback = ui_name_normalized
        print(f"[DEBUG] No match found, using fallback: {fallback}")
        return fallback
    
    def _on_plot_script_changed(self):
        """Handle plot script text changes"""
        try:
            current_script = self.plot_script_text.toPlainText()
            is_modified = self.script_tracker.is_plot_script_modified(current_script)
            
            if is_modified:
                self.plot_script_status_label.setText("✏️ Modified")
                self.plot_script_status_label.setStyleSheet("color: orange; font-weight: bold;")
                self.script_tracker.mark_plot_script_modified()
            else:
                self.plot_script_status_label.setText("Default")
                self.plot_script_status_label.setStyleSheet("color: gray; font-style: italic;")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling plot script change: {e}")

    def _on_stats_script_changed(self):
        """Handle stats script text changes"""
        try:
            current_script = self.stats_script_text.toPlainText()
            is_modified = self.script_tracker.is_stats_script_modified(current_script)
            
            if is_modified:
                self.stats_script_status_label.setText("✏️ Modified")
                self.stats_script_status_label.setStyleSheet("color: orange; font-weight: bold;")
                self.script_tracker.mark_stats_script_modified()
            else:
                self.stats_script_status_label.setText("Default")
                self.stats_script_status_label.setStyleSheet("color: gray; font-style: italic;")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling stats script change: {e}")

    def _reset_plot_script(self):
        """Reset plot script to default"""
        try:
            if self.script_tracker.original_plot_script:
                self.plot_script_text.setPlainText(self.script_tracker.original_plot_script)
                self.plot_script_status_label.setText("Default")
                self.plot_script_status_label.setStyleSheet("color: gray; font-style: italic;")
                print("[ComparisonWizard] Plot script reset to default")
        except Exception as e:
            print(f"[ComparisonWizard] Error resetting plot script: {e}")

    def _reset_stats_script(self):
        """Reset stats script to default"""
        try:
            if self.script_tracker.original_stats_script:
                self.stats_script_text.setPlainText(self.script_tracker.original_stats_script)
                self.stats_script_status_label.setText("Default")
                self.stats_script_status_label.setStyleSheet("color: gray; font-style: italic;")
                print("[ComparisonWizard] Stats script reset to default")
        except Exception as e:
            print(f"[ComparisonWizard] Error resetting stats script: {e}")

    def _get_pair_data_shape_from_object(self, pair):
        """Get the data shape of a pair from its aligned data"""
        try:
            # Check if pair has aligned data
            if hasattr(pair, 'aligned_ref_data') and hasattr(pair, 'aligned_test_data'):
                ref_data = pair.aligned_ref_data
                test_data = pair.aligned_test_data
                
                if ref_data is not None and test_data is not None:
                    ref_shape = ref_data.shape if hasattr(ref_data, 'shape') else f"({len(ref_data)},)"
                    test_shape = test_data.shape if hasattr(test_data, 'shape') else f"({len(test_data)},)"
                    
                    # Format the shape information
                    if len(ref_shape) == 1 and len(test_shape) == 1:
                        # Both are 1D arrays, show as "N samples"
                        n_samples = len(ref_data)
                        shape_text = f"({n_samples},)"
                    else:
                        # Show both shapes
                        shape_text = f"ref: {ref_shape}, test: {test_shape}"
                    
                    print(f"[ComparisonWizard] Pair '{pair.name}' data shape: {shape_text}")
                    return shape_text
                else:
                    print(f"[ComparisonWizard] Pair '{pair.name}' has no aligned data")
                    return "No data"
            else:
                print(f"[ComparisonWizard] Pair '{pair.name}' missing aligned data attributes")
                return "No data"
        except Exception as e:
            print(f"[ComparisonWizard] Error getting pair data shape for '{pair.name}': {e}")
            return "Error"
    
    def _delete_pair(self, row):
        """Delete a pair from the channels table"""
        try:
            self.channels_table.removeRow(row)
            self.pair_deleted.emit()
            
            if hasattr(self, 'info_output'):
                self.info_output.append("🗑️ Pair deleted")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error deleting pair: {e}")

    def get_performance_options(self):
        """Get current performance options from UI"""
        try:
            return {
                'max_points_enabled': self.max_points_checkbox.isChecked(),
                'max_points': self.max_points_input.value(),
                'density_mode': self.density_combo.currentText().lower(),  # "scatter", "hexbin", "kde"
                'bins': self.bins_spinbox.value()
            }
        except Exception as e:
            print(f"[ComparisonWizard] Error getting performance options: {e}")
            return {
                'max_points_enabled': False,
                'max_points': 5000,
                'density_mode': 'scatter',
                'bins': 50
            }


def main():
    """Run the streamlined comparison wizard demo"""
    app = QApplication([])
    
    # Create and show the wizard
    wizard = ComparisonWizardWindow()
    wizard.show()
    
    app.exec()


if __name__ == "__main__":
    main()