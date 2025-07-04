from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor, QIntValidator, QColor, QFont
import pandas as pd
from copy import deepcopy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import time

class ComparisonWizardWindow(QMainWindow):
    """
    Data Comparison Wizard - Following process/mixer wizard patterns
    Single panel interface with modern UI/UX consistent with other wizards
    """
    
    pair_added = Signal(dict)
    pair_deleted = Signal()
    plot_generated = Signal(dict)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        
        # Initialize data
        self.active_pairs = []
        
        # UI update timer for performance
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)
        self.update_delay = 300  # ms
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        self._populate_file_combos()
        
        # Initialize with available data
        self._validate_initialization()
        
        # Auto-populate alignment parameters with initial selection
        self._auto_populate_alignment_parameters()
        
    def _validate_initialization(self):
        """Validate that the wizard has necessary data to function"""
        if not self.file_manager:
            print("[ComparisonWizard] Warning: No file manager available")
            return
            
        files = self.file_manager.get_all_files()
        if len(files) < 1:
            print("[ComparisonWizard] Warning: No files available for comparison")
            return
            
        print(f"[ComparisonWizard] Initialized with {len(files)} files available")
        
    def _show_error(self, message):
        """Show error message to user"""
        QMessageBox.critical(self, "Comparison Wizard Error", message)
        
    def _log_state_change(self, message: str):
        """Log state changes for debugging"""
        print(f"[ComparisonWizard] {message}")

    def _init_ui(self):
        """Initialize the user interface following mixer wizard patterns"""
        self.setWindowTitle("Data Comparison Wizard")
        self.setMinimumSize(1200, 800)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Build left and right panels
        self._build_left_panel(main_splitter)
        self._build_right_panel(main_splitter)
        
        # Set splitter proportions to match mixer wizard
        main_splitter.setSizes([400, 800])
        
    def _build_left_panel(self, main_splitter):
        """Build the left panel with two-column layout like mixer wizard"""
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Create horizontal splitter for two columns (like mixer wizard)
        left_splitter = QSplitter(Qt.Horizontal)
        left_layout.addWidget(left_splitter)
        
        # Create left and right column widgets
        left_col_widget = QWidget()
        left_col_layout = QVBoxLayout(left_col_widget)
        left_col_layout.setContentsMargins(5, 5, 5, 5)
        left_col_layout.setSpacing(10)
        
        right_col_widget = QWidget()
        right_col_layout = QVBoxLayout(right_col_widget)
        right_col_layout.setContentsMargins(5, 5, 5, 5)
        right_col_layout.setSpacing(10)
        
        # Left column: Comparison Methods and Method-specific Controls
        self._create_comparison_method_group(left_col_layout)
        self._create_method_controls_group(left_col_layout)
        
        # Add stretch to left column
        left_col_layout.addStretch()
        
        # Right column: Channel Selection, Alignment, and Actions
        self._create_channel_selection_group(right_col_layout)
        self._create_alignment_group(right_col_layout)
        self._create_pairs_management_group(right_col_layout)
        
        # Add stretch to right column
        right_col_layout.addStretch()
        
        # Add columns to splitter
        left_splitter.addWidget(left_col_widget)
        left_splitter.addWidget(right_col_widget)
        
        main_splitter.addWidget(self.left_panel)
        
    def _create_channel_selection_group(self, layout):
        """Create channel selection group box"""
        group = QGroupBox("Channel Selection")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QFormLayout(group)
        
        # Reference file and channel
        self.ref_file_combo = QComboBox()
        self.ref_file_combo.setMinimumWidth(200)
        group_layout.addRow("Reference File:", self.ref_file_combo)
        
        self.ref_channel_combo = QComboBox()
        group_layout.addRow("Reference Channel:", self.ref_channel_combo)
        
        # Test file and channel
        self.test_file_combo = QComboBox()
        group_layout.addRow("Test File:", self.test_file_combo)
        
        self.test_channel_combo = QComboBox()
        group_layout.addRow("Test Channel:", self.test_channel_combo)
        
        layout.addWidget(group)
        
    def _create_comparison_method_group(self, layout):
        """Create comparison method selection group"""
        group = QGroupBox("Comparison Method")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Method list (populated from comparison registry)
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(120)
        self._populate_comparison_methods()
        group_layout.addWidget(self.method_list)
        
        layout.addWidget(group)
        
    def _populate_comparison_methods(self):
        """Populate comparison methods from the registry"""
        try:
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                methods = self.comparison_manager.get_available_comparison_methods()
            else:
                # Fallback methods
                methods = ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis", "Statistical Tests"]
            
            self.method_list.clear()
            self.method_list.addItems(methods)
            if methods:
                self.method_list.setCurrentRow(0)  # Select first method by default
                
        except Exception as e:
            print(f"[ComparisonWizard] Error populating methods: {e}")
            # Fallback
            methods = ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis", "Statistical Tests"]
            self.method_list.clear()
            self.method_list.addItems(methods)
            if methods:
                self.method_list.setCurrentRow(0)
        
    def _create_method_controls_group(self, layout):
        """Create method-specific controls group (replaces parameter table)"""
        group = QGroupBox("Method Options")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.method_controls_layout = QVBoxLayout(group)
        
        # Create stacked widget to hold different method controls
        self.method_controls_stack = QStackedWidget()
        self.method_controls_layout.addWidget(self.method_controls_stack)
        
        # Create controls for each method
        self._create_dynamic_method_controls()
        
        # Add Generate Plot button
        self.generate_plot_button = QPushButton("📊 Generate Plot")
        self.generate_plot_button.clicked.connect(self._on_generate_plot)
        self.method_controls_layout.addWidget(self.generate_plot_button)
        
        layout.addWidget(group)
        
    def _create_dynamic_method_controls(self):
        """Create method controls dynamically from comparison registry"""
        try:
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                try:
                    methods = self.comparison_manager.get_available_comparison_methods()
                    print(f"[ComparisonWizard] Creating dynamic controls for methods: {methods}")
                    
                    for method_name in methods:
                        try:
                            method_info = self.comparison_manager.get_method_info(method_name)
                            widget = self._create_controls_for_method(method_name, method_info)
                            self.method_controls_stack.addWidget(widget)
                            print(f"[ComparisonWizard] Created controls for {method_name}")
                        except Exception as method_error:
                            print(f"[ComparisonWizard] Error creating controls for {method_name}: {method_error}")
                            # Create a simple placeholder widget for this method
                            placeholder = QWidget()
                            placeholder_layout = QVBoxLayout(placeholder)
                            placeholder_layout.addWidget(QLabel(f"Error loading controls for {method_name}"))
                            self.method_controls_stack.addWidget(placeholder)
                    
                    if len(methods) == 0:
                        print("[ComparisonWizard] No methods available, using static controls")
                        self._create_static_method_controls()
                        
                except Exception as manager_error:
                    print(f"[ComparisonWizard] Error accessing comparison manager: {manager_error}")
                    self._create_static_method_controls()
            else:
                print("[ComparisonWizard] No comparison manager available, using static controls")
                # Fallback: create static controls
                self._create_static_method_controls()
                
        except Exception as e:
            print(f"[ComparisonWizard] Error creating dynamic controls: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to static controls
            self._create_static_method_controls()
            
    def _create_static_method_controls(self):
        """Create static method controls as fallback"""
        self._create_correlation_controls()
        self._create_bland_altman_controls()
        self._create_residual_controls()
        self._create_statistical_controls()
        
    def _create_controls_for_method(self, method_name, method_info):
        """Create controls for a specific comparison method"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        parameters = method_info.get('parameters', {})
        
        # Store parameter controls for later retrieval
        if not hasattr(self, '_method_controls'):
            self._method_controls = {}
        self._method_controls[method_name] = {}
        
        for param_name, param_config in parameters.items():
            control = self._create_parameter_control(param_name, param_config)
            if control:
                # Use the shorter description for the label
                label_text = param_config.get('description', param_name)
                layout.addRow(label_text + ":", control)
                self._method_controls[method_name][param_name] = control
        
        # If no parameters, show a message
        if not parameters:
            label = QLabel("No configurable parameters for this method")
            label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(label)
        
        return widget
    
    def _create_parameter_control(self, param_name, param_config):
        """Create appropriate control widget for a parameter"""
        param_type = param_config.get('type', str)
        default_value = param_config.get('default')
        choices = param_config.get('choices')
        tooltip = param_config.get('tooltip', '')
        
        control = None
        
        if choices:
            # Dropdown for choices
            control = QComboBox()
            control.addItems([str(choice) for choice in choices])
            if default_value in choices:
                control.setCurrentText(str(default_value))
                
        elif param_type == bool:
            # Checkbox for boolean
            control = QCheckBox()
            control.setChecked(bool(default_value))
            
        elif param_type == int:
            # Spinbox for integer
            control = QSpinBox()
            control.setRange(param_config.get('min', -999999), param_config.get('max', 999999))
            control.setValue(int(default_value) if default_value is not None else 0)
            
        elif param_type == float:
            # Double spinbox for float
            control = QDoubleSpinBox()
            control.setRange(param_config.get('min', -999999.0), param_config.get('max', 999999.0))
            control.setDecimals(4)
            control.setValue(float(default_value) if default_value is not None else 0.0)
            
        else:
            # Line edit for string or unknown types
            control = QLineEdit()
            if default_value is not None:
                control.setText(str(default_value))
        
        # Set tooltip if available
        if control and tooltip:
            control.setToolTip(tooltip)
            
        return control
        
    def _create_correlation_controls(self):
        """Create controls for correlation analysis"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Correlation type
        self.corr_type_combo = QComboBox()
        self.corr_type_combo.addItems(["pearson", "spearman", "kendall", "all"])
        layout.addRow("Correlation Type:", self.corr_type_combo)
        
        # Confidence level
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setValue(0.95)
        self.confidence_spin.setDecimals(2)
        layout.addRow("Confidence Level:", self.confidence_spin)
        
        # Bootstrap samples
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(100, 10000)
        self.bootstrap_spin.setValue(1000)
        layout.addRow("Bootstrap Samples:", self.bootstrap_spin)
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_bland_altman_controls(self):
        """Create controls for Bland-Altman analysis"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Agreement limits
        self.agreement_spin = QDoubleSpinBox()
        self.agreement_spin.setRange(1.0, 3.0)
        self.agreement_spin.setValue(1.96)
        self.agreement_spin.setDecimals(2)
        layout.addRow("Agreement Limits:", self.agreement_spin)
        
        # Show confidence intervals
        self.show_ci_checkbox = QCheckBox()
        self.show_ci_checkbox.setChecked(True)
        layout.addRow("Show CI:", self.show_ci_checkbox)
        
        # Proportional bias
        self.prop_bias_checkbox = QCheckBox()
        self.prop_bias_checkbox.setChecked(False)
        layout.addRow("Proportional Bias:", self.prop_bias_checkbox)
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_residual_controls(self):
        """Create controls for residual analysis"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Normality test
        self.normality_combo = QComboBox()
        self.normality_combo.addItems(["shapiro", "kstest", "jarque_bera"])
        layout.addRow("Normality Test:", self.normality_combo)
        
        # Outlier detection
        self.outlier_combo = QComboBox()
        self.outlier_combo.addItems(["iqr", "zscore", "modified_zscore"])
        layout.addRow("Outlier Detection:", self.outlier_combo)
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_statistical_controls(self):
        """Create controls for statistical tests"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Alpha level
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.1)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setDecimals(3)
        layout.addRow("Alpha Level:", self.alpha_spin)
        
        # Test suite
        self.test_suite_combo = QComboBox()
        self.test_suite_combo.addItems(["basic", "comprehensive", "nonparametric"])
        layout.addRow("Test Suite:", self.test_suite_combo)
        
        # Equal variance
        self.equal_var_combo = QComboBox()
        self.equal_var_combo.addItems(["assume_equal", "assume_unequal", "test"])
        layout.addRow("Equal Variance:", self.equal_var_combo)
        
        # Normality assumption
        self.normality_assume_combo = QComboBox()
        self.normality_assume_combo.addItems(["assume_normal", "assume_nonnormal", "test"])
        layout.addRow("Normality:", self.normality_assume_combo)
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_alignment_group(self, layout):
        """Create alignment controls group (like mixer wizard alignment section)"""
        group = QGroupBox("Data Alignment")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
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
        
        # Start index
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(0, 999999)
        self.start_index_spin.setValue(0)
        index_layout.addRow("Start Index:", self.start_index_spin)
        
        # End index
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(0, 999999)
        self.end_index_spin.setValue(500)
        index_layout.addRow("End Index:", self.end_index_spin)
        
        # Index offset
        self.index_offset_spin = QSpinBox()
        self.index_offset_spin.setRange(-999999, 999999)
        self.index_offset_spin.setValue(0)
        self.index_offset_spin.setToolTip("Positive: shift test channel forward, Negative: shift reference channel forward")
        index_layout.addRow("Offset:", self.index_offset_spin)
        
        group_layout.addWidget(self.index_group)
        
        # Time-based options
        self.time_group = QGroupBox("Time Options")
        time_layout = QFormLayout(self.time_group)
        
        # Start time
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(-999999.0, 999999.0)
        self.start_time_spin.setValue(0.0)
        self.start_time_spin.setDecimals(3)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        # End time
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
        self.time_offset_spin.setToolTip("Time offset to apply to test channel")
        time_layout.addRow("Time Offset:", self.time_offset_spin)
        
        # Interpolation method
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(["linear", "nearest", "cubic"])
        time_layout.addRow("Interpolation:", self.interpolation_combo)
        
        # Time resolution
        self.round_to_spin = QDoubleSpinBox()
        self.round_to_spin.setRange(0.0001, 10.0)
        self.round_to_spin.setValue(1.0)
        self.round_to_spin.setDecimals(4)
        self.round_to_spin.setToolTip("Time resolution (smaller = more points)")
        time_layout.addRow("Time Resolution:", self.round_to_spin)
        
        group_layout.addWidget(self.time_group)
        
        # Alignment status
        self.alignment_status_label = QLabel("No alignment needed")
        self.alignment_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
        group_layout.addWidget(self.alignment_status_label)
        
        # Show/hide appropriate groups
        self._on_alignment_mode_changed("Index-Based")
        
        layout.addWidget(group)
        
    def _create_pairs_management_group(self, layout):
        """Create pairs management group"""
        group = QGroupBox("Add Pair")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Pair name input with label
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Pair Name:"))
        self.pair_name_input = QLineEdit()
        self.pair_name_input.setPlaceholderText("Auto-generated from channels")
        name_layout.addWidget(self.pair_name_input)
        group_layout.addLayout(name_layout)
        
        # Add pair button (plot updates automatically when clicked)
        self.add_pair_button = QPushButton("Add Comparison Pair")
        group_layout.addWidget(self.add_pair_button)
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with table at top and plot at bottom (like mixer wizard)"""
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Results table at the top (like mixer wizard)
        self._build_results_table(right_layout)
        
        # Plot area at the bottom (like mixer wizard)
        self._build_plot_area(right_layout)
        
        main_splitter.addWidget(self.right_panel)
        
    def _build_results_table(self, layout):
        """Build the results table showing active comparison pairs (like process wizard step table)"""
        # Active pairs table header
        layout.addWidget(QLabel("📋 Active Comparison Pairs:"))
        
        # Active pairs table (similar to process wizard step table)
        self.active_pair_table = QTableWidget()
        self.active_pair_table.setColumnCount(4)
        self.active_pair_table.setHorizontalHeaderLabels(["Show", "Pair Name", "Style", "Status"])
        
        # Set column widths similar to process wizard
        header = self.active_pair_table.horizontalHeader()
        header.setStretchLastSection(True)
        self.active_pair_table.setColumnWidth(0, 60)  # Show column
        self.active_pair_table.setColumnWidth(1, 200)  # Pair name
        self.active_pair_table.setColumnWidth(2, 80)   # Style column
        
        self.active_pair_table.setMaximumHeight(200)  # Match process wizard table height
        self.active_pair_table.setAlternatingRowColors(True)
        self.active_pair_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.active_pair_table.setShowGrid(True)
        self.active_pair_table.setGridStyle(Qt.SolidLine)
        
        # Style similar to process wizard
        self.active_pair_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #ddd;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                padding: 4px;
                border: 1px solid #bdc3c7;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(self.active_pair_table)
        
    def _build_plot_area(self, layout):
        """Build the plot area (like mixer wizard plot area)"""
        # Create tabbed interface for different views (keeping existing functionality)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add tabs
        self._create_comparison_plot_tab()
        self._create_statistics_tab()
        self._create_results_tab()
        
    def _create_comparison_plot_tab(self):
        """Create the comparison plot tab"""
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # Matplotlib canvas (like process wizard - no plot controls)
        self.figure = plt.figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        self.tab_widget.addTab(plot_tab, "📊 Plot")
        
    def _create_statistics_tab(self):
        """Create the statistics tab"""
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        # Statistics display
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.StyledPanel)
        stats_frame_layout = QVBoxLayout(stats_frame)
        
        stats_title = QLabel("📈 Comparison Statistics")
        stats_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        stats_frame_layout.addWidget(stats_title)
        
        self.cumulative_stats_text = QTextEdit()
        self.cumulative_stats_text.setReadOnly(True)
        self.cumulative_stats_text.setMaximumHeight(300)
        self.cumulative_stats_text.setPlainText("No statistics available. Add comparison pairs and generate plots to see results.")
        stats_frame_layout.addWidget(self.cumulative_stats_text)
        
        stats_layout.addWidget(stats_frame)
        
        # Detailed results table
        results_frame = QFrame()
        results_frame.setFrameStyle(QFrame.StyledPanel)
        results_frame_layout = QVBoxLayout(results_frame)
        
        results_title = QLabel("📋 Detailed Results")
        results_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        results_frame_layout.addWidget(results_title)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["Pair", "Method", "Correlation", "P-value", "RMSE", "Bias"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        results_frame_layout.addWidget(self.results_table)
        
        stats_layout.addWidget(results_frame)
        
        self.tab_widget.addTab(stats_tab, "📊 Statistics")
        
    def _create_results_tab(self):
        """Create the results export tab"""
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        # Export options
        export_frame = QFrame()
        export_frame.setFrameStyle(QFrame.StyledPanel)
        export_frame_layout = QVBoxLayout(export_frame)
        
        export_title = QLabel("💾 Export Results")
        export_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        export_frame_layout.addWidget(export_title)
        
        # Export buttons
        export_buttons_layout = QHBoxLayout()
        
        export_plot_button = QPushButton("📊 Export Plot")
        export_data_button = QPushButton("📋 Export Data")
        export_report_button = QPushButton("📄 Export Report")
        
        export_buttons_layout.addWidget(export_plot_button)
        export_buttons_layout.addWidget(export_data_button)
        export_buttons_layout.addWidget(export_report_button)
        export_buttons_layout.addStretch()
        
        export_frame_layout.addLayout(export_buttons_layout)
        results_layout.addWidget(export_frame)
        
        # Results summary
        summary_frame = QFrame()
        summary_frame.setFrameStyle(QFrame.StyledPanel)
        summary_frame_layout = QVBoxLayout(summary_frame)
        
        summary_title = QLabel("📝 Analysis Summary")
        summary_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        summary_frame_layout.addWidget(summary_title)
        
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlainText("Analysis summary will appear here after generating comparisons.")
        summary_frame_layout.addWidget(self.summary_text)
        
        results_layout.addWidget(summary_frame)
        
        self.tab_widget.addTab(results_tab, "📋 Results")
        
    def _populate_file_combos(self):
        """Populate file combo boxes with available files"""
        if not self.file_manager:
            return
            
        files = self.file_manager.get_all_files()
        file_names = [f.filename for f in files]
        
        # Clear and populate reference file combo
        self.ref_file_combo.clear()
        self.ref_file_combo.addItems(file_names)
        
        # Clear and populate test file combo
        self.test_file_combo.clear()
        self.test_file_combo.addItems(file_names)
        
        # Auto-select different files if available
        if len(file_names) >= 2:
            self.ref_file_combo.setCurrentIndex(0)
            self.test_file_combo.setCurrentIndex(1)
        elif len(file_names) == 1:
            self.ref_file_combo.setCurrentIndex(0)
            self.test_file_combo.setCurrentIndex(0)

    def _connect_signals(self):
        """Connect UI signals to handlers"""
        # File selection signals
        self.ref_file_combo.currentTextChanged.connect(self._on_ref_file_changed)
        self.test_file_combo.currentTextChanged.connect(self._on_test_file_changed)
        
        # Channel selection signals
        self.ref_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        self.test_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        self.ref_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
        self.test_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
        
        # Method selection signal
        self.method_list.itemClicked.connect(self._on_method_selected)
        
        # Pair management signals
        self.add_pair_button.clicked.connect(self._on_add_pair)
        
        # Alignment signals
        self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
        
        # Update timer for delayed operations
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)
        self.update_delay = 500  # 500ms delay
        
    def _on_alignment_mode_changed(self, text):
        """Handle alignment mode changes (like mixer wizard)"""
        print(f"[ComparisonWizard] Alignment mode changed to: {text}")
        
        # Show/hide appropriate alignment groups
        if text == "Index-Based":
            self.index_group.show()
            self.time_group.hide()
        else:  # Time-Based
            self.index_group.hide()
            self.time_group.show()
        
        # Auto-populate alignment parameters for the new mode
        self._auto_populate_alignment_parameters()
        
        # No automatic plot update - only when Generate Plot button is clicked
        
    def _on_method_selected(self, item):
        """Handle method selection change"""
        if not item:
            return
            
        # Update method controls stack to show controls for selected method
        method_name = item.text()
        method_index = self.method_list.row(item)
        
        if method_index < self.method_controls_stack.count():
            self.method_controls_stack.setCurrentIndex(method_index)
        
        # No automatic plot update - only when Generate Plot button is clicked
        
    def _trigger_plot_update(self):
        """Trigger a delayed plot update for performance"""
        self.update_timer.start(self.update_delay)
            
    def _delayed_update(self):
        """Perform delayed update operations"""
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            self.comparison_manager._update_cumulative_display()
            
    def _force_plot_update(self):
        """Force immediate plot update"""
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            self.comparison_manager._update_cumulative_display()
            
    def _on_generate_plot(self):
        """Generate plot when button is clicked using the selected comparison method"""
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            # Update status for all pairs
            for row in range(self.active_pair_table.rowCount()):
                status_item = self.active_pair_table.item(row, 3)
                if status_item:
                    status_item.setText("Generating...")
            
            # Get the current method and its parameters
            current_item = self.method_list.currentItem()
            if current_item:
                method_name = current_item.text()
                method_params = self._get_method_parameters_from_controls()
                
                # Update all pairs with the current method and parameters
                for pair in self.active_pairs:
                    pair['comparison_method'] = method_name
                    pair['method_parameters'] = method_params
                
                print(f"[ComparisonWizard] Generating plot with method: {method_name}")
                print(f"[ComparisonWizard] Method parameters: {method_params}")
            
            # Get proper plot configuration from the window
            plot_config = self._get_plot_config()
            
            # Generate the plot using the proper plot generation method
            self.comparison_manager._on_plot_generated(plot_config)
            
            # Update status after generation
            for row in range(self.active_pair_table.rowCount()):
                status_item = self.active_pair_table.item(row, 3)
                if status_item:
                    status_item.setText("Complete")
        
    def _get_plot_config(self):
        """Get current plot configuration"""
        config = {
            'show_grid': True,  # Default to True since checkbox removed
            'show_legend': False,  # Remove legend from plot
            'checked_pairs': self.get_checked_pairs()
        }
        
        # Add method-specific parameters based on current method selection
        current_item = self.method_list.currentItem()
        if current_item:
            method_name = current_item.text()
            method_params = self._get_method_parameters_from_controls()
            
            # Get plot type dynamically from comparison manager
            plot_type = 'scatter'  # default
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                method_info = self.comparison_manager.get_method_info(method_name)
                if method_info:
                    plot_type = method_info.get('plot_type', 'scatter')
            else:
                # Fallback mapping for when comparison manager is not available
                method_to_plot_type = {
                    'Correlation Analysis': 'pearson',
                    'Bland-Altman Analysis': 'bland_altman', 
                    'Residual Analysis': 'residual',
                    'Statistical Tests': 'scatter',
                    'Lin\'s CCC': 'ccc',
                    'RMSE': 'rmse',
                    'Intraclass Correlation Coefficient': 'icc',
                    'Cross-Correlation': 'cross_correlation',
                    'Dynamic Time Warping': 'dtw'
                }
                plot_type = method_to_plot_type.get(method_name, 'scatter')
            config['plot_type'] = plot_type
            
            # Add method-specific parameters
            if plot_type == 'bland_altman':
                config['confidence_interval'] = method_params.get('show_ci', True)
                config['agreement_limits'] = method_params.get('agreement_limits', 1.96)
                config['proportional_bias'] = method_params.get('proportional_bias', False)
            elif plot_type == 'scatter' or plot_type == 'pearson':
                config['confidence_level'] = method_params.get('confidence_level', 0.95)
                config['correlation_type'] = method_params.get('correlation_type', 'pearson')
            elif plot_type == 'residual':
                config['normality_test'] = method_params.get('normality_test', 'shapiro')
                config['outlier_detection'] = method_params.get('outlier_detection', 'iqr')
            elif plot_type == 'ccc':
                config['confidence_level'] = method_params.get('confidence_level', 0.95)
                config['bias_correction'] = method_params.get('bias_correction', True)
            elif plot_type == 'rmse':
                config['normalize_by'] = method_params.get('normalize_by', 'none')
                config['percentage_error'] = method_params.get('percentage_error', False)
            elif plot_type == 'icc':
                config['icc_type'] = method_params.get('icc_type', 'ICC(2,1)')
                config['confidence_level'] = method_params.get('confidence_level', 0.95)
            elif plot_type == 'cross_correlation':
                config['max_lag'] = method_params.get('max_lag', 50)
                config['normalize'] = method_params.get('normalize', True)
            elif plot_type == 'dtw':
                config['distance_metric'] = method_params.get('distance_metric', 'euclidean')
                config['window_type'] = method_params.get('window_type', 'sakoe_chiba')
        
        return config

    def _on_ref_file_changed(self, filename):
        """Update reference channel combo when file changes"""
        self._update_channel_combo(filename, self.ref_channel_combo)
        # Auto-populate alignment parameters after channel list changes
        self._auto_populate_alignment_parameters()
        
    def _on_test_file_changed(self, filename):
        """Update test channel combo when file changes"""
        self._update_channel_combo(filename, self.test_channel_combo)
        # Auto-populate alignment parameters after channel list changes
        self._auto_populate_alignment_parameters()
        
    def _update_channel_combo(self, filename, combo):
        """Update a channel combo box with channels from selected file"""
        combo.clear()
        
        if not self.channel_manager or not self.file_manager or not filename:
            return
            
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return
        
        # Get channels for this file using file_id
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        if channels:
            # Get unique channel names (by legend_label or channel_id)
            channel_names = []
            seen = set()
            for ch in channels:
                name = ch.legend_label or ch.channel_id
                if name not in seen:
                    channel_names.append(name)
                    seen.add(name)
            
            combo.addItems(channel_names)
            
    def _update_default_pair_name(self):
        """Update default pair name based on selected channels"""
        try:
            ref_channel = self.ref_channel_combo.currentText()
            test_channel = self.test_channel_combo.currentText()
            
            if ref_channel and test_channel and not self.pair_name_input.text().strip():
                # Only update if the field is empty (user hasn't entered custom name)
                default_name = f"{ref_channel} vs {test_channel}"
                self.pair_name_input.setPlaceholderText(default_name)
        except Exception as e:
            print(f"[ComparisonWizard] Error updating default pair name: {str(e)}")
    
    def _on_channel_selection_changed(self):
        """Handle channel selection changes to auto-populate alignment parameters"""
        try:
            self._auto_populate_alignment_parameters()
        except Exception as e:
            print(f"[ComparisonWizard] Error auto-populating alignment parameters: {str(e)}")
    
    def _auto_populate_alignment_parameters(self):
        """Auto-populate alignment parameters based on selected channels"""
        # Get selected channels
        ref_channel = self._get_selected_channel('ref')
        test_channel = self._get_selected_channel('test')
        
        if not ref_channel or not test_channel:
            # Update status label
            self.alignment_status_label.setText("Select both channels to configure alignment")
            self.alignment_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
            return
        
        # Always reset offsets to 0 (as requested)
        self.index_offset_spin.setValue(0)
        self.time_offset_spin.setValue(0.0)
        
        # Get alignment mode
        alignment_mode = self.alignment_mode_combo.currentText()
        
        if alignment_mode == "Index-Based":
            self._auto_populate_index_parameters(ref_channel, test_channel)
        else:  # Time-Based
            self._auto_populate_time_parameters(ref_channel, test_channel)
    
    def _get_selected_channel(self, channel_type):
        """Get the selected channel object (ref or test)"""
        if channel_type == 'ref':
            filename = self.ref_file_combo.currentText()
            channel_name = self.ref_channel_combo.currentText()
        else:  # test
            filename = self.test_file_combo.currentText()
            channel_name = self.test_channel_combo.currentText()
        
        if not filename or not channel_name:
            return None
        
        # Find file by filename
        if not self.file_manager or not self.channel_manager:
            return None
            
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return None
        
        # Get channels for this file using file_id
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        # Find matching channel by legend_label or channel_id
        for channel in channels:
            name = channel.legend_label or channel.channel_id
            if name == channel_name:
                return channel
        return None
    
    def _auto_populate_index_parameters(self, ref_channel, test_channel):
        """Auto-populate index-based alignment parameters"""
        try:
            # Get data lengths
            ref_length = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
            test_length = len(test_channel.ydata) if test_channel.ydata is not None else 0
            
            if ref_length == 0 or test_length == 0:
                self.alignment_status_label.setText("Selected channels have no data")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px;")
                return
            
            # Calculate optimal range
            start_index = 0  # Always start from 0
            end_index = min(ref_length - 1, test_length - 1)  # End at the shortest channel's last index
            
            # Update spin boxes
            self.start_index_spin.setValue(start_index)
            self.end_index_spin.setValue(end_index)
            
            # Update maximum values for the spin boxes
            max_index = min(ref_length - 1, test_length - 1)
            self.start_index_spin.setMaximum(max_index)
            self.end_index_spin.setMaximum(max_index)
            
            # Update status
            if ref_length == test_length:
                self.alignment_status_label.setText(f"✓ Channels aligned (length: {ref_length})")
                self.alignment_status_label.setStyleSheet("color: #2e7d32; font-size: 10px; padding: 5px;")
            else:
                self.alignment_status_label.setText(f"⚠ Length mismatch - Ref: {ref_length}, Test: {test_length}")
                self.alignment_status_label.setStyleSheet("color: #f57c00; font-size: 10px; padding: 5px;")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error auto-populating index parameters: {str(e)}")
            self.alignment_status_label.setText("Error configuring index alignment")
            self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px;")
    
    def _auto_populate_time_parameters(self, ref_channel, test_channel):
        """Auto-populate time-based alignment parameters"""
        try:
            # Check if channels have time data
            ref_has_time = hasattr(ref_channel, 'xdata') and ref_channel.xdata is not None and len(ref_channel.xdata) > 0
            test_has_time = hasattr(test_channel, 'xdata') and test_channel.xdata is not None and len(test_channel.xdata) > 0
            
            if ref_has_time and test_has_time:
                # Both channels have time data
                ref_start = float(ref_channel.xdata[0])
                ref_end = float(ref_channel.xdata[-1])
                test_start = float(test_channel.xdata[0])
                test_end = float(test_channel.xdata[-1])
                
                # Calculate overlap region
                overlap_start = max(ref_start, test_start)
                overlap_end = min(ref_end, test_end)
                
                if overlap_start < overlap_end:
                    # Valid overlap exists
                    self.start_time_spin.setValue(overlap_start)
                    self.end_time_spin.setValue(overlap_end)
                    
                    # Update ranges
                    min_time = min(ref_start, test_start)
                    max_time = max(ref_end, test_end)
                    self.start_time_spin.setRange(min_time, max_time)
                    self.end_time_spin.setRange(min_time, max_time)
                    
                    # Update status
                    self.alignment_status_label.setText(f"✓ Time overlap: {overlap_start:.3f}s to {overlap_end:.3f}s")
                    self.alignment_status_label.setStyleSheet("color: #2e7d32; font-size: 10px; padding: 5px;")
                else:
                    # No overlap
                    self.alignment_status_label.setText("⚠ No time overlap between channels")
                    self.alignment_status_label.setStyleSheet("color: #f57c00; font-size: 10px; padding: 5px;")
                    
            elif ref_has_time or test_has_time:
                # Only one channel has time data - use its range
                if ref_has_time:
                    start_time = float(ref_channel.xdata[0])
                    end_time = float(ref_channel.xdata[-1])
                    channel_name = "reference"
                else:
                    start_time = float(test_channel.xdata[0])
                    end_time = float(test_channel.xdata[-1])
                    channel_name = "test"
                
                self.start_time_spin.setValue(start_time)
                self.end_time_spin.setValue(end_time)
                self.start_time_spin.setRange(start_time, end_time)
                self.end_time_spin.setRange(start_time, end_time)
                
                # Update status
                self.alignment_status_label.setText(f"ℹ Using {channel_name} channel time range")
                self.alignment_status_label.setStyleSheet("color: #1976d2; font-size: 10px; padding: 5px;")
                
            else:
                # Neither channel has time data
                self.alignment_status_label.setText("⚠ No time data - will create from indices")
                self.alignment_status_label.setStyleSheet("color: #f57c00; font-size: 10px; padding: 5px;")
                
                # Set default time range
                self.start_time_spin.setValue(0.0)
                self.end_time_spin.setValue(10.0)
                
        except Exception as e:
            print(f"[ComparisonWizard] Error auto-populating time parameters: {str(e)}")
            self.alignment_status_label.setText("Error configuring time alignment")
            self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px;")
    
    def _on_add_pair(self):
        """Add a new comparison pair"""
        pair_config = self._get_current_pair_config()
        if pair_config:
            self.active_pairs.append(pair_config)
            self._update_active_pairs_table()
            self.pair_added.emit(pair_config)
            
            # Clear pair name for next entry and update placeholder
            self.pair_name_input.clear()
            self._update_default_pair_name()
    
    def _get_current_pair_config(self):
        """Get configuration for current pair selection"""
        # Validate inputs
        ref_file = self.ref_file_combo.currentText()
        test_file = self.test_file_combo.currentText()
        ref_channel = self.ref_channel_combo.currentText()
        test_channel = self.test_channel_combo.currentText()
        
        if not all([ref_file, test_file, ref_channel, test_channel]):
            QMessageBox.warning(self, "Incomplete Selection", 
                              "Please select both reference and test files and channels.")
            return None
            
        # Get pair name
        pair_name = self.pair_name_input.text().strip()
        if not pair_name:
            pair_name = self.pair_name_input.placeholderText() or f"{ref_channel} vs {test_channel}"
            
        # Get current method (method doesn't matter for adding pairs, only for plot generation)
        current_item = self.method_list.currentItem()
        method_name = current_item.text() if current_item else "Correlation Analysis"
        
        # Get alignment configuration
        alignment_config = self._get_alignment_config()
        
        # Create pair config
        pair_config = {
            'name': pair_name,
            'ref_file': ref_file,
            'test_file': test_file,
            'ref_channel': ref_channel,
            'test_channel': test_channel,
            'comparison_method': method_name,  # Stored but not used until plot generation
            'alignment_config': alignment_config,
            'method_parameters': {},  # Will be filled during plot generation
            'marker_type': None,  # Will be set in _update_active_pairs_table
            'marker_color': None   # Will be set in _update_active_pairs_table
        }
        
        return pair_config
        
    def _get_alignment_config(self):
        """Get alignment configuration from alignment controls"""
        config = {}
        
        if self.alignment_mode_combo.currentText() == "Index-Based":
            config['start_index'] = self.start_index_spin.value()
            config['end_index'] = self.end_index_spin.value()
            config['offset'] = self.index_offset_spin.value()
        else:  # Time-Based
            config['start_time'] = self.start_time_spin.value()
            config['end_time'] = self.end_time_spin.value()
            config['round_to'] = self.round_to_spin.value()
            config['interpolation'] = self.interpolation_combo.currentText()
            config['time_offset'] = self.time_offset_spin.value()
                    
        return config
        
    def _get_method_parameters_from_controls(self):
        """Extract method-specific parameters from method controls"""
        parameters = {}
        
        # Get current method
        current_item = self.method_list.currentItem()
        if not current_item:
            return parameters
            
        method_name = current_item.text()
        
        # Try to get parameters from dynamic controls first
        if hasattr(self, '_method_controls') and method_name in self._method_controls:
            for param_name, control in self._method_controls[method_name].items():
                try:
                    if isinstance(control, QComboBox):
                        parameters[param_name] = control.currentText()
                    elif isinstance(control, QCheckBox):
                        parameters[param_name] = control.isChecked()
                    elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
                        parameters[param_name] = control.value()
                    elif isinstance(control, QLineEdit):
                        parameters[param_name] = control.text()
                except Exception as e:
                    print(f"[ComparisonWizard] Error getting parameter {param_name}: {e}")
            return parameters
        
        # Fallback to static controls (with safety checks)
        try:
            if method_name == "Correlation Analysis":
                if hasattr(self, 'corr_type_combo'):
                    parameters['correlation_type'] = self.corr_type_combo.currentText()
                if hasattr(self, 'confidence_spin'):
                    parameters['confidence_level'] = self.confidence_spin.value()
                if hasattr(self, 'bootstrap_spin'):
                    parameters['bootstrap_samples'] = self.bootstrap_spin.value()
            elif method_name in ["Bland-Altman", "Bland-Altman Analysis"]:
                if hasattr(self, 'agreement_spin'):
                    parameters['agreement_limits'] = self.agreement_spin.value()
                if hasattr(self, 'show_ci_checkbox'):
                    parameters['show_ci'] = self.show_ci_checkbox.isChecked()
                if hasattr(self, 'prop_bias_checkbox'):
                    parameters['proportional_bias'] = self.prop_bias_checkbox.isChecked()
            elif method_name == "Residual Analysis":
                if hasattr(self, 'normality_combo'):
                    parameters['normality_test'] = self.normality_combo.currentText()
                if hasattr(self, 'outlier_combo'):
                    parameters['outlier_detection'] = self.outlier_combo.currentText()
            elif method_name == "Statistical Tests":
                if hasattr(self, 'alpha_spin'):
                    parameters['alpha_level'] = self.alpha_spin.value()
                if hasattr(self, 'test_suite_combo'):
                    parameters['test_suite'] = self.test_suite_combo.currentText()
                if hasattr(self, 'equal_var_combo'):
                    parameters['equal_variance'] = self.equal_var_combo.currentText()
                if hasattr(self, 'normality_assume_combo'):
                    parameters['normality_assumption'] = self.normality_assume_combo.currentText()
        except Exception as e:
            print(f"[ComparisonWizard] Error getting static control parameters: {e}")
                    
        return parameters
        
    def _update_active_pairs_table(self):
        """Update the active pairs table"""
        self.active_pair_table.setRowCount(len(self.active_pairs))
        
        for i, pair in enumerate(self.active_pairs):
            # Update pair style information
            pair['marker_type'] = self._get_style_for_pair(pair)
            pair['marker_color'] = self._get_color_for_pair(pair)
            
            # Show checkbox
            show_cb = QCheckBox()
            show_cb.setChecked(True)
            # Connect checkbox signal to update plots
            show_cb.stateChanged.connect(self._on_show_checkbox_changed)
            self.active_pair_table.setCellWidget(i, 0, show_cb)
            
            # Pair name with tooltip
            pair_name_item = QTableWidgetItem(pair['name'])
            self._set_pair_name_tooltip_on_item(pair_name_item, pair)
            self.active_pair_table.setItem(i, 1, pair_name_item)
            
            # Style - show marker with color information
            # Extract just the symbol from the marker type (e.g., "○" from "○ Circle")
            marker_symbol = pair['marker_type'].split()[0] if pair['marker_type'] else '○'
            style_text = marker_symbol
            style_item = QTableWidgetItem(style_text)
            style_item.setToolTip(f"Marker: {pair['marker_type']}\nColor: {pair['marker_color']}")
            
            # Set text color to match the pair color
            try:
                from PySide6.QtGui import QColor
                # Map emoji color names to actual colors
                color_display_map = {
                    '🔵 Blue': '#1f77b4',
                    '🔴 Red': '#d62728',
                    '🟢 Green': '#2ca02c',
                    '🟣 Purple': '#9467bd',
                    '🟠 Orange': '#ff7f0e',
                    '🟤 Brown': '#8c564b',
                    '🩷 Pink': '#e377c2',
                    '⚫ Gray': '#7f7f7f',
                    '🟡 Yellow': '#bcbd22',
                    '🔶 Cyan': '#17becf'
                }
                color_hex = color_display_map.get(pair['marker_color'], '#1f77b4')
                color = QColor(color_hex)
                style_item.setForeground(color)
            except:
                pass  # If color parsing fails, use default color
            self.active_pair_table.setItem(i, 2, style_item)
            
            # Status
            self.active_pair_table.setItem(i, 3, QTableWidgetItem("Ready"))  # Status
            
        # Enable generate plot button if pairs exist
        if hasattr(self, 'generate_plot_button'):
            self.generate_plot_button.setEnabled(len(self.active_pairs) > 0)
    
    def _on_show_checkbox_changed(self, state):
        """Handle Show checkbox state changes"""
        # No automatic plot update - only when Generate Plot button is clicked
        pass
    
    def _set_pair_name_tooltip_on_item(self, item, pair_config):
        """Set tooltip for a pair name table item"""
        try:
            pair_name = pair_config['name']
            
            # Build detailed tooltip
            tooltip_lines = []
            tooltip_lines.append(f"Comparison Pair: {pair_name}")
            tooltip_lines.append("")  # Empty line for spacing
            
            # Reference channel info
            ref_file = pair_config.get('ref_file', 'Unknown')
            ref_channel = pair_config.get('ref_channel', 'Unknown')
            tooltip_lines.append(f"📊 Reference:")
            tooltip_lines.append(f"   File: {ref_file}")
            tooltip_lines.append(f"   Channel: {ref_channel}")
            
            # Test channel info
            test_file = pair_config.get('test_file', 'Unknown')
            test_channel = pair_config.get('test_channel', 'Unknown')
            tooltip_lines.append(f"📊 Test:")
            tooltip_lines.append(f"   File: {test_file}")
            tooltip_lines.append(f"   Channel: {test_channel}")
            
            # Method and alignment info
            method = pair_config.get('comparison_method', 'Unknown')
            alignment_mode = pair_config.get('alignment_mode', 'index')
            tooltip_lines.append("")
            tooltip_lines.append(f"🔬 Method: {method}")
            tooltip_lines.append(f"⚙️ Alignment: {alignment_mode.title()}-based")
            
            # Join all lines and set tooltip
            tooltip_text = "\n".join(tooltip_lines)
            item.setToolTip(tooltip_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting pair tooltip: {str(e)}")
            # Fallback simple tooltip
            item.setToolTip(f"Pair: {pair_config.get('name', 'Unknown')}")
    
    def get_checked_pairs(self):
        """Get list of pairs that have their Show checkbox checked"""
        checked_pairs = []
        for row in range(self.active_pair_table.rowCount()):
            checkbox = self.active_pair_table.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                pair_name_item = self.active_pair_table.item(row, 1)
                if pair_name_item:
                    pair_name = pair_name_item.text()
                    # Find the corresponding pair config
                    for pair in self.active_pairs:
                        if pair['name'] == pair_name:
                            checked_pairs.append(pair)
                            break
        return checked_pairs
    
    def get_active_pairs(self):
        """Get list of active comparison pairs"""
        return self.active_pairs.copy()

    def update_cumulative_stats(self, stats_text):
        """Update the cumulative statistics display"""
        if hasattr(self, 'cumulative_stats_text'):
            self.cumulative_stats_text.setText(stats_text)
            
    def _refresh_comparison_data(self):
        """Refresh comparison data from the registry after manager is set"""
        try:
            print("[ComparisonWizard] Starting comparison data refresh...")
            
            # Repopulate comparison methods
            self._populate_comparison_methods()
            
            # Recreate dynamic method controls
            if hasattr(self, 'method_controls_stack'):
                print(f"[ComparisonWizard] Clearing {self.method_controls_stack.count()} existing controls")
                
                # Clear existing controls
                while self.method_controls_stack.count() > 0:
                    widget = self.method_controls_stack.widget(0)
                    self.method_controls_stack.removeWidget(widget)
                    if widget:
                        widget.deleteLater()
                
                # Recreate controls
                self._create_dynamic_method_controls()
                
                # Update current method selection
                if self.method_list.currentItem():
                    current_row = self.method_list.currentRow()
                    if current_row < self.method_controls_stack.count():
                        self.method_controls_stack.setCurrentIndex(current_row)
                        print(f"[ComparisonWizard] Set method controls to index {current_row}")
                        
            print("[ComparisonWizard] Successfully refreshed comparison data from registry")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing comparison data: {e}")
            import traceback
            traceback.print_exc()

    def _get_style_for_pair(self, pair):
        """Get style marker for a comparison pair"""
        # Define available markers similar to main window - these match the manager's marker_map
        marker_types = [
            '○ Circle', '□ Square', '△ Triangle', '◇ Diamond', '▽ Inverted Triangle',
            '◁ Left Triangle', '▷ Right Triangle', '⬟ Pentagon', '✦ Star', '⬢ Hexagon',
            '○ Circle', '□ Square'  # Repeat first two for more pairs
        ]
        
        # Use pair index to assign marker (cycling through available markers)
        pair_index = self.active_pairs.index(pair) if pair in self.active_pairs else 0
        marker_type = marker_types[pair_index % len(marker_types)]
        
        return marker_type
    
    def _get_color_for_pair(self, pair):
        """Get color for a comparison pair"""
        # Define color palette that matches the manager's color_map
        color_types = [
            '🔵 Blue', '🔴 Red', '🟢 Green', '🟣 Purple', '🟠 Orange', 
            '🟤 Brown', '🩷 Pink', '⚫ Gray', '🟡 Yellow', '🔶 Cyan'
        ]
        
        # Use pair index to assign color (cycling through available colors)
        pair_index = self.active_pairs.index(pair) if pair in self.active_pairs else 0
        color_type = color_types[pair_index % len(color_types)]
        
        return color_type