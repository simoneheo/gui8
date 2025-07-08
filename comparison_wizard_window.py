from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QDoubleSpinBox, QAbstractItemView, QHeaderView, QFileDialog
)
from PySide6.QtCore import Qt, Signal, QTimer, QPoint
from PySide6.QtGui import QTextCursor, QIntValidator, QColor, QFont, QPixmap, QPainter, QPen, QBrush, QPolygon
import pandas as pd
from copy import deepcopy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import time
import numpy as np
from comparison.comparison_registry import ComparisonRegistry, load_all_comparisons

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
        
        # Initialize overlay options for the default selected method
        self._initialize_default_method_selection()
        
        # Set initial helpful console message
        self._set_initial_console_message()
        
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
        
        # Set splitter proportions to match other wizards
        left_splitter.setSizes([350, 500])
        
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
        self.method_list.setMaximumHeight(300)  # Extended height to match right panel
        self._populate_comparison_methods()
        group_layout.addWidget(self.method_list)
        
        layout.addWidget(group)
        
    def _populate_comparison_methods(self):
        """Populate comparison methods from the registry"""
        try:
            # Initialize comparison registry if not already done
            if not ComparisonRegistry._initialized:
                load_all_comparisons()
            
            # Get methods from registry
            methods = ComparisonRegistry.get_all_methods()
            
            if not methods:
                # Fallback methods if registry is empty
                methods = ["Bland-Altman Analysis", "Correlation Analysis", "Residual Analysis", "Statistical Tests", "Cross-Correlation"]
                print("[ComparisonWizard] Using fallback methods - comparison registry may not be loaded")
            
            self.method_list.clear()
            self.method_list.addItems(methods)
            if methods:
                self.method_list.setCurrentRow(0)  # Select first method by default
                
            print(f"[ComparisonWizard] Loaded {len(methods)} comparison methods from registry")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error populating methods: {e}")
            # Fallback methods
            methods = ["Bland-Altman Analysis", "Correlation Analysis", "Residual Analysis", "Statistical Tests", "Cross-Correlation"]
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

        # Overlay Options (new section)
        self._create_overlay_options(self.method_controls_layout)

        # Performance Options (from old version)
        performance_group = QGroupBox("Performance Options")
        performance_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        performance_layout = QVBoxLayout(performance_group)

        # Downsampling option
        downsample_layout = QHBoxLayout()
        self.downsample_checkbox = QCheckBox("Limit Max Points:")
        self.downsample_checkbox.setToolTip("Reduce data points for better performance with large datasets")
        self.downsample_input = QLineEdit("5000")
        self.downsample_input.setMaximumWidth(80)
        self.downsample_input.setPlaceholderText("5000")
        self.downsample_input.setEnabled(False)  # Initially disabled

        # Connect checkbox to enable/disable input
        self.downsample_checkbox.toggled.connect(self.downsample_input.setEnabled)

        downsample_layout.addWidget(self.downsample_checkbox)
        downsample_layout.addWidget(self.downsample_input)
        downsample_layout.addStretch()

        performance_layout.addLayout(downsample_layout)

        # Additional performance options
        other_perf_layout = QHBoxLayout()

        self.fast_render_checkbox = QCheckBox("Fast Rendering")
        self.fast_render_checkbox.setToolTip("Use simplified rendering for better performance")
        self.fast_render_checkbox.setChecked(False)

        self.cache_results_checkbox = QCheckBox("Cache Results")
        self.cache_results_checkbox.setToolTip("Cache computation results to avoid recalculation")
        self.cache_results_checkbox.setChecked(True)

        other_perf_layout.addWidget(self.fast_render_checkbox)
        other_perf_layout.addWidget(self.cache_results_checkbox)
        other_perf_layout.addStretch()

        performance_layout.addLayout(other_perf_layout)

        # Density Display Options (from old version)
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density Display:"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["Scatter", "Hexbin", "KDE"])
        self.density_combo.setToolTip("Choose display method for high-density data")
        self.density_combo.setCurrentText("Scatter")

        # Bin size for hexbin
        self.bin_size_label = QLabel("Bin Size:")
        self.bin_size_input = QLineEdit("20")
        self.bin_size_input.setMaximumWidth(50)
        self.bin_size_input.setToolTip("Number of hexagonal bins for hexbin display")

        density_layout.addWidget(self.density_combo)
        density_layout.addWidget(self.bin_size_label)
        density_layout.addWidget(self.bin_size_input)
        density_layout.addStretch()

        performance_layout.addLayout(density_layout)

        self.method_controls_layout.addWidget(performance_group)

        layout.addWidget(group)
        
    def _create_dynamic_method_controls(self):
        """Create method controls dynamically from comparison registry"""
        try:
            # Initialize comparison registry if not already done
            if not ComparisonRegistry._initialized:
                load_all_comparisons()
            
            # Get methods from registry
            methods = ComparisonRegistry.get_all_methods()
            
            if methods:
                print(f"[ComparisonWizard] Creating dynamic controls for methods: {methods}")
                
                for method_name in methods:
                    try:
                        method_info = ComparisonRegistry.get_method_info(method_name)
                        if method_info:
                            widget = self._create_controls_for_method(method_name, method_info)
                            self.method_controls_stack.addWidget(widget)
                            print(f"[ComparisonWizard] Created controls for {method_name}")
                        else:
                            print(f"[ComparisonWizard] No method info found for {method_name}")
                            # Create a simple placeholder widget for this method
                            placeholder = QWidget()
                            placeholder_layout = QVBoxLayout(placeholder)
                            placeholder_layout.addWidget(QLabel(f"No parameters available for {method_name}"))
                            self.method_controls_stack.addWidget(placeholder)
                    except Exception as method_error:
                        print(f"[ComparisonWizard] Error creating controls for {method_name}: {method_error}")
                        # Create a simple placeholder widget for this method
                        placeholder = QWidget()
                        placeholder_layout = QVBoxLayout(placeholder)
                        placeholder_layout.addWidget(QLabel(f"Error loading controls for {method_name}"))
                        self.method_controls_stack.addWidget(placeholder)
            else:
                print("[ComparisonWizard] No methods available from registry, using static controls")
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
        
        # List of overlay-related parameters to skip (now handled in overlay section)
        overlay_params = {
            'show_ci', 'confidence_intervals', 'compute_confidence_intervals',
            'outlier_detection', 'confidence_bands', 'confidence_level',
            'bootstrap_ci', 'confidence_interval'
        }
        
        for param_name, param_config in parameters.items():
            # Skip overlay-related parameters
            if param_name in overlay_params:
                continue
                
            control = self._create_parameter_control(param_name, param_config)
            if control:
                # Use the shorter description for the label
                label_text = param_config.get('description', param_name)
                layout.addRow(label_text + ":", control)
                self._method_controls[method_name][param_name] = control
        
        # If no parameters after filtering, show a message
        if layout.rowCount() == 0:
            label = QLabel("Method-specific parameters (overlays configured separately)")
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
        """Create controls for correlation analysis - computational parameters only"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Correlation type (computational parameter)
        self.corr_type_combo = QComboBox()
        self.corr_type_combo.addItems(["pearson", "spearman", "kendall", "all"])
        layout.addRow("Correlation Type:", self.corr_type_combo)
        
        # Bootstrap samples (computational parameter)
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(100, 10000)
        self.bootstrap_spin.setValue(1000)
        layout.addRow("Bootstrap Samples:", self.bootstrap_spin)
        
        # Detrend method (computational parameter)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["none", "linear", "polynomial"])
        layout.addRow("Detrend Method:", self.detrend_combo)
        
        # Note: removed remove_outliers_checkbox - outlier detection is now display-only in overlay section
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_bland_altman_controls(self):
        """Create controls for Bland-Altman analysis - computational parameters only"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Agreement limits (computational parameter for calculating limits)
        self.agreement_spin = QDoubleSpinBox()
        self.agreement_spin.setRange(1.0, 3.0)
        self.agreement_spin.setValue(1.96)
        self.agreement_spin.setDecimals(2)
        layout.addRow("Agreement Multiplier:", self.agreement_spin)
        
        # Percentage difference option (computational parameter)
        self.percentage_diff_checkbox = QCheckBox()
        self.percentage_diff_checkbox.setChecked(False)
        layout.addRow("Percentage Differences:", self.percentage_diff_checkbox)
        
        # Log transform option (computational parameter)
        self.log_transform_checkbox = QCheckBox()
        self.log_transform_checkbox.setChecked(False)
        layout.addRow("Log Transform:", self.log_transform_checkbox)
        
        # Proportional bias test (computational parameter)
        self.prop_bias_checkbox = QCheckBox()
        self.prop_bias_checkbox.setChecked(True)
        layout.addRow("Test Proportional Bias:", self.prop_bias_checkbox)
        
        # Note: All display options (bias line, limits of agreement, etc.) moved to overlay section
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_residual_controls(self):
        """Create controls for residual analysis - computational parameters only"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Residual type (computational parameter)
        self.residual_type_combo = QComboBox()
        self.residual_type_combo.addItems(["absolute", "relative", "standardized", "studentized"])
        layout.addRow("Residual Type:", self.residual_type_combo)
        
        # Normality test (computational parameter)
        self.normality_combo = QComboBox()
        self.normality_combo.addItems(["shapiro", "kstest", "jarque_bera", "anderson", "all"])
        layout.addRow("Normality Test:", self.normality_combo)
        
        # Trend analysis (computational parameter)
        self.trend_analysis_checkbox = QCheckBox()
        self.trend_analysis_checkbox.setChecked(True)
        layout.addRow("Trend Analysis:", self.trend_analysis_checkbox)
        
        # Autocorrelation test (computational parameter)
        self.autocorr_checkbox = QCheckBox()
        self.autocorr_checkbox.setChecked(True)
        layout.addRow("Autocorrelation Test:", self.autocorr_checkbox)
        
        # Note: All display options (outliers, trend lines, statistics) moved to overlay section
        
        self.method_controls_stack.addWidget(widget)
        
    def _create_statistical_controls(self):
        """Create controls for statistical tests - computational parameters only"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Alpha level (significance level) - computational parameter
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 0.1)
        self.alpha_spin.setValue(0.05)
        self.alpha_spin.setDecimals(3)
        layout.addRow("Significance Level (α):", self.alpha_spin)
        
        # Test suite (computational parameter)
        self.test_suite_combo = QComboBox()
        self.test_suite_combo.addItems(["basic", "comprehensive", "nonparametric", "robust"])
        layout.addRow("Test Suite:", self.test_suite_combo)
        
        # Equal variance assumption (computational parameter)
        self.equal_var_combo = QComboBox()
        self.equal_var_combo.addItems(["assume_equal", "assume_unequal", "test"])
        layout.addRow("Equal Variance:", self.equal_var_combo)
        
        # Normality assumption (computational parameter)
        self.normality_assume_combo = QComboBox()
        self.normality_assume_combo.addItems(["assume_normal", "assume_nonnormal", "test"])
        layout.addRow("Normality:", self.normality_assume_combo)
        
        # Multiple comparisons correction
        self.multiple_comp_combo = QComboBox()
        self.multiple_comp_combo.addItems(["none", "bonferroni", "holm", "fdr_bh"])
        layout.addRow("Multiple Comparisons:", self.multiple_comp_combo)
        
        # Effect size measures
        self.effect_size_checkbox = QCheckBox()
        self.effect_size_checkbox.setChecked(True)
        layout.addRow("Effect Size Measures:", self.effect_size_checkbox)
        
        # Note: confidence_intervals moved to overlay section
        
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
        
        # Index mode
        self.index_mode_combo = QComboBox()
        self.index_mode_combo.addItems(["Truncate to Shortest", "Custom Range"])
        self.index_mode_combo.currentTextChanged.connect(self._on_index_mode_changed)
        index_layout.addRow("Mode:", self.index_mode_combo)
        
        # Custom range controls
        self.start_index_spin = QSpinBox()
        self.start_index_spin.setRange(0, 999999)
        self.start_index_spin.setValue(0)
        self.start_index_spin.valueChanged.connect(self._on_alignment_parameter_changed)
        index_layout.addRow("Start Index:", self.start_index_spin)
        
        self.end_index_spin = QSpinBox()
        self.end_index_spin.setRange(0, 999999)
        self.end_index_spin.setValue(500)
        self.end_index_spin.valueChanged.connect(self._on_alignment_parameter_changed)
        index_layout.addRow("End Index:", self.end_index_spin)
        
        # Index offset
        self.index_offset_spin = QSpinBox()
        self.index_offset_spin.setRange(-999999, 999999)
        self.index_offset_spin.setValue(0)
        self.index_offset_spin.setToolTip("Positive: shift test channel forward, Negative: shift reference channel forward")
        self.index_offset_spin.valueChanged.connect(self._on_alignment_parameter_changed)
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
        self.start_time_spin.valueChanged.connect(self._on_alignment_parameter_changed)
        time_layout.addRow("Start Time:", self.start_time_spin)
        
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(-999999.0, 999999.0)
        self.end_time_spin.setValue(10.0)
        self.end_time_spin.setDecimals(3)
        self.end_time_spin.valueChanged.connect(self._on_alignment_parameter_changed)
        time_layout.addRow("End Time:", self.end_time_spin)
        
        # Time offset
        self.time_offset_spin = QDoubleSpinBox()
        self.time_offset_spin.setRange(-999999.0, 999999.0)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setDecimals(3)
        self.time_offset_spin.setToolTip("Time offset to apply to test channel")
        self.time_offset_spin.valueChanged.connect(self._on_alignment_parameter_changed)
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
        
    def _create_pairs_management_group(self, layout):
        """Create pairs management group"""
        group = QGroupBox("Add Pair")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Console output (like mixer wizard)
        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setPlaceholderText("Method information and guidance will appear here")
        self.console_output.setMaximumHeight(300)  # Extended height to match right panel
        group_layout.addWidget(self.console_output)
        
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
        
        # Generate Plot button - moved here for better workflow
        self.generate_plot_button = QPushButton("Generate Plot")
        self.generate_plot_button.clicked.connect(self._on_generate_plot)
        self.generate_plot_button.setStyleSheet("""
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
        group_layout.addWidget(self.generate_plot_button)
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right panel with table at top, legend, and plot at bottom"""
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Results table at the top (like mixer wizard)
        self._build_results_table(right_layout)
        
        # Legend section (separate from plot)
        self._build_legend_section(right_layout)
        
        # Plot area at the bottom (like mixer wizard plot area)
        self._build_plot_area(right_layout)
        
        main_splitter.addWidget(self.right_panel)
        
    def _build_results_table(self, layout):
        """Build the results table showing active comparison pairs (like mixer wizard)"""
        layout.addWidget(QLabel("Channels:"))
        
        self.active_pair_table = QTableWidget(0, 5)
        self.active_pair_table.setHorizontalHeaderLabels(["Show", "Style", "Pair Name", "Shape", "Actions"])
        self.active_pair_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.active_pair_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.active_pair_table.setMaximumHeight(200)
        
        # Set column resize modes for better layout (matching mixer wizard)
        header = self.active_pair_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width
        header.setSectionResizeMode(2, QHeaderView.Fixed)     # Pair Name - fixed width
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Shape - fixed width
        header.setSectionResizeMode(4, QHeaderView.Stretch)   # Actions - stretches
        
        # Set specific column widths (matching mixer wizard)
        self.active_pair_table.setColumnWidth(0, 60)   # Show checkbox
        self.active_pair_table.setColumnWidth(1, 80)   # Style preview
        self.active_pair_table.setColumnWidth(2, 120)  # Pair Name column
        self.active_pair_table.setColumnWidth(3, 80)   # Shape column
        # Actions column will stretch
        
        self.active_pair_table.setAlternatingRowColors(True)
        self.active_pair_table.setShowGrid(True)
        self.active_pair_table.setGridStyle(Qt.SolidLine)
        
        # Style similar to mixer wizard
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
        
    def _build_legend_section(self, layout):
        """Build the legend section outside of plot area"""
        legend_frame = QFrame()
        legend_frame.setFrameStyle(QFrame.StyledPanel)
        legend_frame.setMaximumHeight(80)
        legend_frame_layout = QVBoxLayout(legend_frame)
        legend_frame_layout.setContentsMargins(5, 5, 5, 5)
        
        # Legend title
        legend_title = QLabel("📊 Plot Legend")
        legend_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #2c3e50;")
        legend_frame_layout.addWidget(legend_title)
        
        # Legend container with horizontal layout spanning 2 columns
        self.legend_container = QWidget()
        self.legend_layout = QHBoxLayout(self.legend_container)
        self.legend_layout.setContentsMargins(0, 0, 0, 0)
        self.legend_layout.setSpacing(15)
        
        # Add scroll area for legend if many items
        legend_scroll = QScrollArea()
        legend_scroll.setWidget(self.legend_container)
        legend_scroll.setWidgetResizable(True)
        legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        legend_scroll.setMaximumHeight(50)
        
        # Initial empty state
        self._update_legend_display()
        
        legend_frame_layout.addWidget(legend_scroll)
        layout.addWidget(legend_frame)
        
    def _update_legend_display(self):
        """Update the legend display with current active overlay options only"""
        try:
            # Clear existing legend items
            for i in reversed(range(self.legend_layout.count())):
                child = self.legend_layout.takeAt(i)
                if child.widget():
                    child.widget().deleteLater()
            
            # Get active overlay options instead of pairs
            active_overlays = self._get_active_overlay_legend_items()
            
            if not active_overlays:
                # Show empty state
                empty_label = QLabel("No active overlay options")
                empty_label.setStyleSheet("color: #666; font-style: italic; font-size: 10px;")
                self.legend_layout.addWidget(empty_label)
                self.legend_layout.addStretch()
                return
            
            # Add legend items for each active overlay
            for overlay_item in active_overlays:
                legend_widget = self._create_overlay_legend_item(overlay_item)
                self.legend_layout.addWidget(legend_widget)
            
            # Add stretch to push items to the left
            self.legend_layout.addStretch()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error updating legend display: {e}")
    
    def _get_active_overlay_legend_items(self):
        """Get list of active overlay options for legend display"""
        active_overlays = []
        
        # Check each overlay option and add to list if active and visible
        if hasattr(self, 'y_equals_x_checkbox') and self.y_equals_x_checkbox.isVisible() and self.y_equals_x_checkbox.isChecked():
            active_overlays.append({'name': 'y = x Line', 'style': 'line', 'color': 'red', 'linestyle': '--'})
            
        if hasattr(self, 'ci_checkbox') and self.ci_checkbox.isVisible() and self.ci_checkbox.isChecked():
            active_overlays.append({'name': 'Confidence Interval (95%)', 'style': 'fill', 'color': 'lightblue', 'alpha': 0.3})
            
        if hasattr(self, 'outlier_checkbox') and self.outlier_checkbox.isVisible() and self.outlier_checkbox.isChecked():
            active_overlays.append({'name': 'Outliers', 'style': 'marker', 'color': 'red', 'marker': 'x'})
            
        if hasattr(self, 'bias_line_checkbox') and self.bias_line_checkbox.isVisible() and self.bias_line_checkbox.isChecked():
            active_overlays.append({'name': 'Bias Line', 'style': 'line', 'color': 'green', 'linestyle': '-'})
            
        if hasattr(self, 'loa_checkbox') and self.loa_checkbox.isVisible() and self.loa_checkbox.isChecked():
            active_overlays.append({'name': 'Limits of Agreement', 'style': 'line', 'color': 'orange', 'linestyle': '--'})
            
        if hasattr(self, 'regression_line_checkbox') and self.regression_line_checkbox.isVisible() and self.regression_line_checkbox.isChecked():
            active_overlays.append({'name': 'Regression Line', 'style': 'line', 'color': 'purple', 'linestyle': '-'})
            
        if hasattr(self, 'error_bands_checkbox') and self.error_bands_checkbox.isVisible() and self.error_bands_checkbox.isChecked():
            active_overlays.append({'name': 'Error Bands', 'style': 'fill', 'color': 'yellow', 'alpha': 0.3})
            
        if hasattr(self, 'confidence_bands_checkbox') and self.confidence_bands_checkbox.isVisible() and self.confidence_bands_checkbox.isChecked():
            active_overlays.append({'name': 'Confidence Bands', 'style': 'fill', 'color': 'lightgray', 'alpha': 0.3})
            
        if hasattr(self, 'custom_line_checkbox') and self.custom_line_checkbox.isVisible() and self.custom_line_checkbox.isChecked():
            try:
                custom_value = float(self.custom_line_edit.text())
                active_overlays.append({'name': f'Custom Line (y = {custom_value})', 'style': 'line', 'color': 'magenta', 'linestyle': ':'})
            except (ValueError, AttributeError):
                active_overlays.append({'name': 'Custom Line (y = 0)', 'style': 'line', 'color': 'magenta', 'linestyle': ':'})
        
        return active_overlays
    
    def _create_overlay_legend_item(self, overlay_info):
        """Create a legend item widget for an overlay option"""
        from PySide6.QtWidgets import QLabel, QHBoxLayout, QWidget
        from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QBrush
        from PySide6.QtCore import Qt
        
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 2, 5, 2)
        item_layout.setSpacing(5)
        
        # Create visual representation based on overlay style
        pixmap = QPixmap(20, 12)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        color = QColor(overlay_info.get('color', 'blue'))
        
        if overlay_info['style'] == 'line':
            # Draw line representation
            pen = QPen(color, 2)
            linestyle = overlay_info.get('linestyle', '-')
            if linestyle == '--':
                pen.setStyle(Qt.DashLine)
            elif linestyle == ':':
                pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            painter.drawLine(2, 6, 18, 6)
            
        elif overlay_info['style'] == 'fill':
            # Draw filled rectangle representation
            color.setAlphaF(overlay_info.get('alpha', 0.3))
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color.darker(), 1))
            painter.drawRect(2, 3, 16, 6)
            
        elif overlay_info['style'] == 'marker':
            # Draw marker representation
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(color))
            if overlay_info.get('marker', 'o') == 'x':
                # Draw X marker
                painter.drawLine(6, 3, 14, 9)
                painter.drawLine(14, 3, 6, 9)
            else:
                # Draw circle marker
                painter.drawEllipse(8, 4, 4, 4)
        
        painter.end()
        
        # Create label with the pixmap
        icon_label = QLabel()
        icon_label.setPixmap(pixmap)
        icon_label.setFixedSize(20, 12)
        item_layout.addWidget(icon_label)
        
        # Overlay name
        name_label = QLabel(overlay_info['name'])
        name_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        item_layout.addWidget(name_label)
        
        # Style the item
        item_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 2px;
            }
            QWidget:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
        """)
        
        return item_widget
    
    def _create_legend_item(self, pair):
        """Create a single legend item widget for a pair"""
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(5, 2, 5, 2)
        item_layout.setSpacing(5)
        
        # Marker preview
        marker_widget = self._create_marker_widget(pair)
        item_layout.addWidget(marker_widget)
        
        # Pair name
        pair_name = pair.get('name', 'Unknown Pair')
        name_label = QLabel(pair_name)
        name_label.setStyleSheet("font-size: 10px; font-weight: bold;")
        item_layout.addWidget(name_label)
        
        # R² info if available
        if pair.get('r_squared') is not None:
            r_squared_label = QLabel(f"R²={pair['r_squared']:.3f}")
            r_squared_label.setStyleSheet("font-size: 9px; color: #666;")
            item_layout.addWidget(r_squared_label)
        
        # Set tooltip with pair information
        tooltip = self._create_pair_tooltip(pair)
        item_widget.setToolTip(tooltip)
        
        # Style the item
        item_widget.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 3px;
                padding: 2px;
            }
            QWidget:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
        """)
        
        return item_widget
        
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
        
        self.export_plot_button = QPushButton("📊 Export Plot")
        self.export_data_button = QPushButton("📋 Export Data")
        self.export_report_button = QPushButton("📄 Export Report")
        
        export_buttons_layout.addWidget(self.export_plot_button)
        export_buttons_layout.addWidget(self.export_data_button)
        export_buttons_layout.addWidget(self.export_report_button)
        export_buttons_layout.addStretch()
        
        export_frame_layout.addLayout(export_buttons_layout)
        
        # Comprehensive export button
        comprehensive_layout = QHBoxLayout()
        self.export_all_button = QPushButton("💾 Export Comparison Result")
        self.export_all_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        comprehensive_layout.addWidget(self.export_all_button)
        comprehensive_layout.addStretch()
        
        export_frame_layout.addLayout(comprehensive_layout)
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
        self.method_list.itemSelectionChanged.connect(self._on_method_selection_changed)
        
        # Pair management signals
        self.add_pair_button.clicked.connect(self._on_add_pair)
        
        # Alignment signals
        self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
        
        # Connect overlay option signals after widgets are created
        self._connect_overlay_signals()
        
        # Performance option signals (from old version)
        if hasattr(self, 'density_combo'):
            self.density_combo.currentTextChanged.connect(self._on_density_display_changed)
        if hasattr(self, 'bin_size_input'):
            self.bin_size_input.textChanged.connect(self._trigger_plot_update)
        if hasattr(self, 'downsample_checkbox'):
            self.downsample_checkbox.stateChanged.connect(self._trigger_plot_update)
        if hasattr(self, 'downsample_input'):
            self.downsample_input.textChanged.connect(self._trigger_plot_update)
        if hasattr(self, 'fast_render_checkbox'):
            self.fast_render_checkbox.stateChanged.connect(self._trigger_plot_update)
        if hasattr(self, 'cache_results_checkbox'):
            self.cache_results_checkbox.stateChanged.connect(self._trigger_plot_update)
        
        # Update timer for delayed operations
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._delayed_update)
        self.update_delay = 500  # 500ms delay
        
        # Export button connections
        self.export_plot_button.clicked.connect(self._export_plot)
        self.export_data_button.clicked.connect(self._export_data)
        self.export_report_button.clicked.connect(self._export_report)
        self.export_all_button.clicked.connect(self._export_comparison_result)
        
        # Update export button states initially
        self._update_export_button_states()

    def _connect_overlay_signals(self):
        """Connect overlay option signals to plot update"""
        overlay_widgets = [
            'y_equals_x_checkbox', 'ci_checkbox', 'confidence_bands_checkbox', 'outlier_checkbox',
            'bias_line_checkbox', 'loa_checkbox', 'regression_line_checkbox', 'trend_line_checkbox',
            'error_bands_checkbox', 'residual_stats_checkbox', 'density_overlay_checkbox',
            'histogram_overlay_checkbox', 'stats_results_checkbox', 'custom_line_checkbox'
        ]
        
        for widget_name in overlay_widgets:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                widget.stateChanged.connect(self._on_overlay_changed)
                print(f"[ComparisonWizard] Connected {widget_name} to overlay change handler")
        
        # Connect custom line edit
        if hasattr(self, 'custom_line_edit'):
            self.custom_line_edit.textChanged.connect(self._on_overlay_changed)
            print("[ComparisonWizard] Connected custom_line_edit to overlay change handler")

    def _on_method_selection_changed(self):
        """Handle method selection changes (for programmatic selection)"""
        selected_items = self.method_list.selectedItems()
        if selected_items:
            self._on_method_selected(selected_items[0])

    def _on_overlay_changed(self):
        """Handle overlay option changes with immediate plot update"""
        print("[ComparisonWizard] Overlay option changed, triggering immediate plot update...")
        
        # Update legend display immediately when overlay options change
        self._update_legend_display()
        
        # Only update if we have active pairs to avoid unnecessary updates
        if len(self.active_pairs) > 0:
            self._trigger_plot_update()
        else:
            print("[ComparisonWizard] No active pairs, skipping plot update")

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
        
    def _on_index_mode_changed(self, mode):
        """Handle index mode change"""
        enable_custom = (mode == "Custom Range")
        self.start_index_spin.setEnabled(enable_custom)
        self.end_index_spin.setEnabled(enable_custom)
        
        # Auto-populate alignment parameters when mode changes
        self._auto_populate_alignment_parameters()
        
    def _on_time_mode_changed(self, mode):
        """Handle time mode change"""
        enable_custom = (mode == "Custom Range")
        self.start_time_spin.setEnabled(enable_custom)
        self.end_time_spin.setEnabled(enable_custom)
        
        # Auto-populate alignment parameters when mode changes
        self._auto_populate_alignment_parameters()

    def _on_alignment_parameter_changed(self):
        """Handle alignment parameter value changes"""
        # Update alignment status display if needed
        self._update_alignment_status()
    
    def _on_density_display_changed(self):
        """Handle changes in density display type (from old version)"""
        if hasattr(self, 'density_combo'):
            density_type = self.density_combo.currentText().lower()
            
            # Enable/disable bin size input based on density type
            if hasattr(self, 'bin_size_input') and hasattr(self, 'bin_size_label'):
                enable_bin_size = density_type == 'hexbin'
                self.bin_size_input.setEnabled(enable_bin_size)
                self.bin_size_label.setEnabled(enable_bin_size)
        
        # Trigger plot update if needed
        self._trigger_plot_update()
    
    def _update_alignment_status(self):
        """Update alignment status label with current validation information"""
        ref_channel = self._get_selected_channel("reference")
        test_channel = self._get_selected_channel("test")
        
        if not ref_channel or not test_channel:
            self.alignment_status_label.setText("Select channels to configure alignment")
            self.alignment_status_label.setStyleSheet("color: #666; font-size: 10px; padding: 5px;")
            return
        
        # Check for empty channels first
        ref_data = getattr(ref_channel, 'ydata', None)
        test_data = getattr(test_channel, 'ydata', None)
        
        len_ref = len(ref_data) if ref_data is not None else 0
        len_test = len(test_data) if test_data is not None else 0
        
        # Enhanced empty channel validation
        if len_ref == 0 and len_test == 0:
            self.alignment_status_label.setText("⚠ Both channels are empty - plot will be blank")
            self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
            return
        elif len_ref == 0:
            self.alignment_status_label.setText("⚠ Reference channel is empty - plot will be blank")
            self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
            return
        elif len_test == 0:
            self.alignment_status_label.setText("⚠ Test channel is empty - plot will be blank")
            self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
            return
        
        # Get current alignment configuration
        alignment_config = self._get_alignment_config()
        
        # Perform basic validation (similar to mixer wizard)
        try:
            if len_ref == len_test:
                status_msg = f"Channels compatible: {len_ref} samples each"
                if alignment_config.get('offset', 0) != 0:
                    status_msg += f" - Offset: {alignment_config['offset']}"
                    self.alignment_status_label.setStyleSheet("color: orange; font-size: 10px; padding: 5px;")
                else:
                    status_msg += " - No alignment needed"
                    self.alignment_status_label.setStyleSheet("color: green; font-size: 10px; padding: 5px;")
            else:
                status_msg = f"Channels compatible: Ref({len_ref}) + Test({len_test}) samples"
                if alignment_config.get('alignment_method') == 'index':
                    if alignment_config.get('mode') == 'truncate':
                        status_msg += " - Will truncate to shortest"
                    else:
                        start_idx = alignment_config.get('start_index', 0)
                        end_idx = alignment_config.get('end_index', 500)
                        status_msg += f" - Custom range: {start_idx}-{end_idx}"
                else:  # time-based
                    if alignment_config.get('mode') == 'overlap':
                        status_msg += " - Will use overlap region"
                    else:
                        start_time = alignment_config.get('start_time', 0.0)
                        end_time = alignment_config.get('end_time', 10.0)
                        status_msg += f" - Custom range: {start_time:.3f}-{end_time:.3f}s"
                self.alignment_status_label.setStyleSheet("color: orange; font-size: 10px; padding: 5px;")
            
            self.alignment_status_label.setText(status_msg)
            
        except Exception as e:
            self.alignment_status_label.setText(f"Validation error: {str(e)}")
            self.alignment_status_label.setStyleSheet("color: red; font-size: 10px; padding: 5px;")
        
    def _on_method_selected(self, item):
        """Handle method selection change"""
        if not item:
            return
            
        # Update method controls stack to show controls for selected method
        method_name = item.text()
        method_index = self.method_list.row(item)
        
        if method_index < self.method_controls_stack.count():
            self.method_controls_stack.setCurrentIndex(method_index)
        
        # Update console with method information
        self._update_console_for_method(method_name)
        
        # Update overlay options based on method
        self._update_overlay_options(method_name)
        
        # Set default overlay states for the method
        self._set_default_overlay_states(method_name)
        
        # No automatic plot update - only when Generate Plot button is clicked

    def _set_default_overlay_states(self, method_name):
        """Set default overlay states based on the selected method"""
        try:
            print(f"[ComparisonWizard] Setting default overlay states for: {method_name}")
            
            # Reset all to unchecked first
            overlay_checkboxes = [
                'y_equals_x_checkbox', 'ci_checkbox', 'confidence_bands_checkbox', 'outlier_checkbox',
                'bias_line_checkbox', 'loa_checkbox', 'regression_line_checkbox', 'trend_line_checkbox',
                'error_bands_checkbox', 'residual_stats_checkbox', 'density_overlay_checkbox',
                'histogram_overlay_checkbox', 'stats_results_checkbox', 'custom_line_checkbox'
            ]
            
            for checkbox_name in overlay_checkboxes:
                if hasattr(self, checkbox_name):
                    checkbox = getattr(self, checkbox_name)
                    checkbox.setChecked(False)
            
            # Set method-specific defaults
            if method_name == "Bland-Altman Analysis":
                print("[ComparisonWizard] Setting Bland-Altman defaults: bias line and LoA")
                if hasattr(self, 'bias_line_checkbox') and self.bias_line_checkbox.isVisible():
                    self.bias_line_checkbox.setChecked(True)
                    print("[ComparisonWizard] ✓ Bias line checkbox set to checked")
                if hasattr(self, 'loa_checkbox') and self.loa_checkbox.isVisible():
                    self.loa_checkbox.setChecked(True)
                    print("[ComparisonWizard] ✓ LoA checkbox set to checked")
                    
            elif method_name == "Correlation Analysis":
                print("[ComparisonWizard] Setting Correlation defaults: y=x line and regression line")
                if hasattr(self, 'y_equals_x_checkbox') and self.y_equals_x_checkbox.isVisible():
                    self.y_equals_x_checkbox.setChecked(True)
                if hasattr(self, 'regression_line_checkbox') and self.regression_line_checkbox.isVisible():
                    self.regression_line_checkbox.setChecked(True)
                    
            elif method_name == "Cross-Correlation":
                print("[ComparisonWizard] Setting Cross-Correlation defaults: confidence bands")
                if hasattr(self, 'confidence_bands_checkbox') and self.confidence_bands_checkbox.isVisible():
                    self.confidence_bands_checkbox.setChecked(True)
                    
            print(f"[ComparisonWizard] Completed setting default overlay states for {method_name}")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error setting default overlay states: {e}")
            import traceback
            traceback.print_exc()

    def _update_overlay_options(self, method_name):
        """Show or hide overlay options based on the selected method"""
        # Hide all overlay options first
        for widget in self.overlay_widgets.values():
            widget.hide()

        # Show specific overlays based on method
        if method_name == "Bland-Altman Analysis":
            self.overlay_widgets['ci'].show()
            self.overlay_widgets['bias_line'].show()
            self.overlay_widgets['loa'].show()
            self.overlay_widgets['outlier'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()
            
        elif method_name == "Correlation Analysis":
            self.overlay_widgets['y_equals_x'].show()
            self.overlay_widgets['ci'].show()
            self.overlay_widgets['regression_line'].show()
            self.overlay_widgets['outlier'].show()
            self.overlay_widgets['density_overlay'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()
            
        elif method_name == "Residual Analysis":
            self.overlay_widgets['outlier'].show()
            self.overlay_widgets['ci'].show()
            self.overlay_widgets['trend_line'].show()
            self.overlay_widgets['residual_stats'].show()
            self.overlay_widgets['histogram_overlay'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()
            
        elif method_name == "Statistical Tests":
            self.overlay_widgets['ci'].show()
            self.overlay_widgets['outlier'].show()
            self.overlay_widgets['y_equals_x'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()
            
        elif method_name == "Cross-Correlation":
            self.overlay_widgets['confidence_bands'].show()
            self.overlay_widgets['trend_line'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()
            
        # Default case - show basic overlays
        else:
            self.overlay_widgets['y_equals_x'].show()
            self.overlay_widgets['ci'].show()
            self.overlay_widgets['outlier'].show()
            self.overlay_widgets['stats_results'].show()
            self.overlay_widgets['custom_line'].show()

    def _create_overlay_options(self, layout):
        """Create comprehensive overlay options group - only display-related toggles"""
        self.overlay_group = QGroupBox("Overlay Options")
        self.overlay_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.overlay_layout = QVBoxLayout(self.overlay_group)

        # Identity/reference lines
        self.y_equals_x_checkbox = QCheckBox("Show y = x Line")
        self.y_equals_x_checkbox.setToolTip("Show identity line for perfect agreement reference")
        self.overlay_layout.addWidget(self.y_equals_x_checkbox)

        # Confidence intervals and bands
        self.ci_checkbox = QCheckBox("Show Confidence Intervals")
        self.ci_checkbox.setToolTip("Show confidence intervals around statistics")
        self.overlay_layout.addWidget(self.ci_checkbox)

        self.confidence_bands_checkbox = QCheckBox("Show Confidence Bands")
        self.confidence_bands_checkbox.setToolTip("Show confidence bands (for cross-correlation and time series)")
        self.overlay_layout.addWidget(self.confidence_bands_checkbox)

        # Outlier detection and highlighting
        self.outlier_checkbox = QCheckBox("Highlight Outliers")
        self.outlier_checkbox.setToolTip("Identify and highlight statistical outliers on the plot")
        self.overlay_layout.addWidget(self.outlier_checkbox)

        # Bland-Altman specific overlays
        self.bias_line_checkbox = QCheckBox("Show Bias Line")
        self.bias_line_checkbox.setToolTip("Show mean bias line (horizontal line at mean difference)")
        self.overlay_layout.addWidget(self.bias_line_checkbox)

        self.loa_checkbox = QCheckBox("Show Limits of Agreement")
        self.loa_checkbox.setToolTip("Show limits of agreement (±1.96×SD of differences)")
        self.overlay_layout.addWidget(self.loa_checkbox)

        # Regression and trend lines
        self.regression_line_checkbox = QCheckBox("Show Regression Line")
        self.regression_line_checkbox.setToolTip("Show linear regression line (best fit)")
        self.overlay_layout.addWidget(self.regression_line_checkbox)

        self.trend_line_checkbox = QCheckBox("Show Trend Line")
        self.trend_line_checkbox.setToolTip("Show trend line for time series or residual patterns")
        self.overlay_layout.addWidget(self.trend_line_checkbox)

        # Error analysis overlays
        self.error_bands_checkbox = QCheckBox("Show Error Bands")
        self.error_bands_checkbox.setToolTip("Show ±RMSE error bands around identity line")
        self.overlay_layout.addWidget(self.error_bands_checkbox)

        self.residual_stats_checkbox = QCheckBox("Show Residual Statistics")
        self.residual_stats_checkbox.setToolTip("Display residual statistics on the plot")
        self.overlay_layout.addWidget(self.residual_stats_checkbox)

        # Distribution overlays
        self.density_overlay_checkbox = QCheckBox("Show Density Overlay")
        self.density_overlay_checkbox.setToolTip("Show kernel density estimation overlay")
        self.overlay_layout.addWidget(self.density_overlay_checkbox)

        self.histogram_overlay_checkbox = QCheckBox("Show Histogram Overlay")
        self.histogram_overlay_checkbox.setToolTip("Show histogram overlay for distributions")
        self.overlay_layout.addWidget(self.histogram_overlay_checkbox)

        # Statistical test results
        self.stats_results_checkbox = QCheckBox("Show Statistical Results")
        self.stats_results_checkbox.setToolTip("Display statistical test results on the plot")
        self.overlay_layout.addWidget(self.stats_results_checkbox)

        # Custom line option (general purpose)
        self.custom_line_widget = QWidget()
        custom_line_layout = QHBoxLayout(self.custom_line_widget)
        custom_line_layout.setContentsMargins(0, 0, 0, 0)
        self.custom_line_checkbox = QCheckBox("Custom Line (y = ")
        self.custom_line_checkbox.setToolTip("Add a custom horizontal or diagonal line to the plot")
        self.custom_line_edit = QLineEdit("0.0")
        self.custom_line_edit.setMaximumWidth(60)
        custom_line_layout.addWidget(self.custom_line_checkbox)
        custom_line_layout.addWidget(self.custom_line_edit)
        custom_line_layout.addWidget(QLabel(")"))
        custom_line_layout.addStretch()
        self.overlay_layout.addWidget(self.custom_line_widget)

        # Store all overlay widgets for easy show/hide management
        self.overlay_widgets = {
            'y_equals_x': self.y_equals_x_checkbox,
            'ci': self.ci_checkbox,
            'confidence_bands': self.confidence_bands_checkbox,
            'outlier': self.outlier_checkbox,
            'bias_line': self.bias_line_checkbox,
            'loa': self.loa_checkbox,
            'regression_line': self.regression_line_checkbox,
            'trend_line': self.trend_line_checkbox,
            'error_bands': self.error_bands_checkbox,
            'residual_stats': self.residual_stats_checkbox,
            'density_overlay': self.density_overlay_checkbox,
            'histogram_overlay': self.histogram_overlay_checkbox,
            'stats_results': self.stats_results_checkbox,
            'custom_line': self.custom_line_widget
        }

        layout.addWidget(self.overlay_group)

    def _trigger_plot_update(self):
        """Trigger a plot update when overlay options change"""
        print("[ComparisonWizard] Overlay option changed, updating plot...")
        
        # Update the plot immediately when overlay options change
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            # Get current plot configuration with updated overlay parameters
            plot_config = self._get_plot_config()
            
            # Regenerate the plot with updated overlay settings
            self.comparison_manager._on_plot_generated(plot_config)
            
            print(f"[ComparisonWizard] Plot updated with overlay parameters: {plot_config.get('overlay_params', {})}")
        else:
            # If no comparison manager, just start the timer for delayed update
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
        # If no pairs exist, try to auto-add one from current selections
        if not self.active_pairs:
            print("[ComparisonWizard] No comparison pairs found - attempting to auto-add pair from current selections")
            try:
                # Try to add a pair using current channel selections and alignment settings
                self._on_add_pair()
                
                # Check if the pair was successfully added
                if not self.active_pairs:
                    print("[ComparisonWizard] Could not auto-add comparison pair - missing channel selections")
                    QMessageBox.warning(self, "No Comparison Pairs", 
                                      "No comparison pairs available for plotting.\n\n"
                                      "Please select reference and test channels, then click 'Add Comparison Pair' first.")
                    return
                else:
                    print(f"[ComparisonWizard] Successfully auto-added comparison pair: {self.active_pairs[-1]['name']}")
                    
            except Exception as e:
                print(f"[ComparisonWizard] Error auto-adding comparison pair: {e}")
                QMessageBox.warning(self, "Error Adding Pair", 
                                  f"Could not automatically add comparison pair: {str(e)}\n\n"
                                  "Please manually select channels and click 'Add Comparison Pair'.")
                return
        
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
            
            # Add computed results to console
            checked_pairs = self.get_checked_pairs()
            self._output_comparison_results_to_console(checked_pairs, method_name, method_params)
            
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
        
        # Add overlay parameters from the overlay section
        overlay_params = self._get_overlay_parameters()
        config.update(overlay_params)
        
        # Add performance options
        if hasattr(self, 'downsample_checkbox') and self.downsample_checkbox.isChecked():
            try:
                max_points = int(self.downsample_input.text())
                config['downsample'] = max_points
            except (ValueError, AttributeError):
                config['downsample'] = 5000  # Default fallback
        
        if hasattr(self, 'fast_render_checkbox'):
            config['fast_render'] = self.fast_render_checkbox.isChecked()
            
        if hasattr(self, 'cache_results_checkbox'):
            config['cache_results'] = self.cache_results_checkbox.isChecked()
            
        # Add density display options
        if hasattr(self, 'density_combo'):
            config['density_display'] = self.density_combo.currentText().lower()
            
        if hasattr(self, 'bin_size_input'):
            try:
                bin_size = int(self.bin_size_input.text())
                config['bin_size'] = bin_size
            except (ValueError, AttributeError):
                config['bin_size'] = 20  # Default fallback
                
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
                    'Cross-Correlation': 'cross_correlation'
                }
                plot_type = method_to_plot_type.get(method_name, 'scatter')
            config['plot_type'] = plot_type
            
            # Add method-specific parameters
            if plot_type == 'bland_altman':
                config['agreement_limits'] = method_params.get('agreement_multiplier', 1.96)
                config['proportional_bias'] = method_params.get('test_proportional_bias', False)
                config['percentage_difference'] = method_params.get('percentage_difference', False)
            elif plot_type == 'scatter' or plot_type == 'pearson':
                config['confidence_level'] = method_params.get('confidence_level', 0.95)
                config['correlation_type'] = method_params.get('correlation_type', 'pearson')
                config['include_rmse'] = method_params.get('include_rmse', True)
            elif plot_type == 'residual':
                config['fit_method'] = method_params.get('fit_method', 'linear')
                config['polynomial_degree'] = method_params.get('polynomial_degree', 2)
                config['detect_outliers'] = method_params.get('detect_outliers', True)
            elif plot_type == 'cross_correlation':
                config['max_lag'] = method_params.get('max_lag', 50)
                config['normalize'] = method_params.get('normalize', True)
                config['find_peak'] = method_params.get('find_peak', True)
        
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
        ref_channel = self._get_selected_channel('reference')
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
        """Get the selected channel object (ref/reference or test)"""
        if channel_type in ['ref', 'reference']:
            filename = self.ref_file_combo.currentText()
            channel_name = self.ref_channel_combo.currentText()
        else:  # test
            filename = self.test_file_combo.currentText()
            channel_name = self.test_channel_combo.currentText()
        
        return self._get_channel_by_name(filename, channel_name)
    
    def _get_channel_by_name(self, filename, channel_name):
        """Get channel object by filename and channel name"""
        if not self.channel_manager or not self.file_manager or not filename or not channel_name:
            return None
            
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return None
        
        # Get channels for this file
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        if channels:
            # Find channel by name (legend_label or channel_id)
            for ch in channels:
                if (ch.legend_label == channel_name) or (ch.channel_id == channel_name):
                    return ch
        
        return None
    
    def _auto_populate_index_parameters(self, ref_channel, test_channel):
        """Auto-populate index-based alignment parameters"""
        try:
            # Get data lengths
            ref_length = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
            test_length = len(test_channel.ydata) if test_channel.ydata is not None else 0
            
            # Enhanced empty channel validation
            if ref_length == 0 and test_length == 0:
                self.alignment_status_label.setText("⚠ Both channels are empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
                return
            elif ref_length == 0:
                self.alignment_status_label.setText("⚠ Reference channel is empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
                return
            elif test_length == 0:
                self.alignment_status_label.setText("⚠ Test channel is empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
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
            # First check for empty channels
            ref_data_length = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
            test_data_length = len(test_channel.ydata) if test_channel.ydata is not None else 0
            
            # Enhanced empty channel validation
            if ref_data_length == 0 and test_data_length == 0:
                self.alignment_status_label.setText("⚠ Both channels are empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
                return
            elif ref_data_length == 0:
                self.alignment_status_label.setText("⚠ Reference channel is empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
                return
            elif test_data_length == 0:
                self.alignment_status_label.setText("⚠ Test channel is empty - plot will be blank")
                self.alignment_status_label.setStyleSheet("color: #d32f2f; font-size: 10px; padding: 5px; font-weight: bold;")
                return
            
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
            # Compute R² for the pair
            self._compute_pair_r_squared(pair_config)
            
            self.active_pairs.append(pair_config)
            self._update_active_pairs_table()
            self.pair_added.emit(pair_config)
            
            # Add success message to console
            r_squared_info = ""
            if pair_config.get('r_squared') is not None:
                r_squared_info = f" (R² = {pair_config['r_squared']:.4f})"
            
            alignment_info = pair_config.get('alignment_config', {}).get('alignment_method', 'index')
            console_msg = f"✅ Added comparison pair: '{pair_config['name']}' using {alignment_info}-based alignment{r_squared_info}"
            print(f"[ComparisonWizard] {console_msg}")
            self.console_output.append(console_msg)
            
            # Clear pair name for next entry and update placeholder
            self.pair_name_input.clear()
            self._update_default_pair_name()
            
            # Update legend display
            self._update_legend_display()
            
            # Update export button states
            self._update_export_button_states()
    
    def _compute_pair_r_squared(self, pair_config):
        """Compute R² (coefficient of determination) for a pair"""
        try:
            # Get the channel data for both reference and test
            ref_channel = self._get_selected_channel('reference')
            test_channel = self._get_selected_channel('test')
            
            if not ref_channel or not test_channel:
                pair_config['r_squared'] = None
                return
            
            # Get the data arrays
            ref_data = ref_channel.ydata
            test_data = test_channel.ydata
            
            if ref_data is None or test_data is None:
                pair_config['r_squared'] = None
                return
            
            # Align the data arrays (simple truncation for now)
            min_len = min(len(ref_data), len(test_data))
            ref_aligned = ref_data[:min_len]
            test_aligned = test_data[:min_len]
            
            # Remove invalid data points
            valid_mask = np.isfinite(ref_aligned) & np.isfinite(test_aligned)
            ref_clean = ref_aligned[valid_mask]
            test_clean = test_aligned[valid_mask]
            
            if len(ref_clean) < 2:
                pair_config['r_squared'] = None
                return
            
            # Compute Pearson correlation coefficient
            correlation_matrix = np.corrcoef(ref_clean, test_clean)
            if correlation_matrix.shape == (2, 2):
                r = correlation_matrix[0, 1]
                r_squared = r ** 2
                pair_config['r_squared'] = r_squared
            else:
                pair_config['r_squared'] = None
                
        except Exception as e:
            print(f"[ComparisonWizard] Error computing R² for pair: {e}")
            pair_config['r_squared'] = None
    
    def _get_current_pair_config(self):
        """Get configuration for current pair selection"""
        # Validate inputs
        ref_file = self.ref_file_combo.currentText()
        test_file = self.test_file_combo.currentText()
        ref_channel = self.ref_channel_combo.currentText()
        test_channel = self.test_channel_combo.currentText()
        
        if not all([ref_file, test_file, ref_channel, test_channel]):
            missing_items = []
            if not ref_file: missing_items.append("reference file")
            if not test_file: missing_items.append("test file") 
            if not ref_channel: missing_items.append("reference channel")
            if not test_channel: missing_items.append("test channel")
            
            console_msg = f"Cannot add comparison pair - missing: {', '.join(missing_items)}"
            print(f"[ComparisonWizard] {console_msg}")
            self.console_output.append(f"⚠️ {console_msg}")
            return None
        
        # Validate that channels are not empty
        ref_channel_obj = self._get_selected_channel("reference")
        test_channel_obj = self._get_selected_channel("test")
        
        if ref_channel_obj and test_channel_obj:
            ref_data = getattr(ref_channel_obj, 'ydata', None)
            test_data = getattr(test_channel_obj, 'ydata', None)
            
            ref_length = len(ref_data) if ref_data is not None else 0
            test_length = len(test_data) if test_data is not None else 0
            
            if ref_length == 0 and test_length == 0:
                console_msg = "Cannot add comparison pair - both channels are empty"
                print(f"[ComparisonWizard] {console_msg}")
                self.console_output.append(f"⚠️ {console_msg}")
                return None
            elif ref_length == 0:
                console_msg = f"Cannot add comparison pair - reference channel '{ref_channel}' is empty"
                print(f"[ComparisonWizard] {console_msg}")
                self.console_output.append(f"⚠️ {console_msg}")
                return None
            elif test_length == 0:
                console_msg = f"Cannot add comparison pair - test channel '{test_channel}' is empty"
                print(f"[ComparisonWizard] {console_msg}")
                self.console_output.append(f"⚠️ {console_msg}")
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
        """Get current alignment configuration from UI controls (like mixer wizard)"""
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
        
    def _get_method_parameters_from_controls(self):
        """Extract method-specific parameters from method controls (excluding overlay parameters)"""
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
        
        # Fallback to static controls (with safety checks) - excluding overlay parameters
        try:
            if method_name == "Correlation Analysis":
                if hasattr(self, 'corr_type_combo'):
                    parameters['correlation_type'] = self.corr_type_combo.currentText()
                if hasattr(self, 'bootstrap_spin'):
                    parameters['bootstrap_samples'] = self.bootstrap_spin.value()
                if hasattr(self, 'remove_outliers_checkbox'):
                    parameters['remove_outliers'] = self.remove_outliers_checkbox.isChecked()
                if hasattr(self, 'detrend_combo'):
                    parameters['detrend_method'] = self.detrend_combo.currentText()
                    
            elif method_name in ["Bland-Altman", "Bland-Altman Analysis"]:
                if hasattr(self, 'agreement_spin'):
                    parameters['agreement_multiplier'] = self.agreement_spin.value()
                if hasattr(self, 'percentage_diff_checkbox'):
                    parameters['percentage_difference'] = self.percentage_diff_checkbox.isChecked()
                if hasattr(self, 'log_transform_checkbox'):
                    parameters['log_transform'] = self.log_transform_checkbox.isChecked()
                if hasattr(self, 'prop_bias_checkbox'):
                    parameters['test_proportional_bias'] = self.prop_bias_checkbox.isChecked()
                    
            elif method_name == "Residual Analysis":
                if hasattr(self, 'residual_type_combo'):
                    parameters['residual_type'] = self.residual_type_combo.currentText()
                if hasattr(self, 'normality_combo'):
                    parameters['normality_test'] = self.normality_combo.currentText()
                if hasattr(self, 'trend_analysis_checkbox'):
                    parameters['trend_analysis'] = self.trend_analysis_checkbox.isChecked()
                if hasattr(self, 'autocorr_checkbox'):
                    parameters['autocorrelation_test'] = self.autocorr_checkbox.isChecked()
                    
            elif method_name == "Statistical Tests":
                if hasattr(self, 'alpha_spin'):
                    parameters['significance_level'] = self.alpha_spin.value()
                if hasattr(self, 'test_suite_combo'):
                    parameters['test_suite'] = self.test_suite_combo.currentText()
                if hasattr(self, 'equal_var_combo'):
                    parameters['equal_variance_assumption'] = self.equal_var_combo.currentText()
                if hasattr(self, 'normality_assume_combo'):
                    parameters['normality_assumption'] = self.normality_assume_combo.currentText()
                if hasattr(self, 'multiple_comp_combo'):
                    parameters['multiple_comparisons'] = self.multiple_comp_combo.currentText()
                if hasattr(self, 'effect_size_checkbox'):
                    parameters['effect_size_measures'] = self.effect_size_checkbox.isChecked()
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error getting static control parameters: {e}")
        
        # Add overlay parameters from the overlay section
        overlay_params = self._get_overlay_parameters()
        parameters.update(overlay_params)
                    
        return parameters
    
    def _get_overlay_parameters(self):
        """Get overlay parameters from the overlay section"""
        overlay_params = {}
        
        try:
            # Get current method to determine which overlays are relevant
            current_item = self.method_list.currentItem()
            method_name = current_item.text() if current_item else ""
            
            # Only include visible overlay options (changed from isEnabled to isVisible)
            if hasattr(self, 'y_equals_x_checkbox') and self.y_equals_x_checkbox.isVisible():
                overlay_params['show_identity_line'] = self.y_equals_x_checkbox.isChecked()
                
            if hasattr(self, 'ci_checkbox') and self.ci_checkbox.isVisible():
                overlay_params['confidence_interval'] = self.ci_checkbox.isChecked()  # Match expected name
                overlay_params['confidence_level'] = 0.95  # Default confidence level
                
            if hasattr(self, 'confidence_bands_checkbox') and self.confidence_bands_checkbox.isVisible():
                overlay_params['show_confidence_bands'] = self.confidence_bands_checkbox.isChecked()
                
            if hasattr(self, 'outlier_checkbox') and self.outlier_checkbox.isVisible():
                overlay_params['highlight_outliers'] = self.outlier_checkbox.isChecked()
                
            if hasattr(self, 'bias_line_checkbox') and self.bias_line_checkbox.isVisible():
                overlay_params['show_bias_line'] = self.bias_line_checkbox.isChecked()
                
            if hasattr(self, 'loa_checkbox') and self.loa_checkbox.isVisible():
                overlay_params['show_limits_of_agreement'] = self.loa_checkbox.isChecked()
                
            if hasattr(self, 'regression_line_checkbox') and self.regression_line_checkbox.isVisible():
                overlay_params['show_regression_line'] = self.regression_line_checkbox.isChecked()
                
            if hasattr(self, 'trend_line_checkbox') and self.trend_line_checkbox.isVisible():
                overlay_params['show_trend_line'] = self.trend_line_checkbox.isChecked()
                
            if hasattr(self, 'error_bands_checkbox') and self.error_bands_checkbox.isVisible():
                overlay_params['show_error_bands'] = self.error_bands_checkbox.isChecked()
                
            if hasattr(self, 'residual_stats_checkbox') and self.residual_stats_checkbox.isVisible():
                overlay_params['show_residual_statistics'] = self.residual_stats_checkbox.isChecked()
                
            if hasattr(self, 'density_overlay_checkbox') and self.density_overlay_checkbox.isVisible():
                overlay_params['show_density_overlay'] = self.density_overlay_checkbox.isChecked()
                
            if hasattr(self, 'histogram_overlay_checkbox') and self.histogram_overlay_checkbox.isVisible():
                overlay_params['show_histogram_overlay'] = self.histogram_overlay_checkbox.isChecked()
                
            if hasattr(self, 'stats_results_checkbox') and self.stats_results_checkbox.isVisible():
                overlay_params['show_statistical_results'] = self.stats_results_checkbox.isChecked()
                
            if hasattr(self, 'custom_line_checkbox') and self.custom_line_checkbox.isVisible() and self.custom_line_checkbox.isChecked():
                try:
                    custom_value = float(self.custom_line_edit.text())
                    overlay_params['custom_line'] = custom_value  # Match expected name
                except (ValueError, AttributeError):
                    overlay_params['custom_line'] = 0.0
            
            print(f"[ComparisonWizard] Overlay parameters for {method_name}: {overlay_params}")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error getting overlay parameters: {e}")
            
        return overlay_params

    def _update_active_pairs_table(self):
        """Update the active pairs table (like mixer wizard)"""
        self.active_pair_table.setRowCount(len(self.active_pairs))
        
        for i, pair in enumerate(self.active_pairs):
            # Update pair style information
            pair['marker_type'] = self._get_style_for_pair(pair)
            pair['marker_color'] = self._get_color_for_pair(pair)
            
            # Create tooltip for the pair
            tooltip = self._create_pair_tooltip(pair)
            
            # Column 0: Show checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, row=i: self._on_show_checkbox_changed(state))
            
            # Center the checkbox in the cell
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            checkbox_widget.setToolTip(tooltip)
            self.active_pair_table.setCellWidget(i, 0, checkbox_widget)
            
            # Column 1: Style (show actual marker as it appears in plot legend)
            style_widget = self._create_marker_widget(pair)
            style_widget.setToolTip(tooltip)
            self.active_pair_table.setCellWidget(i, 1, style_widget)
            
            # Column 2: Pair Name
            pair_name_item = QTableWidgetItem(pair['name'])
            pair_name_item.setToolTip(tooltip)
            self.active_pair_table.setItem(i, 2, pair_name_item)
            
            # Column 3: Shape
            shape_text = self._get_pair_shape_text(pair)
            shape_item = QTableWidgetItem(shape_text)
            shape_item.setToolTip(tooltip)
            self.active_pair_table.setItem(i, 3, shape_item)
            
            # Column 4: Actions (like mixer wizard)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Info button (shows individual pair statistics)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(25)
            info_button.setMaximumHeight(25)
            info_button.setToolTip("View individual pair statistics")
            info_button.clicked.connect(lambda checked=False, p=pair: self._show_pair_info(p))
            actions_layout.addWidget(info_button)
            
            # Inspect button
            inspect_button = QPushButton("🔍")
            inspect_button.setMaximumWidth(25)
            inspect_button.setMaximumHeight(25)
            inspect_button.setToolTip("Inspect pair data")
            inspect_button.clicked.connect(lambda checked=False, p=pair: self._inspect_pair_data(p))
            actions_layout.addWidget(inspect_button)
            
            # Style button
            style_button = QPushButton("🎨")
            style_button.setMaximumWidth(25)
            style_button.setMaximumHeight(25)
            style_button.setToolTip("Pair styling")
            style_button.clicked.connect(lambda checked=False, p=pair: self._style_pair(p))
            actions_layout.addWidget(style_button)
            
            # Transform button (disabled for comparison pairs)
            transform_button = QPushButton("🔨")
            transform_button.setMaximumWidth(25)
            transform_button.setMaximumHeight(25)
            transform_button.setEnabled(False)
            transform_button.setToolTip("Transform not available for comparison pairs")
            actions_layout.addWidget(transform_button)
            
            # Delete button
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(25)
            delete_button.setMaximumHeight(25)
            delete_button.setToolTip("Delete comparison pair")
            delete_button.clicked.connect(lambda checked=False, idx=i: self._delete_pair(idx))
            actions_layout.addWidget(delete_button)
            
            self.active_pair_table.setCellWidget(i, 4, actions_widget)
    
    def _on_show_checkbox_changed(self, state):
        """Handle Show checkbox state changes"""
        print(f"[ComparisonWizard] Show checkbox changed, state: {state}")
        
        # Update the plot immediately when checkbox state changes
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            # Update cumulative statistics based on checked pairs
            self.comparison_manager._update_cumulative_display()
            
            # Get current plot configuration
            plot_config = self._get_plot_config()
            
            # Regenerate the plot with only checked pairs
            self.comparison_manager._on_plot_generated(plot_config)
            
            print(f"[ComparisonWizard] Plot updated with {len(self.get_checked_pairs())} visible pairs")
            
            # Update legend display
            self._update_legend_display()
    
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
            checkbox_widget = self.active_pair_table.cellWidget(row, 0)
            if checkbox_widget:
                # The checkbox is inside a QWidget with a layout
                checkbox = None
                for child in checkbox_widget.children():
                    if isinstance(child, QCheckBox):
                        checkbox = child
                        break
                
                if checkbox and checkbox.isChecked():
                    pair_name_item = self.active_pair_table.item(row, 2)  # Column 2 is pair name
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

    def _create_pair_tooltip(self, pair):
        """Create a detailed tooltip for a comparison pair"""
        # Get alignment method from alignment config
        alignment_config = pair.get('alignment_config', {})
        alignment_method = alignment_config.get('alignment_method', 'index')
        
        tooltip_lines = [
            f"Pair: {pair['name']}",
            f"Reference: {pair['ref_file']} - {pair['ref_channel']}",
            f"Test: {pair['test_file']} - {pair['test_channel']}",
            f"Method: {pair.get('comparison_method', 'N/A')}",
            f"Alignment: {alignment_method.title()}-based",
            f"R²: {pair.get('r_squared', 'N/A')}",
            f"Shape: {self._get_pair_shape_text(pair)}",
            f"Marker: {pair.get('marker_type', 'Circle')}",
            f"Color: {pair.get('marker_color', 'Blue')}"
        ]
        return "\n".join(tooltip_lines)

    def _get_pair_shape_text(self, pair):
        """Get the data dimensions of the aligned pair (e.g., '100x2')"""
        try:
            # Get the channels for this pair
            ref_channel = self._get_channel_by_name(pair['ref_file'], pair['ref_channel'])
            test_channel = self._get_channel_by_name(pair['test_file'], pair['test_channel'])
            
            if ref_channel and test_channel:
                # Get alignment configuration
                alignment_config = pair.get('alignment_config', {})
                alignment_method = alignment_config.get('alignment_method', 'index')
                
                # Calculate aligned data dimensions
                if alignment_method == 'index':
                    # For index-based alignment
                    ref_len = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
                    test_len = len(test_channel.ydata) if test_channel.ydata is not None else 0
                    
                    mode = alignment_config.get('mode', 'truncate')
                    if mode == 'truncate':
                        # Use minimum length
                        aligned_len = min(ref_len, test_len)
                    else:  # custom
                        # Use custom range
                        start_idx = alignment_config.get('start_index', 0)
                        end_idx = alignment_config.get('end_index', min(ref_len, test_len))
                        aligned_len = max(0, end_idx - start_idx + 1)
                    
                    return f"{aligned_len}x2"
                    
                else:  # time-based alignment
                    # For time-based alignment, we need to estimate based on time range
                    ref_len = len(ref_channel.ydata) if ref_channel.ydata is not None else 0
                    test_len = len(test_channel.ydata) if test_channel.ydata is not None else 0
                    
                    # Use minimum as approximation for time-based alignment
                    aligned_len = min(ref_len, test_len)
                    return f"~{aligned_len}x2"
            
            # Fallback if channels can't be found
            return "N/A"
            
        except Exception as e:
            print(f"[ComparisonWizard] Error calculating pair shape: {e}")
            return "N/A"
    
    def _get_color_hex_for_pair(self, pair):
        """Get the hex color code for a comparison pair"""
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
        color_name = pair.get('marker_color', '🔵 Blue')
        return color_display_map.get(color_name, '#1f77b4')

    def _show_pair_info(self, pair):
        """Show individual pair statistics"""
        try:
            # Create a dialog to show pair statistics
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Pair Statistics: {pair['name']}")
            dialog.setMinimumSize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Pair information
            info_text = QTextEdit()
            info_text.setReadOnly(True)
            
            # Build information text
            info_lines = []
            info_lines.append(f"=== Pair Information ===")
            info_lines.append(f"Name: {pair['name']}")
            info_lines.append(f"Reference: {pair['ref_file']} - {pair['ref_channel']}")
            info_lines.append(f"Test: {pair['test_file']} - {pair['test_channel']}")
            info_lines.append(f"Method: {pair['comparison_method']}")
            info_lines.append(f"Alignment: {pair['alignment_config']['alignment_method'].title()}-based")
            info_lines.append("")
            
            # Basic statistics
            info_lines.append(f"=== Basic Statistics ===")
            r2_value = pair.get('r_squared', None)
            if r2_value is not None:
                info_lines.append(f"R² (Coefficient of Determination): {r2_value:.6f}")
                info_lines.append(f"R (Correlation Coefficient): {np.sqrt(r2_value):.6f}")
                
                # Interpret R² value
                if r2_value >= 0.9:
                    interpretation = "Excellent correlation"
                elif r2_value >= 0.7:
                    interpretation = "Strong correlation"
                elif r2_value >= 0.5:
                    interpretation = "Moderate correlation"
                elif r2_value >= 0.3:
                    interpretation = "Weak correlation"
                else:
                    interpretation = "Very weak correlation"
                info_lines.append(f"Interpretation: {interpretation}")
            else:
                info_lines.append("R² not available (compute by generating plot)")
            
            info_lines.append("")
            
            # Data information
            try:
                ref_channel = self._get_selected_channel('reference')
                test_channel = self._get_selected_channel('test')
                
                if ref_channel and test_channel:
                    info_lines.append(f"=== Data Information ===")
                    info_lines.append(f"Reference data points: {len(ref_channel.ydata) if ref_channel.ydata is not None else 'N/A'}")
                    info_lines.append(f"Test data points: {len(test_channel.ydata) if test_channel.ydata is not None else 'N/A'}")
                    
                    if ref_channel.ydata is not None:
                        info_lines.append(f"Reference range: {np.min(ref_channel.ydata):.3f} to {np.max(ref_channel.ydata):.3f}")
                        info_lines.append(f"Reference mean: {np.mean(ref_channel.ydata):.3f}")
                        info_lines.append(f"Reference std: {np.std(ref_channel.ydata):.3f}")
                    
                    if test_channel.ydata is not None:
                        info_lines.append(f"Test range: {np.min(test_channel.ydata):.3f} to {np.max(test_channel.ydata):.3f}")
                        info_lines.append(f"Test mean: {np.mean(test_channel.ydata):.3f}")
                        info_lines.append(f"Test std: {np.std(test_channel.ydata):.3f}")
                    
            except Exception as e:
                info_lines.append(f"Error loading data information: {str(e)}")
            
            info_lines.append("")
            info_lines.append(f"=== Alignment Configuration ===")
            alignment_config = pair.get('alignment_config', {})
            for key, value in alignment_config.items():
                info_lines.append(f"{key}: {value}")
            
            info_text.setText("\n".join(info_lines))
            layout.addWidget(info_text)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error showing pair info: {e}")
            QMessageBox.warning(self, "Error", f"Could not display pair information: {str(e)}")

    def _inspect_pair_data(self, pair):
        """Inspect pair data"""
        # This method should be implemented based on your specific requirements
        # For example, you can use the pair information to display a dialog with data inspection
        print(f"Inspecting data for pair: {pair['name']}")

    def _style_pair(self, pair):
        """Style a comparison pair using the marker wizard"""
        try:
            from marker_wizard import open_marker_wizard
            
            # Open the marker wizard for this pair
            result = open_marker_wizard(pair, self)
            
            if result:
                # Update the table to reflect changes
                self._update_active_pairs_table()
                print(f"[ComparisonWizard] Updated marker properties for pair: {pair['name']}")
                
                # Optionally trigger plot update if desired
                # self._trigger_plot_update()
                
        except ImportError as e:
            print(f"[ComparisonWizard] Error importing marker wizard: {e}")
            QMessageBox.warning(self, "Error", "Marker wizard not available")
        except Exception as e:
            print(f"[ComparisonWizard] Error opening marker wizard: {e}")
            QMessageBox.warning(self, "Error", f"Could not open marker wizard: {str(e)}")

    def _delete_pair(self, index):
        """Delete a comparison pair"""
        if index >= 0 and index < len(self.active_pairs):
            removed_pair = self.active_pairs.pop(index)
            self._update_active_pairs_table()
            self._update_legend_display()
            self._update_export_button_states()
            self.pair_deleted.emit()
            print(f"[ComparisonWizard] Pair '{removed_pair['name']}' deleted")

    def _update_console_for_method(self, method_name):
        """Update console with helpful information about the selected method"""
        try:
            # Get method information from comparison manager
            method_info = None
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                method_info = self.comparison_manager.get_method_info(method_name)
            
            if method_info:
                # Get helpful information from the method class
                helpful_info = method_info.get('helpful_info', '')
                if helpful_info:
                    self.console_output.append(helpful_info)
                    return
            
            # Fallback to built-in method descriptions
            method_descriptions = {
                'Correlation Analysis': 'Measures linear and monotonic relationships between datasets.\n• Pearson: Linear correlation (assumes normal distribution)\n• Spearman: Rank-based correlation (non-parametric)\n• Kendall: Robust to outliers, good for small samples\n• Use for: Assessing how well two variables move together',
                
                'Bland-Altman Analysis': 'Evaluates agreement between two measurement methods.\n• Plots difference vs average of measurements\n• Shows bias (systematic difference) and limits of agreement\n• Identifies proportional bias and outliers\n• Use for: Method comparison, clinical agreement studies',
                
                'Residual Analysis': 'Analyzes residuals (prediction errors) between datasets.\n• Checks for patterns in residuals (heteroscedasticity)\n• Tests normality of residuals\n• Identifies outliers and influential points\n• Use for: Validating model assumptions, quality control',
                
                'Statistical Tests': 'Performs comprehensive statistical comparisons.\n• t-tests for mean differences\n• F-tests for variance equality\n• Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)\n• Use for: Hypothesis testing, distribution comparison',
                
                'Lin\'s CCC': 'Concordance Correlation Coefficient for agreement assessment.\n• Combines precision (Pearson correlation) and accuracy (bias)\n• Values: 1 = perfect agreement, 0 = no agreement\n• More stringent than Pearson correlation\n• Use for: Method validation, reproducibility studies',
                
                'RMSE': 'Root Mean Square Error - measures prediction accuracy.\n• Lower values indicate better agreement\n• Sensitive to outliers\n• Same units as original data\n• Use for: Model validation, accuracy assessment',
                
                'Intraclass Correlation Coefficient': 'Measures reliability within groups.\n• ICC(1,1): Single measurement, random raters\n• ICC(2,1): Single measurement, fixed raters\n• ICC(3,1): Single measurement, fixed raters (consistency)\n• Use for: Inter-rater reliability, test-retest reliability',
                
                'Cross-Correlation': 'Measures similarity as a function of lag/displacement.\n• Finds optimal alignment between signals\n• Detects time delays and phase shifts\n• Normalized values between -1 and 1\n• Use for: Signal alignment, time series analysis',
                
                'Dynamic Time Warping': 'Aligns time series with different time scales.\n• Finds optimal warping path between sequences\n• Handles non-linear time scaling\n• Robust to timing variations\n• Use for: Pattern matching, sequence alignment'
            }
            
            description = method_descriptions.get(method_name, f'Selected method: {method_name}')
            self.console_output.append(description)
            
        except Exception as e:
            self.console_output.append(f"Error loading method information: {str(e)}")

    def _create_marker_widget(self, pair):
        """Create a widget displaying the marker as it appears in the plot legend"""
        from PySide6.QtWidgets import QLabel
        from PySide6.QtGui import QPainter, QPixmap, QColor, QPen, QBrush
        from PySide6.QtCore import Qt
        
        # Create a small pixmap to draw the marker
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get marker properties
        marker_type = pair.get('marker_type', '○ Circle')
        color_hex = self._get_color_hex_for_pair(pair)
        color = QColor(color_hex)
        
        # Set up painter
        painter.setPen(QPen(color, 1.5))
        painter.setBrush(QBrush(color))
        
        # Draw marker based on type
        center_x, center_y = 10, 10
        size = 6
        
        if '○' in marker_type or 'Circle' in marker_type:
            # Filled circle (like in the plot)
            painter.drawEllipse(center_x - size//2, center_y - size//2, size, size)
        elif '□' in marker_type or 'Square' in marker_type:
            # Filled square (like in the plot)
            painter.drawRect(center_x - size//2, center_y - size//2, size, size)
        elif '△' in marker_type or 'Triangle' in marker_type:
            # Filled triangle
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            triangle = QPolygon([
                QPoint(center_x, center_y - size//2),
                QPoint(center_x - size//2, center_y + size//2),
                QPoint(center_x + size//2, center_y + size//2)
            ])
            painter.drawPolygon(triangle)
        elif '◇' in marker_type or 'Diamond' in marker_type:
            # Filled diamond
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            diamond = QPolygon([
                QPoint(center_x, center_y - size//2),
                QPoint(center_x + size//2, center_y),
                QPoint(center_x, center_y + size//2),
                QPoint(center_x - size//2, center_y)
            ])
            painter.drawPolygon(diamond)
        elif '✦' in marker_type or 'Star' in marker_type:
            # Simple star representation (filled circle with points)
            painter.drawEllipse(center_x - size//2, center_y - size//2, size, size)
            # Add small lines for star effect
            painter.drawLine(center_x, center_y - size//2 - 1, center_x, center_y + size//2 + 1)
            painter.drawLine(center_x - size//2 - 1, center_y, center_x + size//2 + 1, center_y)
        else:
            # Default to filled circle
            painter.drawEllipse(center_x - size//2, center_y - size//2, size, size)
        
        painter.end()
        
        # Create label widget with the pixmap
        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(20, 20)
        
        return label

    def _initialize_default_method_selection(self):
        """Initialize overlay options for the default selected method"""
        try:
            # Get the currently selected method (should be the first one)
            current_item = self.method_list.currentItem()
            if current_item:
                method_name = current_item.text()
                print(f"[ComparisonWizard] Initializing overlay options for default method: {method_name}")
                
                # Manually trigger the method selection logic
                self._on_method_selected(current_item)
                
                # Set the method controls stack to show the first method
                if self.method_controls_stack.count() > 0:
                    self.method_controls_stack.setCurrentIndex(0)
                
                # Ensure overlay signals are connected after everything is set up
                print("[ComparisonWizard] Re-connecting overlay signals after initialization")
                self._connect_overlay_signals()
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error initializing default method: {e}")
            import traceback
            traceback.print_exc()
    
    def _output_comparison_results_to_console(self, checked_pairs, method_name, method_params):
        """Output comparison results and statistics to console"""
        try:
            if not checked_pairs:
                return
                
            self.console_output.append("\n" + "="*50)
            self.console_output.append(f"📊 COMPARISON RESULTS - {method_name}")
            self.console_output.append("="*50)
            
            for i, pair in enumerate(checked_pairs):
                pair_name = pair.get('name', f'Pair {i+1}')
                self.console_output.append(f"\n🔍 {pair_name}:")
                
                # Show basic pair info
                ref_channel = pair.get('ref_channel', 'Unknown')
                test_channel = pair.get('test_channel', 'Unknown')
                self.console_output.append(f"   Reference: {ref_channel}")
                self.console_output.append(f"   Test: {test_channel}")
                
                # Show R² if available
                if pair.get('r_squared') is not None:
                    self.console_output.append(f"   R² = {pair['r_squared']:.4f}")
                
                # Show alignment method
                alignment_info = pair.get('alignment_config', {})
                alignment_method = alignment_info.get('alignment_method', 'index')
                self.console_output.append(f"   Alignment: {alignment_method}-based")
                
                # Method-specific results
                if method_name == "Correlation Analysis":
                    corr_type = method_params.get('correlation_type', 'pearson')
                    self.console_output.append(f"   Correlation Type: {corr_type}")
                    if method_params.get('remove_outliers'):
                        self.console_output.append("   ⚠️ Outliers removed from analysis")
                        
                elif method_name in ["Bland-Altman", "Bland-Altman Analysis"]:
                    agreement_mult = method_params.get('agreement_multiplier', 1.96)
                    self.console_output.append(f"   Agreement Limits: ±{agreement_mult}σ")
                    if method_params.get('percentage_difference'):
                        self.console_output.append("   Using percentage differences")
                    if method_params.get('test_proportional_bias'):
                        self.console_output.append("   Testing for proportional bias")
                        
                elif method_name == "Residual Analysis":
                    residual_type = method_params.get('residual_type', 'linear')
                    self.console_output.append(f"   Residual Type: {residual_type}")
                    if method_params.get('trend_analysis'):
                        self.console_output.append("   ✓ Trend analysis enabled")
                        
                elif method_name == "Statistical Tests":
                    alpha = method_params.get('significance_level', 0.05)
                    test_suite = method_params.get('test_suite', 'basic')
                    self.console_output.append(f"   Significance Level: α = {alpha}")
                    self.console_output.append(f"   Test Suite: {test_suite}")
            
            # Show overlay settings
            overlay_info = []
            if method_params.get('show_identity_line'):
                overlay_info.append("Identity Line")
            if method_params.get('confidence_interval'):
                overlay_info.append("Confidence Intervals")
            if method_params.get('show_regression_line'):
                overlay_info.append("Regression Line")
            if method_params.get('highlight_outliers'):
                overlay_info.append("Outlier Highlighting")
                
            if overlay_info:
                self.console_output.append(f"\n📈 Plot Overlays: {', '.join(overlay_info)}")
            
            self.console_output.append("\n✅ Plot generation completed successfully!")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error outputting results to console: {e}")
            self.console_output.append(f"\n⚠️ Error displaying results: {str(e)}")

    def _set_initial_console_message(self):
        """Set initial helpful console message on wizard startup"""
        try:
            # Clear the console first
            self.console_output.clear()
            
            # Add welcome message and helpful information
            welcome_msg = """Welcome to the Data Comparison Wizard!

=== Data Alignment Options ===

INDEX-BASED ALIGNMENT:
• Compares data points by their position in the dataset
• Point 1 from Reference vs Point 1 from Test
• Use when datasets have same sampling rate and timing
• Best for: synchronized measurements, matched samples

TIME-BASED ALIGNMENT:
• Compares data points by their timestamp values
• Finds closest time matches between datasets
• Use when datasets have different sampling rates
• Best for: different acquisition systems, resampled data

=== Quick Start ===
1. Select your reference and test channels above
2. Choose appropriate alignment method (auto-configured)
3. Click "Add Comparison Pair" to begin analysis
4. The wizard will automatically generate preview plots
5. Click "Generate Plot" for final comparison results

TIP: The wizard auto-configures alignment parameters based on your data.
Just select your channels and click "Add Comparison Pair" to get started!"""
            
            self.console_output.setPlainText(welcome_msg)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting initial console message: {e}")
            # Fallback to a simple message
            self.console_output.setPlainText("Welcome to the Data Comparison Wizard!\n\nSelect channels and click 'Add Comparison Pair' to begin analysis.")

    def _export_plot(self):
        """Export the current comparison plot"""
        try:
            from PySide6.QtWidgets import QFileDialog
            
            # Check if plot exists
            if not hasattr(self, 'comparison_plot_widget') or not self.comparison_plot_widget:
                QMessageBox.warning(self, "No Plot", "No plot to export. Please generate a comparison plot first.")
                return
            
            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Export Comparison Plot", 
                "comparison_plot.png",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*.*)"
            )
            
            if filename:
                # Access the matplotlib figure from the plot widget
                if hasattr(self.comparison_plot_widget, 'figure'):
                    self.comparison_plot_widget.figure.savefig(filename, dpi=300, bbox_inches='tight')
                    QMessageBox.information(self, "Export Complete", f"Plot exported to:\n{filename}")
                else:
                    QMessageBox.warning(self, "Export Error", "Could not access plot figure for export.")
                    
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting plot:\n{str(e)}")
    
    def _export_data(self):
        """Export comparison data to CSV/TXT"""
        try:
            from PySide6.QtWidgets import QFileDialog
            import pandas as pd
            import numpy as np
            
            # Check if we have active pairs
            checked_pairs = self.get_checked_pairs()
            if not checked_pairs:
                QMessageBox.warning(self, "No Data", "No comparison pairs selected. Please add and select pairs first.")
                return
            
            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Comparison Data",
                "comparison_data.csv",
                "CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)"
            )
            
            if filename:
                # Prepare data for export
                export_data = []
                
                for pair in checked_pairs:
                    # Get pair data
                    ref_channel = self._get_channel_by_name(pair['ref_file'], pair['ref_channel'])
                    test_channel = self._get_channel_by_name(pair['test_file'], pair['test_channel'])
                    
                    if ref_channel and test_channel:
                        # Add pair data
                        pair_data = {
                            'Pair_Name': pair['pair_name'],
                            'Reference_File': pair['ref_file'],
                            'Reference_Channel': pair['ref_channel'],
                            'Test_File': pair['test_file'],
                            'Test_Channel': pair['test_channel'],
                            'Method': pair.get('method', 'correlation'),
                            'R_Squared': pair.get('r_squared', 'N/A')
                        }
                        
                        # Add raw data if available
                        if hasattr(ref_channel, 'ydata') and hasattr(test_channel, 'ydata'):
                            ref_data = ref_channel.ydata
                            test_data = test_channel.ydata
                            
                            # Handle different lengths
                            max_len = max(len(ref_data), len(test_data))
                            
                            # Create arrays for this pair
                            ref_padded = np.full(max_len, np.nan)
                            test_padded = np.full(max_len, np.nan)
                            
                            ref_padded[:len(ref_data)] = ref_data
                            test_padded[:len(test_data)] = test_data
                            
                            # Add to export data
                            for i in range(max_len):
                                row = {
                                    **pair_data,
                                    'Data_Index': i,
                                    'Reference_Value': ref_padded[i],
                                    'Test_Value': test_padded[i]
                                }
                                export_data.append(row)
                        else:
                            # Just add metadata
                            export_data.append(pair_data)
                
                # Create DataFrame and export
                df = pd.DataFrame(export_data)
                
                if filename.endswith('.csv'):
                    df.to_csv(filename, index=False)
                else:
                    df.to_csv(filename, index=False, sep='\t')
                    
                QMessageBox.information(self, "Export Complete", f"Data exported to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data:\n{str(e)}")
    
    def _export_report(self):
        """Export comparison report as text"""
        try:
            from PySide6.QtWidgets import QFileDialog
            import datetime
            
            # Check if we have active pairs
            checked_pairs = self.get_checked_pairs()
            if not checked_pairs:
                QMessageBox.warning(self, "No Data", "No comparison pairs selected. Please add and select pairs first.")
                return
            
            # Get filename
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Comparison Report",
                "comparison_report.txt",
                "Text Files (*.txt);;All Files (*.*)"
            )
            
            if filename:
                # Generate report content
                report_lines = []
                report_lines.append("COMPARISON ANALYSIS REPORT")
                report_lines.append("=" * 50)
                report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Total Pairs: {len(checked_pairs)}")
                report_lines.append("")
                
                # Add method information
                if self.method_list.currentItem():
                    method_name = self.method_list.currentItem().text()
                    report_lines.append(f"Analysis Method: {method_name.title()}")
                    report_lines.append("")
                
                # Add pair details
                for i, pair in enumerate(checked_pairs, 1):
                    report_lines.append(f"PAIR {i}: {pair['pair_name']}")
                    report_lines.append("-" * 30)
                    report_lines.append(f"Reference: {pair['ref_file']} - {pair['ref_channel']}")
                    report_lines.append(f"Test: {pair['test_file']} - {pair['test_channel']}")
                    report_lines.append(f"Method: {pair.get('method', 'correlation')}")
                    
                    if 'r_squared' in pair and pair['r_squared'] != 'N/A':
                        report_lines.append(f"R-squared: {pair['r_squared']:.4f}")
                    
                    # Add channel statistics if available
                    ref_channel = self._get_channel_by_name(pair['ref_file'], pair['ref_channel'])
                    test_channel = self._get_channel_by_name(pair['test_file'], pair['test_channel'])
                    
                    if ref_channel and hasattr(ref_channel, 'ydata') and ref_channel.ydata is not None:
                        report_lines.append(f"Reference Data Points: {len(ref_channel.ydata)}")
                        report_lines.append(f"Reference Range: {np.min(ref_channel.ydata):.4f} to {np.max(ref_channel.ydata):.4f}")
                        report_lines.append(f"Reference Mean: {np.mean(ref_channel.ydata):.4f}")
                        report_lines.append(f"Reference Std: {np.std(ref_channel.ydata):.4f}")
                    
                    if test_channel and hasattr(test_channel, 'ydata') and test_channel.ydata is not None:
                        report_lines.append(f"Test Data Points: {len(test_channel.ydata)}")
                        report_lines.append(f"Test Range: {np.min(test_channel.ydata):.4f} to {np.max(test_channel.ydata):.4f}")
                        report_lines.append(f"Test Mean: {np.mean(test_channel.ydata):.4f}")
                        report_lines.append(f"Test Std: {np.std(test_channel.ydata):.4f}")
                    
                    report_lines.append("")
                
                # Add summary statistics
                report_lines.append("SUMMARY STATISTICS")
                report_lines.append("-" * 30)
                
                r_squared_values = [pair.get('r_squared') for pair in checked_pairs 
                                   if pair.get('r_squared') != 'N/A' and pair.get('r_squared') is not None]
                
                if r_squared_values:
                    report_lines.append(f"Average R-squared: {np.mean(r_squared_values):.4f}")
                    report_lines.append(f"R-squared Range: {np.min(r_squared_values):.4f} to {np.max(r_squared_values):.4f}")
                    report_lines.append(f"R-squared Std: {np.std(r_squared_values):.4f}")
                
                # Write report to file
                with open(filename, 'w') as f:
                    f.write('\n'.join(report_lines))
                
                QMessageBox.information(self, "Export Complete", f"Report exported to:\n{filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting report:\n{str(e)}")
    
    def _update_export_button_states(self):
        """Update export button states based on available data"""
        try:
            # Check if we have pairs
            checked_pairs = self.get_checked_pairs()
            has_pairs = len(checked_pairs) > 0
            
            # Check if we have a plot
            has_plot = (hasattr(self, 'comparison_plot_widget') and 
                       self.comparison_plot_widget is not None and
                       hasattr(self.comparison_plot_widget, 'figure'))
            
            # Enable/disable buttons based on available data
            self.export_plot_button.setEnabled(has_plot)
            self.export_data_button.setEnabled(has_pairs)
            self.export_report_button.setEnabled(has_pairs)
            self.export_all_button.setEnabled(has_pairs)
            
        except Exception as e:
            # If there's an error, disable all buttons
            self.export_plot_button.setEnabled(False)
            self.export_data_button.setEnabled(False)
            self.export_report_button.setEnabled(False)
            self.export_all_button.setEnabled(False)
            
    def _export_comparison_result(self):
        """Export comprehensive comparison result (plot + data + report)"""
        try:
            from PySide6.QtWidgets import QFileDialog
            import os
            
            # Check if we have active pairs
            checked_pairs = self.get_checked_pairs()
            if not checked_pairs:
                QMessageBox.warning(self, "No Data", "No comparison pairs selected. Please add and select pairs first.")
                return
            
            # Get directory for export
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Directory for Comparison Export",
                "",
                QFileDialog.ShowDirsOnly
            )
            
            if directory:
                # Create a base filename
                base_name = "comparison_results"
                if len(checked_pairs) == 1:
                    # Use pair name if only one pair
                    pair_name = checked_pairs[0]['pair_name'].replace(' ', '_')
                    base_name = f"{pair_name}_comparison"
                
                # Export plot
                plot_filename = os.path.join(directory, f"{base_name}_plot.png")
                try:
                    if hasattr(self, 'comparison_plot_widget') and hasattr(self.comparison_plot_widget, 'figure'):
                        self.comparison_plot_widget.figure.savefig(plot_filename, dpi=300, bbox_inches='tight')
                except Exception as e:
                    print(f"Warning: Could not export plot: {e}")
                
                # Export data
                data_filename = os.path.join(directory, f"{base_name}_data.csv")
                try:
                    # Use the same logic as _export_data but with fixed filename
                    import pandas as pd
                    import numpy as np
                    
                    export_data = []
                    for pair in checked_pairs:
                        ref_channel = self._get_channel_by_name(pair['ref_file'], pair['ref_channel'])
                        test_channel = self._get_channel_by_name(pair['test_file'], pair['test_channel'])
                        
                        if ref_channel and test_channel:
                            pair_data = {
                                'Pair_Name': pair['pair_name'],
                                'Reference_File': pair['ref_file'],
                                'Reference_Channel': pair['ref_channel'],
                                'Test_File': pair['test_file'],
                                'Test_Channel': pair['test_channel'],
                                'Method': pair.get('method', 'correlation'),
                                'R_Squared': pair.get('r_squared', 'N/A')
                            }
                            
                            # Add raw data if available
                            if hasattr(ref_channel, 'ydata') and hasattr(test_channel, 'ydata'):
                                ref_data = ref_channel.ydata
                                test_data = test_channel.ydata
                                
                                max_len = max(len(ref_data), len(test_data))
                                ref_padded = np.full(max_len, np.nan)
                                test_padded = np.full(max_len, np.nan)
                                
                                ref_padded[:len(ref_data)] = ref_data
                                test_padded[:len(test_data)] = test_data
                                
                                for i in range(max_len):
                                    row = {
                                        **pair_data,
                                        'Data_Index': i,
                                        'Reference_Value': ref_padded[i],
                                        'Test_Value': test_padded[i]
                                    }
                                    export_data.append(row)
                            else:
                                export_data.append(pair_data)
                    
                    df = pd.DataFrame(export_data)
                    df.to_csv(data_filename, index=False)
                    
                except Exception as e:
                    print(f"Warning: Could not export data: {e}")
                
                # Export report
                report_filename = os.path.join(directory, f"{base_name}_report.txt")
                try:
                    import datetime
                    
                    report_lines = []
                    report_lines.append("COMPARISON ANALYSIS REPORT")
                    report_lines.append("=" * 50)
                    report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    report_lines.append(f"Total Pairs: {len(checked_pairs)}")
                    report_lines.append("")
                    
                    if self.method_list.currentItem():
                        method_name = self.method_list.currentItem().text()
                        report_lines.append(f"Analysis Method: {method_name.title()}")
                        report_lines.append("")
                    
                    for i, pair in enumerate(checked_pairs, 1):
                        report_lines.append(f"PAIR {i}: {pair['pair_name']}")
                        report_lines.append("-" * 30)
                        report_lines.append(f"Reference: {pair['ref_file']} - {pair['ref_channel']}")
                        report_lines.append(f"Test: {pair['test_file']} - {pair['test_channel']}")
                        report_lines.append(f"Method: {pair.get('method', 'correlation')}")
                        
                        if 'r_squared' in pair and pair['r_squared'] != 'N/A':
                            report_lines.append(f"R-squared: {pair['r_squared']:.4f}")
                        
                        ref_channel = self._get_channel_by_name(pair['ref_file'], pair['ref_channel'])
                        test_channel = self._get_channel_by_name(pair['test_file'], pair['test_channel'])
                        
                        if ref_channel and hasattr(ref_channel, 'ydata') and ref_channel.ydata is not None:
                            report_lines.append(f"Reference Data Points: {len(ref_channel.ydata)}")
                            report_lines.append(f"Reference Range: {np.min(ref_channel.ydata):.4f} to {np.max(ref_channel.ydata):.4f}")
                            report_lines.append(f"Reference Mean: {np.mean(ref_channel.ydata):.4f}")
                            report_lines.append(f"Reference Std: {np.std(ref_channel.ydata):.4f}")
                        
                        if test_channel and hasattr(test_channel, 'ydata') and test_channel.ydata is not None:
                            report_lines.append(f"Test Data Points: {len(test_channel.ydata)}")
                            report_lines.append(f"Test Range: {np.min(test_channel.ydata):.4f} to {np.max(test_channel.ydata):.4f}")
                            report_lines.append(f"Test Mean: {np.mean(test_channel.ydata):.4f}")
                            report_lines.append(f"Test Std: {np.std(test_channel.ydata):.4f}")
                        
                        report_lines.append("")
                    
                    # Summary statistics
                    report_lines.append("SUMMARY STATISTICS")
                    report_lines.append("-" * 30)
                    
                    r_squared_values = [pair.get('r_squared') for pair in checked_pairs 
                                       if pair.get('r_squared') != 'N/A' and pair.get('r_squared') is not None]
                    
                    if r_squared_values:
                        report_lines.append(f"Average R-squared: {np.mean(r_squared_values):.4f}")
                        report_lines.append(f"R-squared Range: {np.min(r_squared_values):.4f} to {np.max(r_squared_values):.4f}")
                        report_lines.append(f"R-squared Std: {np.std(r_squared_values):.4f}")
                    
                    with open(report_filename, 'w') as f:
                        f.write('\n'.join(report_lines))
                    
                except Exception as e:
                    print(f"Warning: Could not export report: {e}")
                
                # Show success message
                exported_files = []
                if os.path.exists(plot_filename):
                    exported_files.append("Plot (PNG)")
                if os.path.exists(data_filename):
                    exported_files.append("Data (CSV)")
                if os.path.exists(report_filename):
                    exported_files.append("Report (TXT)")
                
                if exported_files:
                    QMessageBox.information(
                        self, 
                        "Export Complete", 
                        f"Comparison results exported to:\n{directory}\n\nFiles created:\n• {chr(10).join(exported_files)}"
                    )
                else:
                    QMessageBox.warning(self, "Export Warning", "No files were successfully exported.")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting comparison results:\n{str(e)}")