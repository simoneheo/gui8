from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup
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
        
        # Initialize parameter table
        self._populate_parameters()
        
        # Initialize with available data
        self._validate_initialization()
        
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
        """Initialize the user interface following process/mixer wizard patterns"""
        self.setWindowTitle("📊 Data Comparison Wizard")
        self.setMinimumSize(1200, 800)
        
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
        
        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([360, 840])
        
    def _build_left_panel(self, main_splitter):
        """Build the left control panel following process wizard patterns"""
        # Create left panel container
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("🔧 Comparison Setup")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        left_layout.addWidget(title_label)
        
        # Create scrollable area for controls
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(10)
        
        # Add control groups
        self._create_channel_selection_group(scroll_layout)
        self._create_comparison_method_group(scroll_layout)
        self._create_parameters_group(scroll_layout)
        self._create_pairs_management_group(scroll_layout)
        self._create_action_buttons_group(scroll_layout)
        
        # Add stretch to push everything to top
        scroll_layout.addStretch()
        
        scroll_area.setWidget(scroll_widget)
        left_layout.addWidget(scroll_area)
        
        main_splitter.addWidget(left_panel)
        
    def _create_channel_selection_group(self, layout):
        """Create channel selection group box"""
        group = QGroupBox("📁 Channel Selection")
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
        
        # Pair name
        self.pair_name_input = QLineEdit()
        self.pair_name_input.setPlaceholderText("Auto-generated from channels")
        group_layout.addRow("Pair Name:", self.pair_name_input)
        
        layout.addWidget(group)
        
    def _create_comparison_method_group(self, layout):
        """Create comparison method selection group"""
        group = QGroupBox("🔬 Comparison Method")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Method list (similar to process wizard filter list)
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(120)
        methods = ["Correlation Analysis", "Bland-Altman", "Residual Analysis", "Statistical Tests"]
        self.method_list.addItems(methods)
        self.method_list.setCurrentRow(0)  # Select first method by default
        group_layout.addWidget(self.method_list)
        
        # Alignment mode
        alignment_layout = QHBoxLayout()
        alignment_layout.addWidget(QLabel("Alignment:"))
        self.alignment_mode_combo = QComboBox()
        self.alignment_mode_combo.addItems(["Index-Based", "Time-Based"])
        alignment_layout.addWidget(self.alignment_mode_combo)
        group_layout.addLayout(alignment_layout)
        
        layout.addWidget(group)
        
    def _create_parameters_group(self, layout):
        """Create parameters configuration group"""
        group = QGroupBox("⚙️ Parameters")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Parameter table (similar to process wizard parameter table)
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.horizontalHeader().setStretchLastSection(True)
        self.param_table.setMaximumHeight(200)
        self.param_table.setAlternatingRowColors(True)
        group_layout.addWidget(self.param_table)
        
        layout.addWidget(group)
        
    def _create_pairs_management_group(self, layout):
        """Create pairs management group (just the add button now)"""
        group = QGroupBox("➕ Add Pair")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Add pair button
        self.add_pair_button = QPushButton("➕ Add Comparison Pair")
        self.add_pair_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
        """)
        group_layout.addWidget(self.add_pair_button)
        
        layout.addWidget(group)
        
    def _create_action_buttons_group(self, layout):
        """Create action buttons group"""
        group = QGroupBox("🚀 Actions")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Generate plot button
        self.generate_plot_button = QPushButton("📊 Generate Plot")
        self.generate_plot_button.setEnabled(False)
        self.generate_plot_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #5dade2;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        group_layout.addWidget(self.generate_plot_button)
        
        layout.addWidget(group)
        
    def _build_right_panel(self, main_splitter):
        """Build the right results panel"""
        # Create right panel container
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)
        
        # Add title
        title_label = QLabel("📈 Results & Visualization")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2c3e50; padding: 5px;")
        right_layout.addWidget(title_label)
        
        # Active pairs table at the top
        pairs_group = QGroupBox("📋 Active Comparison Pairs")
        pairs_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pairs_layout = QVBoxLayout(pairs_group)
        
        # Active pairs table
        self.active_pair_table = QTableWidget()
        self.active_pair_table.setColumnCount(5)
        self.active_pair_table.setHorizontalHeaderLabels(["Show", "Pair Name", "Method", "Correlation", "Status"])
        self.active_pair_table.horizontalHeader().setStretchLastSection(True)
        self.active_pair_table.setMaximumHeight(150)
        self.active_pair_table.setAlternatingRowColors(True)
        self.active_pair_table.setSelectionBehavior(QTableWidget.SelectRows)
        pairs_layout.addWidget(self.active_pair_table)
        
        # Pair management buttons
        button_layout = QHBoxLayout()
        self.delete_pair_button = QPushButton("🗑️ Delete")
        self.delete_pair_button.setEnabled(False)
        self.clear_all_button = QPushButton("🧹 Clear All")
        
        button_layout.addWidget(self.delete_pair_button)
        button_layout.addWidget(self.clear_all_button)
        button_layout.addStretch()
        pairs_layout.addLayout(button_layout)
        
        right_layout.addWidget(pairs_group)
        
        # Create tabbed interface for results
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget)
        
        # Add tabs
        self._create_comparison_plot_tab()
        self._create_statistics_tab()
        self._create_results_tab()
        
        main_splitter.addWidget(right_panel)
        
    def _create_comparison_plot_tab(self):
        """Create the comparison plot tab"""
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        
        # Plot controls (simplified - removed plot type dropdown)
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.StyledPanel)
        controls_layout = QHBoxLayout(controls_frame)
        
        # Plot options
        self.show_grid_checkbox = QCheckBox("Grid")
        self.show_grid_checkbox.setChecked(True)
        self.show_legend_checkbox = QCheckBox("Legend")
        self.show_legend_checkbox.setChecked(True)
        
        controls_layout.addWidget(self.show_grid_checkbox)
        controls_layout.addWidget(self.show_legend_checkbox)
        controls_layout.addStretch()
        
        plot_layout.addWidget(controls_frame)
        
        # Matplotlib canvas
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
        """Connect UI signals following process wizard patterns"""
        # File and channel selection
        self.ref_file_combo.currentTextChanged.connect(self._on_ref_file_changed)
        self.test_file_combo.currentTextChanged.connect(self._on_test_file_changed)
        self.ref_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        self.test_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        
        # Method selection
        self.method_list.itemClicked.connect(self._on_method_selected)
        self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
        
        # Parameter table changes
        self.param_table.itemChanged.connect(self._trigger_plot_update)
        
        # Pair management
        self.add_pair_button.clicked.connect(self._on_add_pair)
        self.delete_pair_button.clicked.connect(self._on_delete_selected_pair)
        self.clear_all_button.clicked.connect(self._on_clear_all_pairs)
        
        # Table selection
        self.active_pair_table.itemSelectionChanged.connect(self._on_pair_selection_changed)
        
        # Plot controls
        self.show_grid_checkbox.stateChanged.connect(self._trigger_plot_update)
        self.show_legend_checkbox.stateChanged.connect(self._trigger_plot_update)
        
        # Action buttons
        self.generate_plot_button.clicked.connect(self._on_generate_plot)
        
    def _on_alignment_mode_changed(self, text):
        """Handle alignment mode changes"""
        # Update parameter table based on alignment mode
        self._populate_parameters()
        
    def _on_method_selected(self, item):
        """Handle comparison method selection (similar to process wizard filter selection)"""
        if not item:
            return
            
        method_name = item.text()
        print(f"[ComparisonWizard] Selected method: {method_name}")
        
        # Update parameter table based on selected method
        self._populate_parameters()
        
        # Trigger plot update when method changes
        self._trigger_plot_update()
        
    def _populate_parameters(self):
        """Populate parameter table based on selected method and alignment mode"""
        # Get selected method
        current_item = self.method_list.currentItem()
        if not current_item:
            return
            
        method_name = current_item.text()
        alignment_mode = self.alignment_mode_combo.currentText()
        
        # Clear existing parameters
        self.param_table.setRowCount(0)
        
        # Add parameters based on method and alignment
        parameters = self._get_method_parameters(method_name, alignment_mode)
        
        self.param_table.setRowCount(len(parameters))
        for i, (param_name, param_config) in enumerate(parameters.items()):
            self.param_table.setItem(i, 0, QTableWidgetItem(param_name))
            
            # Create appropriate widget based on parameter type
            if isinstance(param_config, dict) and 'choices' in param_config:
                # Create dropdown for parameters with choices
                combo = QComboBox()
                combo.addItems(param_config['choices'])
                combo.setCurrentText(str(param_config['default']))
                combo.currentTextChanged.connect(self._trigger_plot_update)
                self.param_table.setCellWidget(i, 1, combo)
            else:
                # Create regular text item for other parameters
                value = param_config['default'] if isinstance(param_config, dict) else param_config
                item = QTableWidgetItem(str(value))
                self.param_table.setItem(i, 1, item)
                # Note: itemChanged signal is connected at table level, not item level
            
    def _get_method_parameters(self, method_name, alignment_mode):
        """Get parameters for the selected method and alignment mode"""
        parameters = {}
        
        # Add alignment-specific parameters
        if alignment_mode == "Index-Based":
            parameters["Start Index"] = {"default": "0", "type": "int"}
            parameters["End Index"] = {"default": "500", "type": "int"}
            parameters["Offset"] = {"default": "0", "type": "int"}
        else:  # Time-Based
            parameters["Start Time"] = {"default": "0.0", "type": "float"}
            parameters["End Time"] = {"default": "10.0", "type": "float"}
            parameters["Time Resolution"] = {"default": "0.01", "type": "float"}
            parameters["Interpolation"] = {
                "default": "linear",
                "choices": ["linear", "cubic", "nearest", "quadratic"],
                "type": "str"
            }
        
        # Add method-specific parameters
        if method_name == "Correlation Analysis":
            parameters["Correlation Type"] = {
                "default": "pearson",
                "choices": ["pearson", "spearman", "kendall", "all"],
                "type": "str"
            }
            parameters["Confidence Level"] = {"default": "0.95", "type": "float"}
            parameters["Bootstrap Samples"] = {"default": "1000", "type": "int"}
        elif method_name == "Bland-Altman":
            parameters["Agreement Limits"] = {"default": "1.96", "type": "float"}
            parameters["Show CI"] = {
                "default": "true",
                "choices": ["true", "false"],
                "type": "bool"
            }
            parameters["Proportional Bias"] = {
                "default": "false",
                "choices": ["true", "false"],
                "type": "bool"
            }
        elif method_name == "Residual Analysis":
            parameters["Normality Test"] = {
                "default": "shapiro",
                "choices": ["shapiro", "kstest", "jarque_bera"],
                "type": "str"
            }
            parameters["Outlier Detection"] = {
                "default": "iqr",
                "choices": ["iqr", "zscore", "modified_zscore"],
                "type": "str"
            }
        elif method_name == "Statistical Tests":
            parameters["Alpha Level"] = {"default": "0.05", "type": "float"}
            parameters["Test Suite"] = {
                "default": "comprehensive",
                "choices": ["basic", "comprehensive", "nonparametric"],
                "type": "str"
            }
            parameters["Equal Variance"] = {
                "default": "test",
                "choices": ["assume_equal", "assume_unequal", "test"],
                "type": "str"
            }
            parameters["Normality"] = {
                "default": "test",
                "choices": ["assume_normal", "assume_nonnormal", "test"],
                "type": "str"
            }
            
        return parameters
        

        
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
        """Handle generate plot button click"""
        checked_pairs = self.get_checked_pairs()
        if not checked_pairs:
            QMessageBox.warning(self, "No Pairs Selected", 
                              "Please select at least one pair to generate a plot.")
            return
            
        # Get plot configuration
        plot_config = self._get_plot_config()
        
        # Emit signal to manager
        self.plot_generated.emit(plot_config)
        
    def _get_plot_config(self):
        """Get current plot configuration"""
        return {
            'show_grid': self.show_grid_checkbox.isChecked(),
            'show_legend': self.show_legend_checkbox.isChecked(),
            'checked_pairs': self.get_checked_pairs()
        }
        
    def _on_clear_all_pairs(self):
        """Clear all comparison pairs"""
        if not self.active_pairs:
            return
            
        reply = QMessageBox.question(
            self, 
            "Clear All Pairs", 
            f"Are you sure you want to remove all {len(self.active_pairs)} comparison pairs?\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.active_pairs.clear()
            self._update_active_pairs_table()
            self.pair_deleted.emit()
            self._log_state_change(f"Cleared all comparison pairs")

    def _on_ref_file_changed(self, filename):
        """Update reference channel combo when file changes"""
        self._update_channel_combo(filename, self.ref_channel_combo)
        
    def _on_test_file_changed(self, filename):
        """Update test channel combo when file changes"""
        self._update_channel_combo(filename, self.test_channel_combo)
        
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
    
    def _on_pair_selection_changed(self):
        """Handle selection changes in the active pairs table"""
        try:
            selected_rows = set()
            for item in self.active_pair_table.selectedItems():
                if item:
                    selected_rows.add(item.row())
            
            # Enable delete button only if exactly one row is selected
            self.delete_pair_button.setEnabled(len(selected_rows) == 1)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error handling pair selection: {str(e)}")
            self.delete_pair_button.setEnabled(False)
    
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
    
    def _on_delete_selected_pair(self):
        """Delete the selected pair from the table"""
        try:
            selected_rows = set()
            for item in self.active_pair_table.selectedItems():
                if item:
                    selected_rows.add(item.row())
            
            if len(selected_rows) != 1:
                QMessageBox.warning(self, "Selection Error", "Please select exactly one pair to delete.")
                return
            
            selected_row = list(selected_rows)[0]
            pair_name_item = self.active_pair_table.item(selected_row, 1)
            
            if not pair_name_item:
                QMessageBox.warning(self, "Delete Error", "Could not identify the selected pair.")
                return
            
            pair_name = pair_name_item.text()
            
            # Find and remove the pair from active_pairs list
            pair_index = -1
            for i, pair in enumerate(self.active_pairs):
                if pair['name'] == pair_name:
                    pair_index = i
                    break
            
            if pair_index >= 0:
                # Confirm deletion
                reply = QMessageBox.question(
                    self, 
                    "Confirm Delete", 
                    f"Are you sure you want to delete the pair:\n\n'{pair_name}'?\n\nThis action cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    # Remove the pair
                    self.active_pairs.pop(pair_index)
                    self._update_active_pairs_table()
                    self.pair_deleted.emit()
                    
                    print(f"[ComparisonWizard] Deleted pair: {pair_name}")
            else:
                QMessageBox.warning(self, "Delete Error", f"Could not find pair '{pair_name}' in the active pairs list.")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error deleting selected pair: {str(e)}")
            QMessageBox.critical(self, "Delete Error", f"An error occurred while deleting the pair:\n\n{str(e)}")
            
    def _get_current_pair_config(self):
        """Get current pair configuration from UI"""
        ref_file = self.ref_file_combo.currentText()
        ref_channel = self.ref_channel_combo.currentText()
        test_file = self.test_file_combo.currentText()
        test_channel = self.test_channel_combo.currentText()
        pair_name = self.pair_name_input.text().strip()
        
        if not all([ref_file, ref_channel, test_file, test_channel]):
            QMessageBox.warning(self, "Warning", "Please select both reference and test channels.")
            return None
            
        if not pair_name:
            # Use placeholder text as default, or generate if placeholder is also empty
            pair_name = self.pair_name_input.placeholderText() or f"{ref_channel} vs {test_channel}"
            
        # Get selected comparison method
        method_item = self.method_list.currentItem()
        method_name = method_item.text() if method_item else "Correlation Analysis"
        
        config = {
            'name': pair_name,
            'ref_file': ref_file,
            'ref_channel': ref_channel,
            'test_file': test_file,
            'test_channel': test_channel,
            'comparison_method': method_name,
            'alignment_mode': 'index' if self.alignment_mode_combo.currentText() == "Index-Based" else 'time',
            'alignment_config': self._get_alignment_config(),
            'method_parameters': self._get_method_parameters_from_table()
        }
        
        return config
        
    def _get_alignment_config(self):
        """Get alignment configuration from parameter table"""
        config = {}
        
        # Extract parameters from the parameter table
        for row in range(self.param_table.rowCount()):
            param_name_item = self.param_table.item(row, 0)
            if not param_name_item:
                continue
                
            param_name = param_name_item.text()
            
            # Get parameter value from either combo box or table item
            widget = self.param_table.cellWidget(row, 1)
            if widget and isinstance(widget, QComboBox):
                # Parameter with dropdown
                param_value = widget.currentText()
            else:
                # Regular text parameter
                param_value_item = self.param_table.item(row, 1)
                param_value = param_value_item.text() if param_value_item else ""
            
            # Convert parameter names to config keys
            if param_name == "Start Index":
                config['start_index'] = int(param_value) if param_value.isdigit() else 0
            elif param_name == "End Index":
                config['end_index'] = int(param_value) if param_value.isdigit() else 500
            elif param_name == "Offset":
                config['offset'] = int(param_value) if param_value.lstrip('-').isdigit() else 0
            elif param_name == "Start Time":
                config['start_time'] = float(param_value) if param_value.replace('.', '').isdigit() else 0.0
            elif param_name == "End Time":
                config['end_time'] = float(param_value) if param_value.replace('.', '').isdigit() else 10.0
            elif param_name == "Time Resolution":
                config['round_to'] = float(param_value) if param_value.replace('.', '').isdigit() else 0.01
            elif param_name == "Interpolation":
                config['interpolation'] = param_value
                    
        return config
        
    def _get_method_parameters_from_table(self):
        """Extract method-specific parameters from the parameter table"""
        parameters = {}
        
        for row in range(self.param_table.rowCount()):
            param_name_item = self.param_table.item(row, 0)
            if not param_name_item:
                continue
                
            param_name = param_name_item.text()
            
            # Skip alignment parameters, only get method-specific ones
            if param_name in ["Start Index", "End Index", "Offset", 
                            "Start Time", "End Time", "Time Resolution", "Interpolation"]:
                continue
            
            # Get parameter value from either combo box or table item
            widget = self.param_table.cellWidget(row, 1)
            if widget and isinstance(widget, QComboBox):
                # Parameter with dropdown
                param_value = widget.currentText()
            else:
                # Regular text parameter
                param_value_item = self.param_table.item(row, 1)
                param_value = param_value_item.text() if param_value_item else ""
            
            parameters[param_name] = param_value
                    
        return parameters
        
    def _update_active_pairs_table(self):
        """Update the active pairs table"""
        self.active_pair_table.setRowCount(len(self.active_pairs))
        
        for i, pair in enumerate(self.active_pairs):
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
            
            # Method
            method_item = QTableWidgetItem(pair.get('comparison_method', 'N/A'))
            self.active_pair_table.setItem(i, 2, method_item)
            
            # Placeholder for statistics
            self.active_pair_table.setItem(i, 3, QTableWidgetItem("--"))  # Correlation
            self.active_pair_table.setItem(i, 4, QTableWidgetItem("Pending"))  # Status
            
        # Enable generate plot button if pairs exist
        self.generate_plot_button.setEnabled(len(self.active_pairs) > 0)
    
    def _on_show_checkbox_changed(self, state):
        """Handle Show checkbox state changes"""
        if hasattr(self, 'comparison_manager'):
            self.comparison_manager._update_cumulative_display()
    
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