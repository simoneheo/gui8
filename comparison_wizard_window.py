from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QDoubleSpinBox, QAbstractItemView, QHeaderView, QFileDialog,
    QColorDialog, QSizePolicy
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
# Import comparison methods from the new comparison folder
try:
    from comparison.comparison_registry import ComparisonRegistry
    from comparison import load_all_comparisons
    COMPARISON_AVAILABLE = True
    print("[ComparisonWizardWindow] Comparison registry imported successfully")
except ImportError as e:
    print(f"[ComparisonWizardWindow] Warning: Could not import comparison registry: {e}")
    COMPARISON_AVAILABLE = False
    
    # Create dummy classes if comparison module is not available
    class ComparisonRegistry:
        @staticmethod
        def get_all_methods():
            return ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis"]
        
        @staticmethod
        def get_all_categories():
            return ["Statistical", "Agreement", "Error Analysis"]
        
        @staticmethod
        def get_methods_by_category(category):
            if category == "Statistical":
                return ["correlation"]
            elif category == "Agreement":
                return ["bland_altman"]
            elif category == "Error Analysis":
                return ["residual"]
            return []
        
        @staticmethod
        def get_method_info(method_name):
            return {
                'name': method_name,
                'description': f'Description for {method_name}',
                'parameters': {},
                'category': 'Statistical'
            }
        
        @staticmethod
        def create_method(method_name, **kwargs):
            return None
            
        @staticmethod
        def get(method_name):
            """Get comparison class by method name"""
            return None
            
        @staticmethod
        def all_comparisons():
            """Get all available comparison methods"""
            return ["correlation", "bland_altman", "residual"]
    
    def load_all_comparisons(directory=None):
        print(f"[ComparisonWizardWindow] Warning: Comparison module not available")
        return False

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
        
        # Script tracking for customization detection
        self.original_plot_script_content = ""
        self.original_stat_script_content = ""
        
        # Add flag to prevent automatic plot updates during method selection
        self._prevent_auto_plot_updates = False
        
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
        
        # Set splitter proportions - wider left panel as requested
        main_splitter.setSizes([800, 400])
        
    def _connect_signals(self):
        """Connect all UI signals to their respective handlers"""
        try:
            # File and channel selection signals
            self.ref_file_combo.currentTextChanged.connect(self._on_ref_file_changed)
            self.test_file_combo.currentTextChanged.connect(self._on_test_file_changed)
            
            # Channel selection change signals for auto-updating pair names
            self.ref_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
            self.test_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
            
            # Method selection signals
            self.method_list.itemClicked.connect(self._on_method_selected)
            self.method_list.currentItemChanged.connect(self._on_method_selection_changed)
            
            # Alignment mode signals
            if hasattr(self, 'alignment_mode_combo'):
                self.alignment_mode_combo.currentTextChanged.connect(self._on_alignment_mode_changed)
            
            # Index mode signals
            if hasattr(self, 'index_mode_combo'):
                self.index_mode_combo.currentTextChanged.connect(self._on_index_mode_changed)
            
            # Time mode signals
            if hasattr(self, 'time_mode_combo'):
                self.time_mode_combo.currentTextChanged.connect(self._on_time_mode_changed)
            
            # Density display signals
            if hasattr(self, 'density_combo'):
                self.density_combo.currentTextChanged.connect(self._on_density_display_changed)
            
            # Bin control signals
            if hasattr(self, 'bins_spinbox'):
                self.bins_spinbox.valueChanged.connect(self._on_density_display_changed)
            
            # Parameter change signals (connect to all parameter controls)
            self._connect_parameter_signals()
            
            # Action button signals
            if hasattr(self, 'add_pair_button'):
                self.add_pair_button.clicked.connect(self._on_add_pair_clicked)
            
            if hasattr(self, 'generate_plot_button'):
                self.generate_plot_button.clicked.connect(self._on_generate_plot)
            
            if hasattr(self, 'refresh_plot_button'):
                self.refresh_plot_button.clicked.connect(self._on_refresh_plot_clicked)
            
            # Export functionality removed - not needed
            
            print("[ComparisonWizard] All signals connected successfully")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error connecting signals: {e}")
    
    def _connect_parameter_signals(self):
        """Connect signals for method parameter controls"""
        try:
            # Connect all parameter controls that might exist
            parameter_controls = [
                'confidence_level_spin', 'correlation_type_combo', 'agreement_limits_spin',
                'proportional_bias_checkbox', 'normality_test_combo', 'outlier_detection_combo',
                'show_ci_checkbox', 'show_loa_checkbox', 'show_bias_checkbox',
                'show_outliers_checkbox', 'show_trend_checkbox', 'show_zero_checkbox'
            ]
            
            for control_name in parameter_controls:
                if hasattr(self, control_name):
                    control = getattr(self, control_name)
                    if hasattr(control, 'valueChanged'):
                        control.valueChanged.connect(self._on_method_parameter_changed)
                    elif hasattr(control, 'currentTextChanged'):
                        control.currentTextChanged.connect(self._on_method_parameter_changed)
                    elif hasattr(control, 'stateChanged'):
                        control.stateChanged.connect(self._on_method_parameter_changed)
                    elif hasattr(control, 'textChanged'):
                        control.textChanged.connect(self._on_method_parameter_changed)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error connecting parameter signals: {e}")
    
    def _populate_file_combos(self):
        """Populate file combo boxes with available files"""
        try:
            if not self.file_manager:
                print("[ComparisonWizard] No file manager available for populating file combos")
                return
            
            files = self.file_manager.get_all_files()
            
            # Clear existing items
            self.ref_file_combo.clear()
            self.test_file_combo.clear()
            
            if not files:
                print("[ComparisonWizard] No files available for comparison")
                return
            
            # Add files to combos
            for file_info in files:
                display_name = file_info.filename
                self.ref_file_combo.addItem(display_name, file_info)
                self.test_file_combo.addItem(display_name, file_info)
            
            print(f"[ComparisonWizard] Loaded {len(files)} files for comparison")
            
            # Set default selections if available
            if len(files) >= 2:
                # Set first file as reference, second as test
                self.ref_file_combo.setCurrentIndex(0)
                self.test_file_combo.setCurrentIndex(1)
            elif len(files) == 1:
                # Set same file for both reference and test
                self.ref_file_combo.setCurrentIndex(0)
                self.test_file_combo.setCurrentIndex(0)
            
            # Update channel combos after file selection
            self._update_channel_combo(self.ref_file_combo.currentText(), self.ref_channel_combo)
            self._update_channel_combo(self.test_file_combo.currentText(), self.test_channel_combo)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error populating file combos: {e}")
    
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
        
        # Left column: Comparison Methods, Method-specific Controls, and Performance Options
        self._create_comparison_method_group(left_col_layout)
        self._create_method_controls_group(left_col_layout)
        self._create_performance_options_group(left_col_layout)
        
        # Add Refresh Plot button at the bottom of left column
        self._create_refresh_plot_button(left_col_layout)
        
        # Remove stretch to let method controls fill all available space
        
        # Right column: Channel Selection, Alignment, and Actions
        self._create_channel_selection_group(right_col_layout)
        self._create_alignment_group(right_col_layout)
        self._create_pairs_management_group(right_col_layout)
        
        # Remove stretch from right column to let console fill available space
        
        # Add columns to splitter
        left_splitter.addWidget(left_col_widget)
        left_splitter.addWidget(right_col_widget)
        
        # Set splitter proportions - method config spans most of the length
        left_splitter.setSizes([200, 650])
        
        main_splitter.addWidget(self.left_panel)
        
    def _create_refresh_plot_button(self, layout):
        """Create refresh plot button at the bottom of the left panel"""
        # Create refresh button
        self.refresh_plot_button = QPushButton("🔄 Refresh Plot")
        self.refresh_plot_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
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
        self.refresh_plot_button.setToolTip("Refresh the plot and table with current method and parameter settings")
        self.refresh_plot_button.clicked.connect(self._on_refresh_plot_clicked)
        
        layout.addWidget(self.refresh_plot_button)
        
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
        
        # Method list (populated from comparison registry) - shorter for ~5 options
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(80)  # Even shorter for compactness
        self._populate_comparison_methods()
        group_layout.addWidget(self.method_list)
        
        layout.addWidget(group)
        
    def _generate_clean_display_name(self, comparison_cls, registry_name):
        """Generate a clean, user-friendly display name from comparison class"""
        if hasattr(comparison_cls, 'description') and comparison_cls.description:
            description = comparison_cls.description
            
            # If description contains ' - ', take the first part
            if ' - ' in description:
                return description.split(' - ')[0]
            
            # If description contains 'analysis' or 'comparison', extract the key part
            desc_lower = description.lower()
            
            # Common patterns to extract clean names
            if desc_lower.startswith('histogram analysis'):
                return 'Error Distribution Histogram'
            elif desc_lower.startswith('time series analysis'):
                return 'Relative Error Time Series'
            elif desc_lower.startswith('cross-correlation analysis'):
                return 'Time Lag Cross-Correlation'
            elif 'correlation' in desc_lower and 'coefficients' in desc_lower:
                return 'Correlation Analysis'
            elif 'bland-altman' in desc_lower or 'bland altman' in desc_lower:
                return 'Bland-Altman Analysis'
            elif 'residual' in desc_lower and 'analysis' in desc_lower:
                return 'Residual Analysis'
            else:
                # Try to extract first few meaningful words
                words = description.split()
                if len(words) >= 2:
                    # Take first 2-3 significant words and add 'Analysis'
                    significant_words = [w for w in words[:3] if len(w) > 2 and w.lower() not in ['of', 'the', 'and', 'with', 'for']]
                    if significant_words:
                        return ' '.join(significant_words[:2]).title() + ' Analysis'
        
        # Fallback: convert registry name to title case
        return registry_name.replace('_', ' ').title() + ' Analysis'
        
    def _populate_comparison_methods(self):
        """Populate comparison methods from the registry"""
        try:
            if COMPARISON_AVAILABLE:
                # Load comparison methods if not already done
                load_all_comparisons()
                
                # Get methods from registry (these are registry names)
                registry_methods = ComparisonRegistry.all_comparisons()
                
                if registry_methods:
                    # Convert registry names to display names using comparison classes
                    display_methods = []
                    for registry_name in registry_methods:
                        try:
                            comparison_cls = ComparisonRegistry.get(registry_name)
                            if comparison_cls:
                                # Generate clean display name
                                display_name = self._generate_clean_display_name(comparison_cls, registry_name)
                                display_methods.append(display_name)
                            else:
                                display_methods.append(registry_name.replace('_', ' ').title() + ' Analysis')
                        except Exception as e:
                            print(f"[ComparisonWizard] Error getting display name for {registry_name}: {e}")
                            display_methods.append(registry_name.replace('_', ' ').title() + ' Analysis')
                    
                    self.method_list.clear()
                    self.method_list.addItems(display_methods)
                    if display_methods:
                        self.method_list.setCurrentRow(0)  # Select first method by default
                        
                    print(f"[ComparisonWizard] Loaded {len(display_methods)} comparison methods from registry")
                    print(f"[ComparisonWizard] Registry names: {registry_methods}")
                    print(f"[ComparisonWizard] Display names: {display_methods}")
                else:
                    # Fallback methods if registry is empty
                    fallback_methods = ["Bland-Altman Analysis", "Correlation Analysis", "Residual Analysis"]
                    self.method_list.clear()
                    self.method_list.addItems(fallback_methods)
                    if fallback_methods:
                        self.method_list.setCurrentRow(0)
                    print("[ComparisonWizard] Using fallback methods - comparison registry may not be loaded")
                
            else:
                # Use fallback methods if comparison module not available
                fallback_methods = ["Bland-Altman Analysis", "Correlation Analysis", "Residual Analysis"]
                self.method_list.clear()
                self.method_list.addItems(fallback_methods)
                if fallback_methods:
                    self.method_list.setCurrentRow(0)
                print(f"[ComparisonWizard] Using {len(fallback_methods)} fallback comparison methods")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error populating methods: {e}")
            # Fallback methods
            methods = ["Bland-Altman Analysis", "Correlation Analysis", "Residual Analysis"]
            self.method_list.clear()
            self.method_list.addItems(methods)
            if methods:
                self.method_list.setCurrentRow(0)
        
    def _create_method_controls_group(self, layout):
        """Create tabbed interface with parameter table, plot script, and stat script"""
        group = QGroupBox("Method Configuration")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Set size policy to expand both horizontally and vertically
        group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Tabbed interface for method configuration - stretch to fill available space
        self.method_tabs = QTabWidget()
        group_layout.addWidget(self.method_tabs)
        
        # Parameters tab - parameter table like process wizard
        self.params_tab = QWidget()
        params_layout = QVBoxLayout(self.params_tab)
        params_layout.setContentsMargins(5, 5, 5, 5)
        
        # Parameter table - 2 columns only, descriptions shown as tooltips
        self.param_table = QTableWidget(0, 2)
        self.param_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.param_table.setAlternatingRowColors(True)
        self.param_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Remove height restriction to let it stretch and fill available space
        
        # Set column resize modes for 2-column layout
        header = self.param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Parameter name - fixed
        header.setSectionResizeMode(1, QHeaderView.Stretch)   # Value - stretches to fill space
        
        # Set column widths
        self.param_table.setColumnWidth(0, 150)  # Parameter - slightly wider since no description column
        
        # Table styling
        self.param_table.setStyleSheet("""
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
        
        params_layout.addWidget(self.param_table)
        self.method_tabs.addTab(self.params_tab, "⚙️ Parameters")
        
        # Plot Script tab - editable plot_script method
        self.plot_script_tab = QWidget()
        plot_script_layout = QVBoxLayout(self.plot_script_tab)
        plot_script_layout.setContentsMargins(5, 5, 5, 5)
        
        # Script editor for plot_script
        self.plot_script_editor = QTextEdit()
        self.plot_script_editor.setPlaceholderText("Select a comparison method to see the plot script")
        self.plot_script_editor.setFont(QFont("Consolas", 10))
        plot_script_layout.addWidget(self.plot_script_editor)
        
        # Plot script controls
        plot_script_controls = QHBoxLayout()
        self.plot_script_readonly = QCheckBox("Read-only")
        self.plot_script_readonly.setChecked(True)
        self.plot_script_readonly.stateChanged.connect(lambda state: self.plot_script_editor.setReadOnly(state == Qt.Checked))
        plot_script_controls.addWidget(self.plot_script_readonly)
        
        plot_script_controls.addStretch()
        
        self.sync_plot_script_btn = QPushButton("Sync from Parameters")
        self.sync_plot_script_btn.clicked.connect(self._sync_plot_script_from_params)
        plot_script_controls.addWidget(self.sync_plot_script_btn)
        
        plot_script_layout.addLayout(plot_script_controls)
        
        self.method_tabs.addTab(self.plot_script_tab, "📊 Plot Script")
        
        # Stat Script tab - editable stats_script method
        self.stat_script_tab = QWidget()
        stat_script_layout = QVBoxLayout(self.stat_script_tab)
        stat_script_layout.setContentsMargins(5, 5, 5, 5)
        
        # Script editor for stats_script
        self.stat_script_editor = QTextEdit()
        self.stat_script_editor.setPlaceholderText("Select a comparison method to see the statistics script")
        self.stat_script_editor.setFont(QFont("Consolas", 10))
        stat_script_layout.addWidget(self.stat_script_editor)
        
        # Stat script controls
        stat_script_controls = QHBoxLayout()
        self.stat_script_readonly = QCheckBox("Read-only")
        self.stat_script_readonly.setChecked(True)
        self.stat_script_readonly.stateChanged.connect(lambda state: self.stat_script_editor.setReadOnly(state == Qt.Checked))
        stat_script_controls.addWidget(self.stat_script_readonly)
        
        stat_script_controls.addStretch()
        
        self.sync_stat_script_btn = QPushButton("Sync from Parameters")
        self.sync_stat_script_btn.clicked.connect(self._sync_stat_script_from_params)
        stat_script_controls.addWidget(self.sync_stat_script_btn)
        
        stat_script_layout.addLayout(stat_script_controls)
        
        self.method_tabs.addTab(self.stat_script_tab, "📈 Stat Script")
        
        # Script tracking for customization detection
        self.original_plot_script_content = ""
        self.original_stat_script_content = ""

        layout.addWidget(group)
        
    def _create_dynamic_method_controls(self):
        """Create method controls dynamically from comparison registry"""
        try:
            if COMPARISON_AVAILABLE:
                # Load comparison methods if not already done
                load_all_comparisons()
                
                # Get methods from registry (these are registry names)
                registry_methods = ComparisonRegistry.all_comparisons()
                
                if registry_methods:
                    # Convert registry names to display names using comparison classes
                    display_methods = []
                    for registry_name in registry_methods:
                        try:
                            comparison_cls = ComparisonRegistry.get(registry_name)
                            if comparison_cls:
                                # Generate clean display name
                                display_name = self._generate_clean_display_name(comparison_cls, registry_name)
                                display_methods.append(display_name)
                            else:
                                display_name = registry_name.replace('_', ' ').title() + ' Analysis'
                                display_methods.append(display_name)
                        except Exception as e:
                            print(f"[ComparisonWizard] Error getting display name for {registry_name}: {e}")
                            display_name = registry_name.replace('_', ' ').title() + ' Analysis'
                            display_methods.append(display_name)
                    
                    print(f"[ComparisonWizard] Creating dynamic controls for methods: {display_methods}")
                    
                    for i, registry_name in enumerate(registry_methods):
                        display_name = display_methods[i]
                        try:
                            # Use registry name directly since we have it
                            comparison_cls = ComparisonRegistry.get(registry_name)
                            if comparison_cls:
                                method_info = comparison_cls().get_info()
                                widget = self._create_controls_for_method(display_name, method_info)
                                self.method_controls_stack.addWidget(widget)
                                print(f"[ComparisonWizard] Created controls for {display_name} ({registry_name})")
                            else:
                                print(f"[ComparisonWizard] No comparison class found for {registry_name}")
                                # Create a simple placeholder widget for this method
                                placeholder = QWidget()
                                placeholder_layout = QVBoxLayout(placeholder)
                                placeholder_layout.addWidget(QLabel(f"No parameters available for {display_name}"))
                                self.method_controls_stack.addWidget(placeholder)
                        except Exception as method_error:
                            print(f"[ComparisonWizard] Error creating controls for {display_name}: {method_error}")
                            # Create a simple placeholder widget for this method
                            placeholder = QWidget()
                            placeholder_layout = QVBoxLayout(placeholder)
                            placeholder_layout.addWidget(QLabel(f"Error loading controls for {display_name}"))
                            self.method_controls_stack.addWidget(placeholder)
                else:
                    print("[ComparisonWizard] No methods available from registry, using static controls")
                    self._create_static_method_controls()
            else:
                print("[ComparisonWizard] Comparison module not available, using static controls")
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
        
    def _create_controls_for_method(self, method_name, method_info):
        """Create controls for a specific comparison method"""
        widget = QWidget()
        layout = QFormLayout(widget)
        
        # Get parameters from the params list (new structure)
        parameters = method_info.get('params', [])
        
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
        
        # Process parameters from params list
        for param_config in parameters:
            param_name = param_config.get('name', '')
            
            # Skip overlay-related parameters
            if param_name in overlay_params:
                continue
                
            control = self._create_parameter_control(param_name, param_config)
            if control:
                # Use the help text as the label
                label_text = param_config.get('help', param_name)
                # Truncate long help text for label
                if len(label_text) > 50:
                    label_text = param_name.replace('_', ' ').title()
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
        param_type = param_config.get('type', 'str')
        default_value = param_config.get('default')
        choices = param_config.get('choices')
        tooltip = param_config.get('help', '')  # Use 'help' field for tooltip
        
        control = None
        
        if choices:
            # Dropdown for choices
            control = QComboBox()
            control.addItems([str(choice) for choice in choices])
            if default_value in choices:
                control.setCurrentText(str(default_value))
                
        elif param_type == 'bool':
            # Checkbox for boolean
            control = QCheckBox()
            control.setChecked(bool(default_value))
            
        elif param_type == 'int':
            # Spinbox for integer
            control = QSpinBox()
            control.setRange(param_config.get('min', -999999), param_config.get('max', 999999))
            control.setValue(int(default_value) if default_value is not None else 0)
            
        elif param_type == 'float':
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
        self.corr_type_combo.currentTextChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Correlation Type:", self.corr_type_combo)
        
        # Bootstrap samples (computational parameter)
        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(100, 10000)
        self.bootstrap_spin.setValue(1000)
        self.bootstrap_spin.valueChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Bootstrap Samples:", self.bootstrap_spin)
        
        # Detrend method (computational parameter)
        self.detrend_combo = QComboBox()
        self.detrend_combo.addItems(["none", "linear", "polynomial"])
        self.detrend_combo.currentTextChanged.connect(self._on_method_parameter_changed)
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
        self.agreement_spin.valueChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Agreement Multiplier:", self.agreement_spin)
        
        # Percentage difference option (computational parameter)
        self.percentage_diff_checkbox = QCheckBox()
        self.percentage_diff_checkbox.setChecked(False)
        self.percentage_diff_checkbox.stateChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Percentage Differences:", self.percentage_diff_checkbox)
        
        # Log transform option (computational parameter)
        self.log_transform_checkbox = QCheckBox()
        self.log_transform_checkbox.setChecked(False)
        self.log_transform_checkbox.stateChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Log Transform:", self.log_transform_checkbox)
        
        # Proportional bias test (computational parameter)
        self.prop_bias_checkbox = QCheckBox()
        self.prop_bias_checkbox.setChecked(True)
        self.prop_bias_checkbox.stateChanged.connect(self._on_method_parameter_changed)
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
        self.residual_type_combo.currentTextChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Residual Type:", self.residual_type_combo)
        
        # Normality test (computational parameter)
        self.normality_combo = QComboBox()
        self.normality_combo.addItems(["shapiro", "kstest", "jarque_bera", "anderson", "all"])
        self.normality_combo.currentTextChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Normality Test:", self.normality_combo)
        
        # Trend analysis (computational parameter)
        self.trend_analysis_checkbox = QCheckBox()
        self.trend_analysis_checkbox.setChecked(True)
        self.trend_analysis_checkbox.stateChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Trend Analysis:", self.trend_analysis_checkbox)
        
        # Autocorrelation test (computational parameter)
        self.autocorr_checkbox = QCheckBox()
        self.autocorr_checkbox.setChecked(True)
        self.autocorr_checkbox.stateChanged.connect(self._on_method_parameter_changed)
        layout.addRow("Autocorrelation Test:", self.autocorr_checkbox)
        
        # Note: All display options (outliers, trend lines, statistics) moved to overlay section
        
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
        self.alignment_status_label.setStyleSheet("color: #666; font-size: 11px; padding: 5px;")
        group_layout.addWidget(self.alignment_status_label)
        
        # Show/hide appropriate groups (call after all widgets are created)
        self._on_alignment_mode_changed("Index-Based")
        
        layout.addWidget(group)
        
    def _create_pairs_management_group(self, layout):
        """Create pairs management group with name input, Add Pair button, and console output"""
        group = QGroupBox("Add Pair")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
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
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        self.add_pair_btn.clicked.connect(self._on_add_pair_clicked)
        group_layout.addWidget(self.add_pair_btn)
        
        # Console output section - moved from right panel
        console_label = QLabel("Console Output:")
        console_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        group_layout.addWidget(console_label)
        
        # Create console output widget - stretched to fill available space
        self.info_output = QTextEdit()
        self.info_output.setReadOnly(True)
        self.info_output.setPlaceholderText("Logs and messages will appear here")
        # Remove maximum height to let it stretch
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
        """Build overlay legend section with table format"""
        # Overlay legend group
        legend_group = QGroupBox("Overlays")
        legend_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        legend_layout = QVBoxLayout(legend_group)
        
        # Create overlay table
        self.overlay_table = QTableWidget(0, 4)
        self.overlay_table.setHorizontalHeaderLabels(["Show", "Style", "Name", "Actions"])
        self.overlay_table.setAlternatingRowColors(True)
        self.overlay_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.overlay_table.setMaximumHeight(200)
        
        # Set column resize modes
        header = self.overlay_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width  
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Name - stretches
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set column widths
        self.overlay_table.setColumnWidth(0, 50)   # Show checkbox
        self.overlay_table.setColumnWidth(1, 80)   # Style preview
        self.overlay_table.setColumnWidth(3, 100)  # Actions buttons
        
        # Table styling
        self.overlay_table.setStyleSheet("""
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
        
        legend_layout.addWidget(self.overlay_table)
        
        # Initialize overlay configurations
        self.overlay_configs = {}
        self.overlay_artists = {}
        
        # Default color schemes for different methods
        self.default_overlay_styles = {
            'bland_altman': {
                'bias_line': {
                    'name': 'Bias Line',
                    'color': '#27ae60',
                    'linestyle': '-',
                    'linewidth': 2,
                    'alpha': 0.8,
                    'default_show': True
                },
                'limits_of_agreement': {
                    'name': 'Limits of Agreement',
                    'color': '#f39c12',
                    'linestyle': '--',
                    'linewidth': 2,
                    'alpha': 0.8,
                    'default_show': True
                },
                'confidence_intervals': {
                    'name': 'Confidence Intervals',
                    'color': '#3498db',
                    'linestyle': '-',
                    'linewidth': 0,
                    'alpha': 0.3,
                    'fill': True,
                    'default_show': True
                },
                'outliers': {
                    'name': 'Outliers',
                    'color': '#e74c3c',
                    'marker': 'o',
                    'markersize': 6,
                    'alpha': 0.8,
                    'default_show': False
                },
                'statistical_results': {
                    'name': 'Statistical Results',
                    'color': '#2c3e50',
                    'fontsize': 10,
                    'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
                    'default_show': False
                }
            },
            'correlation': {
                'identity_line': {
                    'name': 'Identity Line',
                    'color': '#95a5a6',
                    'linestyle': '-',
                    'linewidth': 1,
                    'alpha': 0.8,
                    'default_show': False
                },
                'regression_line': {
                    'name': 'Regression Line',
                    'color': '#3498db',
                    'linestyle': '-',
                    'linewidth': 2,
                    'alpha': 0.8,
                    'default_show': False
                },
                'confidence_bands': {
                    'name': 'Confidence Bands',
                    'color': '#bdc3c7',
                    'linestyle': '-',
                    'linewidth': 0,
                    'alpha': 0.3,
                    'fill': True,
                    'default_show': False
                },
                'outliers': {
                    'name': 'Outliers',
                    'color': '#e74c3c',
                    'marker': 'o',
                    'markersize': 6,
                    'alpha': 0.8,
                    'default_show': False
                },
                'statistical_results': {
                    'name': 'Statistical Results',
                    'color': '#2c3e50',
                    'fontsize': 10,
                    'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
                    'default_show': False
                }
            },
            'residual': {
                'zero_line': {
                    'name': 'Zero Line',
                    'color': '#95a5a6',
                    'linestyle': '--',
                    'linewidth': 1,
                    'alpha': 0.8,
                    'default_show': True
                },
                'trend_line': {
                    'name': 'Trend Line',
                    'color': '#3498db',
                    'linestyle': '-',
                    'linewidth': 2,
                    'alpha': 0.8,
                    'default_show': False
                },
                'outliers': {
                    'name': 'Outliers',
                    'color': '#e74c3c',
                    'marker': 'o',
                    'markersize': 6,
                    'alpha': 0.8,
                    'default_show': False
                },
                'statistical_results': {
                    'name': 'Statistical Results',
                    'color': '#2c3e50',
                    'fontsize': 10,
                    'bbox': {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8},
                    'default_show': False
                }
            }
        }
        
        layout.addWidget(legend_group)
    
    def _build_plot_area(self, layout):
        """Build the plot area with matplotlib canvas and toolbar"""
        # Plot group
        plot_group = QGroupBox("Plot")
        plot_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        plot_layout = QVBoxLayout(plot_group)
        
        # Create matplotlib figure and canvas
        self.figure = plt.figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(400, 300)
        
        # Create navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Create initial empty plot
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Select channels and add pairs to generate plots', 
               ha='center', va='center', transform=ax.transAxes,
               fontsize=12, color='gray')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Create additional plot canvases for different views
        self._create_additional_plot_canvases()
        
        # Add tab widget to layout instead of individual canvas
        plot_layout.addWidget(self.plot_tab_widget)
        
        # Add plot group to main layout
        layout.addWidget(plot_group)
    
    def _create_additional_plot_canvases(self):
        """Create additional plot canvases for histogram and heatmap views"""
        # Histogram canvas
        self.histogram_figure = plt.figure(figsize=(6, 4), dpi=100)
        self.histogram_canvas = FigureCanvas(self.histogram_figure)
        self.histogram_canvas.setMinimumSize(300, 200)
        self.histogram_toolbar = NavigationToolbar(self.histogram_canvas, self)
        
        # Heatmap canvas
        self.heatmap_figure = plt.figure(figsize=(6, 4), dpi=100)
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        self.heatmap_canvas.setMinimumSize(300, 200)
        self.heatmap_toolbar = NavigationToolbar(self.heatmap_canvas, self)
        
        # Create tab widget for multiple plot views - remove height restriction to fill space
        self.plot_tabs = QTabWidget()
        # Remove setMaximumHeight to let it stretch and fill available space
        
        # Main plot tab
        main_plot_widget = QWidget()
        main_plot_layout = QVBoxLayout(main_plot_widget)
        main_plot_layout.addWidget(self.toolbar)
        main_plot_layout.addWidget(self.canvas)
        self.plot_tabs.addTab(main_plot_widget, "Main Plot")
        
        # Histogram tab
        histogram_widget = QWidget()
        histogram_layout = QVBoxLayout(histogram_widget)
        histogram_layout.addWidget(self.histogram_toolbar)
        histogram_layout.addWidget(self.histogram_canvas)
        self.plot_tabs.addTab(histogram_widget, "Histogram")
        
        # Heatmap tab
        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        heatmap_layout.addWidget(self.heatmap_toolbar)
        heatmap_layout.addWidget(self.heatmap_canvas)
        self.plot_tabs.addTab(heatmap_widget, "Heatmap")
        
        # Add tab widget to the plot area (this will be added in _build_plot_area)
        # We'll store it for later use
        self.plot_tab_widget = self.plot_tabs
    
    def _populate_overlay_table(self, method_name):
        """Populate overlay table based on selected comparison method"""
        try:
            # Save current overlay visibility states before clearing
            current_visibility_states = {}
            if hasattr(self, 'overlay_table') and self.overlay_table.rowCount() > 0:
                current_visibility_states = self._get_overlay_visibility_states()
                print(f"[ComparisonWizard] Preserving overlay visibility states: {current_visibility_states}")
            
            # Clear existing rows
            self.overlay_table.setRowCount(0)
            
            # Get method registry name
            method_registry_name = self._get_registry_name_from_display_name(method_name)
            if not method_registry_name:
                return
                
            # Get overlays from the comparison method - no fallback
            method_overlays = {}
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                method_info = self.comparison_manager.get_method_info(method_name)
                if method_info and 'overlay_options' in method_info:
                    # Convert overlay_options to overlay config format
                    overlay_options = method_info['overlay_options']
                    for overlay_id, overlay_option in overlay_options.items():
                        method_overlays[overlay_id] = {
                            'name': overlay_option.get('label', overlay_id.replace('_', ' ').title()),
                            'default_show': overlay_option.get('default', True),
                            'tooltip': overlay_option.get('tooltip', ''),
                            'color': '#3498db',  # Default color
                            'linestyle': '-',    # Default line style
                            'linewidth': 2,      # Default line width
                            'alpha': 0.8         # Default alpha
                        }
            
            # Populate table with overlays
            for overlay_id, overlay_config in method_overlays.items():
                row = self.overlay_table.rowCount()
                self.overlay_table.insertRow(row)
                
                # Show checkbox - preserve current state if available, otherwise use method's default
                show_checkbox = QCheckBox()
                if overlay_id in current_visibility_states:
                    # Use preserved state
                    show_checkbox.setChecked(current_visibility_states[overlay_id])
                    print(f"[ComparisonWizard] Restored overlay {overlay_id} visibility: {current_visibility_states[overlay_id]}")
                else:
                    # Use method's default or True
                    show_checkbox.setChecked(overlay_config.get('default_show', True))
                    print(f"[ComparisonWizard] Using default overlay {overlay_id} visibility: {overlay_config.get('default_show', True)}")
                
                show_checkbox.stateChanged.connect(lambda state, oid=overlay_id: self._on_overlay_visibility_changed(oid, state))
                self.overlay_table.setCellWidget(row, 0, show_checkbox)
                
                # Style preview
                style_preview = self._create_style_preview_widget(overlay_config)
                self.overlay_table.setCellWidget(row, 1, style_preview)
                
                # Name
                name_item = QTableWidgetItem(overlay_config.get('name', overlay_id))
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                name_item.setToolTip(overlay_config.get('tooltip', ''))
                self.overlay_table.setItem(row, 2, name_item)
                
                # Actions
                actions_widget = self._create_overlay_actions_widget(overlay_id)
                self.overlay_table.setCellWidget(row, 3, actions_widget)
                
                # Store overlay configuration
                self.overlay_configs[overlay_id] = overlay_config.copy()
                
            print(f"[ComparisonWizard] Populated overlay table with {len(method_overlays)} overlays for {method_name}")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error populating overlay table for {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_style_preview_widget(self, overlay_config):
        """Create style preview widget showing color and line style"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Color swatch
        color_swatch = QLabel()
        color_swatch.setFixedSize(16, 16)
        color = overlay_config.get('color', '#000000')
        color_swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #ccc;")
        layout.addWidget(color_swatch)
        
        # Line style indicator
        line_style = overlay_config.get('linestyle', '-')
        if line_style == '-':
            style_text = "——"
        elif line_style == '--':
            style_text = "- -"
        elif line_style == ':':
            style_text = "···"
        elif line_style == '-.':
            style_text = "-·-"
        else:
            style_text = "——"
            
        # Handle fill overlays
        if overlay_config.get('fill', False):
            style_text = "▓▓▓"
        elif 'marker' in overlay_config:
            style_text = "●●●"
        elif 'fontsize' in overlay_config:  # Text overlay
            style_text = "T"
            
        style_label = QLabel(style_text)
        style_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(style_label)
        
        layout.addStretch()
        return widget
    
    def _create_overlay_actions_widget(self, overlay_id):
        """Create actions widget with paint, info, and trash buttons"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Paint button
        paint_btn = QPushButton("🎨")
        paint_btn.setFixedSize(24, 24)
        paint_btn.setToolTip("Change overlay style")
        paint_btn.clicked.connect(lambda: self._on_overlay_paint_clicked(overlay_id))
        layout.addWidget(paint_btn)
        
        # Info button
        info_btn = QPushButton("ℹ️")
        info_btn.setFixedSize(24, 24)
        info_btn.setToolTip("Show overlay information")
        info_btn.clicked.connect(lambda: self._on_overlay_info_clicked(overlay_id))
        layout.addWidget(info_btn)
        
        # Trash button
        trash_btn = QPushButton("🗑️")
        trash_btn.setFixedSize(24, 24)
        trash_btn.setToolTip("Remove overlay")
        trash_btn.clicked.connect(lambda: self._on_overlay_trash_clicked(overlay_id))
        layout.addWidget(trash_btn)
        
        return widget
    
    def _on_overlay_visibility_changed(self, overlay_id, state):
        """Handle overlay visibility toggle"""
        try:
            is_visible = state == Qt.Checked
            print(f"[ComparisonWizard] Overlay {overlay_id} visibility changed to {is_visible}")
            
            # Update overlay configuration
            if overlay_id in self.overlay_configs:
                self.overlay_configs[overlay_id]['visible'] = is_visible
            
            # Toggle matplotlib artist visibility if it exists
            if overlay_id in self.overlay_artists:
                artist = self.overlay_artists[overlay_id]
                if hasattr(artist, 'set_visible'):
                    artist.set_visible(is_visible)
                elif isinstance(artist, list):
                    # Handle multiple artists
                    for a in artist:
                        if hasattr(a, 'set_visible'):
                            a.set_visible(is_visible)
            
            # Trigger visual-only plot update
            self._trigger_visual_plot_update()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error handling overlay visibility change: {e}")
    
    def _on_overlay_paint_clicked(self, overlay_id):
        """Handle paint button click - open overlay wizard"""
        try:
            print(f"[ComparisonWizard] Paint clicked for overlay: {overlay_id}")
            
            if overlay_id not in self.overlay_configs:
                return
                
            config = self.overlay_configs[overlay_id].copy()
            
            # Import and open overlay wizard
            from overlay_wizard import OverlayWizard
            
            # Create and show overlay wizard
            wizard = OverlayWizard(overlay_id, config, self)
            wizard.overlay_updated.connect(self._on_overlay_wizard_updated)
            
            result = wizard.exec()
            
            if result == QDialog.Accepted:
                print(f"[ComparisonWizard] Overlay wizard accepted for {overlay_id}")
            else:
                print(f"[ComparisonWizard] Overlay wizard cancelled for {overlay_id}")
                
        except ImportError as e:
            print(f"[ComparisonWizard] Could not import overlay wizard: {e}")
            # Fallback to simple color picker
            self._fallback_color_picker(overlay_id)
        except Exception as e:
            print(f"[ComparisonWizard] Error handling paint click: {e}")
    
    def _on_overlay_wizard_updated(self, overlay_id, updated_config):
        """Handle overlay wizard updates"""
        try:
            print(f"[ComparisonWizard] Overlay wizard updated {overlay_id}")
            
            # Update overlay configuration
            self.overlay_configs[overlay_id] = updated_config
            
            # Update style preview in table
            self._update_overlay_style_preview(overlay_id)
            
            # Update matplotlib artist if it exists
            self._update_matplotlib_artist(overlay_id, updated_config)
            
            # Trigger visual-only plot update
            self._trigger_visual_plot_update()
            
        except Exception as e:
            print(f"[ComparisonWizard] Error handling overlay wizard update: {e}")
    
    def _update_matplotlib_artist(self, overlay_id, config):
        """Update matplotlib artist properties"""
        try:
            if overlay_id not in self.overlay_artists:
                return
                
            artist = self.overlay_artists[overlay_id]
            
            # Handle single artist
            if hasattr(artist, 'set_color'):
                if 'color' in config:
                    artist.set_color(config['color'])
                if 'alpha' in config:
                    artist.set_alpha(config['alpha'])
                if 'linewidth' in config and hasattr(artist, 'set_linewidth'):
                    artist.set_linewidth(config['linewidth'])
                if 'linestyle' in config and hasattr(artist, 'set_linestyle'):
                    artist.set_linestyle(config['linestyle'])
            
            # Handle multiple artists
            elif isinstance(artist, list):
                for a in artist:
                    if hasattr(a, 'set_color') and 'color' in config:
                        a.set_color(config['color'])
                    if hasattr(a, 'set_alpha') and 'alpha' in config:
                        a.set_alpha(config['alpha'])
                    if hasattr(a, 'set_linewidth') and 'linewidth' in config:
                        a.set_linewidth(config['linewidth'])
                    if hasattr(a, 'set_linestyle') and 'linestyle' in config:
                        a.set_linestyle(config['linestyle'])
                        
        except Exception as e:
            print(f"[ComparisonWizard] Error updating matplotlib artist: {e}")
    
    def _fallback_color_picker(self, overlay_id):
        """Fallback color picker when overlay wizard is not available"""
        try:
            config = self.overlay_configs[overlay_id]
            
            # Open simple color picker
            current_color = QColor(config.get('color', '#000000'))
            color = QColorDialog.getColor(current_color, self, f"Choose color for {config.get('name', overlay_id)}")
            
            if color.isValid():
                # Update overlay configuration
                config['color'] = color.name()
                
                # Update style preview
                self._update_overlay_style_preview(overlay_id)
                
                # Update matplotlib artist
                self._update_matplotlib_artist(overlay_id, config)
                
                # Trigger visual-only plot update
                self._trigger_visual_plot_update()
                
        except Exception as e:
            print(f"[ComparisonWizard] Error in fallback color picker: {e}")
    
    def _on_overlay_info_clicked(self, overlay_id):
        """Handle info button click - show overlay information"""
        try:
            print(f"[ComparisonWizard] Info clicked for overlay: {overlay_id}")
            
            if overlay_id not in self.overlay_configs:
                return
                
            config = self.overlay_configs[overlay_id]
            
            # Create info popup
            info_text = f"<b>{config.get('name', overlay_id)}</b><br><br>"
            info_text += f"Color: {config.get('color', 'N/A')}<br>"
            info_text += f"Line Style: {config.get('linestyle', 'N/A')}<br>"
            info_text += f"Alpha: {config.get('alpha', 'N/A')}<br>"
            
            # Add method-specific information
            if overlay_id == 'bias_line':
                info_text += "<br>Shows the mean difference between methods."
            elif overlay_id == 'limits_of_agreement':
                info_text += "<br>Shows the range where 95% of differences lie."
            elif overlay_id == 'confidence_intervals':
                info_text += "<br>Shows uncertainty around bias and limits."
            elif overlay_id == 'outliers':
                info_text += "<br>Highlights data points outside normal range."
            elif overlay_id == 'statistical_results':
                info_text += "<br>Displays numerical statistics on the plot."
            
            QMessageBox.information(self, "Overlay Information", info_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error handling info click: {e}")
    
    def _on_overlay_trash_clicked(self, overlay_id):
        """Handle trash button click - remove overlay"""
        try:
            print(f"[ComparisonWizard] Trash clicked for overlay: {overlay_id}")
            
            if overlay_id not in self.overlay_configs:
                return
            
            config = self.overlay_configs[overlay_id]
            
            # Confirm deletion
            reply = QMessageBox.question(self, "Remove Overlay", 
                                       f"Remove {config.get('name', overlay_id)} overlay?",
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                # Find and remove row from table
                for row in range(self.overlay_table.rowCount()):
                    name_item = self.overlay_table.item(row, 2)
                    if name_item and name_item.text() == config.get('name', overlay_id):
                        self.overlay_table.removeRow(row)
                        break
                
                # Remove from configurations
                if overlay_id in self.overlay_configs:
                    del self.overlay_configs[overlay_id]
                
                # Remove matplotlib artist if it exists
                if overlay_id in self.overlay_artists:
                    artist = self.overlay_artists[overlay_id]
                    if hasattr(artist, 'remove'):
                        artist.remove()
                    elif isinstance(artist, list):
                        for a in artist:
                            if hasattr(a, 'remove'):
                                a.remove()
                    del self.overlay_artists[overlay_id]
                
                # Trigger visual-only plot update
                self._trigger_visual_plot_update()
                
        except Exception as e:
            print(f"[ComparisonWizard] Error handling trash click: {e}")
    
    def _update_overlay_style_preview(self, overlay_id):
        """Update style preview widget for specific overlay"""
        try:
            if overlay_id not in self.overlay_configs:
                return
                
            config = self.overlay_configs[overlay_id]
            
            # Find the row for this overlay
            for row in range(self.overlay_table.rowCount()):
                name_item = self.overlay_table.item(row, 2)
                if name_item and name_item.text() == config.get('name', overlay_id):
                    # Update style preview widget
                    style_preview = self._create_style_preview_widget(config)
                    self.overlay_table.setCellWidget(row, 1, style_preview)
                    break
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error updating style preview: {e}")
    
    def _trigger_visual_plot_update(self):
        """Trigger visual-only plot update without recomputation"""
        try:
            # Regenerate the plot with updated overlay settings
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                checked_pairs = self.get_checked_pairs()
                if checked_pairs:
                    plot_config = self._get_plot_config()
                    plot_type = self.comparison_manager._determine_plot_type_from_pairs(checked_pairs)
                    plot_config['plot_type'] = plot_type
                    plot_config['checked_pairs'] = checked_pairs
                    
                    # Include overlay configurations in plot config
                    self.comparison_manager._enhance_plot_config_with_overlays(plot_config)
                    
                    # Regenerate the plot
                    self.comparison_manager._generate_multi_pair_plot(checked_pairs, plot_config)
                    
                    print(f"[ComparisonWizard] Visual plot update complete with overlay changes")
                else:
                    print(f"[ComparisonWizard] No checked pairs for visual update")
            else:
                print(f"[ComparisonWizard] No comparison manager available for visual update")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error triggering visual plot update: {e}")
    
    def _update_legend_display(self):
        """Update legend display when comparison data changes"""
        try:
            # This method is for updating legend display, not overlay table
            # Overlay table is now only updated on refresh plot
            # TODO: Add actual legend display update logic here if needed
            pass
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating legend display: {e}")
    
    def _get_overlay_visibility_states(self):
        """Get current visibility states of all overlays"""
        visibility_states = {}
        
        for row in range(self.overlay_table.rowCount()):
            show_checkbox = self.overlay_table.cellWidget(row, 0)
            name_item = self.overlay_table.item(row, 2)
            
            if show_checkbox and name_item:
                # Convert name back to overlay_id
                overlay_name = name_item.text()
                
                for overlay_id, config in self.overlay_configs.items():
                    config_name = config.get('name', overlay_id)
                    if config_name == overlay_name:
                        visibility_states[overlay_id] = show_checkbox.isChecked()
                        break
        
        return visibility_states

    def _on_method_selection_changed(self):
        """Handle method selection changes (for programmatic selection)"""
        selected_items = self.method_list.selectedItems()
        if selected_items:
            self._on_method_selected(selected_items[0])

    def _on_overlay_changed(self):
        """Handle overlay option changes - visual only, no recomputation"""
        print("[ComparisonWizard] Overlay option changed, updating visuals only...")
        
        # Update legend display immediately when overlay options change
        self._update_legend_display()
        
        # Trigger visual-only plot update (no cache invalidation)
        if hasattr(self, 'comparison_manager') and self.comparison_manager and len(self.active_pairs) > 0:
            checked_pairs = self.get_checked_pairs()
            if checked_pairs:
                plot_config = self._get_plot_config()
                plot_type = self.comparison_manager._determine_plot_type_from_pairs(checked_pairs)
                plot_config['plot_type'] = plot_type
                plot_config['checked_pairs'] = checked_pairs
                self.comparison_manager._generate_scatter_plot(checked_pairs, plot_config)
                print(f"[ComparisonWizard] Visual overlay update complete for {len(checked_pairs)} pairs")
        else:
            print("[ComparisonWizard] No active pairs, skipping plot update")
            
    def _on_method_parameter_changed(self):
        """Handle method parameter changes - triggers cache invalidation"""
        if hasattr(self, 'comparison_manager') and self.comparison_manager:
            current_item = self.method_list.currentItem()
            if current_item:
                method_name = current_item.text()
                method_params = self._get_method_parameters_from_controls()
                self.comparison_manager.on_method_parameters_changed(method_name, method_params)
                print(f"[ComparisonWizard] Method parameters changed, cache invalidated")

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
        """Handle changes in density display type and bin count"""
        if hasattr(self, 'density_combo'):
            density_type = self.density_combo.currentText().lower()
            
            # Enable/disable bin control based on density type
            if hasattr(self, 'bins_spinbox'):
                # Enable bins for hexbin and KDE, disable for scatter
                enable_bins = density_type in ['hexbin', 'kde']
                self.bins_spinbox.setEnabled(enable_bins)
                
                # Update tooltip based on density type
                if density_type == 'hexbin':
                    self.bins_spinbox.setToolTip("Number of hexagonal bins for hexbin display")
                elif density_type == 'kde':
                    self.bins_spinbox.setToolTip("Grid resolution for KDE density estimation")
                else:  # scatter
                    self.bins_spinbox.setToolTip("Bins not applicable for scatter plots")
            
            # Legacy bin size input support
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
        
        # Check for same-channel comparison (highly unfavorable)
        ref_file = self.ref_file_combo.currentText()
        test_file = self.test_file_combo.currentText()
        ref_channel_name = self.ref_channel_combo.currentText()
        test_channel_name = self.test_channel_combo.currentText()
        
        if (ref_file == test_file and ref_channel_name == test_channel_name):
            self.alignment_status_label.setText("⚠ Comparing same channel - unfavorable for analysis (zero variance)")
            self.alignment_status_label.setStyleSheet("color: #ff6b35; font-size: 10px; padding: 5px; font-weight: bold;")
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
        """Handle method selection change - now triggers cache invalidation and script sync"""
        if not item:
            return
            
        # Set flag to prevent automatic plot updates during method selection
        self._prevent_auto_plot_updates = True
        
        try:
            # Get method name
            method_name = item.text()
            
            # Populate parameter table for the selected method
            self._populate_parameter_table(method_name)
            
            # Update console with method information
            self._update_console_for_method(method_name)
            
            # Update pair name to include new method
            self._update_default_pair_name()
            
            # Sync scripts when method is selected
            try:
                print(f"[ComparisonWizard] Syncing scripts for method: {method_name}")
                self._sync_plot_script_from_params()
                self._sync_stat_script_from_params()
                print(f"[ComparisonWizard] Scripts synchronized successfully")
            except Exception as e:
                print(f"[ComparisonWizard] Error syncing scripts: {e}")
                import traceback
                traceback.print_exc()
            
            # Mark cache as needing invalidation but don't trigger plot regeneration
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                method_params = self._get_parameter_table_values()
                # Just invalidate cache without triggering plot regeneration
                from comparison_wizard_manager import create_method_hash
                new_method_hash = create_method_hash(method_name, method_params)
                if self.comparison_manager.current_method_hash != new_method_hash:
                    self.comparison_manager.pair_cache.invalidate_all()
                    self.comparison_manager.current_method_hash = new_method_hash
                    print(f"[ComparisonWizard] Method changed to '{method_name}', cache invalidated (plot regeneration deferred)")
                
        finally:
            # Reset flag to allow plot updates again
            self._prevent_auto_plot_updates = False



    def _get_registry_name_from_display_name(self, display_name):
        """Map display names to registry names using the comparison registry"""
        if COMPARISON_AVAILABLE:
            try:
                # Check each registered method to find matching display name
                for registry_name in ComparisonRegistry.all_comparisons():
                    comparison_cls = ComparisonRegistry.get(registry_name)
                    if comparison_cls:
                        # Generate display name using the same logic as _generate_clean_display_name
                        method_display_name = self._generate_clean_display_name(comparison_cls, registry_name)
                        
                        if method_display_name == display_name:
                            return registry_name
                            
                # Fallback: convert display name to registry name
                fallback_name = display_name.lower().replace(' analysis', '').replace(' ', '_').replace('-', '_')
                return fallback_name
            except Exception as e:
                print(f"[ComparisonWizard] Error converting display name {display_name}: {e}")
                fallback_name = display_name.lower().replace(' analysis', '').replace(' ', '_').replace('-', '_')
                return fallback_name
        
        # Fallback conversion if registry not available
        fallback_name = display_name.lower().replace(' analysis', '').replace(' ', '_').replace('-', '_')
        return fallback_name





        # Store all overlay widgets for easy show/hide management
        self.overlay_widgets = {
            'y_equals_x': self.y_equals_x_checkbox,
            'ci': self.ci_checkbox,
            'outlier': self.outlier_checkbox,
            'bias_line': self.bias_line_checkbox,
            'loa': self.loa_checkbox,
            'regression_line': self.regression_line_checkbox,
            'error_bands': self.error_bands_checkbox,
            'density_overlay': self.density_overlay_checkbox,
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
                
                # Populate overlay table (same as refresh plot)
                self._populate_overlay_table(method_name)
                
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
    
    def _on_refresh_plot_clicked(self):
        """Handle refresh plot button click - refresh table and plot with current settings"""
        try:
            print("[ComparisonWizard] Refresh plot button clicked")
            
            # Check if we have any active pairs
            if not self.active_pairs:
                console_msg = "No comparison pairs to refresh. Please add pairs first."
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"ℹ️ {console_msg}")
                return
            
            # Get checked pairs
            checked_pairs = self.get_checked_pairs()
            if not checked_pairs:
                console_msg = "No pairs selected for display. Please check pairs in the table."
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"ℹ️ {console_msg}")
                return
            
            # Get current method and parameters
            current_item = self.method_list.currentItem()
            if not current_item:
                console_msg = "No comparison method selected. Please select a method."
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"⚠️ {console_msg}")
                return
            
            method_name = current_item.text()
            method_params = self._get_method_parameters_from_controls()
            
            print(f"[ComparisonWizard] Refresh plot - Method: {method_name}")
            print(f"[ComparisonWizard] Refresh plot - Parameters: {method_params}")
            
            # Populate overlay table BEFORE invalidating cache (same as _on_method_selected)
            self._populate_overlay_table(method_name)
            
            # Update all pairs with current method and parameters
            for pair in self.active_pairs:
                pair['comparison_method'] = method_name
                pair['method_parameters'] = method_params
            
            # Notify comparison manager about method parameter changes to invalidate cache
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                self.comparison_manager.on_method_parameters_changed(method_name, method_params)
            
            # Update table display
            self._update_active_pairs_table()
            
            # Use manager's refresh method for comprehensive refresh
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                # Update status for all pairs
                for row in range(self.active_pair_table.rowCount()):
                    status_item = self.active_pair_table.item(row, 3)
                    if status_item:
                        status_item.setText("Refreshing...")
                
                # Get plot config with updated parameters
                plot_config = self._get_plot_config()
                plot_config['method_parameters'] = method_params
                
                # Use the proper plot generation method with updated parameters
                self.comparison_manager._on_plot_generated(plot_config)
                
                # Add results to console
                self._output_comparison_results_to_console(checked_pairs, method_name, method_params)
                
                # Update status after generation
                for row in range(self.active_pair_table.rowCount()):
                    status_item = self.active_pair_table.item(row, 3)
                    if status_item:
                        status_item.setText("Complete")
                
                console_msg = f"Plot refreshed with {method_name} method and {len(checked_pairs)} pairs"
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"🔄 {console_msg}")
            else:
                console_msg = "Comparison manager not available for plot refresh"
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"⚠️ {console_msg}")
                
        except Exception as e:
            error_msg = f"Error refreshing plot: {str(e)}"
            print(f"[ComparisonWizard] {error_msg}")
            self.info_output.append(f"❌ {error_msg}")
            QMessageBox.warning(self, "Refresh Error", error_msg)
        
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
            
        # Add bin count for density displays
        if hasattr(self, 'bins_spinbox'):
            config['bins'] = self.bins_spinbox.value()
            config['hexbin_gridsize'] = self.bins_spinbox.value()  # For hexbin plots
            config['hist_bins'] = self.bins_spinbox.value()  # For histogram plots
            
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
        """Update default pair name based on selected channels and method"""
        try:
            ref_channel = self.ref_channel_combo.currentText()
            test_channel = self.test_channel_combo.currentText()
            
            if ref_channel and test_channel:
                # Get current method name
                method_name = "Analysis"  # Default
                current_item = self.method_list.currentItem()
                if current_item:
                    method_name = current_item.text().replace(" Analysis", "")
                
                # Create method-aware pair name
                default_name = f"{ref_channel}_vs_{test_channel}_{method_name}"
                
                # Only update if the field is empty (user hasn't entered custom name)
                if not self.pair_name_input.text().strip():
                    self.pair_name_input.setPlaceholderText(default_name)
                    
                # Also auto-fill the field with the name (user can edit it)
                self.pair_name_input.setText(default_name)
                
        except Exception as e:
            print(f"[ComparisonWizard] Error updating default pair name: {str(e)}")
    
    def _on_channel_selection_changed(self):
        """Handle channel selection changes to auto-populate alignment parameters and pair name"""
        try:
            self._auto_populate_alignment_parameters()
            self._update_default_pair_name()
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
    
    def _on_add_pair_clicked(self):
        """Add a new comparison pair and automatically generate plot"""
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
            self.info_output.append(console_msg)
            
            # Automatically generate plot with the selected comparison method
            self._on_generate_plot()
            
            # Clear pair name for next entry and update placeholder
            self.pair_name_input.clear()
            self._update_default_pair_name()
            
            # Update legend display
            self._update_legend_display()
    
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
            self.info_output.append(f"⚠️ {console_msg}")
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
                self.info_output.append(f"⚠️ {console_msg}")
                return None
            elif ref_length == 0:
                console_msg = f"Cannot add comparison pair - reference channel '{ref_channel}' is empty"
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"⚠️ {console_msg}")
                return None
            elif test_length == 0:
                console_msg = f"Cannot add comparison pair - test channel '{test_channel}' is empty"
                print(f"[ComparisonWizard] {console_msg}")
                self.info_output.append(f"⚠️ {console_msg}")
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
        """Get current method parameters from parameter table"""
        return self._get_parameter_table_values()
    
    def _get_method_parameters_from_controls_old(self):
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
                    

        except Exception as e:
            print(f"[ComparisonWizard] Error getting static control parameters: {e}")
        
        # Add overlay parameters from the overlay section
        overlay_params = self._get_overlay_parameters()
        parameters.update(overlay_params)
                    
        return parameters
    
    def _get_overlay_parameters(self):
        """Get overlay parameters from overlay table - direct passthrough from method definitions"""
        overlay_params = {}
        
        try:
            # Get overlay visibility states from table
            visibility_states = self._get_overlay_visibility_states()
            
            # Direct passthrough - overlay_id = parameter_name
            overlay_params.update(visibility_states)
            
            # Add default confidence level
            overlay_params['confidence_level'] = 0.95
            
            # Add overlay style configurations
            overlay_params['overlay_styles'] = self.overlay_configs
            
            print(f"[ComparisonWizard] Overlay parameters: {overlay_params}")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error getting overlay parameters: {e}")
            
        return overlay_params

    def _update_active_pairs_table(self):
        """Update the active pairs table (like mixer wizard)"""
        self.active_pair_table.setRowCount(len(self.active_pairs))
        
        for i, pair in enumerate(self.active_pairs):
            # Update pair style information - ensure these are set for channel creation
            pair['marker_type'] = self._get_style_for_pair(pair)
            pair['marker_color'] = self._get_color_for_pair(pair)
            
            # Create tooltip for the pair
            tooltip = self._create_pair_tooltip(pair)
            
            # Column 0: Show checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, row=i, cb=checkbox: self._on_show_checkbox_changed(state, row, cb))
            
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
    
    def _on_show_checkbox_changed(self, state, row=None, checkbox=None):
        """Handle show checkbox change for comparison pairs"""
        print(f"[ComparisonWizard] Show checkbox changed, state: {state}, row: {row}")
        
        # Use the provided checkbox or try to get sender as fallback
        sender = checkbox if checkbox else self.sender()
        if not sender:
            print("[ComparisonWizard] No sender found for checkbox change")
            return
        
        pair_name = None
        
        # Find the pair name from the checkbox
        for table_row in range(self.active_pair_table.rowCount()):
            checkbox_widget = self.active_pair_table.cellWidget(table_row, 0)
            if checkbox_widget:
                for child in checkbox_widget.children():
                    if isinstance(child, QCheckBox) and child == sender:
                        pair_name_item = self.active_pair_table.item(table_row, 2)
                        if pair_name_item:
                            pair_name = pair_name_item.text()
                            break
                if pair_name:
                    break
        
        if pair_name and hasattr(self, 'comparison_manager') and self.comparison_manager:
            visible = state == Qt.Checked
            
            # Direct comparison pair visibility handling - regenerate the plot
            print(f"[ComparisonWizard] Setting pair '{pair_name}' visibility: {visible}")
            
            # Get all checked pairs (this will include/exclude based on checkboxes)
            checked_pairs = self.get_checked_pairs()
            
            if checked_pairs:
                # Regenerate plot with current visibility states
                plot_config = self._get_plot_config()
                plot_config['checked_pairs'] = checked_pairs
                
                # Use the comparison manager to regenerate the plot
                self.comparison_manager._on_plot_generated(plot_config)
                
                print(f"[ComparisonWizard] Regenerated plot with {len(checked_pairs)} visible pairs")
            else:
                # Clear all plots if no pairs are visible
                self.comparison_manager._clear_all_plots()
                print(f"[ComparisonWizard] Cleared plots - no visible pairs")
                
            # Update legend display
            self._update_legend_display()
        else:
            print(f"[ComparisonWizard] Could not find pair name for checkbox change (row: {row})")
    
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
            
            if COMPARISON_AVAILABLE:
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
            else:
                print("[ComparisonWizard] Comparison module not available, using fallback methods")
                self._populate_comparison_methods()
            
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
        
        # Use pair name hash to assign marker consistently (stable across table updates)
        pair_name = pair.get('name', '')
        pair_hash = hash(pair_name) % len(marker_types)
        marker_type = marker_types[pair_hash]
        
        return marker_type
    
    def _get_color_for_pair(self, pair):
        """Get color for a comparison pair"""
        # Define color palette that matches the manager's color_map
        color_types = [
            '🔵 Blue', '🔴 Red', '🟢 Green', '🟣 Purple', '🟠 Orange', 
            '🟤 Brown', '🩷 Pink', '⚫ Gray', '🟡 Yellow', '🔶 Cyan'
        ]
        
        # Use pair name hash to assign color consistently (stable across table updates)
        pair_name = pair.get('name', '')
        pair_hash = hash(pair_name) % len(color_types)
        color_type = color_types[pair_hash]
        
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
            self.pair_deleted.emit()
            print(f"[ComparisonWizard] Pair '{removed_pair['name']}' deleted")
            
            # Automatically refresh the plot to reflect the deletion
            if len(self.active_pairs) > 0:
                # If there are still pairs remaining, refresh the plot
                self._on_refresh_plot_clicked()
                self.info_output.append(f"🗑️ Pair '{removed_pair['name']}' deleted - plot refreshed")
            else:
                # If no pairs remain, clear the plot
                if hasattr(self, 'comparison_manager') and self.comparison_manager:
                    self.comparison_manager._clear_all_plots()
                self.info_output.append(f"🗑️ Pair '{removed_pair['name']}' deleted - plot cleared")

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
                    self.info_output.append(helpful_info)
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
            self.info_output.append(description)
            
        except Exception as e:
            self.info_output.append(f"Error loading method information: {str(e)}")

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
        
        # Use the actual marker and color that will be used in the plot
        marker_text = pair.get('marker_type', '○ Circle')
        color_text = pair.get('marker_color', '🔵 Blue')
        
        # Color mapping (same as in comparison_wizard_manager.py)
        color_map = {
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
        
        # Get the actual color that will be used
        color_hex = color_map.get(color_text, '#1f77b4')
        color = QColor(color_hex)
        
        # Draw the marker based on the type
        pen = QPen(color, 2)
        brush = QBrush(color)
        painter.setPen(pen)
        painter.setBrush(brush)
        
        center_x, center_y = 10, 10
        size = 6
        
        if marker_text == '○ Circle':
            painter.drawEllipse(center_x - size//2, center_y - size//2, size, size)
        elif marker_text == '□ Square':
            painter.drawRect(center_x - size//2, center_y - size//2, size, size)
        elif marker_text == '△ Triangle':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            triangle = QPolygon([
                QPoint(center_x, center_y - size//2),
                QPoint(center_x - size//2, center_y + size//2),
                QPoint(center_x + size//2, center_y + size//2)
            ])
            painter.drawPolygon(triangle)
        elif marker_text == '◇ Diamond':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            diamond = QPolygon([
                QPoint(center_x, center_y - size//2),
                QPoint(center_x + size//2, center_y),
                QPoint(center_x, center_y + size//2),
                QPoint(center_x - size//2, center_y)
            ])
            painter.drawPolygon(diamond)
        elif marker_text == '▽ Inverted Triangle':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            triangle = QPolygon([
                QPoint(center_x, center_y + size//2),
                QPoint(center_x - size//2, center_y - size//2),
                QPoint(center_x + size//2, center_y - size//2)
            ])
            painter.drawPolygon(triangle)
        elif marker_text == '◁ Left Triangle':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            triangle = QPolygon([
                QPoint(center_x - size//2, center_y),
                QPoint(center_x + size//2, center_y - size//2),
                QPoint(center_x + size//2, center_y + size//2)
            ])
            painter.drawPolygon(triangle)
        elif marker_text == '▷ Right Triangle':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            triangle = QPolygon([
                QPoint(center_x + size//2, center_y),
                QPoint(center_x - size//2, center_y - size//2),
                QPoint(center_x - size//2, center_y + size//2)
            ])
            painter.drawPolygon(triangle)
        elif marker_text == '⬟ Pentagon':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            import math
            pentagon = QPolygon()
            for i in range(5):
                angle = 2 * math.pi * i / 5 - math.pi / 2
                x = center_x + size//2 * math.cos(angle)
                y = center_y + size//2 * math.sin(angle)
                pentagon.append(QPoint(int(x), int(y)))
            painter.drawPolygon(pentagon)
        elif marker_text == '✦ Star':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            import math
            star = QPolygon()
            for i in range(10):
                angle = 2 * math.pi * i / 10 - math.pi / 2
                radius = (size//2) if i % 2 == 0 else (size//4)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                star.append(QPoint(int(x), int(y)))
            painter.drawPolygon(star)
        elif marker_text == '⬢ Hexagon':
            from PySide6.QtGui import QPolygon
            from PySide6.QtCore import QPoint
            import math
            hexagon = QPolygon()
            for i in range(6):
                angle = 2 * math.pi * i / 6
                x = center_x + size//2 * math.cos(angle)
                y = center_y + size//2 * math.sin(angle)
                hexagon.append(QPoint(int(x), int(y)))
            painter.drawPolygon(hexagon)
        else:
            # Default to circle for unknown markers
            painter.drawEllipse(center_x - size//2, center_y - size//2, size, size)
        
        painter.end()
        
        # Create label with the pixmap
        label = QLabel()
        label.setPixmap(pixmap)
        label.setFixedSize(20, 20)
        label.setToolTip(f"Marker: {marker_text}, Color: {color_text}")
        
        return label

    def _initialize_default_method_selection(self):
        """Initialize default method selection and populate all related components"""
        try:
            # Ensure methods are populated first
            self._populate_comparison_methods()
            
            # Get the first method if available
            if self.method_list.count() > 0:
                # Set the first method as selected
                self.method_list.setCurrentRow(0)
                current_item = self.method_list.currentItem()
                
                if current_item:
                    method_name = current_item.text()
                    print(f"[ComparisonWizard] Setting default method: {method_name}")
                    
                    # Manually trigger the method selection logic to populate everything
                    self._on_method_selected(current_item)
                    
                    # Set the method controls stack to show the first method
                    if hasattr(self, 'method_controls_stack') and self.method_controls_stack.count() > 0:
                        self.method_controls_stack.setCurrentIndex(0)
                    
                    # Ensure the parameter table is populated with default values
                    self._populate_parameter_table(method_name)
                    
                    # Ensure the overlay table is populated with default values
                    self._populate_overlay_table(method_name)
                    
                    # Sync the scripts for the default method
                    self._sync_plot_script_from_params()
                    self._sync_stat_script_from_params()
                    
                    print(f"[ComparisonWizard] Default method '{method_name}' initialized successfully")
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error initializing default method: {e}")
            import traceback
            traceback.print_exc()

    def _output_comparison_results_to_console(self, checked_pairs, method_name, method_params):
        """Output comparison results and statistics to console"""
        try:
            if not checked_pairs:
                return
                
            self.info_output.append("\n" + "="*50)
            self.info_output.append(f"📊 COMPARISON RESULTS - {method_name}")
            self.info_output.append("="*50)
            
            for i, pair in enumerate(checked_pairs):
                pair_name = pair.get('name', f'Pair {i+1}')
                self.info_output.append(f"\n🔍 {pair_name}:")
                
                # Show basic pair info
                ref_channel = pair.get('ref_channel', 'Unknown')
                test_channel = pair.get('test_channel', 'Unknown')
                self.info_output.append(f"   Reference: {ref_channel}")
                self.info_output.append(f"   Test: {test_channel}")
                
                # Show R² if available
                if pair.get('r_squared') is not None:
                    self.info_output.append(f"   R² = {pair['r_squared']:.4f}")
                
                # Show alignment method
                alignment_info = pair.get('alignment_config', {})
                alignment_method = alignment_info.get('alignment_method', 'index')
                self.info_output.append(f"   Alignment: {alignment_method}-based")
                
                # Method-specific results
                if method_name == "Correlation Analysis":
                    corr_type = method_params.get('correlation_type', 'pearson')
                    self.info_output.append(f"   Correlation Type: {corr_type}")
                    if method_params.get('remove_outliers'):
                        self.info_output.append("   ⚠️ Outliers removed from analysis")
                        
                elif method_name in ["Bland-Altman", "Bland-Altman Analysis"]:
                    agreement_mult = method_params.get('agreement_multiplier', 1.96)
                    self.info_output.append(f"   Agreement Limits: ±{agreement_mult}σ")
                    if method_params.get('percentage_difference'):
                        self.info_output.append("   Using percentage differences")
                    if method_params.get('test_proportional_bias'):
                        self.info_output.append("   Testing for proportional bias")
                        
                elif method_name == "Residual Analysis":
                    residual_type = method_params.get('residual_type', 'linear')
                    self.info_output.append(f"   Residual Type: {residual_type}")
                    if method_params.get('trend_analysis'):
                        self.info_output.append("   ✓ Trend analysis enabled")
                        

            
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
                self.info_output.append(f"\n📈 Plot Overlays: {', '.join(overlay_info)}")
            
            self.info_output.append("\n✅ Plot generation completed successfully!")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error outputting results to console: {e}")
            self.info_output.append(f"\n⚠️ Error displaying results: {str(e)}")

    def _set_initial_console_message(self):
        """Set initial helpful console message on wizard startup"""
        try:
            # Clear the console first
            self.info_output.clear()
            
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
            
            self.info_output.setPlainText(welcome_msg)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting initial console message: {e}")
            # Fallback to a simple message
            self.info_output.setPlainText("Welcome to the Data Comparison Wizard!\n\nSelect channels and click 'Add Comparison Pair' to begin analysis.")



    def _sync_plot_script_from_params(self):
        """Generate plot script from current method or expose method's plot_script"""
        try:
            print(f"[ComparisonWizard] _sync_plot_script_from_params() called")
            
            # Get current method
            current_item = self.method_list.currentItem()
            if not current_item:
                self.plot_script_editor.setPlainText("# No comparison method selected\n# Select a method from the list to generate script")
                return
            
            method_name = self._get_registry_name_from_display_name(current_item.text())
            print(f"[ComparisonWizard] Generating plot script for method: {method_name}")
            
            # Get current parameters
            try:
                params = self._get_method_parameters_from_controls()
                print(f"[ComparisonWizard] Parameters extracted: {params}")
            except Exception as param_e:
                print(f"[ComparisonWizard] Error getting parameters: {param_e}")
                self.plot_script_editor.setPlainText(f"# Error getting parameters: {param_e}\n# Please check the parameter controls")
                return
            
            # Try to get the method's actual plot_script
            script = self._generate_plot_script_from_method(method_name, params)
            if script:
                self.plot_script_editor.setPlainText(script)
                self.original_plot_script_content = script
                print(f"[ComparisonWizard] Plot script exposed successfully")
                return
            
            # Fallback to generated script
            try:
                script = self._generate_plot_script_template(method_name, params)
                print(f"[ComparisonWizard] Generated plot script template successfully")
            except Exception as gen_e:
                print(f"[ComparisonWizard] Error generating plot script template: {gen_e}")
                script = self._generate_basic_plot_script(method_name, params)
            
            self.plot_script_editor.setPlainText(script)
            self.original_plot_script_content = script
            print(f"[ComparisonWizard] Plot script set in editor successfully")
                    
        except Exception as e:
            print(f"[ComparisonWizard] CRITICAL ERROR in _sync_plot_script_from_params: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.plot_script_editor.setPlainText(f"# Plot script generation failed: {e}")
            except:
                pass

    def _sync_stat_script_from_params(self):
        """Generate stat script from current method or expose method's stats_script"""
        try:
            print(f"[ComparisonWizard] _sync_stat_script_from_params() called")
            
            # Get current method
            current_item = self.method_list.currentItem()
            if not current_item:
                self.stat_script_editor.setPlainText("# No comparison method selected\n# Select a method from the list to generate script")
                return
            
            method_name = self._get_registry_name_from_display_name(current_item.text())
            print(f"[ComparisonWizard] Generating stat script for method: {method_name}")
            
            # Get current parameters
            try:
                params = self._get_method_parameters_from_controls()
                print(f"[ComparisonWizard] Parameters extracted: {params}")
            except Exception as param_e:
                print(f"[ComparisonWizard] Error getting parameters: {param_e}")
                self.stat_script_editor.setPlainText(f"# Error getting parameters: {param_e}\n# Please check the parameter controls")
                return
            
            # Try to get the method's actual stats_script
            script = self._generate_stat_script_from_method(method_name, params)
            if script:
                self.stat_script_editor.setPlainText(script)
                self.original_stat_script_content = script
                print(f"[ComparisonWizard] Stat script exposed successfully")
                return
            
            # Fallback to generated script
            try:
                script = self._generate_stat_script_template(method_name, params)
                print(f"[ComparisonWizard] Generated stat script template successfully")
            except Exception as gen_e:
                print(f"[ComparisonWizard] Error generating stat script template: {gen_e}")
                script = self._generate_basic_stat_script(method_name, params)
            
            self.stat_script_editor.setPlainText(script)
            self.original_stat_script_content = script
            print(f"[ComparisonWizard] Stat script set in editor successfully")
                    
        except Exception as e:
            print(f"[ComparisonWizard] CRITICAL ERROR in _sync_stat_script_from_params: {e}")
            import traceback
            traceback.print_exc()
            try:
                self.stat_script_editor.setPlainText(f"# Stat script generation failed: {e}")
            except:
                pass

    def _generate_plot_script_from_method(self, method_name, params):
        """Extract and format the method's actual plot_script for user editing"""
        try:
            # Get method class and create instance
            method_class = ComparisonRegistry.get(method_name)
            if not method_class:
                return None
            
            # Create instance with parameters
            method_instance = method_class(**params)
            if not method_instance:
                return None
            
            # Check if method has plot_script method
            if not hasattr(method_instance, 'plot_script'):
                return None
            
            # Get the source code of the plot_script method
            import inspect
            try:
                source_lines = inspect.getsource(method_instance.plot_script).split('\n')
            except (OSError, TypeError):
                return None
            
            # Find the method definition
            start_line = None
            for i, line in enumerate(source_lines):
                if line.strip().startswith('def plot_script('):
                    start_line = i
                    break
            
            if start_line is None:
                return None
            
            # Extract method body (skip the def line)
            method_body = source_lines[start_line + 1:]
            
            # Remove method indentation
            adjusted_lines = self._remove_method_indentation(method_body)
            
            # Format parameters for display
            param_info = []
            if params:
                for key, value in params.items():
                    param_info.append(f"# {key}: {value}")
            else:
                param_info.append("# No parameters")
            
            # Convert method body to standalone script by replacing return statements
            standalone_lines = []
            for line in adjusted_lines:
                line_stripped = line.strip()
                if line_stripped.startswith('return '):
                    # Convert return statement to variable assignments
                    return_expr = line_stripped[7:]  # Remove 'return '
                    if ',' in return_expr:
                        # Multiple return values (e.g., return x_data, y_data, metadata)
                        standalone_lines.append(f"{line[:line.find(line_stripped)]}# Return values assigned to variables:")
                        standalone_lines.append(f"{line[:line.find(line_stripped)]}x_data, y_data, metadata = {return_expr}")
                    else:
                        # Single return value
                        standalone_lines.append(f"{line[:line.find(line_stripped)]}# Return value assigned to result:")
                        standalone_lines.append(f"{line[:line.find(line_stripped)]}result = {return_expr}")
                else:
                    standalone_lines.append(line)
            
            # Create the user-friendly script
            script_parts = [
                f"# {method_name} plot_script - edit to customize plotting",
                "# Available: ref_data, test_data, params",
                "# Must define: x_data, y_data, metadata",
                "",
                "import numpy as np",
                "import scipy.stats as stats",
                "",
                "# Parameters:",
                *param_info,
                "",
                "# Original plot_script logic (converted to standalone script):",
            ]
            
            # Add the converted method body
            script_parts.extend(standalone_lines)
            
            return "\n".join(script_parts)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error extracting plot script: {e}")
            return None

    def _generate_stat_script_from_method(self, method_name, params):
        """Extract and format the method's actual stats_script for user editing"""
        try:
            # Get method class and create instance
            method_class = ComparisonRegistry.get(method_name)
            if not method_class:
                return None
            
            # Create instance with parameters
            method_instance = method_class(**params)
            if not method_instance:
                return None
            
            # Check if method has stats_script method
            if not hasattr(method_instance, 'stats_script'):
                return None
            
            # Get the source code of the stats_script method
            import inspect
            try:
                source_lines = inspect.getsource(method_instance.stats_script).split('\n')
            except (OSError, TypeError):
                return None
            
            # Find the method definition and skip the entire signature
            start_line = None
            for i, line in enumerate(source_lines):
                if line.strip().startswith('def stats_script('):
                    start_line = i
                    break
            
            if start_line is None:
                return None
            
            # Find the end of the method signature (look for the line ending with ':')
            signature_end = start_line
            for i in range(start_line, len(source_lines)):
                if source_lines[i].rstrip().endswith(':'):
                    signature_end = i
                    break
            
            # Extract method body (skip the entire method signature)
            method_body = source_lines[signature_end + 1:]
            
            # Remove docstring if present
            cleaned_body = self._remove_docstring_from_method_body(method_body)
            
            # Remove method indentation
            adjusted_lines = self._remove_method_indentation(cleaned_body)
            
            # Format parameters for display
            param_info = []
            if params:
                for key, value in params.items():
                    param_info.append(f"# {key}: {value}")
            else:
                param_info.append("# No parameters")
            
            # Convert method body to standalone script by replacing return statements
            standalone_lines = []
            for line in adjusted_lines:
                line_stripped = line.strip()
                if line_stripped.startswith('return '):
                    # Convert return statement to variable assignment
                    return_expr = line_stripped[7:]  # Remove 'return '
                    standalone_lines.append(f"{line[:line.find(line_stripped)]}# Return value assigned to stats_results:")
                    standalone_lines.append(f"{line[:line.find(line_stripped)]}stats_results = {return_expr}")
                else:
                    standalone_lines.append(line)
            
            # Create the user-friendly script
            script_parts = [
                f"# {method_name} stats_script - edit to customize statistics",
                "# Available: x_data, y_data, ref_data, test_data, params",
                "# Must define: stats_results (dict with statistical results)",
                "",
                "import numpy as np",
                "import scipy.stats as stats",
                "",
                "# Parameters:",
                *param_info,
                "",
                "# Original stats_script logic (converted to standalone script):",
            ]
            
            # Add the converted method body
            script_parts.extend(standalone_lines)
            
            return "\n".join(script_parts)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error extracting stat script: {e}")
            return None

    def _remove_docstring_from_method_body(self, method_body):
        """Remove docstring from the beginning of method body"""
        try:
            if not method_body:
                return method_body
            
            # Find the first non-empty line
            first_code_line = 0
            for i, line in enumerate(method_body):
                if line.strip():
                    first_code_line = i
                    break
            
            # Check if the first non-empty line starts a docstring
            first_line = method_body[first_code_line].strip()
            if first_line.startswith('"""') or first_line.startswith("'''"):
                quote_type = first_line[:3]
                
                # Check if docstring ends on the same line
                if first_line.count(quote_type) >= 2:
                    # Single-line docstring
                    return method_body[first_code_line + 1:]
                else:
                    # Multi-line docstring - find the end
                    for i in range(first_code_line + 1, len(method_body)):
                        if quote_type in method_body[i]:
                            return method_body[i + 1:]
                    
                    # If we can't find the end, return original
                    return method_body
            
            return method_body
            
        except Exception as e:
            print(f"[ComparisonWizard] Error removing docstring: {e}")
            return method_body

    def _remove_method_indentation(self, method_body):
        """Remove common indentation from method body"""
        try:
            # Find minimum indentation from non-empty lines
            min_indent = float('inf')
            non_empty_lines = [line for line in method_body if line.strip()]
            
            if not non_empty_lines:
                return method_body
            
            for line in non_empty_lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    min_indent = min(min_indent, indent)
            
            # Remove common indentation
            if min_indent < float('inf') and min_indent > 0:
                adjusted_lines = []
                for line in method_body:
                    if line.strip():
                        # Only remove min_indent characters if the line has at least that much indentation
                        if len(line) >= min_indent and line[:min_indent].isspace():
                            adjusted_lines.append(line[min_indent:])
                        else:
                            # Line has less indentation than expected, keep as is
                            adjusted_lines.append(line.lstrip())
                    else:
                        # Empty line - keep as is
                        adjusted_lines.append('')
            else:
                adjusted_lines = method_body
            
            # Remove empty lines at the end
            while adjusted_lines and not adjusted_lines[-1].strip():
                adjusted_lines.pop()
            
            return adjusted_lines
            
        except Exception as e:
            print(f"[ComparisonWizard] Error removing indentation: {e}")
            return method_body

    def _generate_plot_script_template(self, method_name, params):
        """Generate a template plot script when method doesn't have plot_script"""
        try:
            param_lines = []
            for key, value in params.items():
                if isinstance(value, str):
                    param_lines.append(f"    '{key}': '{value}'")
                else:
                    param_lines.append(f"    '{key}': {value}")
            
            params_str = "{\n" + ",\n".join(param_lines) + "\n}" if param_lines else "{}"
            
            script = f"""# {method_name} plot script template
# Edit this script to customize the plotting transformation
# Available: ref_data, test_data, params
# Must return: (x_data, y_data, metadata)

import numpy as np

# Parameters:
params = {params_str}

# Basic plot transformation (customize as needed)
# Example for scatter plot:
x_data = ref_data  # or (ref_data + test_data) / 2 for mean
y_data = test_data  # or test_data - ref_data for differences

# Metadata for plotting
metadata = {{
    'x_label': 'Reference Data',
    'y_label': 'Test Data',
    'title': '{method_name} Plot',
    'plot_type': 'scatter'
}}

# Return tuple: (x_data, y_data, metadata)
"""
            return script
            
        except Exception as e:
            print(f"[ComparisonWizard] Error generating plot script template: {e}")
            return self._generate_basic_plot_script(method_name, params)

    def _generate_stat_script_template(self, method_name, params):
        """Generate a template stat script when method doesn't have stats_script"""
        try:
            param_lines = []
            for key, value in params.items():
                if isinstance(value, str):
                    param_lines.append(f"    '{key}': '{value}'")
                else:
                    param_lines.append(f"    '{key}': {value}")
            
            params_str = "{\n" + ",\n".join(param_lines) + "\n}" if param_lines else "{}"
            
            script = f"""# {method_name} statistics script template
# Edit this script to customize the statistical calculations
# Available: x_data, y_data, ref_data, test_data, params
# Must return: dict with statistical results

import numpy as np
import scipy.stats as stats

# Parameters:
params = {params_str}

# Basic statistical calculations (customize as needed)
# Example calculations:
correlation = np.corrcoef(ref_data, test_data)[0, 1]
bias = np.mean(test_data - ref_data)
rmse = np.sqrt(np.mean((test_data - ref_data)**2))

# Statistical results dictionary
stats_results = {{
    'correlation': correlation,
    'bias': bias,
    'rmse': rmse,
    'n_samples': len(ref_data),
    'method': '{method_name}'
}}

# Return statistics dictionary
"""
            return script
            
        except Exception as e:
            print(f"[ComparisonWizard] Error generating stat script template: {e}")
            return self._generate_basic_stat_script(method_name, params)

    def _generate_basic_plot_script(self, method_name, params):
        """Generate basic fallback plot script"""
        return f"""# Basic plot script for {method_name}
# Script generation failed, using simplified template

import numpy as np

# Basic transformation
x_data = ref_data
y_data = test_data

# Basic metadata
metadata = {{
    'x_label': 'Reference',
    'y_label': 'Test',
    'title': '{method_name}',
    'plot_type': 'scatter'
}}

# Parameters: {params}
"""

    def _generate_basic_stat_script(self, method_name, params):
        """Generate basic fallback stat script"""
        return f"""# Basic statistics script for {method_name}
# Script generation failed, using simplified template

import numpy as np

# Basic statistics
correlation = np.corrcoef(ref_data, test_data)[0, 1]
bias = np.mean(test_data - ref_data)

# Basic results
stats_results = {{
    'correlation': correlation,
    'bias': bias,
    'n_samples': len(ref_data),
    'method': '{method_name}'
}}

# Parameters: {params}
"""

    def _is_plot_script_customized(self):
        """Check if plot script has been customized by user"""
        try:
            current_script = self.plot_script_editor.toPlainText().strip()
            original_script = self.original_plot_script_content.strip()
            
            print(f"[ComparisonWizard] DEBUG: Checking plot script customization")
            print(f"[ComparisonWizard] DEBUG: Current script length: {len(current_script)}")
            print(f"[ComparisonWizard] DEBUG: Original script length: {len(original_script)}")
            
            # Normalize whitespace for comparison
            current_normalized = '\n'.join(line.strip() for line in current_script.split('\n') if line.strip())
            original_normalized = '\n'.join(line.strip() for line in original_script.split('\n') if line.strip())
            
            is_different = current_normalized != original_normalized
            print(f"[ComparisonWizard] DEBUG: Plot script is different from original: {is_different}")
            
            return is_different
            
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Error checking plot script customization: {e}")
            return False

    def _is_stat_script_customized(self):
        """Check if stat script has been customized by user"""
        try:
            current_script = self.stat_script_editor.toPlainText().strip()
            original_script = self.original_stat_script_content.strip()
            
            print(f"[ComparisonWizard] DEBUG: Checking stat script customization")
            print(f"[ComparisonWizard] DEBUG: Current stat script length: {len(current_script)}")
            print(f"[ComparisonWizard] DEBUG: Original stat script length: {len(original_script)}")
            
            # Normalize whitespace for comparison
            current_normalized = '\n'.join(line.strip() for line in current_script.split('\n') if line.strip())
            original_normalized = '\n'.join(line.strip() for line in original_script.split('\n') if line.strip())
            
            is_different = current_normalized != original_normalized
            print(f"[ComparisonWizard] DEBUG: Stat script is different from original: {is_different}")
            
            return is_different
            
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Error checking stat script customization: {e}")
            return False

    def _should_use_custom_plot_script(self):
        """Check if we should use custom plot script execution"""
        # Check if script editor exists and has content
        if not hasattr(self, 'plot_script_editor'):
            print("[ComparisonWizard] DEBUG: No plot_script_editor attribute - not using custom script")
            return False
        
        script_text = self.plot_script_editor.toPlainText().strip()
        if not script_text or script_text.startswith("# No comparison method selected"):
            print("[ComparisonWizard] DEBUG: Empty or placeholder script - not using custom script")
            return False
        
        # Check if script has been customized by user
        is_customized = self._is_plot_script_customized()
        print(f"[ComparisonWizard] DEBUG: Plot script customized: {is_customized}")
        return is_customized

    def _should_use_custom_stat_script(self):
        """Check if we should use custom stat script execution"""
        # Check if script editor exists and has content
        if not hasattr(self, 'stat_script_editor'):
            print("[ComparisonWizard] DEBUG: No stat_script_editor attribute - not using custom script")
            return False
        
        script_text = self.stat_script_editor.toPlainText().strip()
        if not script_text or script_text.startswith("# No comparison method selected"):
            print("[ComparisonWizard] DEBUG: Empty or placeholder stat script - not using custom script")
            return False
        
        # Check if script has been customized by user
        is_customized = self._is_stat_script_customized()
        print(f"[ComparisonWizard] DEBUG: Stat script customized: {is_customized}")
        return is_customized

    def _execute_custom_plot_script(self, ref_data, test_data, params):
        """Execute custom plot script with fallback to original method"""
        if not hasattr(self, 'plot_script_editor'):
            print("[ComparisonWizard] DEBUG: No plot_script_editor found - falling back to original method")
            return None
            
        script_text = self.plot_script_editor.toPlainText()
        
        if not script_text.strip():
            print("[ComparisonWizard] DEBUG: Empty plot script - falling back to original method")
            return None
            
        print(f"[ComparisonWizard] DEBUG: Attempting to execute custom plot script ({len(script_text)} chars)")
        
        # Try to validate the script syntax
        try:
            compile(script_text, '<plot_script>', 'exec')
            print("[ComparisonWizard] DEBUG: Custom plot script syntax validation passed")
        except SyntaxError as e:
            print(f"[ComparisonWizard] DEBUG: Plot script syntax error at line {e.lineno}: {e.msg}")
            print("[ComparisonWizard] DEBUG: Falling back to original plot method")
            return None
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Plot script validation error: {str(e)}")
            print("[ComparisonWizard] DEBUG: Falling back to original plot method")
            return None
        
        # Execute the validated script
        try:
            result = self._execute_plot_script_safely(script_text, ref_data, test_data, params)
            if result:
                print("[ComparisonWizard] DEBUG: Custom plot script executed successfully")
                return result
            else:
                print("[ComparisonWizard] DEBUG: Custom plot script returned None - falling back to original method")
                return None
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Plot script execution error: {str(e)}")
            print("[ComparisonWizard] DEBUG: Falling back to original plot method")
            return None

    def _execute_custom_stat_script(self, x_data, y_data, ref_data, test_data, params):
        """Execute custom stat script with fallback to original method"""
        if not hasattr(self, 'stat_script_editor'):
            print("[ComparisonWizard] DEBUG: No stat_script_editor found - falling back to original method")
            return None
            
        script_text = self.stat_script_editor.toPlainText()
        
        if not script_text.strip():
            print("[ComparisonWizard] DEBUG: Empty stat script - falling back to original method")
            return None
            
        print(f"[ComparisonWizard] DEBUG: Attempting to execute custom stat script ({len(script_text)} chars)")
        
        # Try to validate the script syntax
        try:
            compile(script_text, '<stat_script>', 'exec')
            print("[ComparisonWizard] DEBUG: Custom stat script syntax validation passed")
        except SyntaxError as e:
            print(f"[ComparisonWizard] DEBUG: Stat script syntax error at line {e.lineno}: {e.msg}")
            print(f"[ComparisonWizard] DEBUG: Problematic line: {e.text}")
            # Show the lines around the error for debugging
            script_lines = script_text.split('\n')
            start_line = max(0, e.lineno - 3)
            end_line = min(len(script_lines), e.lineno + 2)
            print(f"[ComparisonWizard] DEBUG: Script context (lines {start_line+1}-{end_line}):")
            for i in range(start_line, end_line):
                marker = " --> " if i == e.lineno - 1 else "     "
                print(f"[ComparisonWizard] DEBUG: {marker}{i+1:3d}: {script_lines[i]}")
            print("[ComparisonWizard] DEBUG: Falling back to original stat method")
            return None
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Stat script validation error: {str(e)}")
            print("[ComparisonWizard] DEBUG: Falling back to original stat method")
            return None
        
        # Execute the validated script
        try:
            result = self._execute_stat_script_safely(script_text, x_data, y_data, ref_data, test_data, params)
            if result:
                print("[ComparisonWizard] DEBUG: Custom stat script executed successfully")
                return result
            else:
                print("[ComparisonWizard] DEBUG: Custom stat script returned None - falling back to original method")
                return None
        except Exception as e:
            print(f"[ComparisonWizard] DEBUG: Stat script execution error: {str(e)}")
            print("[ComparisonWizard] DEBUG: Falling back to original stat method")
            return None

    def _execute_plot_script_safely(self, script_text, ref_data, test_data, params):
        """Execute plot script in a controlled environment"""
        import numpy as np
        import scipy.signal
        import copy
        
        # Create safe global variables for the script
        safe_globals = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'print': print,
                'isinstance': isinstance,
                'type': type,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                '__import__': __import__,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
            },
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'copy': copy,
            'ref_data': ref_data,
            'test_data': test_data,
            'params': params,
        }
        
        # Create local variables for the script
        safe_locals = {}
        
        # Execute the script
        exec(script_text, safe_globals, safe_locals)
        
        # Look for expected return variables
        if 'x_data' in safe_locals and 'y_data' in safe_locals and 'metadata' in safe_locals:
            return safe_locals['x_data'], safe_locals['y_data'], safe_locals['metadata']
        else:
            raise ValueError("Script must define x_data, y_data, and metadata variables")

    def _execute_stat_script_safely(self, script_text, x_data, y_data, ref_data, test_data, params):
        """Execute stat script in a controlled environment"""
        import numpy as np
        import scipy.signal
        import copy
        from scipy import stats
        
        # Create safe global variables for the script
        safe_globals = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'print': print,
                'isinstance': isinstance,
                'type': type,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                '__import__': __import__,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
            },
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'copy': copy,
            'stats': stats,
            'x_data': x_data,
            'y_data': y_data,
            'ref_data': ref_data,
            'test_data': test_data,
            'params': params,
        }
        
        # Create local variables for the script
        safe_locals = {}
        
        # Execute the script
        exec(script_text, safe_globals, safe_locals)
        
        # Look for expected return variables
        if 'stats_results' in safe_locals:
            return safe_locals['stats_results']
        else:
            raise ValueError("Script must define stats_results variable")

    def _create_performance_options_group(self, layout):
        """Create performance options group underneath alignment"""
        group = QGroupBox("Performance Options")
        group.setStyleSheet("QGroupBox { font-weight: bold; }")
        group_layout = QVBoxLayout(group)
        
        # Max points option
        max_points_layout = QHBoxLayout()
        self.max_points_checkbox = QCheckBox("Max Points:")
        self.max_points_checkbox.setToolTip("Limit data points for better performance")
        self.max_points_input = QSpinBox()
        self.max_points_input.setRange(100, 50000)
        self.max_points_input.setValue(5000)
        self.max_points_input.setMaximumWidth(80)
        self.max_points_input.setEnabled(False)
        
        # Connect checkbox to enable/disable input
        self.max_points_checkbox.toggled.connect(self.max_points_input.setEnabled)
        
        max_points_layout.addWidget(self.max_points_checkbox)
        max_points_layout.addWidget(self.max_points_input)
        max_points_layout.addStretch()
        
        group_layout.addLayout(max_points_layout)
        
        # Fast rendering and density display
        options_layout = QHBoxLayout()
        
        self.fast_render_checkbox = QCheckBox("Fast Rendering")
        self.fast_render_checkbox.setToolTip("Use simplified rendering for better performance")
        self.fast_render_checkbox.setChecked(False)
        
        options_layout.addWidget(self.fast_render_checkbox)
        options_layout.addStretch()
        
        group_layout.addLayout(options_layout)
        
        # Density display options
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density_combo = QComboBox()
        self.density_combo.addItems(["Scatter", "Hexbin", "KDE"])
        self.density_combo.setToolTip("Display method for high-density data")
        self.density_combo.setCurrentText("Scatter")
        density_layout.addWidget(self.density_combo)
        
        # Add bin control next to density
        density_layout.addWidget(QLabel("Bins:"))
        self.bins_spinbox = QSpinBox()
        self.bins_spinbox.setRange(5, 200)
        self.bins_spinbox.setValue(50)  # Default bin count
        self.bins_spinbox.setToolTip("Number of bins for hexbin and histogram displays")
        self.bins_spinbox.setMaximumWidth(70)
        # Initially disabled since default is "Scatter"
        self.bins_spinbox.setEnabled(False)
        density_layout.addWidget(self.bins_spinbox)
        
        density_layout.addStretch()
        
        group_layout.addLayout(density_layout)
        
        layout.addWidget(group)

    def _populate_parameter_table(self, method_name):
        """Populate parameter table based on selected method"""
        try:
            print(f"[ComparisonWizard] Populating parameter table for method: {method_name}")
            
            # Clear existing rows
            self.param_table.setRowCount(0)
            
            # Initialize parameter name mapping
            self._param_name_mapping = {}
            
            # Get method registry name
            method_registry_name = self._get_registry_name_from_display_name(method_name)
            if not method_registry_name:
                print(f"[ComparisonWizard] No registry name found for {method_name}")
                return
                
            # Get comparison class
            comparison_cls = ComparisonRegistry.get(method_registry_name)
            if not comparison_cls:
                print(f"[ComparisonWizard] No comparison class found for {method_registry_name}")
                return
                
            # Get method parameters - handle both 'parameters' dict and 'params' list formats
            method_params = getattr(comparison_cls, 'parameters', {})
            
            if not method_params:
                # Try the 'params' list format (used by processing steps and newer comparison methods)
                params_list = getattr(comparison_cls, 'params', [])
                method_params = {param['name']: param for param in params_list}
            
            if not method_params:
                print(f"[ComparisonWizard] No parameters found for {method_name}")
                return
            
            print(f"[ComparisonWizard] Found {len(method_params)} parameters to populate")
            
            # Populate table with parameters
            for param_name, param_config in method_params.items():
                row = self.param_table.rowCount()
                self.param_table.insertRow(row)
                
                # Parameter name with tooltip description
                param_label = param_config.get('label', param_name.replace('_', ' ').title())
                name_item = QTableWidgetItem(param_label)
                name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
                
                # Set tooltip with description (handle both 'description' and 'help' keys)
                description = param_config.get('description', param_config.get('help', 'No description available'))
                name_item.setToolTip(description)
                
                self.param_table.setItem(row, 0, name_item)
                
                # Store parameter name mapping for later retrieval
                self._param_name_mapping[param_label] = param_name
                
                # Parameter value control
                value_widget = self._create_parameter_value_widget(param_name, param_config)
                self.param_table.setCellWidget(row, 1, value_widget)
                
            print(f"[ComparisonWizard] Successfully populated {self.param_table.rowCount()} parameter rows")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error populating parameter table for {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_parameter_value_widget(self, param_name, param_config):
        """Create appropriate widget for parameter value based on type"""
        param_type = param_config.get('type', 'string')
        default_value = param_config.get('default', None)
        
        if param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(default_value if default_value is not None else False)
            # Connect to parameter change handler
            widget.stateChanged.connect(self._on_method_parameter_changed)
            return widget
            
        elif param_type == 'int':
            widget = QSpinBox()
            range_vals = param_config.get('range', (0, 100))
            widget.setRange(range_vals[0], range_vals[1])
            widget.setValue(default_value if default_value is not None else range_vals[0])
            widget.valueChanged.connect(self._on_method_parameter_changed)
            return widget
            
        elif param_type == 'float':
            widget = QDoubleSpinBox()
            range_vals = param_config.get('range', (0.0, 1.0))
            widget.setRange(range_vals[0], range_vals[1])
            widget.setDecimals(param_config.get('decimals', 2))
            widget.setValue(default_value if default_value is not None else range_vals[0])
            widget.valueChanged.connect(self._on_method_parameter_changed)
            return widget
            
        elif param_type == 'combo':
            widget = QComboBox()
            options = param_config.get('options', [])
            widget.addItems(options)
            if default_value and default_value in options:
                widget.setCurrentText(default_value)
            widget.currentTextChanged.connect(self._on_method_parameter_changed)
            return widget
            
        elif param_type == 'str' or param_name == 'outlier_method':
            # Handle string type parameters, with special handling for outlier_method
            if param_name == 'outlier_method':
                widget = QComboBox()
                widget.addItems(['iqr', 'zscore'])
                widget.setCurrentText(default_value if default_value else 'iqr')
                widget.currentTextChanged.connect(self._on_method_parameter_changed)
                return widget
            else:
                widget = QLineEdit()
                widget.setText(str(default_value) if default_value is not None else "")
                widget.textChanged.connect(self._on_method_parameter_changed)
                return widget
                
        else:  # string or other
            widget = QLineEdit()
            widget.setText(str(default_value) if default_value is not None else "")
            widget.textChanged.connect(self._on_method_parameter_changed)
            return widget
    
    def _get_parameter_table_values(self):
        """Get current parameter values from the parameter table"""
        params = {}
        
        # Store parameter name mapping during table population
        if not hasattr(self, '_param_name_mapping'):
            self._param_name_mapping = {}
        
        for row in range(self.param_table.rowCount()):
            # Get parameter name from the display name (column 0)
            name_item = self.param_table.item(row, 0)
            if not name_item:
                continue
                
            param_label = name_item.text()
            
            # Get the widget from the value column (column 1)
            value_widget = self.param_table.cellWidget(row, 1)
            if not value_widget:
                continue
                
            # Extract value based on widget type
            if isinstance(value_widget, QCheckBox):
                value = value_widget.isChecked()
            elif isinstance(value_widget, QSpinBox):
                value = value_widget.value()
            elif isinstance(value_widget, QDoubleSpinBox):
                value = value_widget.value()
            elif isinstance(value_widget, QComboBox):
                value = value_widget.currentText()
            elif isinstance(value_widget, QLineEdit):
                value = value_widget.text()
            else:
                continue
                
            # Use stored mapping if available, otherwise convert label back to parameter name
            if param_label in self._param_name_mapping:
                param_name = self._param_name_mapping[param_label]
            else:
                param_name = param_label.lower().replace(' ', '_')
                
            params[param_name] = value
            
        return params

    def _find_channel_id_for_pair(self, pair_name):
        """Find the channel ID that corresponds to a comparison pair name"""
        try:
            if not hasattr(self, 'channel_manager') or not self.channel_manager:
                return None
            
            # Search through all channels to find one with matching pair name
            for channel in self.channel_manager.get_all_channels():
                # Check if this is a comparison channel with matching pair name
                if (hasattr(channel, 'metadata') and 
                    channel.metadata and 
                    isinstance(channel.metadata, dict)):
                    
                    # Check if the pair info matches
                    pair_info = channel.metadata.get('pair_info', {})
                    if pair_info and pair_info.get('name') == pair_name:
                        return channel.channel_id
                    
                    # Also check legend_label as fallback
                    if channel.legend_label == pair_name:
                        return channel.channel_id
            
            return None
        except Exception as e:
            print(f"[ComparisonWizard] Error finding channel ID for pair '{pair_name}': {e}")
            return None
    
    def _refresh_plot_with_visibility_changes(self):
        """Refresh the plot to reflect channel visibility changes"""
        try:
            # Force a plot refresh by calling the refresh plot method
            if hasattr(self, 'comparison_manager') and self.comparison_manager:
                # Get all active pairs (not just checked ones)
                all_pairs = self.get_active_pairs()
                
                # Filter pairs based on channel visibility
                visible_pairs = []
                for pair in all_pairs:
                    channel_id = self._find_channel_id_for_pair(pair['name'])
                    if channel_id and hasattr(self, 'channel_manager') and self.channel_manager:
                        channel = self.channel_manager.get_channel(channel_id)
                        if channel and channel.show:
                            visible_pairs.append(pair)
                    else:
                        # If no channel found, check if checkbox is checked (fallback)
                        # Find the corresponding checkbox for this pair
                        for row in range(self.active_pair_table.rowCount()):
                            pair_name_item = self.active_pair_table.item(row, 2)
                            if pair_name_item and pair_name_item.text() == pair['name']:
                                checkbox_widget = self.active_pair_table.cellWidget(row, 0)
                                if checkbox_widget:
                                    checkbox = None
                                    for child in checkbox_widget.children():
                                        if isinstance(child, QCheckBox):
                                            checkbox = child
                                            break
                                    if checkbox and checkbox.isChecked():
                                        visible_pairs.append(pair)
                                break
                
                # Refresh the plot with only visible pairs
                if visible_pairs:
                    plot_config = self._get_plot_config()
                    plot_config['checked_pairs'] = visible_pairs
                    
                    # Clear all plots first to ensure clean state
                    self.comparison_manager._clear_all_plots()
                    
                    # Regenerate plots with visible pairs
                    self.comparison_manager._generate_all_visualizations(visible_pairs, plot_config)
                    
                    print(f"[ComparisonWizard] Refreshed plot with {len(visible_pairs)} visible pairs")
                else:
                    # Clear all plots if no pairs are visible
                    self.comparison_manager._clear_all_plots()
                    print(f"[ComparisonWizard] Cleared plots - no visible pairs")
                        
        except Exception as e:
            print(f"[ComparisonWizard] Error refreshing plot with visibility changes: {e}")