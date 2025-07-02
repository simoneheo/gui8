from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor, QIntValidator, QColor
import pandas as pd
from copy import deepcopy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class ComparisonWizardWindow(QMainWindow):
    """
    Two-step wizard for comparing data channels:
    Step 1: Channel selection and alignment
    Step 2: Plot configuration and visualization
    """
    
    pair_added = Signal(dict)
    pair_deleted = Signal()
    plot_generated = Signal(dict)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers with consistent naming
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state
        self.current_step = 0
        self.active_pairs = []  # Store pair configurations
        self.comparison_manager = None  # Will be set by manager
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        self._populate_file_combos()
        
        # Initialize with available data
        self._validate_initialization()
        
    def _validate_initialization(self):
        """Validate that required managers are available"""
        if not self.file_manager:
            self._show_error("File manager not available")
            return False
            
        if not self.channel_manager:
            self._show_error("Channel manager not available")
            return False
            
        return True
        
    def _show_error(self, message):
        """Show error message to user"""
        if hasattr(self, 'console_output'):
            self.console_output.append(f"ERROR: {message}")
        else:
            print(f"[ComparisonWizard] ERROR: {message}")
        
    def _init_ui(self):
        self.setWindowTitle("Data Comparison Wizard")
        self.setMinimumSize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Step navigation
        nav_widget = self._create_navigation()
        layout.addWidget(nav_widget)
        
        # Main content area
        self.steps = QStackedWidget()
        self._build_step1()
        self._build_step2()
        layout.addWidget(self.steps)
        
        # Bottom navigation
        bottom_nav = self._create_bottom_navigation()
        layout.addWidget(bottom_nav)
        
    def _create_navigation(self):
        """Create step navigation bar"""
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        
        step1_label = QLabel("1. Channel Selection & Alignment")
        step1_label.setStyleSheet("font-weight: bold; color: #2c3e50; padding: 10px;")
        
        step2_label = QLabel("2. Plot Configuration")
        step2_label.setStyleSheet("color: #7f8c8d; padding: 10px;")
        
        nav_layout.addWidget(step1_label)
        nav_layout.addWidget(QLabel("→"))
        nav_layout.addWidget(step2_label)
        nav_layout.addStretch()
        
        self.step_labels = [step1_label, step2_label]
        return nav_widget
        
    def _create_bottom_navigation(self):
        """Create bottom navigation buttons"""
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        
        self.back_button = QPushButton("← Back")
        self.next_button = QPushButton("Next →")
        self.cancel_button = QPushButton("Cancel")
        
        nav_layout.addWidget(self.back_button)
        nav_layout.addStretch()
        nav_layout.addWidget(self.cancel_button)
        nav_layout.addWidget(self.next_button)
        
        return nav_widget
        
    def _build_step1(self):
        """Build step 1: Channel selection and alignment"""
        step = QWidget()
        layout = QHBoxLayout(step)

        # LEFT PANEL
        left = QVBoxLayout()

        # Reference & Test Channel
        left.addWidget(QLabel("Reference Channel"))
        ref_row = QHBoxLayout()
        self.ref_file_combo = QComboBox()
        self.ref_channel_combo = QComboBox()
        ref_row.addWidget(self.ref_file_combo)
        ref_row.addWidget(self.ref_channel_combo)
        left.addLayout(ref_row)

        left.addWidget(QLabel("Test Channel"))
        test_row = QHBoxLayout()
        self.test_file_combo = QComboBox()
        self.test_channel_combo = QComboBox()
        test_row.addWidget(self.test_file_combo)
        test_row.addWidget(self.test_channel_combo)
        left.addLayout(test_row)

        # Alignment Mode
        alignment_box = QGroupBox("Alignment Mode")
        alignment_layout = QVBoxLayout()
        self.alignment_mode_index = QRadioButton("Index-Based")
        self.alignment_mode_time = QRadioButton("Time-Based")
        self.alignment_mode_index.setChecked(True)
        alignment_layout.addWidget(self.alignment_mode_index)
        alignment_layout.addWidget(self.alignment_mode_time)
        alignment_box.setLayout(alignment_layout)
        left.addWidget(alignment_box)

        # Index-based Options
        index_group = QGroupBox("Index-Based Options")
        index_form = QFormLayout()
        self.index_mode_truncate = QRadioButton("Truncate to Full Length")
        self.index_mode_custom = QRadioButton("Custom Index Range")
        self.index_mode_truncate.setChecked(True)
        self.index_start_edit = QLineEdit("0")
        self.index_end_edit = QLineEdit("499")
        self.index_offset_checkbox = QCheckBox("Apply Offset (samples)")
        self.index_offset_input = QLineEdit("0")
        index_form.addRow(self.index_mode_truncate)
        index_form.addRow(self.index_mode_custom)
        index_form.addRow("Start Index:", self.index_start_edit)
        index_form.addRow("End Index:", self.index_end_edit)
        index_form.addRow(self.index_offset_checkbox)
        index_form.addRow("Offset:", self.index_offset_input)
        index_group.setLayout(index_form)
        left.addWidget(index_group)

        # Time-based Options
        time_group = QGroupBox("Time-Based Options")
        time_form = QFormLayout()
        self.time_mode_overlap = QRadioButton("Use Overlapping Times Only")
        self.time_mode_custom = QRadioButton("Custom Time Window")
        self.time_mode_overlap.setChecked(True)
        self.time_round_combo = QComboBox()
        self.time_round_combo.addItems(["0.01", "0.05", "0.1", "0.5", "1.0"])
        self.time_start_edit = QLineEdit("0.000")
        self.time_end_edit = QLineEdit("12.500")
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["linear", "nearest", "cubic"])
        self.time_offset_checkbox = QCheckBox("Apply Offset (seconds)")
        self.time_offset_input = QLineEdit("0.000")
        time_form.addRow(self.time_mode_overlap)
        time_form.addRow("Round to Nearest:", self.time_round_combo)
        time_form.addRow(self.time_mode_custom)
        time_form.addRow("Start Time:", self.time_start_edit)
        time_form.addRow("End Time:", self.time_end_edit)
        time_form.addRow("Interpolation:", self.interp_combo)
        time_form.addRow(self.time_offset_checkbox)
        time_form.addRow("Offset:", self.time_offset_input)
        time_group.setLayout(time_form)
        left.addWidget(time_group)

        # Cumulative Statistics Display (moved to bottom of left panel)
        stats_group = QGroupBox("Cumulative Statistics")
        stats_layout = QVBoxLayout()
        
        self.cumulative_stats_label = QLabel("Cumulative Stats: No pairs added")
        self.cumulative_stats_label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                color: #495057;
            }
        """)
        self.cumulative_stats_label.setWordWrap(True)
        self.cumulative_stats_label.setMaximumHeight(120)
        stats_layout.addWidget(self.cumulative_stats_label)
        
        stats_group.setLayout(stats_layout)
        left.addWidget(stats_group)

        # RIGHT PANEL
        right = QVBoxLayout()

        # Pair Name + Buttons (moved to top of right panel)
        pair_control_group = QGroupBox("Add/Remove Pairs")
        pair_control_layout = QVBoxLayout()
        
        name_row = QHBoxLayout()
        self.pair_name_input = QLineEdit("")
        self.pair_name_input.setPlaceholderText("Enter pair name or leave blank for auto-naming")
        self.add_pair_button = QPushButton("+ Add Pair")
        self.delete_pair_button = QPushButton("🗑️ Delete Selected")
        self.delete_pair_button.setToolTip("Select a pair from the table below, then click to delete")
        self.delete_pair_button.setEnabled(False)  # Initially disabled
        
        name_row.addWidget(QLabel("Pair Name:"))
        name_row.addWidget(self.pair_name_input)
        name_row.addWidget(self.add_pair_button)
        name_row.addWidget(self.delete_pair_button)
        
        pair_control_layout.addLayout(name_row)
        pair_control_group.setLayout(pair_control_layout)
        right.addWidget(pair_control_group)

        # Active Pairs Table
        self.active_pair_table = QTableWidget()
        self.active_pair_table.setColumnCount(5)
        self.active_pair_table.setHorizontalHeaderLabels(["Show", "Pair Name", "r", "RMS", "N"])
        self.active_pair_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.active_pair_table.setSelectionMode(QTableWidget.SingleSelection)
        right.addWidget(QLabel("Active Pairs"))
        right.addWidget(self.active_pair_table)

        # Matplotlib Plot
        self.canvas = FigureCanvas(plt.figure())
        self.toolbar = NavigationToolbar(self.canvas, self)
        right.addWidget(self.toolbar)
        right.addWidget(self.canvas)

        layout.addLayout(left, 1)
        layout.addLayout(right, 2)
        self.steps.addWidget(step)
        self.step1_ui = step

    def _build_step2(self):
        """Build step 2: Plot configuration"""
        step = QWidget()
        layout = QHBoxLayout(step)

        # LEFT PANEL - Plot Type & Overlay Options
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # Plot Type Selection
        plot_type_group = QGroupBox("Plot Type")
        plot_type_layout = QVBoxLayout()
        self.plot_type_bland = QRadioButton("Bland-Altman")
        self.plot_type_scatter = QRadioButton("Scatter Plot")
        self.plot_type_residual = QRadioButton("Residual Plot")
        self.plot_type_pearson = QRadioButton("Pearson Correlation")
        self.plot_type_bland.setChecked(True)
        
        # Connect plot type changes to reactive updates
        self.plot_type_bland.toggled.connect(self._on_plot_config_changed)
        self.plot_type_scatter.toggled.connect(self._on_plot_config_changed)
        self.plot_type_residual.toggled.connect(self._on_plot_config_changed)
        self.plot_type_pearson.toggled.connect(self._on_plot_config_changed)
        
        plot_type_layout.addWidget(self.plot_type_bland)
        plot_type_layout.addWidget(self.plot_type_scatter)
        plot_type_layout.addWidget(self.plot_type_residual)
        plot_type_layout.addWidget(self.plot_type_pearson)
        plot_type_group.setLayout(plot_type_layout)
        left_layout.addWidget(plot_type_group)
        
        # Overlay Options
        overlay_group = QGroupBox("Overlay Options")
        overlay_layout = QVBoxLayout()
        
        self.ci_checkbox = QCheckBox("Confidence Interval (95%)")
        self.ci_checkbox.stateChanged.connect(self._on_plot_config_changed)
        self.outlier_checkbox = QCheckBox("Highlight Outliers")
        self.outlier_checkbox.stateChanged.connect(self._on_plot_config_changed)
        overlay_layout.addWidget(self.ci_checkbox)
        overlay_layout.addWidget(self.outlier_checkbox)
        
        # Custom line option
        custom_row = QHBoxLayout()
        self.custom_line_checkbox = QCheckBox("Custom Line (y = ")
        self.custom_line_checkbox.stateChanged.connect(self._on_plot_config_changed)
        self.custom_line_edit = QLineEdit("0.0")
        self.custom_line_edit.setMaximumWidth(60)
        self.custom_line_edit.textChanged.connect(self._on_plot_config_changed)
        custom_label = QLabel(")")
        custom_row.addWidget(self.custom_line_checkbox)
        custom_row.addWidget(self.custom_line_edit)
        custom_row.addWidget(custom_label)
        custom_row.addStretch()
        overlay_layout.addLayout(custom_row)
        
        # Performance options
        perf_row = QHBoxLayout()
        self.downsample_checkbox = QCheckBox("Max Points:")
        self.downsample_checkbox.stateChanged.connect(self._on_plot_config_changed)
        self.downsample_input = QLineEdit("5000")
        self.downsample_input.setMaximumWidth(60)
        self.downsample_input.textChanged.connect(self._on_plot_config_changed)
        perf_row.addWidget(self.downsample_checkbox)
        perf_row.addWidget(self.downsample_input)
        perf_row.addStretch()
        overlay_layout.addLayout(perf_row)
        
        overlay_group.setLayout(overlay_layout)
        left_layout.addWidget(overlay_group)
        
        # Density Display Options (Scatter/Hexbin)
        density_group = QGroupBox("Density Display")
        density_layout = QVBoxLayout()
        
        self.density_combo = QComboBox()
        self.density_combo.addItems(["Scatter", "Hexbin", "KDE"])
        self.density_combo.currentTextChanged.connect(self._on_density_display_changed)
        density_layout.addWidget(self.density_combo)
        
        # Bin size for hexbin
        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel("Bin Size:"))
        self.bin_size_input = QLineEdit("20")
        self.bin_size_input.setMaximumWidth(60)
        self.bin_size_input.textChanged.connect(self._on_plot_config_changed)
        bin_row.addWidget(self.bin_size_input)
        bin_row.addStretch()
        density_layout.addLayout(bin_row)
        
        # KDE bandwidth
        kde_row = QHBoxLayout()
        kde_row.addWidget(QLabel("KDE Bandwidth:"))
        self.kde_bw_input = QLineEdit("0.2")
        self.kde_bw_input.setMaximumWidth(60)
        self.kde_bw_input.textChanged.connect(self._on_plot_config_changed)
        kde_row.addWidget(self.kde_bw_input)
        kde_row.addStretch()
        density_layout.addLayout(kde_row)
        
        density_group.setLayout(density_layout)
        left_layout.addWidget(density_group)
        
        # Cumulative Statistics Display (moved from right panel)
        stats_group = QGroupBox("Cumulative Statistics")
        stats_layout = QVBoxLayout()
        
        self.cumulative_stats_step2 = QLabel("No pairs selected for analysis")
        self.cumulative_stats_step2.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 5px;
                padding: 6px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 9px;
                color: #495057;
            }
        """)
        self.cumulative_stats_step2.setWordWrap(True)
        self.cumulative_stats_step2.setMaximumHeight(150)
        stats_layout.addWidget(self.cumulative_stats_step2)
        
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        left_layout.addStretch()

        # RIGHT PANEL - Active Pairs, Plot Config, Canvas, Stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Active Pairs Table
        pairs_group = QGroupBox("Active Pairs")
        pairs_layout = QVBoxLayout()
        
        self.active_pair_table_step2 = QTableWidget()
        self.active_pair_table_step2.setColumnCount(4)
        self.active_pair_table_step2.setHorizontalHeaderLabels(["Show", "Pair Name", "Marker Type", "Marker Color"])
        self.active_pair_table_step2.setMaximumHeight(120)
        pairs_layout.addWidget(self.active_pair_table_step2)
        
        pairs_group.setLayout(pairs_layout)
        right_layout.addWidget(pairs_group)

        # Plot Configuration
        config_group = QGroupBox("Plot Configuration")
        config_layout = QVBoxLayout()
        
        # First row: Labels and basic options
        config_row1 = QHBoxLayout()
        config_row1.addWidget(QLabel("X Label:"))
        self.xlabel_input = QLineEdit("Reference")
        self.xlabel_input.textChanged.connect(self._on_plot_config_changed)
        config_row1.addWidget(self.xlabel_input)
        
        config_row1.addWidget(QLabel("Y Label:"))
        self.ylabel_input = QLineEdit("Test")
        self.ylabel_input.textChanged.connect(self._on_plot_config_changed)
        config_row1.addWidget(self.ylabel_input)
        
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self._on_plot_config_changed)
        config_row1.addWidget(self.grid_checkbox)
        
        self.legend_checkbox = QCheckBox("Legend")
        self.legend_checkbox.setChecked(True)
        self.legend_checkbox.stateChanged.connect(self._on_plot_config_changed)
        config_row1.addWidget(self.legend_checkbox)
        
        config_layout.addLayout(config_row1)
        
        # Second row: Ranges
        config_row2 = QHBoxLayout()
        config_row2.addWidget(QLabel("X Range:"))
        self.x_range_input = QLineEdit("Auto")
        self.x_range_input.textChanged.connect(self._on_plot_config_changed)
        config_row2.addWidget(self.x_range_input)
        
        config_row2.addWidget(QLabel("Y Range:"))
        self.y_range_input = QLineEdit("Auto")
        self.y_range_input.textChanged.connect(self._on_plot_config_changed)
        config_row2.addWidget(self.y_range_input)
        
        config_layout.addLayout(config_row2)
        
        config_group.setLayout(config_layout)
        right_layout.addWidget(config_group)

        # Plot Area
        self.canvas2 = FigureCanvas(plt.figure())
        self.toolbar2 = NavigationToolbar(self.canvas2, self)
        right_layout.addWidget(self.toolbar2)
        right_layout.addWidget(self.canvas2)

        # Add panels to main layout
        layout.addWidget(left_panel, 1)   # Left panel takes 1/4
        layout.addWidget(right_panel, 3)  # Right panel takes 3/4

        self.steps.addWidget(step)
        self.step2_ui = step
        
    def _connect_signals(self):
        """Connect UI signals"""
        # Navigation
        self.back_button.clicked.connect(self._on_back_clicked)
        self.next_button.clicked.connect(self._on_next_clicked)
        self.cancel_button.clicked.connect(self.close)
        
        # Step 1 signals
        self.ref_file_combo.currentTextChanged.connect(self._on_ref_file_changed)
        self.test_file_combo.currentTextChanged.connect(self._on_test_file_changed)
        self.ref_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        self.test_channel_combo.currentTextChanged.connect(self._update_default_pair_name)
        self.add_pair_button.clicked.connect(self._on_add_pair)
        self.delete_pair_button.clicked.connect(self._on_delete_selected_pair)
        
        # Active pairs table selection
        self.active_pair_table.itemSelectionChanged.connect(self._on_pair_selection_changed)
        
        # Step 2 signals
        # (Generate plot button removed - using reactive plotting)
        
        self._update_navigation()
        
    def _populate_file_combos(self):
        """Populate file combo boxes with successfully parsed files"""
        if not self.file_manager or not self.channel_manager:
            return
            
        # Get all files and filter for successfully parsed ones (have channels)
        all_files = self.file_manager.get_all_files()
        parsed_files = []
        
        for file_info in all_files:
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            if channels:  # File has channels, meaning it was parsed successfully
                parsed_files.append(file_info)
        
        # Clear existing items
        self.ref_file_combo.clear()
        self.test_file_combo.clear()
        
        # Add parsed files to combo boxes
        file_names = [f.filename for f in parsed_files]
        self.ref_file_combo.addItems(file_names)
        self.test_file_combo.addItems(file_names)
        
        # Populate initial channels
        if file_names:
            self._on_ref_file_changed(file_names[0])
            self._on_test_file_changed(file_names[0])
            # Set initial default pair name
            self._update_default_pair_name()
            
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
            
            if len(selected_rows) == 1:
                selected_row = list(selected_rows)[0]
                pair_name_item = self.active_pair_table.item(selected_row, 1)
                if pair_name_item:
                    pair_name = pair_name_item.text()
                    self.delete_pair_button.setToolTip(f"Delete pair: {pair_name}")
            else:
                self.delete_pair_button.setToolTip("Select a pair from the table below, then click to delete")
                
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
            
        config = {
            'name': pair_name,
            'ref_file': ref_file,
            'ref_channel': ref_channel,
            'test_file': test_file,
            'test_channel': test_channel,
            'alignment_mode': 'index' if self.alignment_mode_index.isChecked() else 'time',
            'alignment_config': self._get_alignment_config()
        }
        
        return config
        
    def _get_alignment_config(self):
        """Get alignment configuration from UI"""
        if self.alignment_mode_index.isChecked():
            return {
                'mode': 'truncate' if self.index_mode_truncate.isChecked() else 'custom',
                'start_index': int(self.index_start_edit.text()) if self.index_start_edit.text() else 0,
                'end_index': int(self.index_end_edit.text()) if self.index_end_edit.text() else 499,
                'offset': int(self.index_offset_input.text()) if self.index_offset_checkbox.isChecked() and self.index_offset_input.text() else 0
            }
        else:
            return {
                'mode': 'overlap' if self.time_mode_overlap.isChecked() else 'custom',
                'round_to': float(self.time_round_combo.currentText()),
                'start_time': float(self.time_start_edit.text()) if self.time_start_edit.text() else 0.0,
                'end_time': float(self.time_end_edit.text()) if self.time_end_edit.text() else 12.5,
                'interpolation': self.interp_combo.currentText(),
                'offset': float(self.time_offset_input.text()) if self.time_offset_checkbox.isChecked() and self.time_offset_input.text() else 0.0
            }
            
    def _update_active_pairs_table(self):
        """Update the active pairs table"""
        self.active_pair_table.setRowCount(len(self.active_pairs))
        
        for i, pair in enumerate(self.active_pairs):
            # Show checkbox
            show_cb = QCheckBox()
            show_cb.setChecked(True)
            # Connect checkbox signal to update cumulative stats
            show_cb.stateChanged.connect(self._on_show_checkbox_changed)
            self.active_pair_table.setCellWidget(i, 0, show_cb)
            
            # Pair name with tooltip
            pair_name_item = QTableWidgetItem(pair['name'])
            self._set_pair_name_tooltip_on_item(pair_name_item, pair)
            self.active_pair_table.setItem(i, 1, pair_name_item)
            
            # Placeholder for statistics (r, RMS, N)
            self.active_pair_table.setItem(i, 2, QTableWidgetItem("--"))
            self.active_pair_table.setItem(i, 3, QTableWidgetItem("--"))
            self.active_pair_table.setItem(i, 4, QTableWidgetItem("--"))
            
        # Update step 2 active pairs table
        self._update_step2_active_pairs_table()
        
    def _update_step2_active_pairs_table(self):
        """Update the step 2 active pairs table"""
        self.active_pair_table_step2.setRowCount(len(self.active_pairs))
        
        # Marker types for selection
        marker_types = ['○ Circle', '□ Square', '△ Triangle', '◇ Diamond', '▽ Inverted Triangle', 
                       '◁ Left Triangle', '▷ Right Triangle', '⬟ Pentagon', '✦ Star', '⬢ Hexagon']
        
        # Color options for selection
        color_options = ['🔵 Blue', '🔴 Red', '🟢 Green', '🟣 Purple', '🟠 Orange', 
                        '🟤 Brown', '🩷 Pink', '⚫ Gray', '🟡 Yellow', '🔶 Cyan']
        
        for i, pair in enumerate(self.active_pairs):
            # Show checkbox
            show_cb = QCheckBox()
            show_cb.setChecked(True)
            # Connect checkbox signal to update plot
            show_cb.stateChanged.connect(self._on_step2_table_changed)
            self.active_pair_table_step2.setCellWidget(i, 0, show_cb)
            
            # Pair name with tooltip
            pair_name_item = QTableWidgetItem(pair['name'])
            self._set_pair_name_tooltip_on_item(pair_name_item, pair)
            self.active_pair_table_step2.setItem(i, 1, pair_name_item)
            
            # Marker type combo box
            marker_combo = QComboBox()
            marker_combo.addItems(marker_types)
            marker_combo.setCurrentIndex(i % len(marker_types))  # Default to different markers
            marker_combo.currentTextChanged.connect(self._on_step2_table_changed)
            self.active_pair_table_step2.setCellWidget(i, 2, marker_combo)
            
            # Marker color combo box
            color_combo = QComboBox()
            color_combo.addItems(color_options)
            color_combo.setCurrentIndex(i % len(color_options))  # Default to different colors
            color_combo.currentTextChanged.connect(self._on_step2_table_changed)
            self.active_pair_table_step2.setCellWidget(i, 3, color_combo)
            
        # Update enabled state based on density display
        self._update_marker_controls_state()
        
    def _on_show_checkbox_changed(self, state):
        """Handle Show checkbox state changes"""
        if hasattr(self, 'comparison_manager'):
            self.comparison_manager._update_cumulative_display()
            
    def _on_step2_show_checkbox_changed(self, state):
        """Handle Show checkbox state changes in step 2"""
        if hasattr(self, 'comparison_manager'):
            self.comparison_manager._update_step2_cumulative_display()
            
    def _on_step2_table_changed(self):
        """Handle any changes in the step 2 active pairs table - reactive plot update"""
        if hasattr(self, 'comparison_manager'):
            # Update cumulative statistics
            self.comparison_manager._update_step2_cumulative_display()
            # Auto-generate plot with current settings
            self._auto_generate_plot()
            
    def _on_plot_config_changed(self):
        """Handle changes in plot configuration - reactive plot update"""
        # Update cumulative statistics when plot type changes
        if hasattr(self, 'comparison_manager'):
            self.comparison_manager._update_step2_cumulative_display()
        # Auto-generate plot with current settings
        self._auto_generate_plot()
        
    def _on_density_display_changed(self):
        """Handle changes in density display type"""
        self._update_marker_controls_state()
        self._auto_generate_plot()
        
    def _update_marker_controls_state(self):
        """Enable/disable marker controls based on density display type"""
        density_type = self.density_combo.currentText().lower()
        is_scatter = density_type == 'scatter'
        
        # Enable/disable marker type and color controls
        for row in range(self.active_pair_table_step2.rowCount()):
            marker_combo = self.active_pair_table_step2.cellWidget(row, 2)
            color_combo = self.active_pair_table_step2.cellWidget(row, 3)
            if marker_combo:
                marker_combo.setEnabled(is_scatter)
            if color_combo:
                color_combo.setEnabled(is_scatter)
                
    def _auto_generate_plot(self):
        """Automatically generate plot with current settings"""
        if hasattr(self, 'comparison_manager') and self.get_step2_checked_pairs():
            plot_config = self._get_plot_config()
            self.comparison_manager._generate_multi_pair_plot(plot_config.get('checked_pairs', []), plot_config)
            
    def _on_marker_type_changed(self, marker_text):
        """Handle marker type changes"""
        # This could trigger plot updates if needed
        pass
        
    def get_step2_checked_pairs(self):
        """Get list of pairs that have their Show checkbox checked in step 2"""
        checked_pairs = []
        for row in range(self.active_pair_table_step2.rowCount()):
            checkbox = self.active_pair_table_step2.cellWidget(row, 0)
            if checkbox and checkbox.isChecked():
                pair_name_item = self.active_pair_table_step2.item(row, 1)
                if pair_name_item:
                    pair_name = pair_name_item.text()
                    # Get marker type and color
                    marker_combo = self.active_pair_table_step2.cellWidget(row, 2)
                    color_combo = self.active_pair_table_step2.cellWidget(row, 3)
                    marker_text = marker_combo.currentText() if marker_combo else '○ Circle'
                    color_text = color_combo.currentText() if color_combo else '🔵 Blue'
                    
                    # Find the corresponding pair config
                    for pair in self.active_pairs:
                        if pair['name'] == pair_name:
                            pair_with_styling = pair.copy()
                            pair_with_styling['marker_type'] = marker_text
                            pair_with_styling['marker_color'] = color_text
                            checked_pairs.append(pair_with_styling)
                            break
        return checked_pairs
        
    def update_step2_cumulative_stats(self, stats_text):
        """Update the step 2 cumulative statistics display"""
        self.cumulative_stats_step2.setText(stats_text)
        
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
            
            # Alignment info
            alignment_mode = pair_config.get('alignment_mode', 'index')
            tooltip_lines.append("")
            tooltip_lines.append(f"⚙️ Alignment: {alignment_mode.title()}-based")
            
            # Join all lines and set tooltip
            tooltip_text = "\n".join(tooltip_lines)
            item.setToolTip(tooltip_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting initial pair tooltip: {str(e)}")
            # Fallback simple tooltip
            item.setToolTip(f"Pair: {pair_config.get('name', 'Unknown')}")
        
    def update_cumulative_stats(self, stats_text):
        """Update the cumulative statistics display"""
        self.cumulative_stats_label.setText(stats_text)
        
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
        
    def _on_back_clicked(self):
        """Handle back button click"""
        if self.current_step > 0:
            self.current_step -= 1
            self.steps.setCurrentIndex(self.current_step)
            self._update_navigation()
            
    def _on_next_clicked(self):
        """Handle next button click"""
        if self.current_step == 0:
            # Validate step 1
            if not self.active_pairs:
                QMessageBox.warning(self, "Warning", "Please add at least one comparison pair.")
                return
                
            self.current_step += 1
            self.steps.setCurrentIndex(self.current_step)
            self._update_navigation()
            
            # Update step 2 cumulative statistics when entering step 2
            if hasattr(self, 'comparison_manager'):
                self.comparison_manager._update_step2_cumulative_display()
                # Automatically generate initial plot when entering step 2
                self._auto_generate_plot()
        else:
            # Finish
            self.close()
            
    def _update_navigation(self):
        """Update navigation button states and step labels"""
        self.back_button.setEnabled(self.current_step > 0)
        
        if self.current_step == 0:
            self.next_button.setText("Next →")
            self.step_labels[0].setStyleSheet("font-weight: bold; color: #2c3e50; padding: 10px;")
            self.step_labels[1].setStyleSheet("color: #7f8c8d; padding: 10px;")
        else:
            self.next_button.setText("Finish")
            self.step_labels[0].setStyleSheet("color: #7f8c8d; padding: 10px;")
            self.step_labels[1].setStyleSheet("font-weight: bold; color: #2c3e50; padding: 10px;")
            
    def _get_plot_config(self):
        """Get current plot configuration"""
        # Determine plot type
        if self.plot_type_bland.isChecked():
            plot_type = 'bland_altman'
        elif self.plot_type_scatter.isChecked():
            plot_type = 'scatter'
        elif self.plot_type_residual.isChecked():
            plot_type = 'residual'
        elif self.plot_type_pearson.isChecked():
            plot_type = 'pearson'
        else:
            plot_type = 'scatter'  # Default fallback
            
        config = {
            'plot_type': plot_type,
            'xlabel': self.xlabel_input.text(),
            'ylabel': self.ylabel_input.text(),
            'show_grid': self.grid_checkbox.isChecked(),
            'show_legend': self.legend_checkbox.isChecked(),
            'x_range': self.x_range_input.text(),
            'y_range': self.y_range_input.text(),
            'confidence_interval': self.ci_checkbox.isChecked(),
            'highlight_outliers': self.outlier_checkbox.isChecked(),
            'custom_line': self.custom_line_edit.text() if self.custom_line_checkbox.isChecked() else None,
            'downsample': int(self.downsample_input.text()) if self.downsample_checkbox.isChecked() and self.downsample_input.text().isdigit() else None,
            'density_display': self.density_combo.currentText().lower(),
            'bin_size': int(self.bin_size_input.text()) if self.bin_size_input.text().isdigit() else 20,
            'kde_bandwidth': float(self.kde_bw_input.text()) if self.kde_bw_input.text().replace('.', '').isdigit() else 0.2,
            'checked_pairs': self.get_step2_checked_pairs()
        }
        return config
        
    def get_active_pairs(self):
        """Get list of active comparison pairs"""
        return self.active_pairs.copy()
        
    def get_plot_config(self):
        """Get current plot configuration"""
        return self._get_plot_config()