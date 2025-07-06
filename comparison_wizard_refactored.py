from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, 
    QCheckBox, QTextEdit, QGroupBox, QFormLayout, QSplitter, QApplication, QListWidget, QSpinBox,
    QTableWidget, QRadioButton, QTableWidgetItem, QDialog, QStackedWidget, QMessageBox, QScrollArea,
    QTabWidget, QFrame, QButtonGroup, QDoubleSpinBox, QAbstractItemView, QHeaderView
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
from typing import List, Dict, Any, Optional

from base_plot_wizard import BasePlotWizard
from comparison.comparison_registry import ComparisonRegistry, load_all_comparisons
from channel import Channel
from plot_manager import StylePreviewWidget


class ComparisonWizardRefactored(BasePlotWizard):
    """
    Refactored Comparison Wizard inheriting from BasePlotWizard
    Demonstrates consolidation of plotting functionality
    """
    
    # Additional signals specific to comparison
    pair_added = Signal(dict)
    pair_deleted = Signal()
    plot_generated = Signal(dict)
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        # Initialize comparison-specific state
        self.active_pairs = []
        self.method_controls = {}
        self.overlay_controls = {}
        
        # UI components specific to comparison wizard
        self.ref_file_combo = None
        self.ref_channel_combo = None
        self.test_file_combo = None
        self.test_channel_combo = None
        self.method_list = None
        self.method_controls_widget = None
        self.pairs_table = None
        self.add_pair_btn = None
        self.generate_plot_btn = None
        self.pair_name_input = None
        
        # Alignment controls
        self.alignment_mode_combo = None
        self.index_mode_combo = None
        self.time_mode_combo = None
        self.alignment_parameter_spin = None
        
        # Statistics display
        self.statistics_text = None
        
        # Initialize base class
        super().__init__(file_manager, channel_manager, signal_bus, parent)
        
        # Set window properties
        self.setWindowTitle("Data Comparison Wizard")
        self.setMinimumSize(1200, 800)
        
        # Initialize comparison-specific components
        self._initialize_comparison_components()
        
        # Initialize UI with data
        self._initialize_ui()
        
        self._log_message("Comparison wizard initialized successfully")
    
    def _get_wizard_type(self) -> str:
        """Get the wizard type name"""
        return "Comparison"
    
    def _initialize_comparison_components(self):
        """Initialize comparison-specific components"""
        try:
            # Initialize comparison registry
            if not ComparisonRegistry._initialized:
                load_all_comparisons()
            
            self._log_message("Comparison registry loaded successfully")
            
        except Exception as e:
            self._log_message(f"Error initializing comparison components: {str(e)}")
    
    def _create_main_content(self, layout: QVBoxLayout):
        """Create the main content area specific to comparison wizard"""
        # Create horizontal splitter for left panel columns
        left_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(left_splitter)
        
        # Left column - Methods and Controls
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        self._create_comparison_methods_section(left_column_layout)
        self._create_method_controls_section(left_column_layout)
        
        # Right column - Channel Selection and Alignment
        right_column = QWidget()
        right_column_layout = QVBoxLayout(right_column)
        self._create_channel_selection_section(right_column_layout)
        self._create_alignment_section(right_column_layout)
        self._create_pairs_management_section(right_column_layout)
        
        # Add columns to splitter
        left_splitter.addWidget(left_column)
        left_splitter.addWidget(right_column)
        left_splitter.setSizes([350, 500])
    
    def _create_comparison_methods_section(self, layout: QVBoxLayout):
        """Create the comparison methods section"""
        group = QGroupBox("Comparison Method")
        group_layout = QVBoxLayout(group)
        
        # Method list
        self.method_list = QListWidget()
        self.method_list.setMaximumHeight(120)
        self.method_list.itemClicked.connect(self._on_method_selected)
        self._populate_comparison_methods()
        
        group_layout.addWidget(self.method_list)
        layout.addWidget(group)
    
    def _create_method_controls_section(self, layout: QVBoxLayout):
        """Create the method-specific controls section"""
        group = QGroupBox("Method Controls")
        group_layout = QVBoxLayout(group)
        
        # Stacked widget for method-specific controls
        self.method_controls_widget = QStackedWidget()
        group_layout.addWidget(self.method_controls_widget)
        
        # Create default controls page
        default_page = QWidget()
        default_layout = QVBoxLayout(default_page)
        default_layout.addWidget(QLabel("Select a comparison method to see controls"))
        self.method_controls_widget.addWidget(default_page)
        
        layout.addWidget(group)
    
    def _create_channel_selection_section(self, layout: QVBoxLayout):
        """Create the channel selection section"""
        group = QGroupBox("Channel Selection")
        group_layout = QFormLayout(group)
        
        # Reference file and channel
        self.ref_file_combo = QComboBox()
        self.ref_file_combo.setMinimumWidth(200)
        self.ref_file_combo.currentTextChanged.connect(self._on_ref_file_changed)
        group_layout.addRow("Reference File:", self.ref_file_combo)
        
        self.ref_channel_combo = QComboBox()
        self.ref_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
        group_layout.addRow("Reference Channel:", self.ref_channel_combo)
        
        # Test file and channel
        self.test_file_combo = QComboBox()
        self.test_file_combo.currentTextChanged.connect(self._on_test_file_changed)
        group_layout.addRow("Test File:", self.test_file_combo)
        
        self.test_channel_combo = QComboBox()
        self.test_channel_combo.currentTextChanged.connect(self._on_channel_selection_changed)
        group_layout.addRow("Test Channel:", self.test_channel_combo)
        
        layout.addWidget(group)
    
    def _create_alignment_section(self, layout: QVBoxLayout):
        """Create the alignment section"""
        group = QGroupBox("Alignment")
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
    
    def _create_pairs_management_section(self, layout: QVBoxLayout):
        """Create the pairs management section"""
        group = QGroupBox("Pairs Management")
        group_layout = QVBoxLayout(group)
        
        # Pair name input
        pair_name_layout = QHBoxLayout()
        pair_name_layout.addWidget(QLabel("Pair Name:"))
        self.pair_name_input = QLineEdit()
        self.pair_name_input.setPlaceholderText("Auto-generated")
        pair_name_layout.addWidget(self.pair_name_input)
        group_layout.addLayout(pair_name_layout)
        
        # Action buttons
        buttons_layout = QHBoxLayout()
        
        self.add_pair_btn = QPushButton("Add Pair")
        self.add_pair_btn.clicked.connect(self._on_add_pair)
        
        self.generate_plot_btn = QPushButton("Generate Plot")
        self.generate_plot_btn.clicked.connect(self._on_generate_plot)
        
        buttons_layout.addWidget(self.add_pair_btn)
        buttons_layout.addWidget(self.generate_plot_btn)
        group_layout.addLayout(buttons_layout)
        
        layout.addWidget(group)
    
    def _create_results_tab(self) -> Optional[QWidget]:
        """Override to create comparison pairs results table"""
        tab_widget = QTabWidget()
        
        # Pairs table tab
        pairs_tab = QWidget()
        pairs_layout = QVBoxLayout(pairs_tab)
        
        self.pairs_table = QTableWidget()
        self.pairs_table.setColumnCount(6)
        self.pairs_table.setHorizontalHeaderLabels([
            "Show", "Pair Name", "Reference", "Test", "Method", "Actions"
        ])
        self.pairs_table.setAlternatingRowColors(True)
        self.pairs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        
        # Configure column widths
        header = self.pairs_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Fixed)   # Show
        header.setSectionResizeMode(1, QHeaderView.Stretch) # Pair Name
        header.setSectionResizeMode(2, QHeaderView.Fixed)   # Reference
        header.setSectionResizeMode(3, QHeaderView.Fixed)   # Test
        header.setSectionResizeMode(4, QHeaderView.Fixed)   # Method
        header.setSectionResizeMode(5, QHeaderView.Fixed)   # Actions
        
        self.pairs_table.setColumnWidth(0, 50)   # Show
        self.pairs_table.setColumnWidth(2, 150)  # Reference
        self.pairs_table.setColumnWidth(3, 150)  # Test
        self.pairs_table.setColumnWidth(4, 120)  # Method
        self.pairs_table.setColumnWidth(5, 100)  # Actions
        
        pairs_layout.addWidget(self.pairs_table)
        tab_widget.addTab(pairs_tab, "Pairs")
        
        # Statistics tab
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.statistics_text = QTextEdit()
        self.statistics_text.setReadOnly(True)
        self.statistics_text.setFont(QFont("Consolas", 9))
        stats_layout.addWidget(self.statistics_text)
        
        tab_widget.addTab(stats_tab, "Statistics")
        
        return tab_widget
    
    def _get_channels_to_plot(self) -> List[Channel]:
        """Get channels to plot based on active pairs"""
        channels = []
        
        # Add reference and test channels from active pairs
        for pair in self.active_pairs:
            if pair.get('show', True):
                ref_channel = pair.get('reference_channel')
                test_channel = pair.get('test_channel')
                
                if ref_channel:
                    channels.append(ref_channel)
                if test_channel:
                    channels.append(test_channel)
        
        return channels
    
    def _populate_comparison_methods(self):
        """Populate comparison methods from registry"""
        try:
            methods = ComparisonRegistry.get_all_methods()
            
            if not methods:
                # Fallback methods
                methods = [
                    "Bland-Altman Analysis",
                    "Correlation Analysis", 
                    "Residual Analysis",
                    "Statistical Tests",
                    "Cross-Correlation"
                ]
                self._log_message("Using fallback methods - comparison registry may not be loaded")
            
            self.method_list.clear()
            for method in methods:
                self.method_list.addItem(method)
            
            # Select first method by default
            if self.method_list.count() > 0:
                self.method_list.setCurrentRow(0)
                self._on_method_selected(self.method_list.item(0))
                
        except Exception as e:
            self._log_message(f"Error populating comparison methods: {str(e)}")
    
    def _initialize_ui(self):
        """Initialize UI with data"""
        # Populate file combos
        self._populate_file_combos()
        
        # Auto-populate alignment parameters
        self._auto_populate_alignment_parameters()
    
    def _populate_file_combos(self):
        """Populate file combo boxes"""
        if not self.file_manager:
            return
        
        files = self.file_manager.get_all_files()
        
        # Clear existing items
        if self.ref_file_combo:
            self.ref_file_combo.clear()
        if self.test_file_combo:
            self.test_file_combo.clear()
        
        if not files:
            return
        
        # Add files to combos
        for file_info in files:
            display_name = file_info.filename
            if self.ref_file_combo:
                self.ref_file_combo.addItem(display_name, file_info)
            if self.test_file_combo:
                self.test_file_combo.addItem(display_name, file_info)
        
        self._log_message(f"Loaded {len(files)} files for comparison")
    
    def _auto_populate_alignment_parameters(self):
        """Auto-populate alignment parameters"""
        # Set default alignment parameters
        if self.alignment_mode_combo:
            self.alignment_mode_combo.setCurrentText("None")
        if self.index_mode_combo:
            self.index_mode_combo.setCurrentText("Start")
        if self.time_mode_combo:
            self.time_mode_combo.setCurrentText("Start")
        if self.alignment_parameter_spin:
            self.alignment_parameter_spin.setValue(0.0)
    
    def _update_active_pairs_table(self):
        """Update the active pairs table"""
        if not self.pairs_table:
            return
        
        self.pairs_table.setRowCount(len(self.active_pairs))
        
        for row, pair in enumerate(self.active_pairs):
            # Show checkbox
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(pair.get('show', True))
            show_checkbox.toggled.connect(
                lambda checked, p=pair: self._on_pair_visibility_changed(p, checked)
            )
            self.pairs_table.setCellWidget(row, 0, show_checkbox)
            
            # Pair name
            name_item = QTableWidgetItem(pair.get('name', f'Pair {row + 1}'))
            self.pairs_table.setItem(row, 1, name_item)
            
            # Reference channel
            ref_channel = pair.get('reference_channel')
            ref_text = ref_channel.legend_label if ref_channel else "N/A"
            ref_item = QTableWidgetItem(ref_text)
            self.pairs_table.setItem(row, 2, ref_item)
            
            # Test channel
            test_channel = pair.get('test_channel')
            test_text = test_channel.legend_label if test_channel else "N/A"
            test_item = QTableWidgetItem(test_text)
            self.pairs_table.setItem(row, 3, test_item)
            
            # Method
            method_item = QTableWidgetItem(pair.get('method', 'Unknown'))
            self.pairs_table.setItem(row, 4, method_item)
            
            # Actions
            actions_btn = QPushButton("⚙")
            actions_btn.clicked.connect(lambda: self._show_pair_info(pair))
            self.pairs_table.setCellWidget(row, 5, actions_btn)
    
    def _get_current_pair_config(self) -> Dict[str, Any]:
        """Get current pair configuration"""
        ref_channel = self.ref_channel_combo.currentData() if self.ref_channel_combo else None
        test_channel = self.test_channel_combo.currentData() if self.test_channel_combo else None
        
        method_item = self.method_list.currentItem()
        method = method_item.text() if method_item else "Unknown"
        
        pair_name = self.pair_name_input.text() if self.pair_name_input else ""
        if not pair_name:
            pair_name = f"Pair {len(self.active_pairs) + 1}"
        
        return {
            'name': pair_name,
            'reference_channel': ref_channel,
            'test_channel': test_channel,
            'method': method,
            'alignment_config': self._get_alignment_config(),
            'method_parameters': self._get_method_parameters(),
            'show': True
        }
    
    def _get_alignment_config(self) -> Dict[str, Any]:
        """Get alignment configuration"""
        return {
            'mode': self.alignment_mode_combo.currentText() if self.alignment_mode_combo else "None",
            'index_mode': self.index_mode_combo.currentText() if self.index_mode_combo else "Start",
            'time_mode': self.time_mode_combo.currentText() if self.time_mode_combo else "Start",
            'parameter': self.alignment_parameter_spin.value() if self.alignment_parameter_spin else 0.0
        }
    
    def _get_method_parameters(self) -> Dict[str, Any]:
        """Get method-specific parameters"""
        # This would be implemented based on the selected method
        # For now, return empty dict
        return {}
    
    def _update_channel_combo(self, filename, combo):
        """Update channel combo based on selected file"""
        if not combo or not self.channel_manager:
            return
        
        combo.clear()
        
        # Find file by filename
        files = self.file_manager.get_all_files()
        selected_file = next((f for f in files if f.filename == filename), None)
        
        if selected_file:
            channels = self.channel_manager.get_channels_by_file(selected_file.file_id)
            for channel in channels:
                if channel.show and channel.ydata is not None:
                    display_name = channel.legend_label or channel.channel_id
                    combo.addItem(display_name, channel)
    
    # Event handlers
    def _on_method_selected(self, item):
        """Handle method selection"""
        if not item:
            return
        
        method_name = item.text()
        self._log_message(f"Selected comparison method: {method_name}")
        
        # Update method controls (simplified for now)
        # In a full implementation, this would show method-specific controls
    
    def _on_ref_file_changed(self, filename):
        """Handle reference file change"""
        self._update_channel_combo(filename, self.ref_channel_combo)
        self._on_channel_selection_changed()
    
    def _on_test_file_changed(self, filename):
        """Handle test file change"""
        self._update_channel_combo(filename, self.test_channel_combo)
        self._on_channel_selection_changed()
    
    def _on_channel_selection_changed(self):
        """Handle channel selection change"""
        self._auto_populate_alignment_parameters()
        self._update_default_pair_name()
    
    def _update_default_pair_name(self):
        """Update default pair name based on selected channels"""
        if not self.pair_name_input:
            return
        
        ref_channel = self.ref_channel_combo.currentData() if self.ref_channel_combo else None
        test_channel = self.test_channel_combo.currentData() if self.test_channel_combo else None
        
        if ref_channel and test_channel:
            ref_name = ref_channel.legend_label or "Ref"
            test_name = test_channel.legend_label or "Test"
            default_name = f"{ref_name} vs {test_name}"
            self.pair_name_input.setPlaceholderText(default_name)
    
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
    
    def _on_add_pair(self):
        """Handle add pair button click"""
        try:
            pair_config = self._get_current_pair_config()
            
            if not pair_config['reference_channel'] or not pair_config['test_channel']:
                self._log_message("Please select both reference and test channels")
                return
            
            # Add pair to active pairs
            self.active_pairs.append(pair_config)
            
            # Add channels to plot
            self.add_channel(pair_config['reference_channel'], visible=True)
            self.add_channel(pair_config['test_channel'], visible=True)
            
            # Update UI
            self._update_active_pairs_table()
            self._schedule_plot_update()
            
            self._log_message(f"Added pair: {pair_config['name']}")
            self.pair_added.emit(pair_config)
            
            # Clear pair name input
            if self.pair_name_input:
                self.pair_name_input.clear()
            
        except Exception as e:
            self._log_message(f"Error adding pair: {str(e)}")
    
    def _on_generate_plot(self):
        """Handle generate plot button click"""
        try:
            if not self.active_pairs:
                self._log_message("No pairs available for plotting")
                return
            
            # Force plot update
            self._force_plot_update()
            
            # Generate statistics
            self._generate_statistics()
            
            self._log_message("Plot generated successfully")
            self.plot_generated.emit({'pairs': self.active_pairs})
            
        except Exception as e:
            self._log_message(f"Error generating plot: {str(e)}")
    
    def _on_pair_visibility_changed(self, pair, visible):
        """Handle pair visibility change"""
        pair['show'] = visible
        
        # Update channel visibility
        ref_channel = pair.get('reference_channel')
        test_channel = pair.get('test_channel')
        
        if ref_channel:
            if visible:
                self.visible_channels.add(ref_channel.channel_id)
            else:
                self.visible_channels.discard(ref_channel.channel_id)
        
        if test_channel:
            if visible:
                self.visible_channels.add(test_channel.channel_id)
            else:
                self.visible_channels.discard(test_channel.channel_id)
        
        self._schedule_plot_update()
    
    def _show_pair_info(self, pair):
        """Show pair information"""
        info_text = f"""
        Pair: {pair.get('name', 'Unknown')}
        Method: {pair.get('method', 'Unknown')}
        Reference: {pair.get('reference_channel', {}).get('legend_label', 'N/A')}
        Test: {pair.get('test_channel', {}).get('legend_label', 'N/A')}
        Alignment: {pair.get('alignment_config', {}).get('mode', 'None')}
        """
        
        self._log_message(info_text.strip())
    
    def _generate_statistics(self):
        """Generate comparison statistics"""
        if not self.statistics_text:
            return
        
        stats_text = "Comparison Statistics\n"
        stats_text += "=" * 50 + "\n\n"
        
        for i, pair in enumerate(self.active_pairs):
            if pair.get('show', True):
                stats_text += f"Pair {i+1}: {pair.get('name', 'Unknown')}\n"
                stats_text += f"Method: {pair.get('method', 'Unknown')}\n"
                
                ref_channel = pair.get('reference_channel')
                test_channel = pair.get('test_channel')
                
                if ref_channel and test_channel:
                    # Calculate basic statistics
                    if hasattr(ref_channel, 'ydata') and hasattr(test_channel, 'ydata'):
                        if ref_channel.ydata is not None and test_channel.ydata is not None:
                            ref_mean = np.mean(ref_channel.ydata)
                            test_mean = np.mean(test_channel.ydata)
                            correlation = np.corrcoef(ref_channel.ydata, test_channel.ydata)[0, 1]
                            
                            stats_text += f"Reference Mean: {ref_mean:.3f}\n"
                            stats_text += f"Test Mean: {test_mean:.3f}\n"
                            stats_text += f"Correlation: {correlation:.3f}\n"
                
                stats_text += "\n"
        
        self.statistics_text.setPlainText(stats_text)
    
    def get_active_pairs(self) -> List[Dict[str, Any]]:
        """Get active comparison pairs"""
        return self.active_pairs
    
    def get_checked_pairs(self) -> List[Dict[str, Any]]:
        """Get checked (visible) comparison pairs"""
        return [pair for pair in self.active_pairs if pair.get('show', True)] 