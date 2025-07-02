# Placeholder for PlotWizardWindow - to be filled based on prior GUI structure and ChannelInfo class

from PySide6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QCheckBox, QLabel, QSplitter,
    QFrame, QTextEdit, QTabWidget, QSpinBox, QColorDialog, QDialog, QFileDialog,
    QDoubleSpinBox, QGridLayout, QGroupBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class PlotWizardWindow(QMainWindow):
    def __init__(self, file_manager, channel_manager):
        super().__init__()
        self.setWindowTitle("Custom Plot Builder")
        self.setMinimumSize(1400, 800)

        self.file_manager = file_manager
        self.channel_manager = channel_manager

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT PANEL: Channel Selector + Config Tables
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)

        # Channel Selector Row
        selector_layout = QHBoxLayout()
        self.file_dropdown = QComboBox()
        self.channel_dropdown = QComboBox()
        self.type_dropdown = QComboBox()
        self.type_dropdown.addItems(["Line", "Scatter", "Spectrogram", "Bar"])
        self.legend_entry = QLineEdit()
        self.legend_entry.setPlaceholderText("Legend Name")
        self.add_btn = QPushButton("Add to Plot")

        selector_layout.addWidget(self.file_dropdown)
        selector_layout.addWidget(self.channel_dropdown)
        selector_layout.addWidget(self.type_dropdown)
        selector_layout.addWidget(self.legend_entry)
        selector_layout.addWidget(self.add_btn)

        left_layout.addLayout(selector_layout)

        # Config Tables
        self.line_config_table = QTableWidget(0, 6)
        self.line_config_table.setHorizontalHeaderLabels(["Subplot#", "Legend", "Color", "Line", "Marker", "Y Axis"])
        self.subplot_config_table = QTableWidget(0, 7)
        self.subplot_config_table.setHorizontalHeaderLabels(["Subplot#", "Xlabel", "Ylabel", "Legend Label", "Legend", "Legend Pos", "Advanced"])

        left_layout.addWidget(QLabel("Line Configurations"))
        left_layout.addWidget(self.line_config_table)
        left_layout.addWidget(QLabel("Subplot Configurations"))
        left_layout.addWidget(self.subplot_config_table)

        # Subplot Dimension Controls
        dimension_layout = QHBoxLayout()
        dimension_label = QLabel("Subplot Dimension:")
        self.rows_spinbox = QSpinBox()
        self.rows_spinbox.setMinimum(1)
        self.rows_spinbox.setMaximum(10)
        self.rows_spinbox.setValue(1)
        self.rows_spinbox.setToolTip("Number of subplot rows")
        
        dimension_x_label = QLabel("×")
        
        self.cols_spinbox = QSpinBox()
        self.cols_spinbox.setMinimum(1)
        self.cols_spinbox.setMaximum(10)
        self.cols_spinbox.setValue(1)
        self.cols_spinbox.setToolTip("Number of subplot columns")
        
        dimension_layout.addWidget(dimension_label)
        dimension_layout.addWidget(self.rows_spinbox)
        dimension_layout.addWidget(dimension_x_label)
        dimension_layout.addWidget(self.cols_spinbox)
        dimension_layout.addStretch()  # Push controls to the left
        
        left_layout.addLayout(dimension_layout)

        # Plot Configuration Panel (Structured GUI)
        config_group = QGroupBox("Plot Configuration")
        config_layout = QGridLayout(config_group)
        
        # Create configuration controls
        row = 0
        
        # Share Axes (start from row 0)
        # (removed Figure Size and DPI controls)
        
        # Share Axes
        config_layout.addWidget(QLabel("Share X Axis:"), row, 0)
        self.sharex_checkbox = QCheckBox()
        config_layout.addWidget(self.sharex_checkbox, row, 1)
        
        config_layout.addWidget(QLabel("Share Y Axis:"), row, 2)
        self.sharey_checkbox = QCheckBox()
        config_layout.addWidget(self.sharey_checkbox, row, 3)
        row += 1
        
        # Tick Settings
        config_layout.addWidget(QLabel("Tick Direction:"), row, 0)
        self.tick_direction_combo = QComboBox()
        self.tick_direction_combo.addItems(["in", "out", "inout"])
        config_layout.addWidget(self.tick_direction_combo, row, 1)
        
        config_layout.addWidget(QLabel("Tick Width:"), row, 2)
        self.tick_width_spinbox = QDoubleSpinBox()
        self.tick_width_spinbox.setRange(0.1, 5.0)
        self.tick_width_spinbox.setSingleStep(0.1)
        self.tick_width_spinbox.setValue(1.0)
        config_layout.addWidget(self.tick_width_spinbox, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Tick Length:"), row, 0)
        self.tick_length_spinbox = QDoubleSpinBox()
        self.tick_length_spinbox.setRange(1.0, 20.0)
        self.tick_length_spinbox.setValue(4.0)
        config_layout.addWidget(self.tick_length_spinbox, row, 1)
        
        config_layout.addWidget(QLabel("Line Width:"), row, 2)
        self.line_width_spinbox = QDoubleSpinBox()
        self.line_width_spinbox.setRange(0.1, 10.0)
        self.line_width_spinbox.setSingleStep(0.1)
        self.line_width_spinbox.setValue(2.0)
        config_layout.addWidget(self.line_width_spinbox, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Axis Line Width:"), row, 0)
        self.axis_linewidth_spinbox = QDoubleSpinBox()
        self.axis_linewidth_spinbox.setRange(0.1, 5.0)
        self.axis_linewidth_spinbox.setSingleStep(0.1)
        self.axis_linewidth_spinbox.setValue(1.5)
        config_layout.addWidget(self.axis_linewidth_spinbox, row, 1)
        
        config_layout.addWidget(QLabel("Font Size:"), row, 2)
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(6, 72)
        self.font_size_spinbox.setValue(12)
        config_layout.addWidget(self.font_size_spinbox, row, 3)
        row += 1
        
        # Font Settings
        config_layout.addWidget(QLabel("Font Weight:"), row, 0)
        self.font_weight_combo = QComboBox()
        self.font_weight_combo.addItems(["normal", "bold", "light", "ultralight", "heavy", "black"])
        config_layout.addWidget(self.font_weight_combo, row, 1)
        
        config_layout.addWidget(QLabel("Font Family:"), row, 2)
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["sans-serif", "serif", "monospace", "cursive", "fantasy"])
        config_layout.addWidget(self.font_family_combo, row, 3)
        row += 1
        
        # Grid and Box Style
        config_layout.addWidget(QLabel("Grid:"), row, 0)
        self.grid_checkbox = QCheckBox()
        self.grid_checkbox.setChecked(True)
        config_layout.addWidget(self.grid_checkbox, row, 1)
        
        config_layout.addWidget(QLabel("Box Style:"), row, 2)
        self.box_style_combo = QComboBox()
        self.box_style_combo.addItems(["full", "left_bottom", "none"])
        config_layout.addWidget(self.box_style_combo, row, 3)
        row += 1
        
        # Legend Settings
        config_layout.addWidget(QLabel("Legend Font Size:"), row, 0)
        self.legend_fontsize_spinbox = QSpinBox()
        self.legend_fontsize_spinbox.setRange(6, 72)
        self.legend_fontsize_spinbox.setValue(10)
        config_layout.addWidget(self.legend_fontsize_spinbox, row, 1)
        
        config_layout.addWidget(QLabel("Legend Columns:"), row, 2)
        self.legend_ncol_spinbox = QSpinBox()
        self.legend_ncol_spinbox.setRange(1, 10)
        self.legend_ncol_spinbox.setValue(1)
        config_layout.addWidget(self.legend_ncol_spinbox, row, 3)
        row += 1
        
        config_layout.addWidget(QLabel("Legend Frame:"), row, 0)
        self.legend_frameon_checkbox = QCheckBox()
        self.legend_frameon_checkbox.setChecked(True)
        config_layout.addWidget(self.legend_frameon_checkbox, row, 1)
        
        config_layout.addWidget(QLabel("Tight Layout:"), row, 2)
        self.tight_layout_checkbox = QCheckBox()
        self.tight_layout_checkbox.setChecked(True)
        config_layout.addWidget(self.tight_layout_checkbox, row, 3)
        
        left_layout.addWidget(config_group)

        splitter.addWidget(self.left_panel)

        # RIGHT PANEL: Plot Area
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self.right_panel)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(self.right_panel)

        splitter.setSizes([500, 900])

        # TODO: Hook up logic for dropdown population, channel selection, adding to plot, rendering, saving layout