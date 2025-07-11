from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QColorDialog, QFrame,
    QGroupBox, QButtonGroup, QRadioButton, QDialogButtonBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSlider, QTabWidget, QWidget
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette
from typing import Optional, Dict, Any

from plot_manager import PLOT_STYLES


class ColorButton(QPushButton):
    """Custom button that displays and allows selection of colors"""
    
    color_changed = Signal(str)  # hex color string
    
    def __init__(self, initial_color: str = "#1f77b4"):
        super().__init__()
        self.current_color = initial_color
        self.setFixedSize(40, 30)
        self.update_button_color()
        self.clicked.connect(self.select_color)
    
    def update_button_color(self):
        """Update button appearance to show current color"""
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_color};
                border: 2px solid #333;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                border: 2px solid #666;
            }}
        """)
    
    def select_color(self):
        """Open color dialog and update color"""
        color = QColorDialog.getColor(QColor(self.current_color), self)
        if color.isValid():
            self.current_color = color.name()
            self.update_button_color()
            self.color_changed.emit(self.current_color)
    
    def set_color(self, color: str):
        """Set color programmatically"""
        self.current_color = color
        self.update_button_color()
    
    def get_color(self) -> str:
        """Get current color as hex string"""
        return self.current_color


class MarkerWizard(QDialog):
    """
    Dialog for editing scatter plot marker properties and density plot colorschemes
    Designed for use with comparison wizard pairs
    """
    
    # Signals
    marker_updated = Signal(dict)  # marker properties when changes are applied
    
    def __init__(self, pair_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.pair_config = pair_config
        self.original_properties = self._backup_marker_properties()
        
        pair_name = pair_config.get('name', 'Unnamed Pair')
        self.setWindowTitle(f"Marker Properties - {pair_name}")
        self.setModal(True)
        self.setFixedSize(500, 700)
        
        self.init_ui()
        self.load_marker_properties()
    
    def _backup_marker_properties(self) -> dict:
        """Backup original marker properties for cancel operation"""
        return {
            'marker_type': self.pair_config.get('marker_type', 'o'),
            'marker_color': self.pair_config.get('marker_color', '🔵 Blue'),
            'marker_size': self.pair_config.get('marker_size', 50),
            'marker_alpha': self.pair_config.get('marker_alpha', 0.7),
            'edge_color': self.pair_config.get('edge_color', '#000000'),
            'edge_width': self.pair_config.get('edge_width', 1.0),
            'fill_style': self.pair_config.get('fill_style', 'full'),
            'z_order': self.pair_config.get('z_order', 0),
            'x_axis': self.pair_config.get('x_axis', 'bottom'),
            'colormap': self.pair_config.get('colormap', 'viridis'),
            'colormap_alpha': self.pair_config.get('colormap_alpha', 0.7),
            'colormap_reverse': self.pair_config.get('colormap_reverse', False),
            'show_colorbar': self.pair_config.get('show_colorbar', True),
            'colorbar_position': self.pair_config.get('colorbar_position', 'right'),
            'colorbar_label': self.pair_config.get('colorbar_label', 'Density'),
            'colorbar_pad': self.pair_config.get('colorbar_pad', 0.05),
            'colorbar_shrink': self.pair_config.get('colorbar_shrink', 0.8),
            'colorbar_aspect': self.pair_config.get('colorbar_aspect', 20),
            'colorbar_ticks': self.pair_config.get('colorbar_ticks', 5),
            'tick_format': self.pair_config.get('tick_format', '%.1f'),
            'colorbar_label_fontsize': self.pair_config.get('colorbar_label_fontsize', 10),
            'colorbar_tick_fontsize': self.pair_config.get('colorbar_tick_fontsize', 8),
            'gridsize': self.pair_config.get('gridsize', 50),
            'kde_bandwidth': self.pair_config.get('kde_bandwidth', 1.0),
            'contour_levels': self.pair_config.get('contour_levels', 15)
        }
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        pair_name = self.pair_config.get('name', 'Unnamed Pair')
        title = QLabel(f"Editing Marker: {pair_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Pair info
        info_text = f"{self.pair_config.get('ref_channel', 'Unknown')} vs {self.pair_config.get('test_channel', 'Unknown')}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Create tab widget for organized settings
        tab_widget = QTabWidget()
        
        # Scatter Markers tab
        scatter_tab = self._create_scatter_tab()
        tab_widget.addTab(scatter_tab, "Scatter Markers")
        
        # Density Plot tab
        density_tab = self._create_density_tab()
        tab_widget.addTab(density_tab, "Density Plots")
        
        layout.addWidget(tab_widget)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
    
    def _create_scatter_tab(self) -> QWidget:
        """Create the scatter markers tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Main form
        form_layout = QFormLayout()
        
        # Marker Type
        self.marker_combo = QComboBox()
        marker_types = [
            ('○ Circle', 'o'),
            ('□ Square', 's'),
            ('△ Triangle', '^'),
            ('◇ Diamond', 'D'),
            ('▽ Inverted Triangle', 'v'),
            ('◁ Left Triangle', '<'),
            ('▷ Right Triangle', '>'),
            ('⬟ Pentagon', 'p'),
            ('✦ Star', '*'),
            ('⬢ Hexagon', 'h'),
            ('+ Plus', '+'),
            ('× Cross', 'x'),
            ('| Vertical Line', '|'),
            ('— Horizontal Line', '_')
        ]
        for name, value in marker_types:
            self.marker_combo.addItem(name, value)
        form_layout.addRow("Marker Type:", self.marker_combo)
        
        # Color Selection
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_combo = QComboBox()
        color_options = [
            ('🔵 Blue', '#1f77b4'),
            ('🔴 Red', '#d62728'),
            ('🟢 Green', '#2ca02c'),
            ('🟣 Purple', '#9467bd'),
            ('🟠 Orange', '#ff7f0e'),
            ('🟤 Brown', '#8c564b'),
            ('🩷 Pink', '#e377c2'),
            ('⚫ Gray', '#7f7f7f'),
            ('🟡 Yellow', '#bcbd22'),
            ('🔶 Cyan', '#17becf'),
            ('Custom', 'custom')
        ]
        for name, value in color_options:
            self.color_combo.addItem(name, value)
        self.color_combo.currentTextChanged.connect(self.on_color_combo_changed)
        self.color_button.color_changed.connect(self.on_color_button_changed)
        
        color_layout.addWidget(self.color_button)
        color_layout.addWidget(self.color_combo)
        form_layout.addRow("Marker Color:", color_layout)
        
        # Size
        size_layout = QHBoxLayout()
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 500)
        self.size_spin.setValue(50)
        self.size_spin.setSuffix(" pts")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 200)
        self.size_slider.setValue(50)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        
        size_layout.addWidget(self.size_spin)
        size_layout.addWidget(self.size_slider)
        form_layout.addRow("Marker Size:", size_layout)
        
        # Alpha (Transparency)
        alpha_layout = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(0.7)
        self.alpha_spin.setDecimals(2)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(70)
        self.alpha_spin.valueChanged.connect(lambda v: self.alpha_slider.setValue(int(v * 100)))
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_spin.setValue(v / 100.0))
        
        alpha_layout.addWidget(self.alpha_spin)
        alpha_layout.addWidget(self.alpha_slider)
        form_layout.addRow("Transparency:", alpha_layout)
        
        layout.addLayout(form_layout)
        
        # Edge Properties Group
        edge_group = QGroupBox("Edge Properties")
        edge_layout = QFormLayout(edge_group)
        
        # Edge Color
        self.edge_color_button = ColorButton("#000000")
        edge_layout.addRow("Edge Color:", self.edge_color_button)
        
        # Edge Width
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 10.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(1.0)
        self.edge_width_spin.setDecimals(1)
        edge_layout.addRow("Edge Width:", self.edge_width_spin)
        
        # Fill Style
        self.fill_combo = QComboBox()
        fill_options = [
            ('Full', 'full'),
            ('Left Half', 'left'),
            ('Right Half', 'right'),
            ('Bottom Half', 'bottom'),
            ('Top Half', 'top'),
            ('None (Outline Only)', 'none')
        ]
        for name, value in fill_options:
            self.fill_combo.addItem(name, value)
        edge_layout.addRow("Fill Style:", self.fill_combo)
        
        layout.addWidget(edge_group)
        
        # Display Options Group
        display_group = QGroupBox("Display Options")
        display_layout = QFormLayout(display_group)
        
        # Bring to Front checkbox
        self.bring_to_front_checkbox = QCheckBox("Bring to Front")
        self.bring_to_front_checkbox.setToolTip("Bring this marker to the top layer in the plot")
        display_layout.addRow("", self.bring_to_front_checkbox)
        
        # Axis Position Group
        axis_group = QGroupBox("Axis Position")
        axis_layout = QFormLayout(axis_group)
        
        # X-Axis Selection
        x_axis_layout = QHBoxLayout()
        self.x_axis_button_group = QButtonGroup()
        self.bottom_x_axis_radio = QRadioButton("Bottom")
        self.top_x_axis_radio = QRadioButton("Top")
        
        self.x_axis_button_group.addButton(self.bottom_x_axis_radio, 0)
        self.x_axis_button_group.addButton(self.top_x_axis_radio, 1)
        
        x_axis_layout.addWidget(self.bottom_x_axis_radio)
        x_axis_layout.addWidget(self.top_x_axis_radio)
        axis_layout.addRow("X-Axis:", x_axis_layout)
        
        layout.addWidget(axis_group)
        layout.addWidget(display_group)
        
        # Connect change signals for live preview
        self.marker_combo.currentTextChanged.connect(self.update_preview)
        self.size_spin.valueChanged.connect(self.update_preview)
        self.alpha_spin.valueChanged.connect(self.update_preview)
        self.edge_color_button.color_changed.connect(self.update_preview)
        self.edge_width_spin.valueChanged.connect(self.update_preview)
        self.fill_combo.currentTextChanged.connect(self.update_preview)
        self.bring_to_front_checkbox.toggled.connect(self.update_preview)
        self.x_axis_button_group.buttonToggled.connect(self.update_preview)
        
        return tab
    
    def _create_density_tab(self) -> QWidget:
        """Create the density plot colorscheme tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Colorscheme Selection
        colorscheme_group = QGroupBox("Colorscheme Selection")
        colorscheme_layout = QFormLayout(colorscheme_group)
        
        # Colormap selection
        self.colormap_combo = QComboBox()
        colormaps = [
            # Sequential colormaps (good for density plots)
            ('Viridis', 'viridis'),
            ('Plasma', 'plasma'),
            ('Inferno', 'inferno'),
            ('Magma', 'magma'),
            ('Cividis', 'cividis'),
            ('Blues', 'Blues'),
            ('Greens', 'Greens'),
            ('Reds', 'Reds'),
            ('Purples', 'Purples'),
            ('Oranges', 'Oranges'),
            ('Greys', 'Greys'),
            ('YlOrRd', 'YlOrRd'),
            ('YlGnBu', 'YlGnBu'),
            ('RdPu', 'RdPu'),
            ('BuPu', 'BuPu'),
            ('GnBu', 'GnBu'),
            ('OrRd', 'OrRd'),
            ('PuBu', 'PuBu'),
            ('PuRd', 'PuRd'),
            ('RdYlBu', 'RdYlBu'),
            ('Spectral', 'Spectral'),
            ('RdBu', 'RdBu'),
            ('PiYG', 'PiYG'),
            ('PRGn', 'PRGn'),
            ('BrBG', 'BrBG'),
            ('PuOr', 'PuOr'),
            ('RdGy', 'RdGy'),
            ('RdYlGn', 'RdYlGn'),
            ('Set1', 'Set1'),
            ('Set2', 'Set2'),
            ('Set3', 'Set3'),
            ('Paired', 'Paired'),
            ('Accent', 'Accent'),
            ('Dark2', 'Dark2'),
            ('Pastel1', 'Pastel1'),
            ('Pastel2', 'Pastel2'),
            ('tab10', 'tab10'),
            ('tab20', 'tab20'),
            ('tab20b', 'tab20b'),
            ('tab20c', 'tab20c'),
            ('flag', 'flag'),
            ('prism', 'prism'),
            ('ocean', 'ocean'),
            ('gist_earth', 'gist_earth'),
            ('terrain', 'terrain'),
            ('gist_stern', 'gist_stern'),
            ('gnuplot', 'gnuplot'),
            ('gnuplot2', 'gnuplot2'),
            ('CMRmap', 'CMRmap'),
            ('cubehelix', 'cubehelix'),
            ('brg', 'brg'),
            ('hsv', 'hsv'),
            ('jet', 'jet'),
            ('rainbow', 'rainbow'),
            ('nipy_spectral', 'nipy_spectral'),
            ('gist_ncar', 'gist_ncar'),
            ('gist_rainbow', 'gist_rainbow'),
            ('gist_heat', 'gist_heat'),
            ('copper', 'copper'),
            ('bone', 'bone'),
            ('pink', 'pink'),
            ('spring', 'spring'),
            ('summer', 'summer'),
            ('autumn', 'autumn'),
            ('winter', 'winter'),
            ('cool', 'cool'),
            ('Wistia', 'Wistia'),
            ('hot', 'hot'),
            ('afmhot', 'afmhot'),
            ('gist_heat', 'gist_heat'),
            ('copper', 'copper')
        ]
        for name, value in colormaps:
            self.colormap_combo.addItem(name, value)
        colorscheme_layout.addRow("Colormap:", self.colormap_combo)
        
        # Reverse colormap
        self.reverse_colormap_checkbox = QCheckBox("Reverse Colormap")
        self.reverse_colormap_checkbox.setToolTip("Reverse the colormap direction")
        colorscheme_layout.addRow("", self.reverse_colormap_checkbox)
        
        # Colormap transparency
        cmap_alpha_layout = QHBoxLayout()
        self.cmap_alpha_spin = QDoubleSpinBox()
        self.cmap_alpha_spin.setRange(0.0, 1.0)
        self.cmap_alpha_spin.setSingleStep(0.1)
        self.cmap_alpha_spin.setValue(0.7)
        self.cmap_alpha_spin.setDecimals(2)
        self.cmap_alpha_slider = QSlider(Qt.Horizontal)
        self.cmap_alpha_slider.setRange(0, 100)
        self.cmap_alpha_slider.setValue(70)
        self.cmap_alpha_spin.valueChanged.connect(lambda v: self.cmap_alpha_slider.setValue(int(v * 100)))
        self.cmap_alpha_slider.valueChanged.connect(lambda v: self.cmap_alpha_spin.setValue(v / 100.0))
        
        cmap_alpha_layout.addWidget(self.cmap_alpha_spin)
        cmap_alpha_layout.addWidget(self.cmap_alpha_slider)
        colorscheme_layout.addRow("Colormap Transparency:", cmap_alpha_layout)
        
        layout.addWidget(colorscheme_group)
        
        # Colorbar Controls
        colorbar_group = QGroupBox("Colorbar Controls")
        colorbar_layout = QVBoxLayout(colorbar_group)
        
        # Basic colorbar settings
        basic_colorbar_layout = QFormLayout()
        
        # Show Colorbar
        self.show_colorbar_checkbox = QCheckBox("Show Colorbar")
        self.show_colorbar_checkbox.setChecked(True)
        basic_colorbar_layout.addRow("", self.show_colorbar_checkbox)
        
        # Colorbar Position
        self.colorbar_position_combo = QComboBox()
        positions = ['right', 'left', 'top', 'bottom']
        self.colorbar_position_combo.addItems(positions)
        basic_colorbar_layout.addRow("Colorbar Position:", self.colorbar_position_combo)
        
        # Colorbar Label
        self.colorbar_label_edit = QLineEdit()
        self.colorbar_label_edit.setPlaceholderText("e.g., Density")
        basic_colorbar_layout.addRow("Colorbar Label:", self.colorbar_label_edit)
        
        colorbar_layout.addLayout(basic_colorbar_layout)
        
        # Precise positioning group
        position_group = QGroupBox("Precise Positioning")
        position_layout = QFormLayout(position_group)
        
        # Padding
        self.colorbar_pad_spin = QDoubleSpinBox()
        self.colorbar_pad_spin.setRange(0.0, 1.0)
        self.colorbar_pad_spin.setSingleStep(0.01)
        self.colorbar_pad_spin.setValue(0.05)
        self.colorbar_pad_spin.setDecimals(3)
        self.colorbar_pad_spin.setToolTip("Distance between colorbar and plot")
        position_layout.addRow("Padding:", self.colorbar_pad_spin)
        
        # Shrink
        self.colorbar_shrink_spin = QDoubleSpinBox()
        self.colorbar_shrink_spin.setRange(0.1, 1.0)
        self.colorbar_shrink_spin.setSingleStep(0.05)
        self.colorbar_shrink_spin.setValue(0.8)
        self.colorbar_shrink_spin.setDecimals(2)
        self.colorbar_shrink_spin.setToolTip("Fraction of original size")
        position_layout.addRow("Shrink:", self.colorbar_shrink_spin)
        
        # Aspect ratio
        self.colorbar_aspect_spin = QSpinBox()
        self.colorbar_aspect_spin.setRange(5, 50)
        self.colorbar_aspect_spin.setValue(20)
        self.colorbar_aspect_spin.setToolTip("Ratio of long to short dimensions")
        position_layout.addRow("Aspect Ratio:", self.colorbar_aspect_spin)
        
        colorbar_layout.addWidget(position_group)
        
        # Tick and label controls
        tick_group = QGroupBox("Ticks and Labels")
        tick_layout = QFormLayout(tick_group)
        
        # Number of ticks
        self.colorbar_ticks_spin = QSpinBox()
        self.colorbar_ticks_spin.setRange(2, 20)
        self.colorbar_ticks_spin.setValue(5)
        self.colorbar_ticks_spin.setToolTip("Number of major ticks on colorbar")
        tick_layout.addRow("Number of Ticks:", self.colorbar_ticks_spin)
        
        # Tick label format
        self.tick_format_combo = QComboBox()
        formats = ['%.1f', '%.2f', '%.0f', '%.1e', '%.2e', '{:.1f}', '{:.2f}', '{:.0f}']
        self.tick_format_combo.addItems(formats)
        self.tick_format_combo.setCurrentText('%.1f')
        self.tick_format_combo.setEditable(True)
        self.tick_format_combo.setToolTip("Format string for tick labels")
        tick_layout.addRow("Tick Format:", self.tick_format_combo)
        
        # Label font size
        self.colorbar_label_fontsize_spin = QSpinBox()
        self.colorbar_label_fontsize_spin.setRange(6, 24)
        self.colorbar_label_fontsize_spin.setValue(10)
        tick_layout.addRow("Label Font Size:", self.colorbar_label_fontsize_spin)
        
        # Tick font size
        self.colorbar_tick_fontsize_spin = QSpinBox()
        self.colorbar_tick_fontsize_spin.setRange(6, 20)
        self.colorbar_tick_fontsize_spin.setValue(8)
        tick_layout.addRow("Tick Font Size:", self.colorbar_tick_fontsize_spin)
        
        colorbar_layout.addWidget(tick_group)
        
        layout.addWidget(colorbar_group)
        
        # Density Plot Options
        density_options_group = QGroupBox("Density Plot Options")
        density_options_layout = QFormLayout(density_options_group)
        
        # Grid size for hexbin
        self.gridsize_spin = QSpinBox()
        self.gridsize_spin.setRange(10, 200)
        self.gridsize_spin.setValue(50)
        self.gridsize_spin.setSuffix(" bins")
        self.gridsize_spin.setToolTip("Number of hexagonal bins for hexbin plots")
        density_options_layout.addRow("Hexbin Grid Size:", self.gridsize_spin)
        
        # KDE bandwidth
        self.kde_bandwidth_spin = QDoubleSpinBox()
        self.kde_bandwidth_spin.setRange(0.1, 10.0)
        self.kde_bandwidth_spin.setSingleStep(0.1)
        self.kde_bandwidth_spin.setValue(1.0)
        self.kde_bandwidth_spin.setDecimals(2)
        self.kde_bandwidth_spin.setToolTip("Bandwidth for KDE plots (higher = smoother)")
        density_options_layout.addRow("KDE Bandwidth:", self.kde_bandwidth_spin)
        
        # Contour levels
        self.contour_levels_spin = QSpinBox()
        self.contour_levels_spin.setRange(5, 50)
        self.contour_levels_spin.setValue(15)
        self.contour_levels_spin.setToolTip("Number of contour levels for KDE plots")
        density_options_layout.addRow("Contour Levels:", self.contour_levels_spin)
        
        layout.addWidget(density_options_group)
        
        # Connect change signals for live preview
        self.colormap_combo.currentTextChanged.connect(self.update_preview)
        self.reverse_colormap_checkbox.toggled.connect(self.update_preview)
        self.cmap_alpha_spin.valueChanged.connect(self.update_preview)
        self.show_colorbar_checkbox.toggled.connect(self.update_preview)
        self.colorbar_position_combo.currentTextChanged.connect(self.update_preview)
        self.colorbar_label_edit.textChanged.connect(self.update_preview)
        self.colorbar_pad_spin.valueChanged.connect(self.update_preview)
        self.colorbar_shrink_spin.valueChanged.connect(self.update_preview)
        self.colorbar_aspect_spin.valueChanged.connect(self.update_preview)
        self.colorbar_ticks_spin.valueChanged.connect(self.update_preview)
        self.tick_format_combo.currentTextChanged.connect(self.update_preview)
        self.colorbar_label_fontsize_spin.valueChanged.connect(self.update_preview)
        self.colorbar_tick_fontsize_spin.valueChanged.connect(self.update_preview)
        self.gridsize_spin.valueChanged.connect(self.update_preview)
        self.kde_bandwidth_spin.valueChanged.connect(self.update_preview)
        self.contour_levels_spin.valueChanged.connect(self.update_preview)
        
        return tab
    
    def load_marker_properties(self):
        """Load current marker properties into the UI"""
        # Marker type - handle both old display names and new matplotlib values
        marker_type = self.pair_config.get('marker_type', 'o')
        
        # Try to find by data value first (new format)
        found = False
        for i in range(self.marker_combo.count()):
            if self.marker_combo.itemData(i) == marker_type:
                self.marker_combo.setCurrentIndex(i)
                found = True
                break
        
        # Fallback to text search (old format)
        if not found:
            for i in range(self.marker_combo.count()):
                if self.marker_combo.itemText(i) == marker_type:
                    self.marker_combo.setCurrentIndex(i)
                    break
        
        # Color
        marker_color = self.pair_config.get('marker_color', '🔵 Blue')
        for i in range(self.color_combo.count()):
            if self.color_combo.itemText(i) == marker_color:
                self.color_combo.setCurrentIndex(i)
                break
        
        # Set color button based on color combo
        self.on_color_combo_changed(self.color_combo.currentText())
        
        # Size
        size = self.pair_config.get('marker_size', 50)
        self.size_spin.setValue(size)
        
        # Alpha
        alpha = self.pair_config.get('marker_alpha', 0.7)
        self.alpha_spin.setValue(alpha)
        
        # Edge properties
        edge_color = self.pair_config.get('edge_color', '#000000')
        self.edge_color_button.set_color(edge_color)
        
        edge_width = self.pair_config.get('edge_width', 1.0)
        self.edge_width_spin.setValue(edge_width)
        
        fill_style = self.pair_config.get('fill_style', 'full')
        for i in range(self.fill_combo.count()):
            if self.fill_combo.itemData(i) == fill_style:
                self.fill_combo.setCurrentIndex(i)
                break
        
        # Z-order (Bring to Front)
        z_order = self.pair_config.get('z_order', 0)
        self.bring_to_front_checkbox.setChecked(z_order > 0)
        
        # X-axis position
        x_axis = self.pair_config.get('x_axis', 'bottom')
        if x_axis == 'bottom':
            self.bottom_x_axis_radio.setChecked(True)
        else:
            self.top_x_axis_radio.setChecked(True)
        
        # Density plot properties
        colormap = self.pair_config.get('colormap', 'viridis')
        for i in range(self.colormap_combo.count()):
            if self.colormap_combo.itemData(i) == colormap:
                self.colormap_combo.setCurrentIndex(i)
                break
        
        colormap_alpha = self.pair_config.get('colormap_alpha', 0.7)
        self.cmap_alpha_spin.setValue(colormap_alpha)
        
        colormap_reverse = self.pair_config.get('colormap_reverse', False)
        self.reverse_colormap_checkbox.setChecked(colormap_reverse)
        
        # Colorbar properties
        show_colorbar = self.pair_config.get('show_colorbar', True)
        self.show_colorbar_checkbox.setChecked(show_colorbar)
        
        colorbar_position = self.pair_config.get('colorbar_position', 'right')
        index = self.colorbar_position_combo.findText(colorbar_position)
        if index >= 0:
            self.colorbar_position_combo.setCurrentIndex(index)
        
        colorbar_label = self.pair_config.get('colorbar_label', 'Density')
        self.colorbar_label_edit.setText(colorbar_label)
        
        # Precise positioning
        colorbar_pad = self.pair_config.get('colorbar_pad', 0.05)
        self.colorbar_pad_spin.setValue(colorbar_pad)
        
        colorbar_shrink = self.pair_config.get('colorbar_shrink', 0.8)
        self.colorbar_shrink_spin.setValue(colorbar_shrink)
        
        colorbar_aspect = self.pair_config.get('colorbar_aspect', 20)
        self.colorbar_aspect_spin.setValue(colorbar_aspect)
        
        # Tick and label settings
        colorbar_ticks = self.pair_config.get('colorbar_ticks', 5)
        self.colorbar_ticks_spin.setValue(colorbar_ticks)
        
        tick_format = self.pair_config.get('tick_format', '%.1f')
        self.tick_format_combo.setCurrentText(tick_format)
        
        colorbar_label_fontsize = self.pair_config.get('colorbar_label_fontsize', 10)
        self.colorbar_label_fontsize_spin.setValue(colorbar_label_fontsize)
        
        colorbar_tick_fontsize = self.pair_config.get('colorbar_tick_fontsize', 8)
        self.colorbar_tick_fontsize_spin.setValue(colorbar_tick_fontsize)
        
        # Density plot options
        gridsize = self.pair_config.get('gridsize', 50)
        self.gridsize_spin.setValue(gridsize)
        
        kde_bandwidth = self.pair_config.get('kde_bandwidth', 1.0)
        self.kde_bandwidth_spin.setValue(kde_bandwidth)
        
        contour_levels = self.pair_config.get('contour_levels', 15)
        self.contour_levels_spin.setValue(contour_levels)
        
        # Initial preview update
        self.update_preview()
    
    def on_color_combo_changed(self, color_name: str):
        """Handle color combo box changes"""
        for i in range(self.color_combo.count()):
            if self.color_combo.itemText(i) == color_name:
                color_value = self.color_combo.itemData(i)
                if color_value and color_value != 'custom':
                    self.color_button.set_color(color_value)
                break
        self.update_preview()
    
    def on_color_button_changed(self, color: str):
        """Handle color button changes"""
        # Set combo to "Custom" for custom color
        self.color_combo.setCurrentText("Custom")
        self.update_preview()
    
    def update_preview(self):
        """Update the preview display"""
        # Get current tab to show appropriate preview
        current_tab = self.findChild(QTabWidget).currentIndex()
        
        if current_tab == 0:  # Scatter tab
            # Get current settings
            marker_type = self.marker_combo.currentText()
            marker_symbol = self.marker_combo.currentData()
            color = self.color_button.get_color()
            size = self.size_spin.value()
            alpha = self.alpha_spin.value()
            edge_color = self.edge_color_button.get_color()
            edge_width = self.edge_width_spin.value()
            fill_style = self.fill_combo.currentText()
            bring_to_front = "Yes" if self.bring_to_front_checkbox.isChecked() else "No"
            x_axis = "Bottom" if self.bottom_x_axis_radio.isChecked() else "Top"
            
            # Create preview text with visual representation
            preview_text = f"""
            <div style="padding: 5px; font-family: monospace;">
                <b>Marker Type:</b> {marker_type}<br>
                <b>Color:</b> <span style="color: {color}; font-size: 16px;">●●●</span> {color}<br>
                <b>Size:</b> {size} pts<br>
                <b>Transparency:</b> {alpha:.2f} ({int(alpha*100)}%)<br>
                <b>Edge Color:</b> <span style="color: {edge_color};">████</span> {edge_color}<br>
                <b>Edge Width:</b> {edge_width}<br>
                <b>Fill Style:</b> {fill_style}<br>
                <b>X-Axis:</b> {x_axis}<br>
                <b>Bring to Front:</b> {bring_to_front}
            </div>
            """
        else:  # Density tab
            colormap = self.colormap_combo.currentText()
            colormap_value = self.colormap_combo.currentData()
            reverse = "Yes" if self.reverse_colormap_checkbox.isChecked() else "No"
            alpha = self.cmap_alpha_spin.value()
            
            show_colorbar = "Yes" if self.show_colorbar_checkbox.isChecked() else "No"
            colorbar_position = self.colorbar_position_combo.currentText()
            colorbar_label = self.colorbar_label_edit.text() or "Density"
            colorbar_pad = self.colorbar_pad_spin.value()
            colorbar_shrink = self.colorbar_shrink_spin.value()
            colorbar_aspect = self.colorbar_aspect_spin.value()
            colorbar_ticks = self.colorbar_ticks_spin.value()
            tick_format = self.tick_format_combo.currentText()
            label_fontsize = self.colorbar_label_fontsize_spin.value()
            tick_fontsize = self.colorbar_tick_fontsize_spin.value()
            
            gridsize = self.gridsize_spin.value()
            bandwidth = self.kde_bandwidth_spin.value()
            levels = self.contour_levels_spin.value()
            
            preview_text = f"""
            <div style="padding: 5px; font-family: monospace; font-size: 10px;">
                <b>Colormap:</b> {colormap} ({colormap_value})<br>
                <b>Reverse:</b> {reverse} | <b>Alpha:</b> {alpha:.2f}<br>
                <br>
                <b>Colorbar:</b> {show_colorbar} at {colorbar_position}<br>
                <b>Label:</b> "{colorbar_label}" (size {label_fontsize})<br>
                <b>Position:</b> pad={colorbar_pad:.3f}, shrink={colorbar_shrink:.2f}<br>
                <b>Ticks:</b> {colorbar_ticks} ({tick_format}, size {tick_fontsize})<br>
                <br>
                <b>Hexbin:</b> {gridsize} bins | <b>KDE:</b> bw={bandwidth}<br>
                <b>Contours:</b> {levels} levels<br>
                <br>
                <b>Preview:</b> <span style="color: #666;">Density plot with {colormap_value} colormap</span>
            </div>
            """
        
        self.preview_label.setText(preview_text)
    
    def apply_changes(self):
        """Apply changes to the pair config without closing dialog"""
        self._update_marker_properties()
        self.marker_updated.emit(self.pair_config)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_marker_properties()
        self.marker_updated.emit(self.pair_config)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original properties"""
        self._restore_marker_properties()
        super().reject()
    
    def _update_marker_properties(self):
        """Update pair config with marker properties from UI"""
        # Marker type - store the matplotlib-compatible value
        self.pair_config['marker_type'] = self.marker_combo.currentData()
        self.pair_config['marker_symbol'] = self.marker_combo.currentData()
        
        # Color
        self.pair_config['marker_color'] = self.color_combo.currentText()
        self.pair_config['marker_color_hex'] = self.color_button.get_color()
        
        # Size and transparency
        self.pair_config['marker_size'] = self.size_spin.value()
        self.pair_config['marker_alpha'] = self.alpha_spin.value()
        
        # Edge properties
        self.pair_config['edge_color'] = self.edge_color_button.get_color()
        self.pair_config['edge_width'] = self.edge_width_spin.value()
        self.pair_config['fill_style'] = self.fill_combo.currentData()
        
        # Z-order (Bring to Front)
        self.pair_config['z_order'] = 10 if self.bring_to_front_checkbox.isChecked() else 0
        
        # X-axis position
        x_axis = "bottom" if self.bottom_x_axis_radio.isChecked() else "top"
        self.pair_config['x_axis'] = x_axis
        
        # Density plot properties
        self.pair_config['colormap'] = self.colormap_combo.currentData()
        self.pair_config['colormap_alpha'] = self.cmap_alpha_spin.value()
        self.pair_config['colormap_reverse'] = self.reverse_colormap_checkbox.isChecked()
        
        # Colorbar properties
        self.pair_config['show_colorbar'] = self.show_colorbar_checkbox.isChecked()
        self.pair_config['colorbar_position'] = self.colorbar_position_combo.currentText()
        self.pair_config['colorbar_label'] = self.colorbar_label_edit.text()
        
        # Precise positioning
        self.pair_config['colorbar_pad'] = self.colorbar_pad_spin.value()
        self.pair_config['colorbar_shrink'] = self.colorbar_shrink_spin.value()
        self.pair_config['colorbar_aspect'] = self.colorbar_aspect_spin.value()
        
        # Tick and label settings
        self.pair_config['colorbar_ticks'] = self.colorbar_ticks_spin.value()
        self.pair_config['tick_format'] = self.tick_format_combo.currentText()
        self.pair_config['colorbar_label_fontsize'] = self.colorbar_label_fontsize_spin.value()
        self.pair_config['colorbar_tick_fontsize'] = self.colorbar_tick_fontsize_spin.value()
        
        # Density plot options
        self.pair_config['gridsize'] = self.gridsize_spin.value()
        self.pair_config['kde_bandwidth'] = self.kde_bandwidth_spin.value()
        self.pair_config['contour_levels'] = self.contour_levels_spin.value()
        
        # Update modification time
        from datetime import datetime
        self.pair_config['modified_at'] = datetime.now().isoformat()
    
    def _restore_marker_properties(self):
        """Restore original marker properties"""
        for key, value in self.original_properties.items():
            self.pair_config[key] = value
    
    @staticmethod
    def edit_marker(pair_config: Dict[str, Any], parent=None) -> bool:
        """
        Static method to edit a pair's marker properties
        
        Args:
            pair_config: Dictionary containing pair configuration
            parent: Parent widget
            
        Returns:
            True if changes were applied, False if cancelled
        """
        dialog = MarkerWizard(pair_config, parent)
        result = dialog.exec()
        return result == QDialog.Accepted


# Convenience function for opening the marker wizard
def open_marker_wizard(pair_config: Dict[str, Any], parent=None) -> bool:
    """
    Open the marker wizard for editing a comparison pair's marker properties
    
    Args:
        pair_config: Dictionary containing pair configuration
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    return MarkerWizard.edit_marker(pair_config, parent) 


class ComparisonPairMarkerWizard(QDialog):
    """
    Focused dialog for editing visual marker properties and legend label for comparison pairs.
    Only includes: marker shape, color, size, edge color/thickness, fill style, and legend label.
    No data transformation, advanced line, animation, or statistical overlay options.
    Purely visual: does not affect underlying data or analysis.
    """
    marker_updated = Signal(dict)

    def __init__(self, pair_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.pair_config = pair_config
        self.original_properties = self._backup_marker_properties()
        self.setWindowTitle(f"Pair Styling - {pair_config.get('name', 'Unnamed Pair')}")
        self.setModal(True)
        self.setFixedSize(420, 420)
        self.init_ui()
        self.load_marker_properties()

    def _backup_marker_properties(self):
        return {
            'marker_type': self.pair_config.get('marker_type', 'o'),
            'marker_color': self.pair_config.get('marker_color', '🔵 Blue'),
            'marker_size': self.pair_config.get('marker_size', 50),
            'marker_alpha': self.pair_config.get('marker_alpha', 0.8),
            'edge_color': self.pair_config.get('edge_color', '#000000'),
            'edge_width': self.pair_config.get('edge_width', 1.0),
            'fill_style': self.pair_config.get('fill_style', 'full'),
            'legend_label': self.pair_config.get('legend_label', ''),
        }

    def init_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel(f"Visual Styling for: {self.pair_config.get('name', 'Unnamed Pair')}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        info = QLabel("Visual changes only. Data and analysis are not affected.")
        info.setStyleSheet("color: #2c5aa0; font-size: 10px; font-style: italic; margin-bottom: 8px;")
        layout.addWidget(info)
        form = QFormLayout()

        # Marker Shape
        self.marker_combo = QComboBox()
        marker_types = [
            ('○ Circle', 'o'),
            ('□ Square', 's'),
            ('△ Triangle', '^'),
            ('◇ Diamond', 'D'),
            ('▽ Inverted Triangle', 'v'),
            ('◁ Left Triangle', '<'),
            ('▷ Right Triangle', '>'),
            ('⬟ Pentagon', 'p'),
            ('✦ Star', '*'),
            ('⬢ Hexagon', 'h'),
            ('+ Plus', '+'),
            ('× Cross', 'x'),
        ]
        for name, value in marker_types:
            self.marker_combo.addItem(name, value)
        form.addRow("Marker Shape:", self.marker_combo)

        # Marker Color
        color_layout = QHBoxLayout()
        self.color_button = ColorButton()
        self.color_combo = QComboBox()
        color_options = [
            ('🔵 Blue', '#1f77b4'),
            ('🔴 Red', '#d62728'),
            ('🟢 Green', '#2ca02c'),
            ('🟣 Purple', '#9467bd'),
            ('🟠 Orange', '#ff7f0e'),
            ('🟤 Brown', '#8c564b'),
            ('🩷 Pink', '#e377c2'),
            ('⚫ Gray', '#7f7f7f'),
            ('🟡 Yellow', '#bcbd22'),
            ('🔶 Cyan', '#17becf'),
            ('Custom', 'custom')
        ]
        for name, value in color_options:
            self.color_combo.addItem(name, value)
        self.color_combo.currentTextChanged.connect(self.on_color_combo_changed)
        self.color_button.color_changed.connect(self.on_color_button_changed)
        color_layout.addWidget(self.color_button)
        color_layout.addWidget(self.color_combo)
        form.addRow("Marker Color:", color_layout)

        # Marker Size
        size_layout = QHBoxLayout()
        self.size_spin = QSpinBox()
        self.size_spin.setRange(6, 200)
        self.size_spin.setValue(50)
        self.size_spin.setSuffix(" pts")
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(6, 200)
        self.size_slider.setValue(50)
        self.size_spin.valueChanged.connect(self.size_slider.setValue)
        self.size_slider.valueChanged.connect(self.size_spin.setValue)
        size_layout.addWidget(self.size_spin)
        size_layout.addWidget(self.size_slider)
        form.addRow("Marker Size:", size_layout)

        # Legend Label
        self.legend_label_edit = QLineEdit()
        self.legend_label_edit.setPlaceholderText("Custom legend label (optional)")
        form.addRow("Legend Label:", self.legend_label_edit)

        layout.addLayout(form)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(button_box)

        # Live preview connections (optional, placeholder)
        self.marker_combo.currentTextChanged.connect(self.update_preview)
        self.size_spin.valueChanged.connect(self.update_preview)
        self.legend_label_edit.textChanged.connect(self.update_preview)

    def load_marker_properties(self):
        # Marker type - handle both old display names and new matplotlib values
        marker_type = self.pair_config.get('marker_type', 'o')
        
        # Try to find by data value first (new format)
        idx = self.marker_combo.findData(marker_type)
        if idx >= 0:
            self.marker_combo.setCurrentIndex(idx)
        else:
            # Fallback to text search (old format)
            idx = self.marker_combo.findText(marker_type)
            if idx >= 0:
                self.marker_combo.setCurrentIndex(idx)
            else:
                # Default to circle if not found
                self.marker_combo.setCurrentIndex(0)
        # Color
        marker_color = self.pair_config.get('marker_color', '🔵 Blue')
        idx = self.color_combo.findText(marker_color)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)
        else:
            self.color_combo.setCurrentText('Custom')
            hex_color = self.pair_config.get('marker_color_hex', '#1f77b4')
            self.color_button.set_color(hex_color)
        # Size
        self.size_spin.setValue(self.pair_config.get('marker_size', 50))
        # Legend label
        self.legend_label_edit.setText(self.pair_config.get('legend_label', ''))

    def on_color_combo_changed(self, color_name: str):
        if color_name == 'Custom':
            return
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
        if color_name in color_map:
            self.color_button.set_color(color_map[color_name])

    def on_color_button_changed(self, color: str):
        self.color_combo.setCurrentText('Custom')

    def update_preview(self):
        pass  # Placeholder for live preview

    def apply_changes(self):
        self._update_marker_properties()
        self.marker_updated.emit(self.pair_config)

    def accept(self):
        self.apply_changes()
        super().accept()

    def reject(self):
        self._restore_marker_properties()
        super().reject()

    def _update_marker_properties(self):
        self.pair_config['marker_type'] = self.marker_combo.currentData()
        self.pair_config['marker_color'] = self.color_combo.currentText()
        self.pair_config['marker_color_hex'] = self.color_button.get_color()
        self.pair_config['marker_size'] = self.size_spin.value()
        self.pair_config['legend_label'] = self.legend_label_edit.text()

    def _restore_marker_properties(self):
        for key, value in self.original_properties.items():
            self.pair_config[key] = value

    @staticmethod
    def edit_pair_marker(pair_config: Dict[str, Any], parent=None) -> bool:
        wizard = ComparisonPairMarkerWizard(pair_config, parent)
        result = wizard.exec()
        return result == QDialog.Accepted

def open_comparison_pair_marker_wizard(pair_config: Dict[str, Any], parent=None) -> bool:
    return ComparisonPairMarkerWizard.edit_pair_marker(pair_config, parent) 