from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QColorDialog, QFrame,
    QGroupBox, QButtonGroup, QRadioButton, QDialogButtonBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSlider, QTabWidget, QWidget, QTextEdit
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette
from typing import Optional, Dict, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime


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


class BaseConfigWizard(QDialog, ABC):
    """
    Base class for configuration wizards (Line, Marker, Spectrogram)
    Provides common functionality and enforces consistent UI presentation
    """
    
    # Common signals - subclasses can add their own
    config_updated = Signal(object)  # Updated object when changes are applied
    
    def __init__(self, config_object: Any, wizard_type: str, parent=None):
        super().__init__(parent)
        self.config_object = config_object
        self.wizard_type = wizard_type
        self.original_properties = self._backup_properties()
        
        # Common UI components (will be created by subclasses)
        self.legend_edit = None
        self.alpha_spin = None
        self.alpha_slider = None
        self.x_axis_button_group = None
        self.bottom_x_axis_radio = None
        self.top_x_axis_radio = None
        self.bring_to_front_checkbox = None
        self.preview_label = None
        
        # Setup window
        self._setup_window()
        self.init_ui()
        self.load_properties()
    
    @abstractmethod
    def _backup_properties(self) -> dict:
        """Backup original properties for cancel operation"""
        pass
    
    @abstractmethod
    def _get_object_name(self) -> str:
        """Get display name for the object being configured"""
        pass
    
    @abstractmethod
    def _get_object_info(self) -> str:
        """Get info text for the object being configured"""
        pass
    
    @abstractmethod
    def _create_main_tabs(self, tab_widget: QTabWidget):
        """Create the main tabs specific to this wizard type"""
        pass
    
    @abstractmethod
    def load_properties(self):
        """Load current properties into the UI"""
        pass
    
    @abstractmethod
    def _update_properties(self):
        """Update properties from UI"""
        pass
    
    @abstractmethod
    def _restore_properties(self):
        """Restore original properties"""
        pass
    
    def _setup_window(self):
        """Setup common window properties"""
        object_name = self._get_object_name()
        self.setWindowTitle(f"{self.wizard_type} Properties - {object_name}")
        self.setModal(True)
        # Default size - subclasses can override
        self.setFixedSize(500, 650)
    
    def init_ui(self):
        """Initialize the UI components - standardized layout"""
        layout = QVBoxLayout(self)
        
        # Title section
        self._create_title_section(layout)
        
        # Tab widget for main content
        tab_widget = QTabWidget()
        
        # Basic tab (common to all wizards)
        basic_tab = self._create_basic_tab()
        tab_widget.addTab(basic_tab, "Basic")
        
        # Axis tab (common to all wizards)
        axis_tab = self._create_axis_tab()
        tab_widget.addTab(axis_tab, "Axis")
        
        # Let subclasses add their specific tabs
        self._create_main_tabs(tab_widget)
        
        layout.addWidget(tab_widget)
        
        # Preview section (common to all wizards)
        self._create_preview_section(layout)
        
        # Dialog buttons (common to all wizards)
        self._create_dialog_buttons(layout)
        
        # Connect common signals
        self._connect_common_signals()
    
    def _create_title_section(self, layout: QVBoxLayout):
        """Create standardized title section"""
        # Main title
        object_name = self._get_object_name()
        title = QLabel(f"Editing {self.wizard_type}: {object_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Info text
        info_text = self._get_object_info()
        if info_text:
            info_label = QLabel(info_text)
            info_label.setStyleSheet("color: #666; font-size: 10px; margin-bottom: 10px;")
            layout.addWidget(info_label)
    
    def _create_basic_tab(self) -> QWidget:
        """Create the basic settings tab - common to all wizards"""
        tab = QWidget()
        layout = QFormLayout(tab)
        
        # Legend Name (common to all)
        self.legend_edit = QLineEdit()
        self.legend_edit.setPlaceholderText("Enter legend label...")
        layout.addRow("Legend Name:", self.legend_edit)
        
        # Transparency (common to all)
        transparency_layout = QHBoxLayout()
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setDecimals(2)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(100)
        self.alpha_spin.valueChanged.connect(lambda v: self.alpha_slider.setValue(int(v * 100)))
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_spin.setValue(v / 100.0))
        
        transparency_layout.addWidget(self.alpha_spin)
        transparency_layout.addWidget(self.alpha_slider)
        layout.addRow("Transparency:", transparency_layout)
        
        # Bring to Front (common to all)
        self.bring_to_front_checkbox = QCheckBox("Bring to Front")
        self.bring_to_front_checkbox.setToolTip("Bring this element to the top layer in the plot")
        layout.addRow("", self.bring_to_front_checkbox)
        
        return tab
    
    def _create_axis_tab(self) -> QWidget:
        """Create the axis settings tab - common to all wizards"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Axis Position Group
        position_group = QGroupBox("Axis Position")
        position_layout = QFormLayout(position_group)
        
        # X-Axis Position (common to all)
        x_axis_layout = QHBoxLayout()
        self.x_axis_button_group = QButtonGroup()
        self.bottom_x_axis_radio = QRadioButton("Bottom")
        self.top_x_axis_radio = QRadioButton("Top")
        
        self.x_axis_button_group.addButton(self.bottom_x_axis_radio, 0)
        self.x_axis_button_group.addButton(self.top_x_axis_radio, 1)
        
        x_axis_layout.addWidget(self.bottom_x_axis_radio)
        x_axis_layout.addWidget(self.top_x_axis_radio)
        position_layout.addRow("X-Axis:", x_axis_layout)
        
        layout.addWidget(position_group)
        
        # Subclasses can add more axis-specific controls
        self._add_axis_specific_controls(layout)
        
        return tab
    
    def _add_axis_specific_controls(self, layout: QVBoxLayout):
        """Override in subclasses to add axis-specific controls"""
        pass
    
    def _create_preview_section(self, layout: QVBoxLayout):
        """Create standardized preview section"""
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel(f"{self.wizard_type} preview will appear here")
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 10px;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
    
    def _create_dialog_buttons(self, layout: QVBoxLayout):
        """Create standardized dialog buttons"""
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
    
    def _connect_common_signals(self):
        """Connect common signals for live preview"""
        if self.legend_edit:
            self.legend_edit.textChanged.connect(self.update_preview)
        if self.alpha_spin:
            self.alpha_spin.valueChanged.connect(self.update_preview)
        if self.bring_to_front_checkbox:
            self.bring_to_front_checkbox.toggled.connect(self.update_preview)
        if self.x_axis_button_group:
            self.x_axis_button_group.buttonToggled.connect(self.update_preview)
    
    def _create_colorbar_controls_group(self) -> QGroupBox:
        """Create standardized colorbar controls - used by spectrograms and density plots"""
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
        self.colorbar_label_edit.setPlaceholderText("e.g., Power (dB), Density")
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
        
        return colorbar_group
    
    def _connect_colorbar_signals(self):
        """Connect colorbar control signals to preview update"""
        if hasattr(self, 'show_colorbar_checkbox'):
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
    
    def _load_common_properties(self):
        """Load common properties - call from subclass load_properties()"""
        # Legend name
        if hasattr(self.config_object, 'legend_label'):
            legend_label = getattr(self.config_object, 'legend_label', None) or ""
            if self.legend_edit:
                self.legend_edit.setText(legend_label)
        elif hasattr(self.config_object, 'get') and 'name' in self.config_object:
            # For dict-like objects (marker wizard)
            legend_label = self.config_object.get('name', '')
            if self.legend_edit:
                self.legend_edit.setText(legend_label)
        
        # Transparency
        alpha = getattr(self.config_object, 'alpha', None)
        if alpha is None and hasattr(self.config_object, 'get'):
            alpha = self.config_object.get('marker_alpha', 1.0)
        if alpha is None:
            alpha = 1.0
        if self.alpha_spin:
            self.alpha_spin.setValue(alpha)
        
        # Bring to front / Z-order
        z_order = getattr(self.config_object, 'z_order', None)
        if z_order is None and hasattr(self.config_object, 'get'):
            z_order = self.config_object.get('z_order', 0)
        if z_order is None:
            z_order = 0
        if self.bring_to_front_checkbox:
            self.bring_to_front_checkbox.setChecked(z_order > 0)
        
        # X-axis position
        x_axis = getattr(self.config_object, 'xaxis', None)
        if x_axis is None and hasattr(self.config_object, 'get'):
            x_axis = self.config_object.get('x_axis', 'bottom')
        if x_axis is None:
            x_axis = 'bottom'
        
        # Normalize x_axis value
        if x_axis in ['x-bottom', 'bottom']:
            if self.bottom_x_axis_radio:
                self.bottom_x_axis_radio.setChecked(True)
        else:
            if self.top_x_axis_radio:
                self.top_x_axis_radio.setChecked(True)
    
    def _load_colorbar_properties(self):
        """Load colorbar properties - call from subclass if using colorbar"""
        if not hasattr(self, 'show_colorbar_checkbox'):
            return
        
        # Basic colorbar settings
        show_colorbar = getattr(self.config_object, 'show_colorbar', True)
        if hasattr(self.config_object, 'get'):
            show_colorbar = self.config_object.get('show_colorbar', True)
        self.show_colorbar_checkbox.setChecked(show_colorbar)
        
        colorbar_position = getattr(self.config_object, 'colorbar_position', 'right')
        if hasattr(self.config_object, 'get'):
            colorbar_position = self.config_object.get('colorbar_position', 'right')
        index = self.colorbar_position_combo.findText(colorbar_position)
        if index >= 0:
            self.colorbar_position_combo.setCurrentIndex(index)
        
        colorbar_label = getattr(self.config_object, 'colorbar_label', '')
        if hasattr(self.config_object, 'get'):
            colorbar_label = self.config_object.get('colorbar_label', '')
        self.colorbar_label_edit.setText(colorbar_label)
        
        # Precise positioning
        self.colorbar_pad_spin.setValue(
            getattr(self.config_object, 'colorbar_pad', None) or
            (self.config_object.get('colorbar_pad', 0.05) if hasattr(self.config_object, 'get') else 0.05)
        )
        
        self.colorbar_shrink_spin.setValue(
            getattr(self.config_object, 'colorbar_shrink', None) or
            (self.config_object.get('colorbar_shrink', 0.8) if hasattr(self.config_object, 'get') else 0.8)
        )
        
        self.colorbar_aspect_spin.setValue(
            getattr(self.config_object, 'colorbar_aspect', None) or
            (self.config_object.get('colorbar_aspect', 20) if hasattr(self.config_object, 'get') else 20)
        )
        
        # Tick and label settings
        self.colorbar_ticks_spin.setValue(
            getattr(self.config_object, 'colorbar_ticks', None) or
            (self.config_object.get('colorbar_ticks', 5) if hasattr(self.config_object, 'get') else 5)
        )
        
        tick_format = getattr(self.config_object, 'tick_format', None) or \
                     (self.config_object.get('tick_format', '%.1f') if hasattr(self.config_object, 'get') else '%.1f')
        self.tick_format_combo.setCurrentText(tick_format)
        
        self.colorbar_label_fontsize_spin.setValue(
            getattr(self.config_object, 'colorbar_label_fontsize', None) or
            (self.config_object.get('colorbar_label_fontsize', 10) if hasattr(self.config_object, 'get') else 10)
        )
        
        self.colorbar_tick_fontsize_spin.setValue(
            getattr(self.config_object, 'colorbar_tick_fontsize', None) or
            (self.config_object.get('colorbar_tick_fontsize', 8) if hasattr(self.config_object, 'get') else 8)
        )
    
    def _update_common_properties(self):
        """Update common properties - call from subclass _update_properties()"""
        # Legend name
        if self.legend_edit and hasattr(self.config_object, 'legend_label'):
            self.config_object.legend_label = self.legend_edit.text() or None
        elif self.legend_edit and hasattr(self.config_object, 'get'):
            # For dict-like objects
            self.config_object['name'] = self.legend_edit.text()
        
        # Transparency
        if self.alpha_spin:
            if hasattr(self.config_object, 'alpha'):
                self.config_object.alpha = self.alpha_spin.value()
            elif hasattr(self.config_object, 'get'):
                self.config_object['marker_alpha'] = self.alpha_spin.value()
        
        # Z-order (Bring to Front)
        if self.bring_to_front_checkbox:
            z_order = 10 if self.bring_to_front_checkbox.isChecked() else 0
            if hasattr(self.config_object, 'z_order'):
                self.config_object.z_order = z_order
            elif hasattr(self.config_object, 'get'):
                self.config_object['z_order'] = z_order
        
        # X-axis position
        if self.x_axis_button_group:
            x_axis = "bottom" if self.bottom_x_axis_radio.isChecked() else "top"
            if hasattr(self.config_object, 'xaxis'):
                self.config_object.xaxis = f"x-{x_axis}"
            elif hasattr(self.config_object, 'get'):
                self.config_object['x_axis'] = x_axis
        
        # Update modification time
        if hasattr(self.config_object, 'get'):
            self.config_object['modified_at'] = datetime.now().isoformat()
        else:
            setattr(self.config_object, 'modified_at', datetime.now())
    
    def _update_colorbar_properties(self):
        """Update colorbar properties - call from subclass if using colorbar"""
        if not hasattr(self, 'show_colorbar_checkbox'):
            return
        
        # Basic colorbar settings
        show_colorbar = self.show_colorbar_checkbox.isChecked()
        colorbar_position = self.colorbar_position_combo.currentText()
        colorbar_label = self.colorbar_label_edit.text()
        
        # Precise positioning
        colorbar_pad = self.colorbar_pad_spin.value()
        colorbar_shrink = self.colorbar_shrink_spin.value()
        colorbar_aspect = self.colorbar_aspect_spin.value()
        
        # Tick and label settings
        colorbar_ticks = self.colorbar_ticks_spin.value()
        tick_format = self.tick_format_combo.currentText()
        colorbar_label_fontsize = self.colorbar_label_fontsize_spin.value()
        colorbar_tick_fontsize = self.colorbar_tick_fontsize_spin.value()
        
        # Update object properties
        if hasattr(self.config_object, 'show_colorbar'):
            # Channel object
            self.config_object.show_colorbar = show_colorbar
            self.config_object.colorbar_position = colorbar_position
            self.config_object.colorbar_label = colorbar_label
            self.config_object.colorbar_pad = colorbar_pad
            self.config_object.colorbar_shrink = colorbar_shrink
            self.config_object.colorbar_aspect = colorbar_aspect
            self.config_object.colorbar_ticks = colorbar_ticks
            self.config_object.tick_format = tick_format
            self.config_object.colorbar_label_fontsize = colorbar_label_fontsize
            self.config_object.colorbar_tick_fontsize = colorbar_tick_fontsize
        elif hasattr(self.config_object, 'get'):
            # Dict-like object
            self.config_object['show_colorbar'] = show_colorbar
            self.config_object['colorbar_position'] = colorbar_position
            self.config_object['colorbar_label'] = colorbar_label
            self.config_object['colorbar_pad'] = colorbar_pad
            self.config_object['colorbar_shrink'] = colorbar_shrink
            self.config_object['colorbar_aspect'] = colorbar_aspect
            self.config_object['colorbar_ticks'] = colorbar_ticks
            self.config_object['tick_format'] = tick_format
            self.config_object['colorbar_label_fontsize'] = colorbar_label_fontsize
            self.config_object['colorbar_tick_fontsize'] = colorbar_tick_fontsize
    
    def update_preview(self):
        """Update the preview display - subclasses should override with specific preview logic"""
        if self.preview_label:
            preview_text = f"""
            <div style="padding: 5px; font-family: monospace; font-size: 10px;">
                <b>{self.wizard_type} Configuration Preview</b><br>
                <b>Legend:</b> {self.legend_edit.text() if self.legend_edit else 'N/A'}<br>
                <b>Transparency:</b> {self.alpha_spin.value():.2f} if self.alpha_spin else 'N/A'}<br>
                <b>X-Axis:</b> {'Bottom' if self.bottom_x_axis_radio and self.bottom_x_axis_radio.isChecked() else 'Top'}<br>
                <b>Bring to Front:</b> {'Yes' if self.bring_to_front_checkbox and self.bring_to_front_checkbox.isChecked() else 'No'}
            </div>
            """
            self.preview_label.setText(preview_text)
    
    def apply_changes(self):
        """Apply changes without closing dialog"""
        self._update_properties()
        self.config_updated.emit(self.config_object)
    
    def accept(self):
        """Apply changes and close dialog"""
        self._update_properties()
        self.config_updated.emit(self.config_object)
        super().accept()
    
    def reject(self):
        """Cancel changes and restore original properties"""
        self._restore_properties()
        super().reject()


# Utility functions for wizard creation
def create_colormap_combo(default: str = 'viridis') -> QComboBox:
    """Create a standardized colormap combo box"""
    combo = QComboBox()
    colormaps = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys',
        'YlOrRd', 'YlGnBu', 'RdPu', 'BuPu', 'GnBu', 'OrRd', 'PuBu', 'PuRd',
        'RdYlBu', 'Spectral', 'RdBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdYlGn',
        'Set1', 'Set2', 'Set3', 'Paired', 'Accent', 'Dark2', 'Pastel1', 'Pastel2',
        'tab10', 'tab20', 'tab20b', 'tab20c',
        'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
        'gray', 'bone', 'copper', 'pink', 'flag', 'prism', 'ocean', 'gist_earth',
        'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix',
        'brg', 'hsv', 'rainbow', 'nipy_spectral', 'gist_ncar', 'gist_rainbow',
        'gist_heat', 'Wistia', 'afmhot'
    ]
    combo.addItems(colormaps)
    index = combo.findText(default)
    if index >= 0:
        combo.setCurrentIndex(index)
    return combo


def create_transparency_controls() -> tuple[QDoubleSpinBox, QSlider]:
    """Create standardized transparency controls"""
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.1)
    spin.setValue(1.0)
    spin.setDecimals(2)
    
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(100)
    
    # Connect them
    spin.valueChanged.connect(lambda v: slider.setValue(int(v * 100)))
    slider.valueChanged.connect(lambda v: spin.setValue(v / 100.0))
    
    return spin, slider 