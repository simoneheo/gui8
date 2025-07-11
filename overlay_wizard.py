from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, 
    QPushButton, QLineEdit, QComboBox, QColorDialog, QFrame,
    QGroupBox, QButtonGroup, QRadioButton, QDialogButtonBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QSlider, QTabWidget, QWidget, QFontComboBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor, QPalette, QFont
from typing import Optional, Dict, Any


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


class OverlayWizard(QDialog):
    """
    Dialog for editing overlay properties
    Supports different overlay types: line, fill, marker, text
    """
    
    # Signals
    overlay_updated = Signal(str, dict)  # overlay_id, properties when changes are applied
    
    def __init__(self, overlay_id: str, overlay_config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.overlay_id = overlay_id
        self.overlay_config = overlay_config.copy()
        self.original_properties = self._backup_overlay_properties()
        
        overlay_name = overlay_config.get('name', overlay_id)
        self.setWindowTitle(f"Overlay Style - {overlay_name}")
        self.setModal(True)
        self.setFixedSize(450, 600)
        
        self.init_ui()
        self.load_overlay_properties()
        self.update_preview()
    
    def _backup_overlay_properties(self) -> dict:
        """Backup original overlay properties for cancel operation"""
        return self.overlay_config.copy()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        overlay_name = self.overlay_config.get('name', self.overlay_id)
        title = QLabel(f"Editing: {overlay_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Create tab widget based on overlay type
        self.tab_widget = QTabWidget()
        
        # Determine overlay type and create appropriate tabs
        overlay_type = self._determine_overlay_type()
        
        if overlay_type in ['line', 'both']:
            line_tab = self._create_line_tab()
            self.tab_widget.addTab(line_tab, "🖍️ Line Style")
        
        if overlay_type in ['fill', 'both']:
            fill_tab = self._create_fill_tab()
            self.tab_widget.addTab(fill_tab, "🎨 Fill Style")
        
        if overlay_type in ['marker', 'both']:
            marker_tab = self._create_marker_tab()
            self.tab_widget.addTab(marker_tab, "⚫ Marker Style")
        
        if overlay_type in ['text', 'both']:
            text_tab = self._create_text_tab()
            self.tab_widget.addTab(text_tab, "📝 Text Style")
        
        layout.addWidget(self.tab_widget)
        
        # Preview section
        preview_group = QGroupBox("Style Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Style preview")
        self.preview_label.setMinimumHeight(60)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
        
        # Connect all controls to preview update
        self._connect_preview_updates()
    
    def _determine_overlay_type(self) -> str:
        """Determine overlay type based on overlay_id and config"""
        overlay_id = self.overlay_id.lower()
        
        # Text overlays
        if 'text' in overlay_id or 'statistical' in overlay_id or 'fontsize' in self.overlay_config:
            return 'text'
        
        # Fill overlays
        elif 'confidence' in overlay_id or 'band' in overlay_id or self.overlay_config.get('fill', False):
            return 'fill'
        
        # Marker overlays
        elif 'outlier' in overlay_id or 'marker' in self.overlay_config:
            return 'marker'
        
        # Line overlays (default)
        else:
            return 'line'
    
    def _create_line_tab(self) -> QWidget:
        """Create the line style tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Color Selection
        color_layout = QHBoxLayout()
        self.line_color_button = ColorButton(self.overlay_config.get('color', '#1f77b4'))
        self.line_color_button.color_changed.connect(self.on_line_color_changed)
        color_layout.addWidget(self.line_color_button)
        color_layout.addStretch()
        form_layout.addRow("Line Color:", color_layout)
        
        # Line Style
        self.line_style_combo = QComboBox()
        line_styles = [
            ('——— Solid', '-'),
            ('- - - Dashed', '--'),
            ('··· Dotted', ':'),
            ('-·- Dash-Dot', '-.'),
            ('None', 'None')
        ]
        for name, value in line_styles:
            self.line_style_combo.addItem(name, value)
        form_layout.addRow("Line Style:", self.line_style_combo)
        
        # Line Width
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setDecimals(1)
        self.line_width_spin.setValue(self.overlay_config.get('linewidth', 2.0))
        form_layout.addRow("Line Width:", self.line_width_spin)
        
        # Alpha/Opacity
        alpha_layout = QHBoxLayout()
        self.line_alpha_slider = QSlider(Qt.Horizontal)
        self.line_alpha_slider.setRange(0, 100)
        self.line_alpha_slider.setValue(int(self.overlay_config.get('alpha', 0.8) * 100))
        self.line_alpha_label = QLabel(f"{self.line_alpha_slider.value()}%")
        self.line_alpha_slider.valueChanged.connect(lambda v: self.line_alpha_label.setText(f"{v}%"))
        
        alpha_layout.addWidget(self.line_alpha_slider)
        alpha_layout.addWidget(self.line_alpha_label)
        form_layout.addRow("Opacity:", alpha_layout)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        
        return tab
    
    def _create_fill_tab(self) -> QWidget:
        """Create the fill style tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Fill Color
        color_layout = QHBoxLayout()
        self.fill_color_button = ColorButton(self.overlay_config.get('color', '#3498db'))
        self.fill_color_button.color_changed.connect(self.on_fill_color_changed)
        color_layout.addWidget(self.fill_color_button)
        color_layout.addStretch()
        form_layout.addRow("Fill Color:", color_layout)
        
        # Fill Alpha
        alpha_layout = QHBoxLayout()
        self.fill_alpha_slider = QSlider(Qt.Horizontal)
        self.fill_alpha_slider.setRange(0, 100)
        self.fill_alpha_slider.setValue(int(self.overlay_config.get('alpha', 0.3) * 100))
        self.fill_alpha_label = QLabel(f"{self.fill_alpha_slider.value()}%")
        self.fill_alpha_slider.valueChanged.connect(lambda v: self.fill_alpha_label.setText(f"{v}%"))
        
        alpha_layout.addWidget(self.fill_alpha_slider)
        alpha_layout.addWidget(self.fill_alpha_label)
        form_layout.addRow("Fill Opacity:", alpha_layout)
        
        # Edge Properties
        edge_group = QGroupBox("Edge Properties")
        edge_layout = QFormLayout(edge_group)
        
        # Edge Color
        edge_color_layout = QHBoxLayout()
        self.edge_color_button = ColorButton(self.overlay_config.get('edgecolor', '#000000'))
        self.edge_color_button.color_changed.connect(self.on_edge_color_changed)
        edge_color_layout.addWidget(self.edge_color_button)
        edge_color_layout.addStretch()
        edge_layout.addRow("Edge Color:", edge_color_layout)
        
        # Edge Width
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 5.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setDecimals(1)
        self.edge_width_spin.setValue(self.overlay_config.get('edgewidth', 0.0))
        edge_layout.addRow("Edge Width:", self.edge_width_spin)
        
        layout.addLayout(form_layout)
        layout.addWidget(edge_group)
        layout.addStretch()
        
        return tab
    
    def _create_marker_tab(self) -> QWidget:
        """Create the marker style tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Marker Type
        self.marker_combo = QComboBox()
        marker_types = [
            ('○ Circle', 'o'),
            ('□ Square', 's'),
            ('△ Triangle', '^'),
            ('◇ Diamond', 'D'),
            ('⬟ Pentagon', 'p'),
            ('✦ Star', '*'),
            ('+ Plus', '+'),
            ('× Cross', 'x')
        ]
        for name, value in marker_types:
            self.marker_combo.addItem(name, value)
        form_layout.addRow("Marker Type:", self.marker_combo)
        
        # Marker Color
        color_layout = QHBoxLayout()
        self.marker_color_button = ColorButton(self.overlay_config.get('color', '#e74c3c'))
        self.marker_color_button.color_changed.connect(self.on_marker_color_changed)
        color_layout.addWidget(self.marker_color_button)
        color_layout.addStretch()
        form_layout.addRow("Marker Color:", color_layout)
        
        # Marker Size
        self.marker_size_spin = QSpinBox()
        self.marker_size_spin.setRange(1, 100)
        self.marker_size_spin.setValue(self.overlay_config.get('markersize', 6))
        form_layout.addRow("Marker Size:", self.marker_size_spin)
        
        # Marker Alpha
        alpha_layout = QHBoxLayout()
        self.marker_alpha_slider = QSlider(Qt.Horizontal)
        self.marker_alpha_slider.setRange(0, 100)
        self.marker_alpha_slider.setValue(int(self.overlay_config.get('alpha', 0.8) * 100))
        self.marker_alpha_label = QLabel(f"{self.marker_alpha_slider.value()}%")
        self.marker_alpha_slider.valueChanged.connect(lambda v: self.marker_alpha_label.setText(f"{v}%"))
        
        alpha_layout.addWidget(self.marker_alpha_slider)
        alpha_layout.addWidget(self.marker_alpha_label)
        form_layout.addRow("Marker Opacity:", alpha_layout)
        
        # Edge Properties
        edge_group = QGroupBox("Marker Edge")
        edge_layout = QFormLayout(edge_group)
        
        edge_color_layout = QHBoxLayout()
        self.marker_edge_color_button = ColorButton(self.overlay_config.get('markeredgecolor', '#000000'))
        self.marker_edge_color_button.color_changed.connect(self.on_marker_edge_color_changed)
        edge_color_layout.addWidget(self.marker_edge_color_button)
        edge_color_layout.addStretch()
        edge_layout.addRow("Edge Color:", edge_color_layout)
        
        self.marker_edge_width_spin = QDoubleSpinBox()
        self.marker_edge_width_spin.setRange(0.0, 5.0)
        self.marker_edge_width_spin.setSingleStep(0.1)
        self.marker_edge_width_spin.setDecimals(1)
        self.marker_edge_width_spin.setValue(self.overlay_config.get('markeredgewidth', 0.5))
        edge_layout.addRow("Edge Width:", self.marker_edge_width_spin)
        
        layout.addLayout(form_layout)
        layout.addWidget(edge_group)
        layout.addStretch()
        
        return tab
    
    def _create_text_tab(self) -> QWidget:
        """Create the text style tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Font Family
        self.font_combo = QFontComboBox()
        current_font = self.overlay_config.get('fontfamily', 'Arial')
        self.font_combo.setCurrentFont(QFont(current_font))
        form_layout.addRow("Font Family:", self.font_combo)
        
        # Font Size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 48)
        self.font_size_spin.setValue(self.overlay_config.get('fontsize', 10))
        form_layout.addRow("Font Size:", self.font_size_spin)
        
        # Text Color
        color_layout = QHBoxLayout()
        self.text_color_button = ColorButton(self.overlay_config.get('color', '#2c3e50'))
        self.text_color_button.color_changed.connect(self.on_text_color_changed)
        color_layout.addWidget(self.text_color_button)
        color_layout.addStretch()
        form_layout.addRow("Text Color:", color_layout)
        
        # Font Weight
        self.font_weight_combo = QComboBox()
        font_weights = [
            ('Normal', 'normal'),
            ('Bold', 'bold')
        ]
        for name, value in font_weights:
            self.font_weight_combo.addItem(name, value)
        form_layout.addRow("Font Weight:", self.font_weight_combo)
        
        # Background Box
        box_group = QGroupBox("Background Box")
        box_layout = QFormLayout(box_group)
        
        self.show_box_checkbox = QCheckBox("Show Background Box")
        bbox_config = self.overlay_config.get('bbox', {})
        self.show_box_checkbox.setChecked(bool(bbox_config))
        box_layout.addRow("", self.show_box_checkbox)
        
        # Box Color
        box_color_layout = QHBoxLayout()
        self.box_color_button = ColorButton(bbox_config.get('facecolor', '#ffffff') if isinstance(bbox_config, dict) else '#ffffff')
        self.box_color_button.color_changed.connect(self.on_box_color_changed)
        box_color_layout.addWidget(self.box_color_button)
        box_color_layout.addStretch()
        box_layout.addRow("Box Color:", box_color_layout)
        
        # Box Alpha
        box_alpha_layout = QHBoxLayout()
        self.box_alpha_slider = QSlider(Qt.Horizontal)
        self.box_alpha_slider.setRange(0, 100)
        box_alpha = bbox_config.get('alpha', 0.8) if isinstance(bbox_config, dict) else 0.8
        self.box_alpha_slider.setValue(int(box_alpha * 100))
        self.box_alpha_label = QLabel(f"{self.box_alpha_slider.value()}%")
        self.box_alpha_slider.valueChanged.connect(lambda v: self.box_alpha_label.setText(f"{v}%"))
        
        box_alpha_layout.addWidget(self.box_alpha_slider)
        box_alpha_layout.addWidget(self.box_alpha_label)
        box_layout.addRow("Box Opacity:", box_alpha_layout)
        
        # Box Style
        self.box_style_combo = QComboBox()
        box_styles = [
            ('Round', 'round'),
            ('Square', 'square'),
            ('Round4', 'round4'),
            ('Sawtooth', 'sawtooth'),
            ('Circle', 'circle')
        ]
        for name, value in box_styles:
            self.box_style_combo.addItem(name, value)
        box_layout.addRow("Box Style:", self.box_style_combo)
        
        layout.addLayout(form_layout)
        layout.addWidget(box_group)
        layout.addStretch()
        
        return tab
    
    def _connect_preview_updates(self):
        """Connect all controls to preview update"""
        # Line tab connections
        if hasattr(self, 'line_style_combo'):
            self.line_style_combo.currentTextChanged.connect(self.update_preview)
            self.line_width_spin.valueChanged.connect(self.update_preview)
            self.line_alpha_slider.valueChanged.connect(self.update_preview)
        
        # Fill tab connections
        if hasattr(self, 'fill_alpha_slider'):
            self.fill_alpha_slider.valueChanged.connect(self.update_preview)
            self.edge_width_spin.valueChanged.connect(self.update_preview)
        
        # Marker tab connections
        if hasattr(self, 'marker_combo'):
            self.marker_combo.currentTextChanged.connect(self.update_preview)
            self.marker_size_spin.valueChanged.connect(self.update_preview)
            self.marker_alpha_slider.valueChanged.connect(self.update_preview)
            self.marker_edge_width_spin.valueChanged.connect(self.update_preview)
        
        # Text tab connections
        if hasattr(self, 'font_combo'):
            self.font_combo.currentFontChanged.connect(self.update_preview)
            self.font_size_spin.valueChanged.connect(self.update_preview)
            self.font_weight_combo.currentTextChanged.connect(self.update_preview)
            self.show_box_checkbox.toggled.connect(self.update_preview)
            self.box_alpha_slider.valueChanged.connect(self.update_preview)
            self.box_style_combo.currentTextChanged.connect(self.update_preview)
    
    def load_overlay_properties(self):
        """Load current overlay properties into controls"""
        try:
            # Load line properties
            if hasattr(self, 'line_style_combo'):
                line_style = self.overlay_config.get('linestyle', '-')
                for i in range(self.line_style_combo.count()):
                    if self.line_style_combo.itemData(i) == line_style:
                        self.line_style_combo.setCurrentIndex(i)
                        break
            
            # Load marker properties
            if hasattr(self, 'marker_combo'):
                marker = self.overlay_config.get('marker', 'o')
                for i in range(self.marker_combo.count()):
                    if self.marker_combo.itemData(i) == marker:
                        self.marker_combo.setCurrentIndex(i)
                        break
            
            # Load text properties
            if hasattr(self, 'font_weight_combo'):
                weight = self.overlay_config.get('fontweight', 'normal')
                for i in range(self.font_weight_combo.count()):
                    if self.font_weight_combo.itemData(i) == weight:
                        self.font_weight_combo.setCurrentIndex(i)
                        break
            
            if hasattr(self, 'box_style_combo'):
                bbox_config = self.overlay_config.get('bbox', {})
                box_style = bbox_config.get('boxstyle', 'round') if isinstance(bbox_config, dict) else 'round'
                for i in range(self.box_style_combo.count()):
                    if self.box_style_combo.itemData(i) == box_style:
                        self.box_style_combo.setCurrentIndex(i)
                        break
                        
        except Exception as e:
            print(f"[OverlayWizard] Error loading properties: {e}")
    
    def on_line_color_changed(self, color: str):
        """Handle line color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_fill_color_changed(self, color: str):
        """Handle fill color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_edge_color_changed(self, color: str):
        """Handle edge color change"""
        self.overlay_config['edgecolor'] = color
        self.update_preview()
    
    def on_marker_color_changed(self, color: str):
        """Handle marker color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_marker_edge_color_changed(self, color: str):
        """Handle marker edge color change"""
        self.overlay_config['markeredgecolor'] = color
        self.update_preview()
    
    def on_text_color_changed(self, color: str):
        """Handle text color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_box_color_changed(self, color: str):
        """Handle box color change"""
        if 'bbox' not in self.overlay_config:
            self.overlay_config['bbox'] = {}
        self.overlay_config['bbox']['facecolor'] = color
        self.update_preview()
    
    def update_preview(self):
        """Update the style preview"""
        try:
            preview_text = self._generate_preview_text()
            self.preview_label.setText(preview_text)
            
            # Apply preview styling based on current settings
            style_sheet = self._generate_preview_style()
            self.preview_label.setStyleSheet(style_sheet)
            
        except Exception as e:
            print(f"[OverlayWizard] Error updating preview: {e}")
            self.preview_label.setText("Preview error")
    
    def _generate_preview_text(self) -> str:
        """Generate preview text based on overlay type"""
        overlay_type = self._determine_overlay_type()
        overlay_name = self.overlay_config.get('name', self.overlay_id)
        
        if overlay_type == 'line':
            style = self.line_style_combo.currentData() if hasattr(self, 'line_style_combo') else '-'
            width = self.line_width_spin.value() if hasattr(self, 'line_width_spin') else 2.0
            alpha = self.line_alpha_slider.value() if hasattr(self, 'line_alpha_slider') else 80
            return f"{overlay_name}\nStyle: {style}  Width: {width}  Opacity: {alpha}%"
            
        elif overlay_type == 'fill':
            alpha = self.fill_alpha_slider.value() if hasattr(self, 'fill_alpha_slider') else 30
            edge_width = self.edge_width_spin.value() if hasattr(self, 'edge_width_spin') else 0.0
            return f"{overlay_name}\nFill Opacity: {alpha}%  Edge Width: {edge_width}"
            
        elif overlay_type == 'marker':
            marker = self.marker_combo.currentText().split()[0] if hasattr(self, 'marker_combo') else '○'
            size = self.marker_size_spin.value() if hasattr(self, 'marker_size_spin') else 6
            alpha = self.marker_alpha_slider.value() if hasattr(self, 'marker_alpha_slider') else 80
            return f"{overlay_name}\n{marker} Size: {size}  Opacity: {alpha}%"
            
        elif overlay_type == 'text':
            font_size = self.font_size_spin.value() if hasattr(self, 'font_size_spin') else 10
            weight = self.font_weight_combo.currentText() if hasattr(self, 'font_weight_combo') else 'Normal'
            show_box = self.show_box_checkbox.isChecked() if hasattr(self, 'show_box_checkbox') else False
            return f"{overlay_name}\nFont: {font_size}pt {weight}\nBackground: {'Yes' if show_box else 'No'}"
            
        return f"{overlay_name}\nPreview"
    
    def _generate_preview_style(self) -> str:
        """Generate CSS style for preview based on current settings"""
        base_style = """
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
            }
        """
        
        overlay_type = self._determine_overlay_type()
        color = self.overlay_config.get('color', '#000000')
        
        if overlay_type == 'text' and hasattr(self, 'font_size_spin'):
            font_size = self.font_size_spin.value()
            weight = 'bold' if hasattr(self, 'font_weight_combo') and 'Bold' in self.font_weight_combo.currentText() else 'normal'
            base_style += f"""
                QLabel {{
                    color: {color};
                    font-size: {font_size}pt;
                    font-weight: {weight};
                }}
            """
        else:
            base_style += f"""
                QLabel {{
                    color: {color};
                }}
            """
            
        return base_style
    
    def apply_changes(self):
        """Apply current settings to overlay configuration"""
        try:
            self._update_overlay_properties()
            self.overlay_updated.emit(self.overlay_id, self.overlay_config)
            print(f"[OverlayWizard] Applied changes for {self.overlay_id}")
            
        except Exception as e:
            print(f"[OverlayWizard] Error applying changes: {e}")
    
    def accept(self):
        """Accept dialog and apply changes"""
        self.apply_changes()
        super().accept()
    
    def reject(self):
        """Cancel dialog and restore original properties"""
        self._restore_overlay_properties()
        super().reject()
    
    def _update_overlay_properties(self):
        """Update overlay configuration from control values"""
        try:
            overlay_type = self._determine_overlay_type()
            
            # Update line properties
            if overlay_type in ['line', 'both'] and hasattr(self, 'line_style_combo'):
                self.overlay_config['linestyle'] = self.line_style_combo.currentData()
                self.overlay_config['linewidth'] = self.line_width_spin.value()
                self.overlay_config['alpha'] = self.line_alpha_slider.value() / 100.0
                self.overlay_config['color'] = self.line_color_button.get_color()
            
            # Update fill properties
            if overlay_type in ['fill', 'both'] and hasattr(self, 'fill_alpha_slider'):
                self.overlay_config['alpha'] = self.fill_alpha_slider.value() / 100.0
                self.overlay_config['color'] = self.fill_color_button.get_color()
                self.overlay_config['edgecolor'] = self.edge_color_button.get_color()
                self.overlay_config['edgewidth'] = self.edge_width_spin.value()
            
            # Update marker properties
            if overlay_type in ['marker', 'both'] and hasattr(self, 'marker_combo'):
                self.overlay_config['marker'] = self.marker_combo.currentData()
                self.overlay_config['markersize'] = self.marker_size_spin.value()
                self.overlay_config['alpha'] = self.marker_alpha_slider.value() / 100.0
                self.overlay_config['color'] = self.marker_color_button.get_color()
                self.overlay_config['markeredgecolor'] = self.marker_edge_color_button.get_color()
                self.overlay_config['markeredgewidth'] = self.marker_edge_width_spin.value()
            
            # Update text properties
            if overlay_type in ['text', 'both'] and hasattr(self, 'font_combo'):
                self.overlay_config['fontfamily'] = self.font_combo.currentFont().family()
                self.overlay_config['fontsize'] = self.font_size_spin.value()
                self.overlay_config['fontweight'] = self.font_weight_combo.currentData()
                self.overlay_config['color'] = self.text_color_button.get_color()
                
                # Update bbox properties
                if self.show_box_checkbox.isChecked():
                    if 'bbox' not in self.overlay_config:
                        self.overlay_config['bbox'] = {}
                    self.overlay_config['bbox']['boxstyle'] = self.box_style_combo.currentData()
                    self.overlay_config['bbox']['facecolor'] = self.box_color_button.get_color()
                    self.overlay_config['bbox']['alpha'] = self.box_alpha_slider.value() / 100.0
                else:
                    self.overlay_config.pop('bbox', None)
                    
        except Exception as e:
            print(f"[OverlayWizard] Error updating properties: {e}")
    
    def _restore_overlay_properties(self):
        """Restore original overlay properties"""
        self.overlay_config = self.original_properties.copy()
    
    @staticmethod
    def edit_overlay(overlay_id: str, overlay_config: Dict[str, Any], parent=None) -> bool:
        """
        Static method to edit overlay properties
        
        Args:
            overlay_id: Unique identifier for the overlay
            overlay_config: Current overlay configuration
            parent: Parent widget
            
        Returns:
            True if changes were accepted, False if cancelled
        """
        dialog = OverlayWizard(overlay_id, overlay_config, parent)
        return dialog.exec() == QDialog.Accepted


# Shared utility functions for overlay auto-detection and styling
def auto_detect_overlay_type(overlay_id: str, overlay_config: dict, comparison_method: str = None) -> str:
    """
    Auto-detect overlay type based on overlay_id, config, and comparison method.
    Returns: 'line', 'fill', 'text', 'marker', 'combination', or 'unknown'
    """
    overlay_id_lower = overlay_id.lower()
    config = overlay_config
    
    # Text overlays
    if any(keyword in overlay_id_lower for keyword in ['text', 'statistical', 'equation', 'results', 'annotation']):
        return 'text'
    if any(keyword in config for keyword in ['fontsize', 'font_family', 'text_color', 'background_color']):
        return 'text'
    
    # Fill/Shading overlays
    if any(keyword in overlay_id_lower for keyword in ['confidence', 'band', 'shading', 'background', 'fill']):
        return 'fill'
    if config.get('fill', False) or 'fill_color' in config:
        return 'fill'
    
    # Marker overlays
    if any(keyword in overlay_id_lower for keyword in ['outlier', 'marker', 'point', 'highlight']):
        return 'marker'
    if any(keyword in config for keyword in ['marker_shape', 'marker_size', 'marker_color']):
        return 'marker'
    
    # Line overlays
    if any(keyword in overlay_id_lower for keyword in ['line', 'regression', 'identity', 'bias', 'limit']):
        return 'line'
    if any(keyword in config for keyword in ['line_style', 'line_width', 'line_color']):
        return 'line'
    
    # Combination overlays (multiple elements)
    if any(keyword in overlay_id_lower for keyword in ['overlay', 'element', 'feature']):
        return 'combination'
    
    # Method-specific detection
    if comparison_method:
        method_lower = comparison_method.lower()
        if 'correlation' in method_lower:
            if any(keyword in overlay_id_lower for keyword in ['regression', 'identity']):
                return 'line'
            if 'statistical' in overlay_id_lower:
                return 'text'
        elif 'bland' in method_lower:
            if any(keyword in overlay_id_lower for keyword in ['bias', 'limit']):
                return 'line'
            if 'confidence' in overlay_id_lower:
                return 'fill'
    
    # Default to line if we can't determine
    return 'line'


def generate_overlay_style_description(overlay_id: str, overlay_config: dict, overlay_type: str = None, comparison_method: str = None) -> str:
    """
    Generate a human-readable style description for an overlay.
    
    Args:
        overlay_id: ID of the overlay
        overlay_config: Overlay configuration dictionary
        overlay_type: Pre-detected overlay type (optional)
        comparison_method: Name of the comparison method (optional)
        
    Returns:
        Human-readable style description
    """
    if overlay_type is None:
        overlay_type = auto_detect_overlay_type(overlay_id, overlay_config, comparison_method)
    
    # Color mapping for readable names
    color_names = {
        '#1f77b4': 'Blue', '#d62728': 'Red', '#2ca02c': 'Green', '#9467bd': 'Purple',
        '#ff7f0e': 'Orange', '#8c564b': 'Brown', '#e377c2': 'Pink', '#7f7f7f': 'Gray',
        '#bcbd22': 'Yellow', '#17becf': 'Cyan', '#000000': 'Black', '#ffffff': 'White'
    }
    
    def get_color_name(color_hex):
        """Convert hex color to readable name"""
        return color_names.get(color_hex.lower(), color_hex)
    
    def get_line_style_name(style):
        """Convert line style to readable name"""
        style_names = {
            '-': 'solid', '--': 'dashed', ':': 'dotted', '-.': 'dash-dot', 'None': 'none'
        }
        return style_names.get(style, style)
    
    def get_marker_name(marker):
        """Convert marker to readable name"""
        marker_names = {
            'o': 'circle', 's': 'square', '^': 'triangle', 'D': 'diamond', 'v': 'inverted triangle',
            '<': 'left triangle', '>': 'right triangle', 'p': 'pentagon', '*': 'star', 'h': 'hexagon'
        }
        return marker_names.get(marker, marker)
    
    try:
        if overlay_type == 'line':
            color = overlay_config.get('color', '#000000')
            style = overlay_config.get('linestyle', '-')
            width = overlay_config.get('linewidth', 1.0)
            
            color_name = get_color_name(color)
            style_name = get_line_style_name(style)
            
            # Method-specific descriptions
            if 'identity' in overlay_id.lower():
                return f"{color_name} {style_name} (y=x)"
            elif 'regression' in overlay_id.lower():
                return f"{color_name} {style_name} (trend)"
            elif 'bias' in overlay_id.lower():
                return f"{color_name} {style_name} (bias)"
            elif 'limit' in overlay_id.lower():
                return f"{color_name} {style_name} (LoA)"
            else:
                return f"{color_name} {style_name} line"
        
        elif overlay_type == 'fill':
            fill_color = overlay_config.get('facecolor', '#1f77b4')
            alpha = overlay_config.get('alpha', 0.3)
            edge_color = overlay_config.get('edgecolor', '#000000')
            
            fill_name = get_color_name(fill_color)
            edge_name = get_color_name(edge_color)
            
            if 'confidence' in overlay_id.lower():
                return f"{fill_name} confidence band ({alpha:.1f})"
            elif 'shading' in overlay_id.lower():
                return f"{fill_name} shading ({alpha:.1f})"
            else:
                return f"{fill_name} fill ({alpha:.1f})"
        
        elif overlay_type == 'text':
            text_color = overlay_config.get('color', '#000000')
            bg_color = overlay_config.get('backgroundcolor', '#ffffff')
            
            text_name = get_color_name(text_color)
            bg_name = get_color_name(bg_color)
            
            if 'statistical' in overlay_id.lower():
                return f"{text_name} text on {bg_name}"
            elif 'equation' in overlay_id.lower():
                return f"{text_name} equation"
            else:
                return f"{text_name} text"
        
        elif overlay_type == 'marker':
            color = overlay_config.get('color', '#1f77b4')
            marker = overlay_config.get('marker', 'o')
            size = overlay_config.get('marker_size', 50)
            
            color_name = get_color_name(color)
            marker_name = get_marker_name(marker)
            
            if 'outlier' in overlay_id.lower():
                return f"{color_name} {marker_name} (outliers)"
            else:
                return f"{color_name} {marker_name} markers"
        
        elif overlay_type == 'combination':
            return "Mixed overlay"
        
        else:
            return "Unknown overlay"
            
    except Exception as e:
        print(f"[OverlayStyle] Error generating style description: {e}")
        return "Style: N/A"


def get_overlay_icon(overlay_type: str) -> str:
    """
    Get an appropriate icon for the overlay type.
    
    Args:
        overlay_type: Type of overlay ('line', 'fill', 'text', 'marker', 'combination')
        
    Returns:
        Unicode icon string
    """
    icons = {
        'line': '🖍️',
        'fill': '🎨', 
        'text': '📝',
        'marker': '⚫',
        'combination': '🔧',
        'unknown': '❓'
    }
    return icons.get(overlay_type, '❓')


class ComparisonPairOverlayWizard(QDialog):
    """
    Dynamic overlay wizard for comparison pairs that auto-detects overlay types
    and shows all relevant paint options (over-detection approach).
    
    Auto-detects: Line, Fill/Shading, Text, Marker, and Combination overlays
    Shows all relevant paint options for the detected type.
    """
    
    # Signals
    overlay_updated = Signal(str, dict)  # overlay_id, properties when changes are applied
    
    def __init__(self, overlay_id: str, overlay_config: Dict[str, Any], comparison_method: str = None, parent=None):
        super().__init__(parent)
        self.overlay_id = overlay_id
        self.overlay_config = overlay_config.copy()
        self.comparison_method = comparison_method
        self.original_properties = self._backup_overlay_properties()
        
        overlay_name = overlay_config.get('name', overlay_id)
        self.setWindowTitle(f"Overlay Styling - {overlay_name}")
        self.setModal(True)
        self.setFixedSize(500, 700)
        
        # Get overlay type
        self.overlay_type = self._get_overlay_type()
        
        self.init_ui()
        self.load_overlay_properties()
        self.update_preview()
    
    def _backup_overlay_properties(self) -> dict:
        """Backup original overlay properties for cancel operation"""
        return self.overlay_config.copy()
    
    def _get_overlay_type(self) -> str:
        """
        Get overlay type from config or determine from overlay_id and comparison method.
        Returns: 'line', 'fill', 'text', 'marker', 'combination', or 'unknown'
        """
        # First, try to get type from config (if it was set by Overlay object)
        if 'type' in self.overlay_config:
            return self.overlay_config['type']
        
        # Fallback: determine from overlay_id and comparison method
        overlay_id = self.overlay_id.lower()
        
        # Method-specific type detection (simplified)
        if self.comparison_method:
            method_lower = self.comparison_method.lower()
            if 'bland' in method_lower:
                if 'bias' in overlay_id or 'limit' in overlay_id:
                    return 'line'
                elif 'confidence' in overlay_id:
                    return 'fill'
                elif 'statistical' in overlay_id:
                    return 'text'
            elif 'correlation' in method_lower:
                if 'regression' in overlay_id or 'identity' in overlay_id:
                    return 'line'
                elif 'statistical' in overlay_id:
                    return 'text'
        
        # Generic type detection (simplified)
        if any(keyword in overlay_id for keyword in ['confidence', 'band', 'shading', 'fill']):
            return 'fill'
        elif any(keyword in overlay_id for keyword in ['text', 'statistical', 'results']):
            return 'text'
        elif any(keyword in overlay_id for keyword in ['marker', 'outlier', 'point']):
            return 'marker'
        
        # Default to line
        return 'line'
    
    def init_ui(self):
        """Initialize the UI components with auto-detected overlay type"""
        layout = QVBoxLayout(self)
        
        # Title with detected type
        overlay_name = self.overlay_config.get('name', self.overlay_id)
        title = QLabel(f"Editing: {overlay_name} ({self.overlay_type.title()} Overlay)")
        title.setStyleSheet("font-size: 14px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Create tab widget based on auto-detected overlay type
        self.tab_widget = QTabWidget()
        
        # Add tabs based on detected type (over-detection approach)
        if self.overlay_type in ['line', 'combination']:
            line_tab = self._create_line_tab()
            self.tab_widget.addTab(line_tab, "🖍️ Line Style")
        
        if self.overlay_type in ['fill', 'combination']:
            fill_tab = self._create_fill_tab()
            self.tab_widget.addTab(fill_tab, "🎨 Fill Style")
        
        if self.overlay_type in ['marker', 'combination']:
            marker_tab = self._create_marker_tab()
            self.tab_widget.addTab(marker_tab, "⚫ Marker Style")
        
        if self.overlay_type in ['text', 'combination']:
            text_tab = self._create_text_tab()
            self.tab_widget.addTab(text_tab, "📝 Text Style")
        
        # Add positioning tab for all types
        position_tab = self._create_position_tab()
        self.tab_widget.addTab(position_tab, "📍 Position")
        
        layout.addWidget(self.tab_widget)
        
        # Preview section
        preview_group = QGroupBox("Style Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_label = QLabel("Style preview")
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                background-color: white;
                padding: 15px;
                border-radius: 4px;
                font-family: monospace;
            }
        """)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group)
        
        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.apply_changes)
        
        layout.addWidget(button_box)
        
        # Connect all controls to preview update
        self._connect_preview_updates()
    
    def _create_line_tab(self) -> QWidget:
        """Create the line style tab with all line-related options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Color Selection
        color_layout = QHBoxLayout()
        self.line_color_button = ColorButton(self.overlay_config.get('color', '#1f77b4'))
        self.line_color_button.color_changed.connect(self.on_line_color_changed)
        color_layout.addWidget(self.line_color_button)
        color_layout.addStretch()
        form_layout.addRow("Line Color:", color_layout)
        
        # Line Style
        self.line_style_combo = QComboBox()
        line_styles = [
            ('——— Solid', '-'),
            ('- - - Dashed', '--'),
            ('··· Dotted', ':'),
            ('-·- Dash-Dot', '-.'),
            ('None', 'None')
        ]
        for name, value in line_styles:
            self.line_style_combo.addItem(name, value)
        form_layout.addRow("Line Style:", self.line_style_combo)
        
        # Line Width
        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setRange(0.1, 10.0)
        self.line_width_spin.setSingleStep(0.1)
        self.line_width_spin.setValue(self.overlay_config.get('linewidth', 2.0))
        form_layout.addRow("Line Width:", self.line_width_spin)
        
        # Alpha/Transparency
        self.line_alpha_spin = QDoubleSpinBox()
        self.line_alpha_spin.setRange(0.0, 1.0)
        self.line_alpha_spin.setSingleStep(0.1)
        self.line_alpha_spin.setValue(self.overlay_config.get('alpha', 0.8))
        form_layout.addRow("Transparency:", self.line_alpha_spin)
        
        # Z-order
        self.line_zorder_spin = QSpinBox()
        self.line_zorder_spin.setRange(-10, 10)
        self.line_zorder_spin.setValue(self.overlay_config.get('zorder', 1))
        form_layout.addRow("Z-Order:", self.line_zorder_spin)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab
    
    def _create_fill_tab(self) -> QWidget:
        """Create the fill style tab with all fill-related options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Fill Color
        fill_color_layout = QHBoxLayout()
        self.fill_color_button = ColorButton(self.overlay_config.get('facecolor', '#1f77b4'))
        self.fill_color_button.color_changed.connect(self.on_fill_color_changed)
        fill_color_layout.addWidget(self.fill_color_button)
        fill_color_layout.addStretch()
        form_layout.addRow("Fill Color:", fill_color_layout)
        
        # Edge Color
        edge_color_layout = QHBoxLayout()
        self.edge_color_button = ColorButton(self.overlay_config.get('edgecolor', '#000000'))
        self.edge_color_button.color_changed.connect(self.on_edge_color_changed)
        edge_color_layout.addWidget(self.edge_color_button)
        edge_color_layout.addStretch()
        form_layout.addRow("Edge Color:", edge_color_layout)
        
        # Edge Style
        self.edge_style_combo = QComboBox()
        edge_styles = [
            ('——— Solid', '-'),
            ('- - - Dashed', '--'),
            ('··· Dotted', ':'),
            ('-·- Dash-Dot', '-.'),
            ('None', 'None')
        ]
        for name, value in edge_styles:
            self.edge_style_combo.addItem(name, value)
        form_layout.addRow("Edge Style:", self.edge_style_combo)
        
        # Edge Width
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setRange(0.0, 5.0)
        self.edge_width_spin.setSingleStep(0.1)
        self.edge_width_spin.setValue(self.overlay_config.get('linewidth', 1.0))
        form_layout.addRow("Edge Width:", self.edge_width_spin)
        
        # Fill Alpha
        self.fill_alpha_spin = QDoubleSpinBox()
        self.fill_alpha_spin.setRange(0.0, 1.0)
        self.fill_alpha_spin.setSingleStep(0.1)
        self.fill_alpha_spin.setValue(self.overlay_config.get('alpha', 0.3))
        form_layout.addRow("Fill Transparency:", self.fill_alpha_spin)
        
        # Fill Pattern
        self.fill_pattern_combo = QComboBox()
        fill_patterns = [
            ('Solid', 'solid'),
            ('Hatched', 'hatched'),
            ('Cross-hatched', 'crosshatched'),
            ('Dotted', 'dotted')
        ]
        for name, value in fill_patterns:
            self.fill_pattern_combo.addItem(name, value)
        form_layout.addRow("Fill Pattern:", self.fill_pattern_combo)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab
    
    def _create_marker_tab(self) -> QWidget:
        """Create the marker style tab with all marker-related options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Marker Color
        marker_color_layout = QHBoxLayout()
        self.marker_color_button = ColorButton(self.overlay_config.get('marker_color', '#1f77b4'))
        self.marker_color_button.color_changed.connect(self.on_marker_color_changed)
        marker_color_layout.addWidget(self.marker_color_button)
        marker_color_layout.addStretch()
        form_layout.addRow("Marker Color:", marker_color_layout)
        
        # Marker Edge Color
        marker_edge_color_layout = QHBoxLayout()
        self.marker_edge_color_button = ColorButton(self.overlay_config.get('marker_edge_color', '#000000'))
        self.marker_edge_color_button.color_changed.connect(self.on_marker_edge_color_changed)
        marker_edge_color_layout.addWidget(self.marker_edge_color_button)
        marker_edge_color_layout.addStretch()
        form_layout.addRow("Edge Color:", marker_edge_color_layout)
        
        # Marker Shape
        self.marker_shape_combo = QComboBox()
        marker_shapes = [
            ('○ Circle', 'o'),
            ('□ Square', 's'),
            ('△ Triangle', '^'),
            ('◇ Diamond', 'D'),
            ('▽ Inverted Triangle', 'v'),
            ('◁ Left Triangle', '<'),
            ('▷ Right Triangle', '>'),
            ('⬟ Pentagon', 'p'),
            ('✦ Star', '*'),
            ('⬢ Hexagon', 'h')
        ]
        for name, value in marker_shapes:
            self.marker_shape_combo.addItem(name, value)
        form_layout.addRow("Marker Shape:", self.marker_shape_combo)
        
        # Marker Size
        self.marker_size_spin = QDoubleSpinBox()
        self.marker_size_spin.setRange(1.0, 100.0)
        self.marker_size_spin.setSingleStep(1.0)
        self.marker_size_spin.setValue(self.overlay_config.get('marker_size', 50.0))
        form_layout.addRow("Marker Size:", self.marker_size_spin)
        
        # Edge Width
        self.marker_edge_width_spin = QDoubleSpinBox()
        self.marker_edge_width_spin.setRange(0.0, 5.0)
        self.marker_edge_width_spin.setSingleStep(0.1)
        self.marker_edge_width_spin.setValue(self.overlay_config.get('marker_edge_width', 1.0))
        form_layout.addRow("Edge Width:", self.marker_edge_width_spin)
        
        # Alpha
        self.marker_alpha_spin = QDoubleSpinBox()
        self.marker_alpha_spin.setRange(0.0, 1.0)
        self.marker_alpha_spin.setSingleStep(0.1)
        self.marker_alpha_spin.setValue(self.overlay_config.get('marker_alpha', 0.8))
        form_layout.addRow("Transparency:", self.marker_alpha_spin)
        
        # Fill Style
        self.marker_fill_style_combo = QComboBox()
        fill_styles = [
            ('Full', 'full'),
            ('Hollow', 'none'),
            ('Top Half', 'top'),
            ('Bottom Half', 'bottom'),
            ('Left Half', 'left'),
            ('Right Half', 'right')
        ]
        for name, value in fill_styles:
            self.marker_fill_style_combo.addItem(name, value)
        form_layout.addRow("Fill Style:", self.marker_fill_style_combo)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab
    
    def _create_text_tab(self) -> QWidget:
        """Create the text style tab with all text-related options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Text Color
        text_color_layout = QHBoxLayout()
        self.text_color_button = ColorButton(self.overlay_config.get('color', '#000000'))
        self.text_color_button.color_changed.connect(self.on_text_color_changed)
        text_color_layout.addWidget(self.text_color_button)
        text_color_layout.addStretch()
        form_layout.addRow("Text Color:", text_color_layout)
        
        # Background Color
        bg_color_layout = QHBoxLayout()
        self.bg_color_button = ColorButton(self.overlay_config.get('backgroundcolor', '#ffffff'))
        self.bg_color_button.color_changed.connect(self.on_box_color_changed)
        bg_color_layout.addWidget(self.bg_color_button)
        bg_color_layout.addStretch()
        form_layout.addRow("Background Color:", bg_color_layout)
        
        # Font Family
        self.font_family_combo = QFontComboBox()
        self.font_family_combo.setCurrentText(self.overlay_config.get('fontfamily', 'Arial'))
        form_layout.addRow("Font Family:", self.font_family_combo)
        
        # Font Size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 72)
        self.font_size_spin.setValue(self.overlay_config.get('fontsize', 10))
        form_layout.addRow("Font Size:", self.font_size_spin)
        
        # Font Weight
        self.font_weight_combo = QComboBox()
        font_weights = [
            ('Normal', 'normal'),
            ('Bold', 'bold'),
            ('Light', 'light')
        ]
        for name, value in font_weights:
            self.font_weight_combo.addItem(name, value)
        form_layout.addRow("Font Weight:", self.font_weight_combo)
        
        # Font Style
        self.font_style_combo = QComboBox()
        font_styles = [
            ('Normal', 'normal'),
            ('Italic', 'italic')
        ]
        for name, value in font_styles:
            self.font_style_combo.addItem(name, value)
        form_layout.addRow("Font Style:", self.font_style_combo)
        
        # Box Style
        self.box_style_combo = QComboBox()
        box_styles = [
            ('Rounded', 'round'),
            ('Square', 'square'),
            ('None', 'none')
        ]
        for name, value in box_styles:
            self.box_style_combo.addItem(name, value)
        form_layout.addRow("Box Style:", self.box_style_combo)
        
        # Box Alpha
        self.box_alpha_spin = QDoubleSpinBox()
        self.box_alpha_spin.setRange(0.0, 1.0)
        self.box_alpha_spin.setSingleStep(0.1)
        self.box_alpha_spin.setValue(self.overlay_config.get('box_alpha', 0.8))
        form_layout.addRow("Box Transparency:", self.box_alpha_spin)
        
        # Border Color
        border_color_layout = QHBoxLayout()
        self.border_color_button = ColorButton(self.overlay_config.get('border_color', '#000000'))
        self.border_color_button.color_changed.connect(self.on_border_color_changed)
        border_color_layout.addWidget(self.border_color_button)
        border_color_layout.addStretch()
        form_layout.addRow("Border Color:", border_color_layout)
        
        # Border Width
        self.border_width_spin = QDoubleSpinBox()
        self.border_width_spin.setRange(0.0, 5.0)
        self.border_width_spin.setSingleStep(0.1)
        self.border_width_spin.setValue(self.overlay_config.get('border_width', 1.0))
        form_layout.addRow("Border Width:", self.border_width_spin)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab
    
    def _create_position_tab(self) -> QWidget:
        """Create the position tab for overlay positioning"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        form_layout = QFormLayout()
        
        # Position
        self.position_combo = QComboBox()
        positions = [
            ('Top Left', 'top-left'),
            ('Top Right', 'top-right'),
            ('Bottom Left', 'bottom-left'),
            ('Bottom Right', 'bottom-right'),
            ('Center', 'center'),
            ('Custom', 'custom')
        ]
        for name, value in positions:
            self.position_combo.addItem(name, value)
        form_layout.addRow("Position:", self.position_combo)
        
        # X Offset
        self.x_offset_spin = QDoubleSpinBox()
        self.x_offset_spin.setRange(-1.0, 1.0)
        self.x_offset_spin.setSingleStep(0.01)
        self.x_offset_spin.setValue(self.overlay_config.get('x_offset', 0.02))
        form_layout.addRow("X Offset:", self.x_offset_spin)
        
        # Y Offset
        self.y_offset_spin = QDoubleSpinBox()
        self.y_offset_spin.setRange(-1.0, 1.0)
        self.y_offset_spin.setSingleStep(0.01)
        self.y_offset_spin.setValue(self.overlay_config.get('y_offset', 0.98))
        form_layout.addRow("Y Offset:", self.y_offset_spin)
        
        # Z-order
        self.position_zorder_spin = QSpinBox()
        self.position_zorder_spin.setRange(-10, 10)
        self.position_zorder_spin.setValue(self.overlay_config.get('zorder', 5))
        form_layout.addRow("Z-Order:", self.position_zorder_spin)
        
        layout.addLayout(form_layout)
        layout.addStretch()
        return tab
    
    def _connect_preview_updates(self):
        """Connect all controls to preview update"""
        # Line controls
        if hasattr(self, 'line_color_button'):
            self.line_color_button.color_changed.connect(self.update_preview)
        if hasattr(self, 'line_style_combo'):
            self.line_style_combo.currentTextChanged.connect(self.update_preview)
        if hasattr(self, 'line_width_spin'):
            self.line_width_spin.valueChanged.connect(self.update_preview)
        if hasattr(self, 'line_alpha_spin'):
            self.line_alpha_spin.valueChanged.connect(self.update_preview)
        
        # Fill controls
        if hasattr(self, 'fill_color_button'):
            self.fill_color_button.color_changed.connect(self.update_preview)
        if hasattr(self, 'edge_color_button'):
            self.edge_color_button.color_changed.connect(self.update_preview)
        if hasattr(self, 'fill_alpha_spin'):
            self.fill_alpha_spin.valueChanged.connect(self.update_preview)
        
        # Marker controls
        if hasattr(self, 'marker_color_button'):
            self.marker_color_button.color_changed.connect(self.update_preview)
        if hasattr(self, 'marker_shape_combo'):
            self.marker_shape_combo.currentTextChanged.connect(self.update_preview)
        if hasattr(self, 'marker_size_spin'):
            self.marker_size_spin.valueChanged.connect(self.update_preview)
        
        # Text controls
        if hasattr(self, 'text_color_button'):
            self.text_color_button.color_changed.connect(self.update_preview)
        if hasattr(self, 'font_family_combo'):
            self.font_family_combo.currentTextChanged.connect(self.update_preview)
        if hasattr(self, 'font_size_spin'):
            self.font_size_spin.valueChanged.connect(self.update_preview)
    
    def load_overlay_properties(self):
        """Load overlay properties into UI controls"""
        try:
            # Load line properties
            if hasattr(self, 'line_color_button'):
                self.line_color_button.set_color(self.overlay_config.get('color', '#1f77b4'))
            if hasattr(self, 'line_style_combo'):
                line_style = self.overlay_config.get('linestyle', '-')
                for i in range(self.line_style_combo.count()):
                    if self.line_style_combo.itemData(i) == line_style:
                        self.line_style_combo.setCurrentIndex(i)
                        break
            if hasattr(self, 'line_width_spin'):
                self.line_width_spin.setValue(self.overlay_config.get('linewidth', 2.0))
            if hasattr(self, 'line_alpha_spin'):
                self.line_alpha_spin.setValue(self.overlay_config.get('alpha', 0.8))
            
            # Load fill properties
            if hasattr(self, 'fill_color_button'):
                self.fill_color_button.set_color(self.overlay_config.get('facecolor', '#1f77b4'))
            if hasattr(self, 'edge_color_button'):
                self.edge_color_button.set_color(self.overlay_config.get('edgecolor', '#000000'))
            if hasattr(self, 'fill_alpha_spin'):
                self.fill_alpha_spin.setValue(self.overlay_config.get('alpha', 0.3))
            
            # Load marker properties
            if hasattr(self, 'marker_color_button'):
                self.marker_color_button.set_color(self.overlay_config.get('marker_color', '#1f77b4'))
            if hasattr(self, 'marker_shape_combo'):
                marker_shape = self.overlay_config.get('marker', 'o')
                for i in range(self.marker_shape_combo.count()):
                    if self.marker_shape_combo.itemData(i) == marker_shape:
                        self.marker_shape_combo.setCurrentIndex(i)
                        break
            if hasattr(self, 'marker_size_spin'):
                self.marker_size_spin.setValue(self.overlay_config.get('marker_size', 50.0))
            
            # Load text properties
            if hasattr(self, 'text_color_button'):
                self.text_color_button.set_color(self.overlay_config.get('color', '#000000'))
            if hasattr(self, 'font_family_combo'):
                self.font_family_combo.setCurrentText(self.overlay_config.get('fontfamily', 'Arial'))
            if hasattr(self, 'font_size_spin'):
                self.font_size_spin.setValue(self.overlay_config.get('fontsize', 10))
            
        except Exception as e:
            print(f"[ComparisonPairOverlayWizard] Error loading overlay properties: {e}")
    
    def on_line_color_changed(self, color: str):
        """Handle line color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_fill_color_changed(self, color: str):
        """Handle fill color change"""
        self.overlay_config['facecolor'] = color
        self.update_preview()
    
    def on_edge_color_changed(self, color: str):
        """Handle edge color change"""
        self.overlay_config['edgecolor'] = color
        self.update_preview()
    
    def on_marker_color_changed(self, color: str):
        """Handle marker color change"""
        self.overlay_config['marker_color'] = color
        self.update_preview()
    
    def on_marker_edge_color_changed(self, color: str):
        """Handle marker edge color change"""
        self.overlay_config['marker_edge_color'] = color
        self.update_preview()
    
    def on_text_color_changed(self, color: str):
        """Handle text color change"""
        self.overlay_config['color'] = color
        self.update_preview()
    
    def on_box_color_changed(self, color: str):
        """Handle box color change"""
        self.overlay_config['backgroundcolor'] = color
        self.update_preview()
    
    def on_border_color_changed(self, color: str):
        """Handle border color change"""
        self.overlay_config['border_color'] = color
        self.update_preview()
    
    def update_preview(self):
        """Update the style preview"""
        try:
            preview_text = self._generate_preview_text()
            self.preview_label.setText(preview_text)
        except Exception as e:
            print(f"[ComparisonPairOverlayWizard] Error updating preview: {e}")
    
    def _generate_preview_text(self) -> str:
        """Generate preview text based on overlay type and current settings"""
        try:
            preview_lines = []
            preview_lines.append(f"Overlay: {self.overlay_id}")
            preview_lines.append(f"Type: {self.overlay_type.title()}")
            preview_lines.append("")
            
            if self.overlay_type in ['line', 'combination']:
                color = self.overlay_config.get('color', '#1f77b4')
                style = self.overlay_config.get('linestyle', '-')
                width = self.overlay_config.get('linewidth', 2.0)
                alpha = self.overlay_config.get('alpha', 0.8)
                preview_lines.append(f"Line: {style} {width}px {color} ({alpha:.1f})")
            
            if self.overlay_type in ['fill', 'combination']:
                fill_color = self.overlay_config.get('facecolor', '#1f77b4')
                edge_color = self.overlay_config.get('edgecolor', '#000000')
                alpha = self.overlay_config.get('alpha', 0.3)
                preview_lines.append(f"Fill: {fill_color} edge:{edge_color} ({alpha:.1f})")
            
            if self.overlay_type in ['marker', 'combination']:
                marker_color = self.overlay_config.get('marker_color', '#1f77b4')
                marker_shape = self.overlay_config.get('marker', 'o')
                marker_size = self.overlay_config.get('marker_size', 50.0)
                preview_lines.append(f"Marker: {marker_shape} {marker_size}px {marker_color}")
            
            if self.overlay_type in ['text', 'combination']:
                text_color = self.overlay_config.get('color', '#000000')
                font_family = self.overlay_config.get('fontfamily', 'Arial')
                font_size = self.overlay_config.get('fontsize', 10)
                preview_lines.append(f"Text: {font_family} {font_size}pt {text_color}")
            
            return "\n".join(preview_lines)
            
        except Exception as e:
            return f"Preview Error: {str(e)}"
    
    def apply_changes(self):
        """Apply changes to overlay properties"""
        try:
            self._update_overlay_properties()
            self.overlay_updated.emit(self.overlay_id, self.overlay_config)
            print(f"[ComparisonPairOverlayWizard] Applied changes to overlay: {self.overlay_id}")
        except Exception as e:
            print(f"[ComparisonPairOverlayWizard] Error applying changes: {e}")
    
    def accept(self):
        """Accept changes and close dialog"""
        self.apply_changes()
        super().accept()
    
    def reject(self):
        """Reject changes and restore original properties"""
        self._restore_overlay_properties()
        super().reject()
    
    def _update_overlay_properties(self):
        """Update overlay properties from UI controls"""
        try:
            # Update line properties
            if hasattr(self, 'line_color_button'):
                self.overlay_config['color'] = self.line_color_button.get_color()
            if hasattr(self, 'line_style_combo'):
                self.overlay_config['linestyle'] = self.line_style_combo.currentData()
            if hasattr(self, 'line_width_spin'):
                self.overlay_config['linewidth'] = self.line_width_spin.value()
            if hasattr(self, 'line_alpha_spin'):
                self.overlay_config['alpha'] = self.line_alpha_spin.value()
            if hasattr(self, 'line_zorder_spin'):
                self.overlay_config['zorder'] = self.line_zorder_spin.value()
            
            # Update fill properties
            if hasattr(self, 'fill_color_button'):
                self.overlay_config['facecolor'] = self.fill_color_button.get_color()
            if hasattr(self, 'edge_color_button'):
                self.overlay_config['edgecolor'] = self.edge_color_button.get_color()
            if hasattr(self, 'edge_style_combo'):
                self.overlay_config['edge_linestyle'] = self.edge_style_combo.currentData()
            if hasattr(self, 'edge_width_spin'):
                self.overlay_config['linewidth'] = self.edge_width_spin.value()
            if hasattr(self, 'fill_alpha_spin'):
                self.overlay_config['alpha'] = self.fill_alpha_spin.value()
            if hasattr(self, 'fill_pattern_combo'):
                self.overlay_config['fill_pattern'] = self.fill_pattern_combo.currentData()
            
            # Update marker properties
            if hasattr(self, 'marker_color_button'):
                self.overlay_config['marker_color'] = self.marker_color_button.get_color()
            if hasattr(self, 'marker_edge_color_button'):
                self.overlay_config['marker_edge_color'] = self.marker_edge_color_button.get_color()
            if hasattr(self, 'marker_shape_combo'):
                self.overlay_config['marker'] = self.marker_shape_combo.currentData()
            if hasattr(self, 'marker_size_spin'):
                self.overlay_config['marker_size'] = self.marker_size_spin.value()
            if hasattr(self, 'marker_edge_width_spin'):
                self.overlay_config['marker_edge_width'] = self.marker_edge_width_spin.value()
            if hasattr(self, 'marker_alpha_spin'):
                self.overlay_config['marker_alpha'] = self.marker_alpha_spin.value()
            if hasattr(self, 'marker_fill_style_combo'):
                self.overlay_config['marker_fill_style'] = self.marker_fill_style_combo.currentData()
            
            # Update text properties
            if hasattr(self, 'text_color_button'):
                self.overlay_config['color'] = self.text_color_button.get_color()
            if hasattr(self, 'bg_color_button'):
                self.overlay_config['backgroundcolor'] = self.bg_color_button.get_color()
            if hasattr(self, 'font_family_combo'):
                self.overlay_config['fontfamily'] = self.font_family_combo.currentText()
            if hasattr(self, 'font_size_spin'):
                self.overlay_config['fontsize'] = self.font_size_spin.value()
            if hasattr(self, 'font_weight_combo'):
                self.overlay_config['fontweight'] = self.font_weight_combo.currentData()
            if hasattr(self, 'font_style_combo'):
                self.overlay_config['fontstyle'] = self.font_style_combo.currentData()
            if hasattr(self, 'box_style_combo'):
                self.overlay_config['boxstyle'] = self.box_style_combo.currentData()
            if hasattr(self, 'box_alpha_spin'):
                self.overlay_config['box_alpha'] = self.box_alpha_spin.value()
            if hasattr(self, 'border_color_button'):
                self.overlay_config['border_color'] = self.border_color_button.get_color()
            if hasattr(self, 'border_width_spin'):
                self.overlay_config['border_width'] = self.border_width_spin.value()
            
            # Update position properties
            if hasattr(self, 'position_combo'):
                self.overlay_config['position'] = self.position_combo.currentData()
            if hasattr(self, 'x_offset_spin'):
                self.overlay_config['x_offset'] = self.x_offset_spin.value()
            if hasattr(self, 'y_offset_spin'):
                self.overlay_config['y_offset'] = self.y_offset_spin.value()
            if hasattr(self, 'position_zorder_spin'):
                self.overlay_config['zorder'] = self.position_zorder_spin.value()
            
        except Exception as e:
            print(f"[ComparisonPairOverlayWizard] Error updating overlay properties: {e}")
    
    def _restore_overlay_properties(self):
        """Restore original overlay properties"""
        self.overlay_config = self.original_properties.copy()


# Convenience function for opening the comparison pair overlay wizard
def open_comparison_pair_overlay_wizard(overlay_id: str, overlay_config: Dict[str, Any], 
                                      comparison_method: str = None, parent=None) -> bool:
    """
    Open the comparison pair overlay wizard for editing overlay properties.
    
    Args:
        overlay_id: ID of the overlay to edit
        overlay_config: Current overlay configuration
        comparison_method: Name of the comparison method (for context)
        parent: Parent widget
        
    Returns:
        True if changes were applied, False if cancelled
    """
    try:
        wizard = ComparisonPairOverlayWizard(overlay_id, overlay_config, comparison_method, parent)
        result = wizard.exec()
        return result == QDialog.Accepted
    except Exception as e:
        print(f"[OverlayWizard] Error opening comparison pair overlay wizard: {e}")
        return False 