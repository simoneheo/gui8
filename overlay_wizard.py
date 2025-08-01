from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QPushButton, QLineEdit, QComboBox, QColorDialog,
    QDialogButtonBox, QSpinBox, QDoubleSpinBox, QSlider, QTextEdit, QWidget, QGroupBox
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from plot_manager import PLOT_STYLES

class ColorButton(QPushButton):
    color_changed = Signal(str)
    def __init__(self, initial_color="#1f77b4"):
        super().__init__()
        self.current_color = initial_color
        self.setFixedSize(40, 24)
        self.update_button_color()
        self.clicked.connect(self.select_color)
    def update_button_color(self):
        self.setStyleSheet(f"background-color: {self.current_color}; border: 1px solid #333;")
    def select_color(self):
        color = QColorDialog.getColor(QColor(self.current_color), self)
        if color.isValid():
            self.current_color = color.name()
            self.update_button_color()
            self.color_changed.emit(self.current_color)
    def set_color(self, color: str):
        self.current_color = color
        self.update_button_color()
    def get_color(self) -> str:
        return self.current_color

class OverlayWizard(QDialog):
    style_updated = Signal(dict)
    
    def __init__(self, overlay_type: str, style: dict, parent=None):
        super().__init__(parent)
        self.overlay_type = overlay_type
        self.overlay_style = style.copy() if style else {}
        
        # Map hline type to line type for UI purposes
        display_type = 'line' if overlay_type == 'hline' or overlay_type == 'vline' else overlay_type
        
        self.setWindowTitle(f"Edit {display_type.title()} Overlay Style")
        self.setModal(True)
        self.setFixedSize(420, 420)
        
        # Custom text state tracking
        self.initial_text = None
        self.is_custom_text = False
        
        self.init_ui()
        self.load_style()
    
    def set_initial_text(self, text: str):
        """Set the initial text content for text overlays."""
        self.initial_text = text
        if hasattr(self, 'text_edit'):
            self.text_edit.setPlainText(text)
            print(f"[OverlayWizard] Set initial text: {len(text)} characters")
    
    def set_custom_state(self, is_custom: bool):
        """Set whether this overlay has custom text."""
        self.is_custom_text = is_custom
        self._update_ui_for_custom_state()
    
    def _update_ui_for_custom_state(self):
        """Update UI elements to reflect custom text state."""
        if self.overlay_type == 'text' and hasattr(self, 'text_edit'):
            if self.is_custom_text:
                # Add visual indicator for custom text
                self.text_edit.setStyleSheet("QTextEdit { border: 2px solid #4CAF50; }")
                self.text_edit.setToolTip("This text has been customized. Click 'Reset to Auto' to restore original.")
                # Add reset button if not already present
                if not hasattr(self, 'reset_button'):
                    self._add_reset_button()
            else:
                # Remove custom styling
                self.text_edit.setStyleSheet("")
                self.text_edit.setToolTip("Auto-generated text. Edit to customize.")
                # Remove reset button if present
                if hasattr(self, 'reset_button'):
                    self.reset_button.hide()
    
    def _add_reset_button(self):
        """Add a reset button to restore auto-generated text."""
        self.reset_button = QPushButton("Reset to Auto")
        self.reset_button.clicked.connect(self._reset_to_auto_text)
        self.reset_button.setStyleSheet("QPushButton { color: #FF6B6B; }")
        self.reset_button.setToolTip("Restore original auto-generated text")
        
        # Insert reset button after the text edit in the form
        for i in range(self.form.rowCount()):
            if self.form.itemAt(i, QFormLayout.ItemRole.FieldRole).widget() == self.text_edit:
                # Create a container widget for text edit and reset button
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(4)
                
                # Remove text edit from form and add to container
                self.form.removeRow(i)
                container_layout.addWidget(self.text_edit)
                container_layout.addWidget(self.reset_button)
                
                # Add container back to form
                self.form.insertRow(i, "Text Content:", container)
                break
    
    def _reset_to_auto_text(self):
        """Reset text content to auto-generated text."""
        if self.initial_text is not None:
            self.text_edit.setPlainText(self.initial_text)
            self.is_custom_text = False
            self._update_ui_for_custom_state()
            print(f"[OverlayWizard] Reset to auto-generated text: {len(self.initial_text)} characters")
    def init_ui(self):
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        
        
        # Treat both 'line', 'hline', and 'vline' as line controls
        if self.overlay_type == 'line' or self.overlay_type == 'hline' or self.overlay_type == 'vline':
            print(f"[OverlayWizard] Adding line controls for type: {self.overlay_type}")
            self._add_line_controls()
        elif self.overlay_type == 'text':
            print(f"[OverlayWizard] Adding text controls for type: {self.overlay_type}")
            self._add_text_controls()
        elif self.overlay_type == 'fill':
            print(f"[OverlayWizard] Adding fill controls for type: {self.overlay_type}")
            self._add_fill_controls()
        else:
            print(f"[OverlayWizard] Unsupported overlay type: {self.overlay_type}")
            label = QLabel(f"Unsupported overlay type: {self.overlay_type}")
            layout.addWidget(label)
        layout.addLayout(self.form)
        self._add_dialog_buttons(layout)
    def _add_line_controls(self):
        self.color_btn = ColorButton(self.overlay_style.get('color', '#808080'))
        self.form.addRow("Line Color:", self.color_btn)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 10)
        self.width_spin.setValue(self.overlay_style.get('linewidth', 2))
        self.form.addRow("Line Width:", self.width_spin)
        # Line Style - Show symbols instead of names
        self.style_combo = QComboBox()
        # Map symbols to values for display
        style_symbols = {
            '─': 'Solid',
            '┄┄': 'Dashed', 
            '┄─┄': 'Dash-dot',
            '┄┄┄': 'Dotted',
            '━━━': 'Solid (thick)',
            '┅┅┅': 'Dashed (long)',
            '┄┄┄┄': 'Dotted (sparse)',
            '┄─┄─': 'Dash-dot-dot',
            '━┄━┄': 'Dash-dash-dot',
            'None': 'None'
        }
        for symbol, name in style_symbols.items():
            if name in PLOT_STYLES['Line Styles']:
                self.style_combo.addItem(symbol, PLOT_STYLES['Line Styles'][name])
            elif name == 'Solid (thick)':
                self.style_combo.addItem(symbol, '-')  # Will be handled by linewidth
            elif name == 'Dashed (long)':
                self.style_combo.addItem(symbol, '--')  # Will be handled by dash pattern
            elif name == 'Dotted (sparse)':
                self.style_combo.addItem(symbol, ':')  # Will be handled by dot spacing
            elif name == 'Dash-dot-dot':
                self.style_combo.addItem(symbol, '-.')  # Will be handled by custom pattern
            elif name == 'Dash-dash-dot':
                self.style_combo.addItem(symbol, '--')  # Will be handled by custom pattern
        
        # Set current style
        current_style = self.overlay_style.get('linestyle', '-')
        for i in range(self.style_combo.count()):
            if self.style_combo.itemData(i) == current_style:
                self.style_combo.setCurrentIndex(i)
                break
        
        self.form.addRow("Line Style:", self.style_combo)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.overlay_style.get('alpha', 0.8))
        self.form.addRow("Alpha (Transparency):", self.alpha_spin)
    def _add_text_controls(self):
        self.text_edit = QTextEdit(self.overlay_style.get('text', ''))
        self.form.addRow("Text Content:", self.text_edit)
        self.fontsize_spin = QSpinBox()
        self.fontsize_spin.setRange(6, 48)
        self.fontsize_spin.setValue(self.overlay_style.get('fontsize', 12))
        self.form.addRow("Font Size:", self.fontsize_spin)
        self.color_btn = ColorButton(self.overlay_style.get('color', '#000000'))
        self.form.addRow("Font Color:", self.color_btn)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.overlay_style.get('alpha', 1.0))
        self.form.addRow("Alpha (Transparency):", self.alpha_spin)
        # Position
        self.position_combo = QComboBox()
        self.position_combo.addItems(['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center', 'custom'])
        pos = self.overlay_style.get('position', 'top-left')
        if isinstance(pos, (tuple, list)):
            self.position_combo.setCurrentText('custom')
        else:
            self.position_combo.setCurrentText(pos if pos in [self.position_combo.itemText(i) for i in range(self.position_combo.count())] else 'top-left')
        self.form.addRow("Position:", self.position_combo)
        self.x_input = QDoubleSpinBox(); self.x_input.setRange(0, 1); self.x_input.setSingleStep(0.01)
        self.y_input = QDoubleSpinBox(); self.y_input.setRange(0, 1); self.y_input.setSingleStep(0.01)
        pos_val = self.overlay_style.get('position', (0.02, 0.98))
        if isinstance(pos_val, (tuple, list)) and len(pos_val) == 2:
            self.x_input.setValue(pos_val[0])
            self.y_input.setValue(pos_val[1])
        else:
            self.x_input.setValue(0.02)
            self.y_input.setValue(0.98)
        pos_widget = QWidget(); pos_layout = QHBoxLayout(pos_widget); pos_layout.setContentsMargins(0,0,0,0)
        pos_layout.addWidget(QLabel("X:")); pos_layout.addWidget(self.x_input)
        pos_layout.addWidget(QLabel("Y:")); pos_layout.addWidget(self.y_input)
        self.form.addRow("Custom Position:", pos_widget)
        self.position_combo.currentTextChanged.connect(lambda t: pos_widget.setVisible(t == 'custom'))
        pos_widget.setVisible(self.position_combo.currentText() == 'custom')
        # Box style
        self.boxstyle_combo = QComboBox()
        self.boxstyle_combo.addItems(['none', 'rounded', 'square'])
        boxstyle = self.overlay_style.get('bbox', {}).get('boxstyle', 'rounded')
        self.boxstyle_combo.setCurrentText('none' if not boxstyle or boxstyle == 'none' else ('rounded' if 'round' in boxstyle else 'square'))
        self.form.addRow("Box Style:", self.boxstyle_combo)
        self.boxcolor_btn = ColorButton(self.overlay_style.get('bbox', {}).get('facecolor', '#ffffff'))
        self.form.addRow("Box Color:", self.boxcolor_btn)
        self.boxalpha_spin = QDoubleSpinBox(); self.boxalpha_spin.setRange(0.0, 1.0); self.boxalpha_spin.setSingleStep(0.05)
        self.boxalpha_spin.setValue(self.overlay_style.get('bbox', {}).get('alpha', 0.8))
        self.form.addRow("Box Alpha:", self.boxalpha_spin)
    def _add_fill_controls(self):
        self.color_btn = ColorButton(self.overlay_style.get('color', '#ff69b4'))
        self.form.addRow("Fill Color:", self.color_btn)
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(self.overlay_style.get('alpha', 0.1))
        self.form.addRow("Alpha (Transparency):", self.alpha_spin)
        self.edgecolor_btn = ColorButton(self.overlay_style.get('edgecolor', '#ff69b4'))
        self.form.addRow("Edge Color:", self.edgecolor_btn)
        self.edgewidth_spin = QSpinBox()
        self.edgewidth_spin.setRange(0, 10)
        self.edgewidth_spin.setValue(self.overlay_style.get('linewidth', 1))
        self.form.addRow("Edge Width:", self.edgewidth_spin)
    def _add_dialog_buttons(self, layout):
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Apply)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        layout.addWidget(btns)
    def load_style(self):
        pass  # Already loaded in widget creation
    def apply_changes(self):
        self._update_style_from_ui()
        self.style_updated.emit(self.overlay_style)
    def accept(self):
        self._update_style_from_ui()
        self.style_updated.emit(self.overlay_style)
        super().accept()
    def _update_style_from_ui(self):
        # Handle both 'line', 'hline', and 'vline' types with line controls
        if self.overlay_type == 'line' or self.overlay_type == 'hline' or self.overlay_type == 'vline':
            self.overlay_style['color'] = self.color_btn.get_color()
            self.overlay_style['linewidth'] = self.width_spin.value()
            # Get the actual linestyle value from the combo box data
            self.overlay_style['linestyle'] = self.style_combo.currentData()
            self.overlay_style['alpha'] = self.alpha_spin.value()
        elif self.overlay_type == 'text':
            current_text = self.text_edit.toPlainText()
            
            # Check if text has been modified from original
            if self.initial_text is not None and current_text != self.initial_text:
                self.overlay_style['text_content'] = current_text  # Special field for custom text
                self.is_custom_text = True
                print(f"[OverlayWizard] Detected custom text change")
            elif self.initial_text is not None and current_text == self.initial_text:
                # Text was reset back to original
                if 'text_content' in self.overlay_style:
                    del self.overlay_style['text_content']
                self.is_custom_text = False
                print(f"[OverlayWizard] Text reset to original")
            
            # Always update the regular text field for style consistency
            self.overlay_style['text'] = current_text
            
            # Update other text style properties
            self.overlay_style['fontsize'] = self.fontsize_spin.value()
            self.overlay_style['color'] = self.color_btn.get_color()
            self.overlay_style['alpha'] = self.alpha_spin.value()
            pos_type = self.position_combo.currentText()
            if pos_type == 'custom':
                self.overlay_style['position'] = (self.x_input.value(), self.y_input.value())
            else:
                self.overlay_style['position'] = pos_type
            boxstyle = self.boxstyle_combo.currentText()
            if boxstyle == 'none':
                self.overlay_style['bbox'] = {}
            else:
                self.overlay_style['bbox'] = {
                    'boxstyle': 'round,pad=0.5' if boxstyle == 'rounded' else 'square,pad=0.5',
                    'facecolor': self.boxcolor_btn.get_color(),
                    'alpha': self.boxalpha_spin.value()
                }
        elif self.overlay_type == 'fill':
            self.overlay_style['color'] = self.color_btn.get_color()
            self.overlay_style['alpha'] = self.alpha_spin.value()
            self.overlay_style['edgecolor'] = self.edgecolor_btn.get_color()
            self.overlay_style['linewidth'] = self.edgewidth_spin.value() 