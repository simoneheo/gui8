from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QGroupBox, QGridLayout, QMessageBox,
    QSplitter, QScrollArea, QFrame, QTabWidget, QWidget,
    QCheckBox, QSpinBox, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QTextCharFormat, QColor, QSyntaxHighlighter
import numpy as np
import re
from typing import Optional, Dict, Any
import traceback
from datetime import datetime

from channel import Channel


class ExpressionHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for mathematical expressions"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Define highlighting rules
        self.highlighting_rules = []
        
        # Variables (x, y)
        variable_format = QTextCharFormat()
        variable_format.setForeground(QColor(0, 100, 200))
        variable_format.setFontWeight(QFont.Bold)
        self.highlighting_rules.append((r'\b[xy]\b', variable_format))
        
        # Numbers
        number_format = QTextCharFormat()
        number_format.setForeground(QColor(200, 0, 0))
        self.highlighting_rules.append((r'\b\d+\.?\d*\b', number_format))
        
        # Functions
        function_format = QTextCharFormat()
        function_format.setForeground(QColor(150, 0, 150))
        function_format.setFontWeight(QFont.Bold)
        functions = [
            'abs', 'round', 'min', 'max', 'sum', 'mean', 'std',
            'sqrt', 'exp', 'log', 'log10', 'log2',
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
            'sinh', 'cosh', 'tanh', 'arcsinh', 'arccosh', 'arctanh',
            'floor', 'ceil', 'sign', 'power', 'square'
        ]
        for func in functions:
            pattern = r'\b' + func + r'\b'
            self.highlighting_rules.append((pattern, function_format))
        
        # Constants
        constant_format = QTextCharFormat()
        constant_format.setForeground(QColor(0, 150, 0))
        constant_format.setFontWeight(QFont.Bold)
        constants = ['pi', 'e', 'inf', 'nan']
        for const in constants:
            pattern = r'\b' + const + r'\b'
            self.highlighting_rules.append((pattern, constant_format))
        
        # Operators
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor(100, 100, 100))
        operator_format.setFontWeight(QFont.Bold)
        operators = [r'\+', r'-', r'\*', r'\/', r'\^', r'\*\*', r'=']
        for op in operators:
            self.highlighting_rules.append((op, operator_format))
    
    def highlightBlock(self, text):
        for pattern, format in self.highlighting_rules:
            expression = re.compile(pattern)
            for match in expression.finditer(text):
                start, end = match.span()
                self.setFormat(start, end - start, format)


class TransformWizard(QDialog):
    """
    Mathematical transformation wizard for channel X,Y data
    """
    
    data_updated = Signal(str)  # Emitted when channel data is updated (channel_id)
    
    # Safe math namespace for expression evaluation
    MATH_NAMESPACE = {
        # Basic functions
        'abs': np.abs,
        'round': np.round,
        'min': np.min,
        'max': np.max,
        'sum': np.sum,
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'median': np.median,
        
        # Math functions
        'sqrt': np.sqrt,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'log2': np.log2,
        'power': np.power,
        'square': np.square,
        
        # Trigonometric functions
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'arcsin': np.arcsin,
        'arccos': np.arccos,
        'arctan': np.arctan,
        'arctan2': np.arctan2,
        
        # Hyperbolic functions
        'sinh': np.sinh,
        'cosh': np.cosh,
        'tanh': np.tanh,
        'arcsinh': np.arcsinh,
        'arccosh': np.arccosh,
        'arctanh': np.arctanh,
        
        # Rounding functions
        'floor': np.floor,
        'ceil': np.ceil,
        'sign': np.sign,
        'clip': np.clip,
        
        # Constants
        'pi': np.pi,
        'e': np.e,
        'inf': np.inf,
        'nan': np.nan,
        
        # Array functions
        'where': np.where,
        'select': np.select,
        'piecewise': np.piecewise,
        
        # Statistical functions
        'percentile': np.percentile,
        'quantile': np.quantile,
        
        # Utility functions
        'len': len,
        'range': range,
        'enumerate': enumerate,
    }
    
    def __init__(self, channel: Channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.original_xdata = channel.xdata.copy() if channel.xdata is not None else None
        self.original_ydata = channel.ydata.copy() if channel.ydata is not None else None
        self.preview_xdata = None
        self.preview_ydata = None
        self.has_unsaved_changes = False
        
        self.setWindowTitle(f"Data Transform - {channel.ylabel or 'Unnamed Channel'}")
        self.setModal(True)
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        
        self.init_ui()
        self.update_data_summary()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel(f"🔨 Data Transform Wizard")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("margin: 10px; color: #2c3e50;")
        layout.addWidget(title)
        
        # Channel info
        info_label = QLabel(f"Channel: {self.channel.ylabel or 'Unnamed'} | "
                           f"Data Points: {len(self.channel.ydata) if self.channel.ydata is not None else 0}")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666; font-size: 11px; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Tab 1: Transform
        self.transform_tab = self.create_transform_tab()
        self.tab_widget.addTab(self.transform_tab, "🔨 Transform")
        
        # Tab 2: Functions Reference
        self.reference_tab = self.create_reference_tab()
        self.tab_widget.addTab(self.reference_tab, "📚 Functions")
        
        # Tab 3: Examples
        self.examples_tab = self.create_examples_tab()
        self.tab_widget.addTab(self.examples_tab, "💡 Examples")
        
        layout.addWidget(self.tab_widget)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        # Left side buttons
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setToolTip("Reset to original data")
        button_layout.addWidget(self.reset_button)
        
        self.preview_button = QPushButton("👁️ Preview")
        self.preview_button.setToolTip("Preview transformations without applying")
        button_layout.addWidget(self.preview_button)
        
        button_layout.addStretch()
        
        # Right side buttons
        self.apply_button = QPushButton("✅ Apply Transform")
        self.apply_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        button_layout.addWidget(self.apply_button)
        
        self.cancel_button = QPushButton("❌ Cancel")
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def create_transform_tab(self) -> QWidget:
        """Create the main transform tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Data summary and transformations
        left_panel = self.create_transform_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Preview and results
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 500])
        layout.addWidget(splitter)
        
        return widget
    
    def create_transform_panel(self) -> QWidget:
        """Create the transformation input panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Data summary group
        summary_group = QGroupBox("Current Data Summary")
        summary_layout = QGridLayout(summary_group)
        
        self.summary_labels = {}
        summary_info = [
            ("Data Points", "points"),
            ("X Range", "x_range"),
            ("Y Range", "y_range"),
            ("X Mean", "x_mean"),
            ("Y Mean", "y_mean")
        ]
        
        for i, (label_text, key) in enumerate(summary_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            value_label.setStyleSheet("color: #666;")
            
            summary_layout.addWidget(label, i, 0)
            summary_layout.addWidget(value_label, i, 1)
            self.summary_labels[key] = value_label
        
        layout.addWidget(summary_group)
        
        # Transformation group
        transform_group = QGroupBox("Mathematical Transformations")
        transform_layout = QVBoxLayout(transform_group)
        
        # X transformation
        x_label = QLabel("X Transformation:")
        x_label.setStyleSheet("font-weight: bold; color: #2980b9;")
        transform_layout.addWidget(x_label)
        
        self.x_transform_input = QTextEdit()
        self.x_transform_input.setPlaceholderText("e.g., x = x / 1000  (leave empty for no change)")
        self.x_transform_input.setStyleSheet("font-family: 'Courier New', monospace; padding: 5px;")
        self.x_transform_input.setMaximumHeight(60)  # Keep it compact
        
        # Add syntax highlighter
        self.x_highlighter = ExpressionHighlighter(self.x_transform_input.document())
        
        transform_layout.addWidget(self.x_transform_input)
        
        # Y transformation
        y_label = QLabel("Y Transformation:")
        y_label.setStyleSheet("font-weight: bold; color: #e74c3c;")
        transform_layout.addWidget(y_label)
        
        self.y_transform_input = QTextEdit()
        self.y_transform_input.setPlaceholderText("e.g., y = abs(y)  (leave empty for no change)")
        self.y_transform_input.setStyleSheet("font-family: 'Courier New', monospace; padding: 5px;")
        self.y_transform_input.setMaximumHeight(60)  # Keep it compact
        
        # Add syntax highlighter
        self.y_highlighter = ExpressionHighlighter(self.y_transform_input.document())
        
        transform_layout.addWidget(self.y_transform_input)
        
        # Quick transform buttons
        quick_layout = QHBoxLayout()
        
        quick_x_label = QLabel("Quick X:")
        quick_x_label.setStyleSheet("font-size: 10px; color: #666;")
        quick_layout.addWidget(quick_x_label)
        
        x_buttons = [
            ("÷1000", "x = x / 1000"),
            ("×1000", "x = x * 1000"),
            ("+Offset", "x = x + "),
            ("Log", "x = log(x)")
        ]
        
        for text, transform in x_buttons:
            btn = QPushButton(text)
            btn.setMaximumWidth(60)
            btn.setStyleSheet("font-size: 10px; padding: 2px;")
            btn.clicked.connect(lambda checked=False, t=transform: self.x_transform_input.setPlainText(t))
            quick_layout.addWidget(btn)
        
        transform_layout.addLayout(quick_layout)
        
        quick_layout2 = QHBoxLayout()
        
        quick_y_label = QLabel("Quick Y:")
        quick_y_label.setStyleSheet("font-size: 10px; color: #666;")
        quick_layout2.addWidget(quick_y_label)
        
        y_buttons = [
            ("Abs", "y = abs(y)"),
            ("1/Y", "y = 1 / y"),
            ("Y²", "y = y ** 2"),
            ("√Y", "y = sqrt(abs(y))")
        ]
        
        for text, transform in y_buttons:
            btn = QPushButton(text)
            btn.setMaximumWidth(60)
            btn.setStyleSheet("font-size: 10px; padding: 2px;")
            btn.clicked.connect(lambda checked=False, t=transform: self.y_transform_input.setPlainText(t))
            quick_layout2.addWidget(btn)
        
        transform_layout.addLayout(quick_layout2)
        
        layout.addWidget(transform_group)
        
        # Options group
        options_group = QGroupBox("Transform Options")
        options_layout = QVBoxLayout(options_group)
        
        self.safe_mode_checkbox = QCheckBox("Safe mode (handle NaN/Inf values)")
        self.safe_mode_checkbox.setChecked(True)
        self.safe_mode_checkbox.setToolTip("Automatically handle NaN and infinite values")
        options_layout.addWidget(self.safe_mode_checkbox)
        
        self.preserve_shape_checkbox = QCheckBox("Preserve data shape")
        self.preserve_shape_checkbox.setChecked(True)
        self.preserve_shape_checkbox.setToolTip("Ensure output has same number of points as input")
        options_layout.addWidget(self.preserve_shape_checkbox)
        
        layout.addWidget(options_group)
        
        return widget
    
    def create_preview_panel(self) -> QWidget:
        """Create the preview and results panel"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Preview group
        preview_group = QGroupBox("Transformation Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.preview_text = QTextEdit()
        self.preview_text.setMaximumHeight(150)
        self.preview_text.setReadOnly(True)
        self.preview_text.setStyleSheet("font-family: 'Courier New', monospace; font-size: 10px;")
        preview_layout.addWidget(self.preview_text)
        
        layout.addWidget(preview_group)
        
        # Results group
        results_group = QGroupBox("Transform Results")
        results_layout = QGridLayout(results_group)
        
        self.result_labels = {}
        result_info = [
            ("Status", "status"),
            ("New X Range", "new_x_range"),
            ("New Y Range", "new_y_range"),
            ("X Change", "x_change"),
            ("Y Change", "y_change"),
            ("Data Points", "new_points"),
            ("Warnings", "warnings")
        ]
        
        for i, (label_text, key) in enumerate(result_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Not previewed")
            value_label.setStyleSheet("color: #666;")
            value_label.setWordWrap(True)
            
            results_layout.addWidget(label, i, 0)
            results_layout.addWidget(value_label, i, 1)
            self.result_labels[key] = value_label
        
        layout.addWidget(results_group)
        
        return widget
    
    def create_reference_tab(self) -> QWidget:
        """Create the functions reference tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Available functions
        functions_data = [
            ("Basic Math", [
                ("abs(x)", "Absolute value"),
                ("round(x)", "Round to nearest integer"),
                ("min(x)", "Minimum value"),
                ("max(x)", "Maximum value"),
                ("sum(x)", "Sum of all values"),
                ("mean(x)", "Average value"),
                ("std(x)", "Standard deviation"),
                ("median(x)", "Median value")
            ]),
            ("Advanced Math", [
                ("sqrt(x)", "Square root"),
                ("exp(x)", "Exponential (e^x)"),
                ("log(x)", "Natural logarithm"),
                ("log10(x)", "Base-10 logarithm"),
                ("log2(x)", "Base-2 logarithm"),
                ("power(x, n)", "x raised to power n"),
                ("square(x)", "x squared")
            ]),
            ("Trigonometric", [
                ("sin(x), cos(x), tan(x)", "Trigonometric functions"),
                ("arcsin(x), arccos(x), arctan(x)", "Inverse trigonometric"),
                ("sinh(x), cosh(x), tanh(x)", "Hyperbolic functions")
            ]),
            ("Rounding & Clipping", [
                ("floor(x)", "Round down to integer"),
                ("ceil(x)", "Round up to integer"),
                ("sign(x)", "Sign of x (-1, 0, or 1)"),
                ("clip(x, min, max)", "Clip values to range")
            ]),
            ("Constants", [
                ("pi", "π ≈ 3.14159"),
                ("e", "Euler's number ≈ 2.71828"),
                ("inf", "Infinity"),
                ("nan", "Not a Number")
            ])
        ]
        
        for category, functions in functions_data:
            group = QGroupBox(category)
            group_layout = QVBoxLayout(group)
            
            for func_name, description in functions:
                func_label = QLabel(f"<b>{func_name}</b> - {description}")
                func_label.setStyleSheet("margin: 2px; padding: 2px;")
                group_layout.addWidget(func_label)
            
            scroll_layout.addWidget(group)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
    
    def create_examples_tab(self) -> QWidget:
        """Create the examples tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        examples_data = [
            ("Unit Conversions", [
                ("x = x / 1000", "Convert X from units to thousands"),
                ("x = x * 3600", "Convert X from hours to seconds"),
                ("y = y * 1e6", "Convert Y to microunits"),
                ("y = y / 9.81", "Convert acceleration to g-force")
            ]),
            ("Mathematical Operations", [
                ("y = abs(y)", "Take absolute value of Y"),
                ("y = sqrt(abs(y))", "Square root (handling negatives)"),
                ("y = log10(abs(y) + 1)", "Log transform (avoid log(0))"),
                ("x = x - min(x)", "Shift X to start at zero"),
                ("y = (y - mean(y)) / std(y)", "Z-score normalization")
            ]),
            ("Signal Processing", [
                ("y = y - mean(y)", "Remove DC offset"),
                ("y = y / max(abs(y))", "Normalize to [-1, 1]"),
                ("x = (x - min(x)) / (max(x) - min(x))", "Normalize X to [0, 1]"),
                ("y = clip(y, -10, 10)", "Clip Y values to range")
            ]),
            ("Conditional Transforms", [
                ("y = where(y > 0, y, 0)", "Set negative values to zero"),
                ("y = where(abs(y) < 0.1, 0, y)", "Set small values to zero"),
                ("x = where(x < 0, nan, x)", "Replace negative X with NaN")
            ])
        ]
        
        for category, examples in examples_data:
            group = QGroupBox(category)
            group_layout = QVBoxLayout(group)
            
            for expression, description in examples:
                example_widget = QWidget()
                example_layout = QHBoxLayout(example_widget)
                example_layout.setContentsMargins(5, 2, 5, 2)
                
                expr_label = QLabel(f"<code>{expression}</code>")
                expr_label.setStyleSheet("font-family: 'Courier New', monospace; background-color: #f0f0f0; padding: 2px; border-radius: 3px;")
                
                desc_label = QLabel(f"→ {description}")
                desc_label.setStyleSheet("color: #666; margin-left: 10px;")
                
                example_layout.addWidget(expr_label)
                example_layout.addWidget(desc_label)
                example_layout.addStretch()
                
                group_layout.addWidget(example_widget)
            
            scroll_layout.addWidget(group)
        
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return widget
    
    def connect_signals(self):
        """Connect all signals"""
        self.preview_button.clicked.connect(self.preview_transform)
        self.apply_button.clicked.connect(self.apply_transform)
        self.reset_button.clicked.connect(self.reset_transform)
        self.cancel_button.clicked.connect(self.reject)
        
        # Auto-preview on text change (with delay)
        self.x_transform_input.textChanged.connect(self.on_transform_changed)
        self.y_transform_input.textChanged.connect(self.on_transform_changed)
    
    def on_transform_changed(self):
        """Handle when transform expressions change"""
        # Clear previous results
        self.result_labels["status"].setText("Expression changed - click Preview")
        self.result_labels["status"].setStyleSheet("color: #f39c12;")
    
    def update_data_summary(self):
        """Update the data summary display"""
        if self.original_xdata is None or self.original_ydata is None:
            return
        
        self.summary_labels["points"].setText(f"{len(self.original_ydata):,}")
        
        x_min, x_max = np.min(self.original_xdata), np.max(self.original_xdata)
        y_min, y_max = np.min(self.original_ydata), np.max(self.original_ydata)
        
        self.summary_labels["x_range"].setText(f"{x_min:.6g} to {x_max:.6g}")
        self.summary_labels["y_range"].setText(f"{y_min:.6g} to {y_max:.6g}")
        self.summary_labels["x_mean"].setText(f"{np.mean(self.original_xdata):.6g}")
        self.summary_labels["y_mean"].setText(f"{np.mean(self.original_ydata):.6g}")
    
    def update_applied_preview(self):
        """Update the preview to show that a transform was successfully applied"""
        preview_lines = [
            "✅ TRANSFORM APPLIED SUCCESSFULLY",
            "",
            f"Channel: {self.channel.ylabel or 'Unnamed'}",
            f"Data points: {len(self.channel.ydata):,}",
            "",
            f"Current X range: {np.min(self.channel.xdata):.6g} to {np.max(self.channel.xdata):.6g}",
            f"Current Y range: {np.min(self.channel.ydata):.6g} to {np.max(self.channel.ydata):.6g}",
            "",
            "Ready for next transformation...",
            "The main plot has been updated with the new data."
        ]
        
        self.preview_text.setPlainText("\n".join(preview_lines))
    
    def preview_transform(self):
        """Preview the transformation without applying it"""
        try:
            x_expr = self.x_transform_input.toPlainText().strip()
            y_expr = self.y_transform_input.toPlainText().strip()
            
            # Start with original data
            x = self.original_xdata.copy()
            y = self.original_ydata.copy()
            
            warnings = []
            
            # Apply X transformation
            if x_expr:
                x = self._safe_eval_transform(x_expr, x, y, 'x')
                if x is None:
                    self.result_labels["status"].setText("❌ X transformation failed")
                    self.result_labels["status"].setStyleSheet("color: #e74c3c;")
                    return
            
            # Apply Y transformation
            if y_expr:
                y = self._safe_eval_transform(y_expr, x, y, 'y')
                if y is None:
                    self.result_labels["status"].setText("❌ Y transformation failed")
                    self.result_labels["status"].setStyleSheet("color: #e74c3c;")
                    return
            
            # Handle safety checks
            if self.safe_mode_checkbox.isChecked():
                x_nan_count = np.sum(np.isnan(x))
                y_nan_count = np.sum(np.isnan(y))
                x_inf_count = np.sum(np.isinf(x))
                y_inf_count = np.sum(np.isinf(y))
                
                if x_nan_count > 0 or y_nan_count > 0:
                    warnings.append(f"Created {x_nan_count + y_nan_count} NaN values")
                if x_inf_count > 0 or y_inf_count > 0:
                    warnings.append(f"Created {x_inf_count + y_inf_count} infinite values")
            
            # Store preview data
            self.preview_xdata = x
            self.preview_ydata = y
            
            # Update results
            self.result_labels["status"].setText("✅ Preview successful")
            self.result_labels["status"].setStyleSheet("color: #27ae60;")
            
            if len(x) > 0 and len(y) > 0:
                self.result_labels["new_x_range"].setText(f"{np.min(x):.6g} to {np.max(x):.6g}")
                self.result_labels["new_y_range"].setText(f"{np.min(y):.6g} to {np.max(y):.6g}")
                
                # Calculate changes
                orig_x_range = np.max(self.original_xdata) - np.min(self.original_xdata)
                new_x_range = np.max(x) - np.min(x)
                orig_y_range = np.max(self.original_ydata) - np.min(self.original_ydata)
                new_y_range = np.max(y) - np.min(y)
                
                x_change = ((new_x_range / orig_x_range) - 1) * 100 if orig_x_range != 0 else 0
                y_change = ((new_y_range / orig_y_range) - 1) * 100 if orig_y_range != 0 else 0
                
                self.result_labels["x_change"].setText(f"{x_change:+.1f}% range change")
                self.result_labels["y_change"].setText(f"{y_change:+.1f}% range change")
            
            self.result_labels["new_points"].setText(f"{len(y):,}")
            self.result_labels["warnings"].setText("; ".join(warnings) if warnings else "None")
            
            # Update preview text
            preview_lines = []
            if x_expr:
                preview_lines.append(f"X Transform: {x_expr}")
            if y_expr:
                preview_lines.append(f"Y Transform: {y_expr}")
            
            preview_lines.append(f"\nFirst 5 original X values: {self.original_xdata[:5]}")
            preview_lines.append(f"First 5 transformed X values: {x[:5]}")
            preview_lines.append(f"\nFirst 5 original Y values: {self.original_ydata[:5]}")
            preview_lines.append(f"First 5 transformed Y values: {y[:5]}")
            
            self.preview_text.setPlainText("\n".join(preview_lines))
            
            self.has_unsaved_changes = True
            
        except Exception as e:
            self.result_labels["status"].setText(f"❌ Error: {str(e)}")
            self.result_labels["status"].setStyleSheet("color: #e74c3c;")
            self.preview_text.setPlainText(f"Error during preview:\n{traceback.format_exc()}")
    
    def _safe_eval_transform(self, expression: str, x: np.ndarray, y: np.ndarray, target_var: str) -> Optional[np.ndarray]:
        """Safely evaluate a transformation expression"""
        try:
            # Create safe namespace
            namespace = self.MATH_NAMESPACE.copy()
            namespace.update({
                'x': x,
                'y': y,
                '__builtins__': {}  # Remove built-in functions for security
            })
            
            # Check for dangerous patterns
            dangerous_patterns = ['import', 'exec', 'eval', 'open', 'file', '__']
            for pattern in dangerous_patterns:
                if pattern in expression.lower():
                    raise ValueError(f"Dangerous pattern '{pattern}' not allowed")
            
            # Execute the expression
            exec(expression, namespace)
            
            # Return the target variable
            result = namespace.get(target_var)
            if result is None:
                raise ValueError(f"Expression must assign to variable '{target_var}'")
            
            # Ensure result is numpy array
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            # Check shape preservation if required
            if self.preserve_shape_checkbox.isChecked():
                if len(result) != len(self.original_ydata):
                    raise ValueError(f"Transform changed data length from {len(self.original_ydata)} to {len(result)}")
            
            return result
            
        except Exception as e:
            QMessageBox.critical(self, "Transform Error", f"Error in {target_var} expression:\n{str(e)}")
            return None
    
    def apply_transform(self):
        """Apply the transformation to the channel"""
        if self.preview_xdata is None or self.preview_ydata is None:
            # Need to preview first
            self.preview_transform()
            if self.preview_xdata is None or self.preview_ydata is None:
                return
        
        try:
            # Apply the transformation
            self.channel.xdata = self.preview_xdata.copy()
            self.channel.ydata = self.preview_ydata.copy()
            self.channel.modified_at = datetime.now()
            
            # Update the original data reference for future transforms
            self.original_xdata = self.channel.xdata.copy()
            self.original_ydata = self.channel.ydata.copy()
            
            # Emit signal that data was updated (this will update the main window plot)
            self.data_updated.emit(self.channel.channel_id)
            
            # Update the data summary with new values
            self.update_data_summary()
            
            # Get the expressions before clearing them
            x_expr = self.x_transform_input.toPlainText().strip()
            y_expr = self.y_transform_input.toPlainText().strip()
            
            # Update preview to show the applied transformation
            self.update_applied_preview()
            
            # Clear the transformation inputs since they've been applied
            self.x_transform_input.clear()
            self.y_transform_input.clear()
            self.preview_xdata = None
            self.preview_ydata = None
            self.has_unsaved_changes = False
            
            # Update apply button state
            self.apply_button.setEnabled(False)
            
            # Clear results
            for key in self.result_labels:
                self.result_labels[key].setText("Transform applied - ready for next")
                self.result_labels[key].setStyleSheet("color: #27ae60;")
            
            # Show success message without closing the dialog
            self.result_labels["status"].setText("✅ Transform applied successfully!")
            self.result_labels["status"].setStyleSheet("color: #27ae60; font-weight: bold;")
            
            # Update window title to remove asterisk
            self.setWindowTitle(f"Data Transform - {self.channel.ylabel or 'Unnamed Channel'}")
            
            # Don't close the dialog - user can apply more transforms
            
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Applying Transform", 
                f"An error occurred while applying the transform:\n{str(e)}"
            )
    
    def reset_transform(self):
        """Reset to original data"""
        self.x_transform_input.clear()
        self.y_transform_input.clear()
        self.preview_xdata = None
        self.preview_ydata = None
        self.has_unsaved_changes = False
        
        # Clear results
        for key in self.result_labels:
            self.result_labels[key].setText("Not previewed")
            self.result_labels[key].setStyleSheet("color: #666;")
        
        self.preview_text.clear()
        
        QMessageBox.information(self, "Reset", "Transform expressions cleared.")
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved transform previews. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Convenience function for opening the transform wizard
def transform_channel_data(channel: Channel, parent=None):
    """
    Open the data transformation wizard for a channel
    
    Args:
        channel: Channel to transform
        parent: Parent widget
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    wizard = TransformWizard(channel, parent)
    return wizard.exec() 