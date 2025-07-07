from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, 
    QCheckBox, QGroupBox, QFormLayout, QListWidget, QListWidgetItem, QMessageBox, 
    QFileDialog, QRadioButton, QButtonGroup, QScrollArea, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import os
import time
from typing import Optional, List

class ExportWizardWindow(QMainWindow):
    """
    Export wizard for exporting channel data to various file formats
    """
    
    export_complete = Signal(dict)
    export_started = Signal(str)  # Emit export file path when started
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers with consistent naming
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state
        self.selected_channels = []
        self.export_manager = None  # Will be set by manager
        self._stats = {
            'channels_selected': 0,
            'last_export_file': None,
            'last_export_time': None
        }
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        self._populate_file_combo()
        
        # Validate initialization
        self._validate_initialization()
        
    def _validate_initialization(self):
        """Validate that required managers are available"""
        if not self.file_manager:
            self._show_error("File manager not available")
            return False
            
        if not self.channel_manager:
            self._show_error("Channel manager not available")
            return False
            
        # Check if files are available
        files = self.file_manager.get_all_files()
        if not files:
            self.channel_info_label.setText("No files available for export")
            
        return True
        
    def _show_error(self, message: str):
        """Show error message to user"""
        print(f"[ExportWizard] ERROR: {message}")
        if hasattr(self, 'channel_info_label'):
            self.channel_info_label.setText(f"Error: {message}")
            
    def get_export_stats(self) -> dict:
        """Get export statistics"""
        return {
            **self._stats,
            'available_files': len(self.file_manager.get_all_files()) if self.file_manager else 0,
            'total_channels': self.channel_manager.get_channel_count() if self.channel_manager else 0
        }
        
    def _init_ui(self):
        self.setWindowTitle("Export Data Wizard")
        self.setMinimumSize(600, 500)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Export Channel Data")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # File selection section
        file_section = self._create_file_selection_section()
        layout.addWidget(file_section)
        
        # Channel selection section
        channel_section = self._create_channel_selection_section()
        layout.addWidget(channel_section)
        
        # Export format section
        format_section = self._create_format_selection_section()
        layout.addWidget(format_section)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.export_button = QPushButton("Export")
        self.export_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        
    def _create_file_selection_section(self):
        """Create file selection section"""
        group = QGroupBox("Select File")
        layout = QFormLayout(group)
        
        self.file_combo = QComboBox()
        self.file_combo.setMinimumWidth(300)
        layout.addRow("File:", self.file_combo)
        
        return group
        
    def _create_channel_selection_section(self):
        """Create channel selection section"""
        group = QGroupBox("Select Channels to Export")
        layout = QVBoxLayout(group)
        
        # Info label
        self.channel_info_label = QLabel("Select a file to see available channels")
        self.channel_info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.channel_info_label)
        

        
        # Scroll area for channel checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(200)
        
        self.channel_widget = QWidget()
        self.channel_layout = QVBoxLayout(self.channel_widget)
        
        scroll_area.setWidget(self.channel_widget)
        layout.addWidget(scroll_area)
        
        # Select all/none buttons
        button_layout = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_none_button = QPushButton("Select None")
        
        button_layout.addWidget(self.select_all_button)
        button_layout.addWidget(self.select_none_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        return group
        
    def _create_format_selection_section(self):
        """Create export format selection section"""
        group = QGroupBox("Export Format")
        layout = QVBoxLayout(group)
        
        # Format radio buttons
        self.format_group = QButtonGroup()
        
        self.csv_radio = QRadioButton("CSV (Comma Separated Values)")
        self.csv_radio.setChecked(True)  # Default selection
        
        self.txt_radio = QRadioButton("TXT (Tab Separated)")
        
        self.format_group.addButton(self.csv_radio)
        self.format_group.addButton(self.txt_radio)
        
        layout.addWidget(self.csv_radio)
        layout.addWidget(self.txt_radio)
        
        return group
        
    def _connect_signals(self):
        """Connect widget signals"""
        self.file_combo.currentTextChanged.connect(self._on_file_changed)
        self.select_all_button.clicked.connect(self._on_select_all)
        self.select_none_button.clicked.connect(self._on_select_none)
        self.export_button.clicked.connect(self._on_export)
        self.cancel_button.clicked.connect(self.close)
        
    def _populate_file_combo(self):
        """Populate the file combo with available files"""
        self.file_combo.clear()
        self.file_combo.addItem("-- Select a file --", None)
        
        if self.file_manager:
            files = self.file_manager.get_all_files()
            for file_info in files:
                self.file_combo.addItem(file_info.filename, file_info.file_id)
                
    def _on_file_changed(self, filename):
        """Handle file selection change"""
        # Clear existing channel checkboxes
        self._clear_channel_checkboxes()
        
        # Get selected file ID
        file_id = self.file_combo.currentData()
        if not file_id:
            self.channel_info_label.setText("Select a file to see available channels")
            self._update_export_button_state()
            return
            
        # Get channels for selected file
        if self.channel_manager:
            channels = self.channel_manager.get_channels_by_file(file_id)
            
            if not channels:
                self.channel_info_label.setText("No channels available for this file")
                self._update_export_button_state()
                return
                
            self.channel_info_label.setText(f"Select channels to export ({len(channels)} available):")
            
            # Create checkboxes for each channel
            for channel in channels:
                self._create_channel_checkbox(channel)
                
        self._update_export_button_state()
        
    def _clear_channel_checkboxes(self):
        """Clear all channel checkboxes"""
        # Remove all widgets from the layout
        while self.channel_layout.count():
            child = self.channel_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
                
    def _create_channel_checkbox(self, channel):
        """Create a checkbox for a channel"""
        # Check if this is a spectrogram channel
        is_spectrogram = "spectrogram" in getattr(channel, 'tags', [])
        
        # Create label with spectrogram indicator
        label_text = channel.legend_label
        if is_spectrogram:
            label_text += " [SPECTROGRAM]"
            
        checkbox = QCheckBox(label_text)
        
        # Create detailed tooltip
        tooltip_parts = [
            f"Channel ID: {channel.channel_id}",
            f"Type: {channel.type.value if channel.type else 'Unknown'}"
        ]
        
        if is_spectrogram:
            tooltip_parts.extend([
                "Format: Spectrogram (2D time-frequency data)",
                "Export: Time, Frequency, and Power columns"
            ])
        else:
            tooltip_parts.extend([
                f"X-axis: {channel.xlabel or 'Unknown'}",
                f"Y-axis: {channel.ylabel or 'Unknown'}",
                "Export: X and Y data columns"
            ])
            
        checkbox.setToolTip("\n".join(tooltip_parts))
        checkbox.channel_info = channel  # Store channel info for later use
        checkbox.stateChanged.connect(self._on_channel_checkbox_changed)
        
        # Style spectrogram checkboxes differently
        if is_spectrogram:
            checkbox.setStyleSheet("""
                QCheckBox {
                    color: #0066cc;
                    font-weight: bold;
                }
                QCheckBox::indicator:checked {
                    background-color: #0066cc;
                }
            """)
        
        self.channel_layout.addWidget(checkbox)
        
    def _on_channel_checkbox_changed(self):
        """Handle channel checkbox state change"""
        self._update_export_button_state()
        
    def _on_select_all(self):
        """Select all channel checkboxes"""
        for i in range(self.channel_layout.count()):
            widget = self.channel_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(True)
                
    def _on_select_none(self):
        """Deselect all channel checkboxes"""
        for i in range(self.channel_layout.count()):
            widget = self.channel_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setChecked(False)
                
    def _update_export_button_state(self):
        """Update the export button enabled state"""
        # Check if at least one channel is selected
        has_selection = False
        for i in range(self.channel_layout.count()):
            widget = self.channel_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                has_selection = True
                break
                
        self.export_button.setEnabled(has_selection)
        
    def _on_export(self):
        """Handle export button click"""
        # Get selected channels
        selected_channels = []
        for i in range(self.channel_layout.count()):
            widget = self.channel_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                selected_channels.append(widget.channel_info)
                
        if not selected_channels:
            QMessageBox.warning(self, "No Selection", "Please select at least one channel to export.")
            return
            
        # Get export format
        if self.csv_radio.isChecked():
            export_format = "csv"
            file_extension = "*.csv"
            filter_text = "CSV Files (*.csv);;All Files (*.*)"
        else:
            export_format = "txt"
            file_extension = "*.txt"
            filter_text = "Text Files (*.txt);;All Files (*.*)"
            
        # Get export file path
        default_filename = f"exported_data.{export_format}"
        if len(selected_channels) == 1:
            # Use channel name if only one channel selected
            channel_name = selected_channels[0].legend_label.replace(" ", "_")
            default_filename = f"{channel_name}.{export_format}"
            
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Data",
            default_filename,
            filter_text
        )
        
        if file_path:
            # Emit export signal with all necessary info
            export_info = {
                'channels': selected_channels,
                'format': export_format,
                'file_path': file_path
            }
            
            self.export_complete.emit(export_info)
            
    def get_selected_channels(self):
        """Get list of selected channels"""
        selected = []
        for i in range(self.channel_layout.count()):
            widget = self.channel_layout.itemAt(i).widget()
            if isinstance(widget, QCheckBox) and widget.isChecked():
                selected.append(widget.channel_info)
        return selected 