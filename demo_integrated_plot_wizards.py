#!/usr/bin/env python3
"""
Demonstration of Integrated Plot Wizards

This script demonstrates how the refactored plot wizards work with integrated
configuration wizards (line, marker, spectrogram) for seamless user experience.
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QComboBox, QTextEdit
from PySide6.QtCore import Qt

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_wizard_integration import create_integrated_plot_wizard, integrate_config_wizards_with_plot_wizard
from base_plot_wizard import BasePlotWizard


class PlotWizardDemo(QMainWindow):
    """
    Demonstration application for integrated plot wizards
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated Plot Wizards Demo")
        self.setMinimumSize(1000, 700)
        
        # Mock managers (in real app these would be actual instances)
        self.file_manager = MockFileManager()
        self.channel_manager = MockChannelManager()
        self.signal_bus = None
        
        # Current plot wizard
        self.current_wizard = None
        
        self._setup_ui()
        self._log("Demo application initialized")
    
    def _setup_ui(self):
        """Setup the demo UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title_label = QLabel("Integrated Plot Wizards Demonstration")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Wizard selection
        selection_layout = QHBoxLayout()
        selection_layout.addWidget(QLabel("Select Wizard Type:"))
        
        self.wizard_combo = QComboBox()
        self.wizard_combo.addItems(["Process Wizard", "Mixer Wizard", "Comparison Wizard", "Plot Wizard"])
        selection_layout.addWidget(self.wizard_combo)
        
        self.create_wizard_btn = QPushButton("Create Wizard")
        self.create_wizard_btn.clicked.connect(self._create_wizard)
        selection_layout.addWidget(self.create_wizard_btn)
        
        selection_layout.addStretch()
        layout.addLayout(selection_layout)
        
        # Demo controls
        controls_layout = QHBoxLayout()
        
        self.demo_line_btn = QPushButton("Demo Line Configuration")
        self.demo_line_btn.clicked.connect(self._demo_line_config)
        self.demo_line_btn.setEnabled(False)
        controls_layout.addWidget(self.demo_line_btn)
        
        self.demo_marker_btn = QPushButton("Demo Marker Configuration")
        self.demo_marker_btn.clicked.connect(self._demo_marker_config)
        self.demo_marker_btn.setEnabled(False)
        controls_layout.addWidget(self.demo_marker_btn)
        
        self.demo_spectrogram_btn = QPushButton("Demo Spectrogram Configuration")
        self.demo_spectrogram_btn.clicked.connect(self._demo_spectrogram_config)
        self.demo_spectrogram_btn.setEnabled(False)
        controls_layout.addWidget(self.demo_spectrogram_btn)
        
        controls_layout.addStretch()
        layout.addLayout(controls_layout)
        
        # Log output
        layout.addWidget(QLabel("Demo Log:"))
        self.log_output = QTextEdit()
        self.log_output.setMaximumHeight(200)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        
        # Wizard container
        self.wizard_container = QWidget()
        self.wizard_container.setMinimumHeight(400)
        layout.addWidget(self.wizard_container)
    
    def _log(self, message: str):
        """Log a message to the demo output"""
        self.log_output.append(f"[Demo] {message}")
    
    def _create_wizard(self):
        """Create the selected wizard type"""
        try:
            wizard_type_map = {
                "Process Wizard": "process",
                "Mixer Wizard": "mixer", 
                "Comparison Wizard": "comparison",
                "Plot Wizard": "plot"
            }
            
            selected_type = self.wizard_combo.currentText()
            wizard_type = wizard_type_map[selected_type]
            
            self._log(f"Creating {selected_type}...")
            
            # Create integrated wizard
            self.current_wizard = create_integrated_plot_wizard(
                wizard_type, 
                self.file_manager, 
                self.channel_manager, 
                self.signal_bus, 
                self
            )
            
            # Add to container
            if self.wizard_container.layout():
                # Clear existing layout
                for i in reversed(range(self.wizard_container.layout().count())):
                    self.wizard_container.layout().itemAt(i).widget().setParent(None)
            else:
                layout = QVBoxLayout(self.wizard_container)
                self.wizard_container.setLayout(layout)
            
            self.wizard_container.layout().addWidget(self.current_wizard)
            
            # Enable demo buttons
            self.demo_line_btn.setEnabled(True)
            self.demo_marker_btn.setEnabled(True)
            self.demo_spectrogram_btn.setEnabled(True)
            
            self._log(f"{selected_type} created successfully with integrated configuration wizards")
            self._log("Click demo buttons to test configuration wizard integration")
            
        except Exception as e:
            self._log(f"Error creating wizard: {str(e)}")
            import traceback
            self._log(f"Traceback: {traceback.format_exc()}")
    
    def _demo_line_config(self):
        """Demonstrate line configuration integration"""
        if not self.current_wizard:
            self._log("No wizard created yet")
            return
        
        try:
            # Create mock channel
            mock_channel = MockChannel("demo_line", "Demo Line Channel")
            
            # Add to wizard
            self.current_wizard.add_channel(mock_channel, visible=True)
            
            # Open line configuration wizard
            if hasattr(self.current_wizard, '_config_wizard_manager'):
                success = self.current_wizard._config_wizard_manager.open_line_wizard(mock_channel)
                if success:
                    self._log("Line configuration wizard opened successfully")
                else:
                    self._log("Failed to open line configuration wizard")
            else:
                self._log("Configuration wizard manager not available")
                
        except Exception as e:
            self._log(f"Error demonstrating line configuration: {str(e)}")
    
    def _demo_marker_config(self):
        """Demonstrate marker configuration integration"""
        if not self.current_wizard:
            self._log("No wizard created yet")
            return
        
        try:
            # Create mock marker configuration
            mock_marker_config = {
                'name': 'Demo Marker',
                'marker_style': 'o',
                'marker_size': 20,
                'marker_color': '#ff0000',
                'marker_alpha': 0.8,
                'edge_color': '#000000',
                'edge_width': 1.0,
                'x_axis': 'bottom',
                'z_order': 1,
                'channel_id': 'demo_marker'
            }
            
            # Open marker configuration wizard
            if hasattr(self.current_wizard, '_config_wizard_manager'):
                success = self.current_wizard._config_wizard_manager.open_marker_wizard(mock_marker_config)
                if success:
                    self._log("Marker configuration wizard opened successfully")
                else:
                    self._log("Failed to open marker configuration wizard")
            else:
                self._log("Configuration wizard manager not available")
                
        except Exception as e:
            self._log(f"Error demonstrating marker configuration: {str(e)}")
    
    def _demo_spectrogram_config(self):
        """Demonstrate spectrogram configuration integration"""
        if not self.current_wizard:
            self._log("No wizard created yet")
            return
        
        try:
            # Create mock spectrogram channel
            mock_channel = MockChannel("demo_spectrogram", "Demo Spectrogram Channel")
            mock_channel.tags = ["spectrogram"]
            mock_channel.metadata = {"Zxx": [[1, 2], [3, 4]]}  # Mock spectrogram data
            
            # Add to wizard
            self.current_wizard.add_channel(mock_channel, visible=True)
            
            # Open spectrogram configuration wizard
            if hasattr(self.current_wizard, '_config_wizard_manager'):
                success = self.current_wizard._config_wizard_manager.open_spectrogram_wizard(mock_channel)
                if success:
                    self._log("Spectrogram configuration wizard opened successfully")
                else:
                    self._log("Failed to open spectrogram configuration wizard")
            else:
                self._log("Configuration wizard manager not available")
                
        except Exception as e:
            self._log(f"Error demonstrating spectrogram configuration: {str(e)}")


class MockFileManager:
    """Mock file manager for demo purposes"""
    
    def get_all_files(self):
        """Return mock files"""
        return [
            MockFile("demo_file_1.csv", "file_1"),
            MockFile("demo_file_2.csv", "file_2")
        ]


class MockChannelManager:
    """Mock channel manager for demo purposes"""
    
    def get_channels_by_file(self, file_id):
        """Return mock channels for a file"""
        return [
            MockChannel(f"channel_1_{file_id}", f"Channel 1 from {file_id}"),
            MockChannel(f"channel_2_{file_id}", f"Channel 2 from {file_id}")
        ]
    
    def get_channel_count(self):
        """Return mock channel count"""
        return 4


class MockFile:
    """Mock file object"""
    
    def __init__(self, filename, file_id):
        self.filename = filename
        self.file_id = file_id


class MockChannel:
    """Mock channel object"""
    
    def __init__(self, channel_id, legend_label):
        self.channel_id = channel_id
        self.legend_label = legend_label
        self.ylabel = legend_label
        self.xlabel = "Time (s)"
        self.show = True
        self.color = "#1f77b4"
        self.style = "-"
        self.marker = None
        self.alpha = 1.0
        self.z_order = 0
        self.xaxis = "x-bottom"
        self.tags = ["time-series"]
        self.metadata = {}
        
        # Mock data
        import numpy as np
        self.xdata = np.linspace(0, 10, 100)
        self.ydata = np.sin(self.xdata) + np.random.normal(0, 0.1, 100)


def main():
    """Main demo function"""
    app = QApplication(sys.argv)
    
    # Create demo window
    demo = PlotWizardDemo()
    demo.show()
    
    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 