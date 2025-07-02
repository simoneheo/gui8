from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QLabel, QPushButton, QHeaderView, QLineEdit, QSpinBox, QCheckBox,
    QGroupBox, QGridLayout, QMessageBox, QProgressBar, QComboBox,
    QDialogButtonBox, QSplitter, QTextEdit, QFrame, QWidget
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QColor, QDoubleValidator, QIntValidator
import numpy as np
from typing import Optional, Tuple
import traceback
from datetime import datetime

from channel import Channel


class DataTableWidget(QTableWidget):
    """
    Custom table widget for displaying and editing X,Y data with enhanced features
    """
    
    data_changed = Signal()  # Emitted when data is modified
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_data = None
        self.modified_rows = set()
        
        # Configure table appearance
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.ExtendedSelection)
        
        # Set up headers
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Index", "X Value", "Y Value"])
        
        # Configure column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)  # Index column
        header.setSectionResizeMode(1, QHeaderView.Stretch)  # X column
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Y column
        
        # Enable sorting
        self.setSortingEnabled(True)
        
        # Connect item changed signal
        self.itemChanged.connect(self._on_item_changed)
    
    def load_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """Load X,Y data into the table"""
        if x_data is None or y_data is None:
            return
        
        # Store original data for comparison
        self.original_data = (x_data.copy(), y_data.copy())
        
        # Clear previous data
        self.clear()
        self.modified_rows.clear()
        
        # Set up table
        data_length = min(len(x_data), len(y_data))
        self.setRowCount(data_length)
        self.setHorizontalHeaderLabels(["Index", "X Value", "Y Value"])
        
        # Populate table
        for i in range(data_length):
            # Index column (read-only)
            index_item = QTableWidgetItem(str(i))
            index_item.setFlags(index_item.flags() & ~Qt.ItemIsEditable)
            index_item.setBackground(QColor(240, 240, 240))
            self.setItem(i, 0, index_item)
            
            # X value (editable)
            x_item = QTableWidgetItem(f"{x_data[i]:.8g}")
            x_item.setToolTip(f"Original: {x_data[i]:.12g}")
            self.setItem(i, 1, x_item)
            
            # Y value (editable)
            y_item = QTableWidgetItem(f"{y_data[i]:.8g}")
            y_item.setToolTip(f"Original: {y_data[i]:.12g}")
            self.setItem(i, 2, y_item)
    
    def _on_item_changed(self, item):
        """Handle when an item in the table is changed"""
        if item.column() in [1, 2]:  # X or Y column
            row = item.row()
            self.modified_rows.add(row)
            
            # Highlight modified row
            for col in range(self.columnCount()):
                cell_item = self.item(row, col)
                if cell_item:
                    cell_item.setBackground(QColor(255, 255, 200))  # Light yellow
            
            # Validate numeric input
            try:
                float(item.text())
                item.setBackground(QColor(255, 255, 200))  # Valid - light yellow
            except ValueError:
                item.setBackground(QColor(255, 200, 200))  # Invalid - light red
                item.setToolTip("Invalid numeric value")
            
            self.data_changed.emit()
    
    def get_modified_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current data from the table"""
        if self.original_data is None:
            return None, None
        
        x_data = []
        y_data = []
        
        for row in range(self.rowCount()):
            try:
                x_item = self.item(row, 1)
                y_item = self.item(row, 2)
                
                if x_item and y_item:
                    x_val = float(x_item.text())
                    y_val = float(y_item.text())
                    x_data.append(x_val)
                    y_data.append(y_val)
                else:
                    # Use original data if item is missing
                    x_data.append(self.original_data[0][row])
                    y_data.append(self.original_data[1][row])
            except (ValueError, IndexError):
                # Use original data if conversion fails
                if row < len(self.original_data[0]):
                    x_data.append(self.original_data[0][row])
                    y_data.append(self.original_data[1][row])
        
        return np.array(x_data), np.array(y_data)
    
    def has_modifications(self) -> bool:
        """Check if any data has been modified"""
        return len(self.modified_rows) > 0
    
    def reset_modifications(self):
        """Reset all modifications and restore original data"""
        if self.original_data is not None:
            self.load_data(self.original_data[0], self.original_data[1])


class InspectionWizard(QDialog):
    """
    Data inspection and editing wizard for channel X,Y data
    """
    
    data_updated = Signal(str)  # Emitted when channel data is updated (channel_id)
    
    def __init__(self, channel: Channel, parent=None):
        super().__init__(parent)
        self.channel = channel
        self.has_unsaved_changes = False
        
        self.setWindowTitle(f"Data Inspector - {channel.ylabel or 'Unnamed Channel'}")
        self.setModal(True)
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)
        
        self.init_ui()
        self.load_channel_data()
        self.connect_signals()
    
    def init_ui(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        
        # Title and info
        header_layout = QHBoxLayout()
        
        title = QLabel(f"🔍 Data Inspector")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Channel info
        info_label = QLabel(f"Channel: {self.channel.ylabel or 'Unnamed'}")
        info_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(info_label)
        
        layout.addLayout(header_layout)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        
        # Top section: Controls and statistics
        controls_widget = self.create_controls_section()
        splitter.addWidget(controls_widget)
        
        # Bottom section: Data table
        table_widget = self.create_table_section()
        splitter.addWidget(table_widget)
        
        # Set splitter proportions (1:3 ratio)
        splitter.setSizes([150, 450])
        layout.addWidget(splitter)
        
        # Dialog buttons
        button_layout = QHBoxLayout()
        
        # Left side buttons
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setToolTip("Reset all changes to original data")
        button_layout.addWidget(self.reset_button)
        
        self.export_button = QPushButton("💾 Export")
        self.export_button.setToolTip("Export current data to file")
        button_layout.addWidget(self.export_button)
        
        button_layout.addStretch()
        
        # Right side buttons
        self.apply_button = QPushButton("✅ Apply Changes")
        self.apply_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.apply_button.setEnabled(False)
        button_layout.addWidget(self.apply_button)
        
        self.cancel_button = QPushButton("❌ Cancel")
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def create_controls_section(self) -> QWidget:
        """Create the controls and statistics section"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        
        # Data statistics group
        stats_group = QGroupBox("Data Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.stats_labels = {}
        stats_info = [
            ("Total Points", "total_points"),
            ("X Range", "x_range"),
            ("Y Range", "y_range"),
            ("Modified Points", "modified_points"),
            ("Invalid Values", "invalid_values")
        ]
        
        for i, (label_text, key) in enumerate(stats_info):
            label = QLabel(f"{label_text}:")
            label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("Computing...")
            value_label.setStyleSheet("color: #666;")
            
            stats_layout.addWidget(label, i, 0)
            stats_layout.addWidget(value_label, i, 1)
            self.stats_labels[key] = value_label
        
        layout.addWidget(stats_group)
        
        # Data operations group
        ops_group = QGroupBox("Data Operations")
        ops_layout = QVBoxLayout(ops_group)
        
        # Row operations
        row_ops_layout = QHBoxLayout()
        
        self.goto_row_input = QSpinBox()
        self.goto_row_input.setMinimum(0)
        self.goto_row_input.setMaximum(0)  # Will be updated when data loads
        self.goto_row_input.setToolTip("Go to specific row")
        row_ops_layout.addWidget(QLabel("Go to row:"))
        row_ops_layout.addWidget(self.goto_row_input)
        
        self.goto_button = QPushButton("Go")
        self.goto_button.clicked.connect(self.goto_row)
        row_ops_layout.addWidget(self.goto_button)
        
        ops_layout.addLayout(row_ops_layout)
        
        # Search operations
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search values...")
        self.search_input.setToolTip("Search for specific values in X or Y data")
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("🔍")
        self.search_button.clicked.connect(self.search_data)
        search_layout.addWidget(self.search_button)
        
        ops_layout.addLayout(search_layout)
        
        # Filter operations
        filter_layout = QHBoxLayout()
        
        self.show_modified_only = QCheckBox("Show modified only")
        self.show_modified_only.toggled.connect(self.filter_table)
        filter_layout.addWidget(self.show_modified_only)
        
        self.show_invalid_only = QCheckBox("Show invalid only")
        self.show_invalid_only.toggled.connect(self.filter_table)
        filter_layout.addWidget(self.show_invalid_only)
        
        ops_layout.addLayout(filter_layout)
        
        layout.addWidget(ops_group)
        
        return widget
    
    def create_table_section(self) -> QWidget:
        """Create the data table section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Table header
        table_header = QLabel("📊 Channel Data (Editable)")
        table_header.setStyleSheet("font-weight: bold; font-size: 12px; margin: 5px;")
        layout.addWidget(table_header)
        
        # Data table
        self.data_table = DataTableWidget()
        layout.addWidget(self.data_table)
        
        # Table status
        self.table_status = QLabel("Ready")
        self.table_status.setStyleSheet("color: #666; font-size: 10px; margin: 2px;")
        layout.addWidget(self.table_status)
        
        return widget
    
    def connect_signals(self):
        """Connect all signals"""
        self.data_table.data_changed.connect(self.on_data_changed)
        self.reset_button.clicked.connect(self.reset_data)
        self.export_button.clicked.connect(self.export_data)
        self.apply_button.clicked.connect(self.apply_changes)
        self.cancel_button.clicked.connect(self.cancel_changes)
        
        # Search as you type
        self.search_input.textChanged.connect(self.search_data)
    
    def load_channel_data(self):
        """Load the channel's X,Y data into the table"""
        if self.channel.xdata is not None and self.channel.ydata is not None:
            self.data_table.load_data(self.channel.xdata, self.channel.ydata)
            
            # Update controls
            self.goto_row_input.setMaximum(len(self.channel.ydata) - 1)
            
            # Update statistics
            self.update_statistics()
            
            self.table_status.setText(f"Loaded {len(self.channel.ydata)} data points")
        else:
            self.table_status.setText("No data available")
            QMessageBox.warning(self, "No Data", "This channel has no data to inspect.")
    
    def update_statistics(self):
        """Update the statistics display"""
        if self.channel.xdata is None or self.channel.ydata is None:
            return
        
        # Basic statistics
        total_points = len(self.channel.ydata)
        self.stats_labels["total_points"].setText(f"{total_points:,}")
        
        # Ranges
        x_min, x_max = np.min(self.channel.xdata), np.max(self.channel.xdata)
        y_min, y_max = np.min(self.channel.ydata), np.max(self.channel.ydata)
        
        self.stats_labels["x_range"].setText(f"{x_min:.6g} to {x_max:.6g}")
        self.stats_labels["y_range"].setText(f"{y_min:.6g} to {y_max:.6g}")
        
        # Modified points
        modified_count = len(self.data_table.modified_rows)
        self.stats_labels["modified_points"].setText(f"{modified_count}")
        
        # Count invalid values
        invalid_count = 0
        for row in range(self.data_table.rowCount()):
            for col in [1, 2]:  # X and Y columns
                item = self.data_table.item(row, col)
                if item:
                    try:
                        float(item.text())
                    except ValueError:
                        invalid_count += 1
        
        self.stats_labels["invalid_values"].setText(f"{invalid_count}")
    
    def on_data_changed(self):
        """Handle when data in the table changes"""
        self.has_unsaved_changes = True
        self.apply_button.setEnabled(True)
        self.update_statistics()
        
        # Update status
        modified_count = len(self.data_table.modified_rows)
        if modified_count > 0:
            self.table_status.setText(f"Modified: {modified_count} rows")
            self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'} *")
        else:
            self.table_status.setText("No modifications")
            self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'}")
    
    def goto_row(self):
        """Go to specific row in the table"""
        row = self.goto_row_input.value()
        if 0 <= row < self.data_table.rowCount():
            self.data_table.selectRow(row)
            self.data_table.scrollToItem(self.data_table.item(row, 0))
    
    def search_data(self):
        """Search for specific values in the data"""
        search_text = self.search_input.text().strip()
        if not search_text:
            # Clear selection if search is empty
            self.data_table.clearSelection()
            return
        
        # Find matching rows
        matching_rows = []
        for row in range(self.data_table.rowCount()):
            for col in [1, 2]:  # X and Y columns
                item = self.data_table.item(row, col)
                if item and search_text.lower() in item.text().lower():
                    matching_rows.append(row)
                    break
        
        # Highlight matching rows
        self.data_table.clearSelection()
        for row in matching_rows:
            self.data_table.selectRow(row)
        
        # Scroll to first match
        if matching_rows:
            self.data_table.scrollToItem(self.data_table.item(matching_rows[0], 0))
            self.table_status.setText(f"Found {len(matching_rows)} matches")
        else:
            self.table_status.setText("No matches found")
    
    def filter_table(self):
        """Filter table based on checkboxes"""
        show_modified = self.show_modified_only.isChecked()
        show_invalid = self.show_invalid_only.isChecked()
        
        if not show_modified and not show_invalid:
            # Show all rows
            for row in range(self.data_table.rowCount()):
                self.data_table.setRowHidden(row, False)
        else:
            # Filter rows
            for row in range(self.data_table.rowCount()):
                should_show = True
                
                if show_modified:
                    should_show = should_show and (row in self.data_table.modified_rows)
                
                if show_invalid:
                    is_invalid = False
                    for col in [1, 2]:
                        item = self.data_table.item(row, col)
                        if item:
                            try:
                                float(item.text())
                            except ValueError:
                                is_invalid = True
                                break
                    should_show = should_show and is_invalid
                
                self.data_table.setRowHidden(row, not should_show)
    
    def reset_data(self):
        """Reset all modifications"""
        if self.data_table.has_modifications():
            reply = QMessageBox.question(
                self, 
                "Reset Data", 
                "Reset all changes to original data?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.data_table.reset_modifications()
                self.has_unsaved_changes = False
                self.apply_button.setEnabled(False)
                self.update_statistics()
                self.table_status.setText("Data reset to original values")
                self.setWindowTitle(f"Data Inspector - {self.channel.ylabel or 'Unnamed Channel'}")
    
    def export_data(self):
        """Export current data to file"""
        # For now, just show a message - could implement actual export
        QMessageBox.information(
            self, 
            "Export Data", 
            "Export functionality would save current data to CSV/TXT file.\n(Feature not yet implemented)"
        )
    
    def apply_changes(self):
        """Apply the changes to the channel"""
        if not self.has_unsaved_changes:
            self.accept()
            return
        
        try:
            # Get modified data
            new_x_data, new_y_data = self.data_table.get_modified_data()
            
            if new_x_data is not None and new_y_data is not None:
                # Update the channel
                self.channel.xdata = new_x_data
                self.channel.ydata = new_y_data
                self.channel.modified_at = datetime.now()
                
                # Emit signal that data was updated
                self.data_updated.emit(self.channel.channel_id)
                
                QMessageBox.information(
                    self, 
                    "Changes Applied", 
                    f"Successfully updated {len(self.data_table.modified_rows)} data points."
                )
                
                self.accept()
            else:
                QMessageBox.warning(self, "Error", "Failed to get modified data.")
        
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error Applying Changes", 
                f"An error occurred while applying changes:\n{str(e)}"
            )
    
    def cancel_changes(self):
        """Cancel and close the dialog"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.reject()
        else:
            self.reject()
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.has_unsaved_changes:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "You have unsaved changes. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# Convenience function for opening the inspection wizard
def inspect_channel_data(channel: Channel, parent=None):
    """
    Open the data inspection wizard for a channel
    
    Args:
        channel: Channel to inspect
        parent: Parent widget
    
    Returns:
        Dialog result (QDialog.Accepted or QDialog.Rejected)
    """
    wizard = InspectionWizard(channel, parent)
    return wizard.exec() 