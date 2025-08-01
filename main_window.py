# main_window.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame, QCheckBox, QSplitter, QTextEdit, QSizePolicy, QMessageBox,
    QFileDialog, QApplication
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPalette, QColor, QFont
from pathlib import Path
import traceback
from datetime import datetime

# Import our manager
from main_window_manager import MainWindowManager
from plot_manager import StylePreviewWidget
from file import FileStatus
from console import get_console, show_console, log_message


class MainWindowUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Dog v1.0")
        self.setMinimumSize(800, 800)
        
        # Position window on left side of screen
        self._position_window_left()
        
        # Enable drag and drop for entire window
        self.setAcceptDrops(True)
        
        # Initialize manager
        self.manager = MainWindowManager(self)
        
        # Track selected file for channel filtering
        self.selected_file_id = None
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Show console by default first
        self.show_floating_console()
        
        # Add a small delay to ensure console is fully initialized
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, self._show_welcome_messages)
    
    def init_ui(self):
        # No main splitter needed - single panel layout

        # Create wizard buttons and styling
        self.load_file_button = QPushButton("Load\nFile")
        self.process_button = QPushButton("Process\nWizard")
        self.mix_button = QPushButton("Mix\nWizard")
        self.compare_button = QPushButton("Compare\nWizard")
        self.plot_button = QPushButton("Plot\nWizard")
        self.export_file_button = QPushButton("Export\nFile")
        
        # Style the wizard buttons for horizontal layout
        wizard_button_style = """
            QPushButton {
                padding: 8px;
                margin: 2px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f8f8;
                color: #333333;
                font-size: 11px;
                text-align: center;
                min-height: 35px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #999;
                color: #000000;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
                color: #000000;
            }
        """

        # Main content panel: Tables and Plot Canvas with vertical splitter
        main_content_panel = QWidget()
        main_content_layout = QVBoxLayout(main_content_panel)
        
        # Create vertical splitter for main content panel
        content_splitter = QSplitter(Qt.Vertical, main_content_panel)
        
        # Top section: Tables (File Manager + Channel Manager)
        tables_widget = QWidget()
        tables_layout = QVBoxLayout(tables_widget)

        # Wizard Buttons (horizontal layout above file manager)
        wizard_layout = QHBoxLayout()
        
        # Create console button
        self.show_console_btn = QPushButton("Info\nConsole")
        
        for button in [self.load_file_button, self.process_button, self.mix_button, self.compare_button, self.plot_button, self.export_file_button, self.show_console_btn]:
            button.setStyleSheet(wizard_button_style)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            wizard_layout.addWidget(button)
        tables_layout.addLayout(wizard_layout)
        
        # File Manager (Enhanced)
        file_manager_layout = QHBoxLayout()
        file_manager_label = QLabel("File Manager")
        self.file_count_label = QLabel("Files: 0")
        self.file_count_label.setStyleSheet("color: #666; font-size: 11px;")
        file_manager_layout.addWidget(file_manager_label)
        file_manager_layout.addStretch()
        file_manager_layout.addWidget(self.file_count_label)
        tables_layout.addLayout(file_manager_layout)
        
        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["File Name", "Size", "Status", "Actions"])
        
        # Set column resize modes for file table
        file_header = self.file_table.horizontalHeader()
        file_header.setSectionResizeMode(0, QHeaderView.Stretch)   # File Name - stretches
        file_header.setSectionResizeMode(1, QHeaderView.Fixed)     # Size - fixed width
        file_header.setSectionResizeMode(2, QHeaderView.Fixed)     # Status - fixed width
        file_header.setSectionResizeMode(3, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set specific column widths for file table
        self.file_table.setColumnWidth(1, 80)   # Size column
        self.file_table.setColumnWidth(2, 100)   # Status column
        self.file_table.setColumnWidth(3, 180)  # Actions column
        
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Enable drag and drop for file table
        self.file_table.setAcceptDrops(True)
        self.file_table.setDragDropMode(QAbstractItemView.DropOnly)

        tables_layout.addWidget(self.file_table)

        # Channel Manager (Enhanced)
        channel_manager_layout = QHBoxLayout()
        channel_manager_label = QLabel("Channel Manager")
        self.channel_count_label = QLabel("Channels: 0 (0 visible)")
        self.channel_count_label.setStyleSheet("color: #666; font-size: 11px;")
        self.selected_file_label = QLabel("No file selected")
        self.selected_file_label.setStyleSheet("color: #888; font-size: 11px; font-style: italic;")
        
        # Add filter toggle
        self.show_all_types_checkbox = QCheckBox("Show all data types")
        self.show_all_types_checkbox.setToolTip("Show all data types (RAW, PROCESSED, MIXED, etc.)\nBy default, only RAW channels are shown\nComparison channels are always excluded from main window")
        self.show_all_types_checkbox.setChecked(False)  # Default to RAW only
        self.show_all_types_checkbox.stateChanged.connect(self._on_channel_filter_changed)
        
        channel_manager_layout.addWidget(channel_manager_label)
        channel_manager_layout.addWidget(self.selected_file_label)
        channel_manager_layout.addStretch()
        channel_manager_layout.addWidget(self.show_all_types_checkbox)
        channel_manager_layout.addWidget(self.channel_count_label)
        tables_layout.addLayout(channel_manager_layout)
        
        self.channel_table = QTableWidget(0, 7)
        self.channel_table.setHorizontalHeaderLabels(["Show", "Style", "Channel Name", "Sampling Rate", "Shape", "Type", "Actions"])
        
        # Set column resize modes for better layout
        header = self.channel_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Channel Name - stretches
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Sampling Rate - fixed width
        header.setSectionResizeMode(4, QHeaderView.Fixed)     # Shape - fixed width
        header.setSectionResizeMode(5, QHeaderView.Fixed)     # Type - fixed width
        header.setSectionResizeMode(6, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set specific column widths
        self.channel_table.setColumnWidth(0, 60)   # Show checkbox
        self.channel_table.setColumnWidth(1, 80)   # Style preview
        self.channel_table.setColumnWidth(3, 100)  # Sampling Rate column
        self.channel_table.setColumnWidth(4, 80)   # Shape column
        self.channel_table.setColumnWidth(5, 100)  # Type column
        self.channel_table.setColumnWidth(6, 180)  # Actions buttons
        self.channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.channel_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        tables_layout.addWidget(self.channel_table)
        
        content_splitter.addWidget(tables_widget)

        # Bottom section: Plot Canvas only
        plot_widget_container = QWidget()
        plot_layout = QVBoxLayout(plot_widget_container)
        
        # Plot Canvas (using PlotManager from manager)
        plot_widget = self.manager.get_plot_manager().get_plot_widget()
        plot_widget.setMinimumHeight(250)
        plot_layout.addWidget(plot_widget)
        
        content_splitter.addWidget(plot_widget_container)
        
        # Set initial splitter sizes: Tables(50%), Plot(50%)
        content_splitter.setSizes([500, 500])
        
        # Add splitter to main layout
        main_content_layout.addWidget(content_splitter)

        # Apply main layout (no more horizontal splitter, just main content panel)
        layout = QVBoxLayout(self)
        layout.addWidget(main_content_panel)
    
    def connect_signals(self):
        """Connect all signal handlers"""
        # Connect manager signals
        self.manager.console_message.connect(self.handle_console_message)
        self.manager.file_selection_changed.connect(self._on_file_selection_changed)
        self.manager.channel_visibility_toggled.connect(self._on_channel_visibility_toggled)
        self.manager.channel_table_updated.connect(self.refresh_channel_table)
        
        # Connect table signals
        self.file_table.cellClicked.connect(self.handle_file_table_click)
        self.file_table.itemSelectionChanged.connect(self.handle_file_selection_changed)
        self.channel_table.cellClicked.connect(self.handle_channel_table_click)
        
        # Connect wizard buttons
        self.load_file_button.clicked.connect(self.show_load_file_dialog)
        self.process_button.clicked.connect(self.show_process_wizard)
        self.mix_button.clicked.connect(self.show_mix_wizard)
        self.compare_button.clicked.connect(self.show_comparison_wizard)
        self.plot_button.clicked.connect(self.show_plot_wizard)
        self.export_file_button.clicked.connect(self.show_export_wizard)
        
        # Connect console button
        self.show_console_btn.clicked.connect(self.show_floating_console)
        
        # Connect plot manager signals
        self.manager.get_plot_manager().legend_item_gear_clicked.connect(self.handle_legend_gear_clicked)
    
    def handle_files_dropped(self, file_paths):
        """Handle files dropped into the drop zone"""
        self.manager.handle_files_dropped(file_paths)
        self.refresh_file_table()
        self.refresh_channel_table()
    
    def refresh_file_table(self):
        """Refresh the file table with current data"""
        files = self.manager.get_files_in_order()
        self.file_table.setRowCount(len(files))
        
        for row, file_obj in enumerate(files):
            # File name
            self.file_table.setItem(row, 0, QTableWidgetItem(file_obj.filename))
            
            # Size (human-readable format)
            size_mb = file_obj.filesize / (1024 * 1024)
            if size_mb >= 1:
                size_str = f"{size_mb:.1f} MB"
            elif file_obj.filesize >= 1024:
                size_str = f"{file_obj.filesize / 1024:.1f} KB"
            else:
                size_str = f"{file_obj.filesize} B"
            self.file_table.setItem(row, 1, QTableWidgetItem(size_str))
            
            # Status with color coding
            status_item = QTableWidgetItem(file_obj.state.status.value)
            if file_obj.state.status == FileStatus.PARSED:
                status_item.setForeground(QColor(34, 139, 34))  # Forest green
                font = QFont()
                font.setBold(True)
                status_item.setFont(font)
            elif file_obj.state.status == FileStatus.ERROR:
                status_item.setForeground(Qt.red)
            elif file_obj.state.status == FileStatus.PROCESSED:
                status_item.setForeground(Qt.blue)
            else:
                status_item.setForeground(QColor(204, 153, 0))  # Dark yellow/orange for better readability
            self.file_table.setItem(row, 2, status_item)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Info button (exclamation mark)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(30)
            info_button.setMaximumHeight(30)
            info_button.setToolTip("Show file information")
            info_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_file_info(f_id))
            actions_layout.addWidget(info_button)
            
            # Preview button (magnifying glass)
            preview_button = QPushButton("🔍")
            preview_button.setMaximumWidth(30)
            preview_button.setMaximumHeight(30)
            preview_button.setToolTip("Preview raw file content before parsing")
            preview_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_raw_file_preview(f_id))
            actions_layout.addWidget(preview_button)
            
            # Scissor button (manual parse)
            scissor_button = QPushButton("✂️")
            scissor_button.setMaximumWidth(30)
            scissor_button.setMaximumHeight(30)
            scissor_button.setToolTip("Manual parse wizard - customize parsing settings or fix autoparsing errors")
            scissor_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_parse_wizard(f_id))
            actions_layout.addWidget(scissor_button)
            
            # Refresh button
            refresh_button = QPushButton("🔄")
            refresh_button.setMaximumWidth(30)
            refresh_button.setMaximumHeight(30)
            refresh_button.setToolTip("Refresh file to original state")
            refresh_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._refresh_file_to_original(f_id))
            actions_layout.addWidget(refresh_button)
            
            # Delete button
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(30)
            delete_button.setMaximumHeight(30)
            delete_button.setToolTip("Delete file and all its channels")
            delete_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._delete_file(f_id))
            actions_layout.addWidget(delete_button)
            
            self.file_table.setCellWidget(row, 3, actions_widget)
        
        # Update count label
        self.file_count_label.setText(f"Files: {len(files)}")
        
        # Ensure column widths are maintained after table refresh
        self.file_table.setColumnWidth(1, 80)   # Size column
        self.file_table.setColumnWidth(2, 100)   # Status column
        self.file_table.setColumnWidth(3, 180)  # Actions column
    
    def handle_file_selection_changed(self):
        """Handle file selection changes in the file table"""
        selected_rows = set()
        for item in self.file_table.selectedItems():
            selected_rows.add(item.row())
        
        if selected_rows:
            # Get the first selected file
            selected_row = min(selected_rows)
            files = self.manager.get_files_in_order()
            if selected_row < len(files):
                selected_file = files[selected_row]
                self.selected_file_id = selected_file.file_id
                self.selected_file_label.setText(f"📁 {selected_file.filename}")
                self.manager.set_selected_file_id(selected_file.file_id)
            else:
                self.selected_file_id = None
                self.selected_file_label.setText("No file selected")
                self.manager.set_selected_file_id(None)
        else:
            self.selected_file_id = None
            self.selected_file_label.setText("No file selected")
            self.manager.set_selected_file_id(None)
        
        # Refresh channel table to show channels from selected file
        self.refresh_channel_table()
        
        # Update plot to show channels from selected file
        self.manager.get_plot_manager().refresh_plot_for_file(self.selected_file_id)
    
    def refresh_channel_table(self):
        """Refresh the channel table with current data - only shows channels from selected file"""
        if self.selected_file_id is None:
            # No file selected, show empty table
            self.channel_table.setRowCount(0)
            self.channel_count_label.setText("Channels: 0 (0 visible)")
            return
        
        # Get filtered channels from selected file
        channels = self._get_filtered_channels()
        all_channels = self.manager.get_channel_manager().get_channels_by_file(self.selected_file_id)
        
        self.channel_table.setRowCount(len(channels))
        
        for row, channel in enumerate(channels):
            # Column 0: Show (checkbox for visibility toggle)
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(channel.show)
            
            # Check if this channel should have disabled Show checkbox
            should_disable_show = self.manager.should_disable_show_checkbox(channel)
            
            if should_disable_show:
                show_checkbox.setEnabled(False)
                show_checkbox.setToolTip("Show/hide disabled for this channel type")
            else:
                show_checkbox.stateChanged.connect(lambda state, ch_id=channel.channel_id: self._toggle_channel_visibility(ch_id))
            
            # Center the checkbox in the cell
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(show_checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.channel_table.setCellWidget(row, 0, checkbox_widget)
            
            # Column 1: Style (visual preview widget)
            style_widget = StylePreviewWidget(
                color=channel.color or '#1f77b4',
                style=channel.style or '-',
                marker=channel.marker if channel.marker != "None" else None
            )
            self.channel_table.setCellWidget(row, 1, style_widget)
            
            # Column 2: Channel Name (legend label)
            channel_name = channel.legend_label or channel.ylabel or "Unnamed"
            self.channel_table.setItem(row, 2, QTableWidgetItem(channel_name))
            
            # Column 3: Sampling Rate
            sampling_rate_str = channel.get_sampling_rate_description()
            self.channel_table.setItem(row, 3, QTableWidgetItem(sampling_rate_str))
            
            # Column 4: Shape (data shape/length)
            if channel.xdata is not None and channel.ydata is not None:
                shape_str = f"({len(channel.xdata)}, 2)"
            elif channel.ydata is not None:
                shape_str = f"({len(channel.ydata)},)"
            else:
                shape_str = "No data"
            self.channel_table.setItem(row, 4, QTableWidgetItem(shape_str))
            
            # Column 5: Type (channel source type)
            type_str = channel.type.value.upper() if hasattr(channel.type, 'value') else str(channel.type).upper()
            self.channel_table.setItem(row, 5, QTableWidgetItem(type_str))
            
            # Column 6: Actions (info, inspect, styling, transform, delete)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Info button (channel information)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(30)
            info_button.setMaximumHeight(30)
            info_button.setToolTip("Channel information and metadata")
            info_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._show_channel_info(ch_id))
            actions_layout.addWidget(info_button)
            
            # Magnifying glass button (inspect data)
            zoom_button = QPushButton("🔍")
            zoom_button.setMaximumWidth(30)
            zoom_button.setMaximumHeight(30)
            zoom_button.setToolTip("Inspect and edit channel data")
            zoom_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._inspect_channel_data(ch_id))
            actions_layout.addWidget(zoom_button)
            
            # Paint brush button (styling)
            style_button = QPushButton("🎨")
            style_button.setMaximumWidth(30)
            style_button.setMaximumHeight(30)
            style_button.setToolTip("Channel styling and appearance settings")
            style_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self.handle_legend_gear_clicked(ch_id))
            actions_layout.addWidget(style_button)
            
            # Tool button (transform data)
            tool_button = QPushButton("🔨")
            tool_button.setMaximumWidth(30)
            tool_button.setMaximumHeight(30)
            tool_button.setToolTip("Transform channel data with math expressions")
            tool_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._transform_channel_data(ch_id))
            actions_layout.addWidget(tool_button)
            
            # Trash button (delete) - always last
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(30)
            delete_button.setMaximumHeight(30)
            delete_button.setToolTip("Delete channel")
            delete_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._delete_channel(ch_id))
            actions_layout.addWidget(delete_button)
            
            self.channel_table.setCellWidget(row, 6, actions_widget)
        
        # Update count label with filtering information
        visible_channels = [ch for ch in channels if ch.show]
        show_all_types = self.show_all_types_checkbox.isChecked()
        
        if show_all_types:
            self.channel_count_label.setText(f"Channels: {len(channels)} ({len(visible_channels)} visible)")
        else:
            self.channel_count_label.setText(f"Channels: {len(channels)} RAW / {len(all_channels)} total ({len(visible_channels)} visible)")
        
        # Update plot
        self.manager.get_plot_manager().update_plot(channels)
        
        # Ensure column widths are maintained after table refresh
        self.channel_table.setColumnWidth(0, 60)   # Show checkbox
        self.channel_table.setColumnWidth(1, 80)   # Style preview
        self.channel_table.setColumnWidth(3, 100)  # Sampling Rate column
        self.channel_table.setColumnWidth(4, 80)   # Shape column
        self.channel_table.setColumnWidth(5, 100)  # Type column
        self.channel_table.setColumnWidth(6, 180)  # Actions buttons
    
    def handle_file_table_click(self, row, column):
        """Handle clicks in the file table (now handled by action buttons)"""
        # File actions are now handled by individual buttons in the actions column
        pass
    
    def handle_channel_table_click(self, row, column):
        """Handle clicks in the channel table (now mostly handled by action buttons)"""
        # Most actions are now handled by individual buttons in the actions column
        pass
    
    def _get_filtered_channels(self, file_id: str = None):
        """Get channels for the selected file with current filter applied"""
        from channel import SourceType
        
        if file_id is None:
            file_id = self.selected_file_id
        
        if not file_id:
            return []
        
        # Get all channels from the file
        all_channels = self.manager.get_channel_manager().get_channels_by_file(file_id)
        
        # ALWAYS exclude comparison channels from main window plot
        # Comparison channels should only appear in the comparison wizard
        valid_channels = [ch for ch in all_channels 
        if not ch.type == SourceType.COMPARISON and not ch.type == SourceType.SPECTROGRAM]
        
        # Apply filter based on "Show all data types" toggle
        show_all_types = self.show_all_types_checkbox.isChecked()
        if show_all_types:
            return valid_channels
        else:
            # Show only RAW channels by default (comparison channels already excluded)
            return [ch for ch in valid_channels if ch.type == SourceType.RAW]

    def _on_channel_filter_changed(self):
        """Handle change in channel type filter"""
        self.refresh_channel_table()
        show_all = self.show_all_types_checkbox.isChecked()

        
        # Warning when showing all types about plot preview limitations
        if show_all:
            log_message("Note: Plot preview is disabled for COMPARISON channels", "warning", "MAIN")

    def _toggle_channel_visibility(self, channel_id: str):
        """Toggle visibility of a channel and update UI"""
        self.manager.toggle_channel_visibility(channel_id)
        self.refresh_channel_table()
    
    def _show_channel_info(self, channel_id: str):
        """Show detailed information about a channel using the metadata wizard"""
        self.manager.show_channel_info(channel_id, self)

    
    def _inspect_channel_data(self, channel_id: str):
        """Open the data inspection wizard for this channel"""
        self.manager.inspect_channel_data(channel_id, self)

    
    def _transform_channel_data(self, channel_id: str):
        """Open the data transformation wizard for this channel"""
        self.manager.transform_channel_data(channel_id, self)

    
    def _delete_channel(self, channel_id: str):
        """Delete a channel with confirmation"""
        self.manager.delete_channel(channel_id, self)
        self.refresh_channel_table()

    
    def _show_file_info(self, file_id: str):
        """Show detailed information about a file"""
        self.manager.show_file_info(file_id, self)

    
    def _show_raw_file_preview(self, file_id: str):
        """Show a preview of the raw file content before parsing"""
        self.manager.show_raw_file_preview(file_id, self)
    
    def _refresh_file_to_original(self, file_id: str):
        """Refresh all channels in a file back to their original state"""
        self.manager.refresh_file_to_original(file_id, self)
        self.refresh_file_table()
        self.refresh_channel_table()
    
    def _delete_file(self, file_id: str):
        """Delete a file and all its channels with confirmation"""
        self.manager.delete_file(file_id, self)
        self.refresh_file_table()
        self.refresh_channel_table()
    
    def handle_legend_gear_clicked(self, channel_id: str):
        """Handle gear button clicks in legend table"""
        self.manager.handle_legend_gear_clicked(channel_id, self)
        self.refresh_channel_table()
    
    def show_mix_wizard(self):
        """Show the Mix Wizard dialog."""
        self.manager.show_mix_wizard(self)

    def show_process_wizard(self):
        """Show the process wizard for channel processing"""
        self.manager.show_process_wizard(self)
        
    def show_comparison_wizard(self):
        """Show the comparison wizard for data analysis"""
        self.manager.show_comparison_wizard(self)
        
    def show_plot_wizard(self):
        """Show the plot wizard for custom plotting"""
        self.manager.show_plot_wizard(self)
    
    def show_load_file_dialog(self):
        """Show file dialog to load data files"""
        # Define supported file types
        file_filter = "Data Files (*.csv *.txt *.tsv *.dat);;CSV Files (*.csv);;Text Files (*.txt);;TSV Files (*.tsv);;DAT Files (*.dat);;All Files (*)"
        
        # Open file dialog
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Data Files",
            "",  # Default directory
            file_filter
        )
        
        if file_paths:
            # Convert string paths to Path objects
            paths = [Path(file_path) for file_path in file_paths]
            log_message(f"Loading {len(paths)} file(s) via file dialog...", "processing")
            
            # Use existing file handling logic
            self.handle_files_dropped(paths)
        else:
            log_message("File loading cancelled")
    
    def show_export_wizard(self):
        """Show the export wizard for saving data"""
        self.manager.show_export_wizard(self)
    
    def _show_parse_wizard(self, file_id: str):
        """Show the parse wizard for manual parsing"""
        self.manager.show_parse_wizard(file_id, self)
        self.refresh_file_table()
        self.refresh_channel_table()
    
    def handle_console_message(self, message, msg_type="info", category="MAIN"):
        """Handle console message from manager (for backward compatibility)"""
        # Messages are already handled by ConsoleManager, this is just for compatibility
        pass
    
    def show_floating_console(self):
        """Show the floating console window"""
        console = get_console()
        console.show_and_focus()
    
    def get_system_status(self):
        """Get current system status for debugging"""
        return self.manager.get_system_status()
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events for file drops"""
        if event.mimeData().hasUrls():
            # Check if any files have supported extensions
            supported_extensions = {'.csv', '.txt', '.tsv', '.dat'}
            has_supported_files = False
            
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    file_path = Path(url.toLocalFile())
                    if file_path.suffix.lower() in supported_extensions:
                        has_supported_files = True
                        break
            
            if has_supported_files:
                event.acceptProposedAction()
                # Visual feedback - highlight file table
                self.file_table.setStyleSheet("""
                    QTableWidget { 
                        border: 3px dashed #4CAF50; 
                        background-color: rgba(76, 175, 80, 0.1);
                    }
                """)
            else:
                event.ignore()
                # Visual feedback for unsupported files
                self.file_table.setStyleSheet("""
                    QTableWidget { 
                        border: 3px dashed #f44336; 
                        background-color: rgba(244, 67, 54, 0.1);
                    }
                """)
                log_message("Only CSV, TXT, TSV, and DAT files are supported", "warning")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Reset visual feedback when drag leaves the window"""
        self.file_table.setStyleSheet("")  # Reset styling
    
    def dropEvent(self, event: QDropEvent):
        """Handle file drops"""
        # Reset visual feedback immediately
        self.file_table.setStyleSheet("")
        
        files = []
        supported_extensions = {'.csv', '.txt', '.tsv', '.dat'}
        
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = Path(url.toLocalFile())
                
                # Check if file exists and has supported extension
                if file_path.exists() and file_path.suffix.lower() in supported_extensions:
                    files.append(file_path)
                elif file_path.exists():
                    log_message(f"Unsupported file type: {file_path.name} (only CSV, TXT, TSV, DAT supported)", "warning")
                else:
                    log_message(f"File not found: {file_path.name}", "error")
        
        if files:
            self.handle_files_dropped(files)
            event.acceptProposedAction()
        else:
            log_message("No valid files to process", "error")
            event.ignore()
    
    def _on_file_selection_changed(self, file_id):
        """Handle file selection change from manager"""
        if file_id:
            file_obj = self.manager.get_file_manager().get_file(file_id)
            if file_obj:
                self.selected_file_label.setText(f"📁 {file_obj.filename}")
        else:
            self.selected_file_label.setText("No file selected")
        
        # Refresh channel table to ensure style column shows correct colors
        self.refresh_channel_table()
    
    def _on_channel_visibility_toggled(self, channel_id, visible):
        """Handle channel visibility toggle from manager"""
        # Update plot if needed
        if self.selected_file_id:
            channels = self._get_filtered_channels()
            self.manager.get_plot_manager().update_plot(channels)
    
    def _show_welcome_messages(self):
        """Show welcome messages in console"""
        welcome_message = """Welcome! Load files via 'Load File' or drag & drop into the table.
Supported formats: CSV, TXT, TSV with time series or tabular data.
Available Wizards:
• Process Wizard – Apply signal processing (resample, smooth, filter)
• Mix Wizard – Combine channels using math (A+B, A-B, A*B, A/B)
• Compare Wizard – EXPERIMENTAL (correlation, Bland-Altman, regression)
• Plot Wizard – Build multi-subplot visualizations with custom styling
• Export Wizard – Save processed data to CSV or TXT
Console window opened for detailed logging."""
        self.manager.log_message(welcome_message, "info", "MAIN")
    
    def _position_window_left(self):
        """Position main window on the left side of the screen"""
        # Get primary screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Calculate dimensions for left half
        window_width = screen_geometry.width() // 2
        window_height = screen_geometry.height()
        
        # Position on left side
        self.setGeometry(
            screen_geometry.x(),  # x position (left edge)
            screen_geometry.y(),  # y position (top edge)  
            window_width,         # width (half screen)
            window_height         # height (full screen)
        )
