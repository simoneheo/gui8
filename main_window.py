# main_window.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame, QCheckBox, QSplitter, QTextEdit, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPalette, QColor, QFont
from pathlib import Path
import traceback
from datetime import datetime

# Import our new managers
from auto_parser import AutoParser
from file_manager import FileManager
from channel_manager import ChannelManager
from file import FileStatus
from plot_manager import PlotManager, StylePreviewWidget
from line_wizard import LineWizard
from metadata_wizard import MetadataWizard
from inspection_wizard import InspectionWizard
from transform_wizard import TransformWizard

# Import the new wizard managers
from comparison_wizard_manager import ComparisonWizardManager
from export_wizard_manager import ExportWizardManager
from signal_mixer_wizard_manager import SignalMixerWizardManager
from process_wizard_manager import ProcessWizardManager
from plot_wizard_manager import PlotWizardManager
from parse_wizard_manager import ParseWizardManager


class MainWindowUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Processing GUI")
        self.setMinimumSize(1200, 800)
        
        # Enable drag and drop for entire window
        self.setAcceptDrops(True)
        
        # Initialize managers
        self.auto_parser = AutoParser()
        self.file_manager = FileManager()
        self.channel_manager = ChannelManager()
        self.plot_manager = PlotManager(self.channel_manager)
        
        # Track selected file for channel filtering
        self.selected_file_id = None
        
        # File size warning settings
        self.large_file_threshold_mb = 50  # MB threshold for large file warning
        self.show_large_file_warnings = True  # Can be disabled via Manual Parse
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI state
        self.log_message("Welcome! Load files by clicking 'Load File' button or drag & drop into the file table")
        self.log_message("Supported data: CSV, TXT, TSV files with time series or tabular data")
        self.log_message("")  # Empty line for separation
        self.log_message("Available Wizards:")
        self.log_message("• Process Wizard - Apply signal processing filters (resample, smooth, filter, transform)")
        self.log_message("• Mix Wizard - Combine multiple channels with arithmetic operations (A+B, A-B, A*B, A/B)")
        self.log_message("• Compare Wizard - Statistical comparison between channels (correlation, Bland-Altman, regression)")
        self.log_message("• Plot Wizard - Create customized multi-subplot visualizations with advanced styling")
    
    def init_ui(self):
        main_splitter = QSplitter(Qt.Horizontal, self)

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
                font-size: 11px;
                text-align: center;
                min-height: 35px;
                max-height: 40px;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border-color: #999;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
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
        for button in [self.load_file_button, self.process_button, self.mix_button, self.compare_button, self.plot_button, self.export_file_button]:
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
        self.show_all_types_checkbox.setToolTip("Show all data types (RAW, PROCESSED, MIXED, etc.)\nBy default, only RAW channels are shown")
        self.show_all_types_checkbox.setChecked(False)  # Default to RAW only
        self.show_all_types_checkbox.stateChanged.connect(self._on_channel_filter_changed)
        
        channel_manager_layout.addWidget(channel_manager_label)
        channel_manager_layout.addWidget(self.selected_file_label)
        channel_manager_layout.addStretch()
        channel_manager_layout.addWidget(self.show_all_types_checkbox)
        channel_manager_layout.addWidget(self.channel_count_label)
        tables_layout.addLayout(channel_manager_layout)
        
        self.channel_table = QTableWidget(0, 6)
        self.channel_table.setHorizontalHeaderLabels(["Show", "Style", "Channel Name", "Shape", "Type", "Actions"])
        
        # Set column resize modes for better layout
        header = self.channel_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)     # Show column - fixed width
        header.setSectionResizeMode(1, QHeaderView.Fixed)     # Style - fixed width
        header.setSectionResizeMode(2, QHeaderView.Stretch)   # Channel Name - stretches
        header.setSectionResizeMode(3, QHeaderView.Fixed)     # Shape - fixed width
        header.setSectionResizeMode(4, QHeaderView.Fixed)     # Type - fixed width
        header.setSectionResizeMode(5, QHeaderView.Fixed)     # Actions - fixed width
        
        # Set specific column widths
        self.channel_table.setColumnWidth(0, 60)   # Show checkbox
        self.channel_table.setColumnWidth(1, 80)   # Style preview
        self.channel_table.setColumnWidth(3, 80)   # Shape column
        self.channel_table.setColumnWidth(4, 100)  # Type column
        self.channel_table.setColumnWidth(5, 180)  # Actions buttons
        self.channel_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.channel_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        tables_layout.addWidget(self.channel_table)
        
        content_splitter.addWidget(tables_widget)

        # Bottom section: Plot Canvas only
        plot_widget_container = QWidget()
        plot_layout = QVBoxLayout(plot_widget_container)
        
        # Plot Canvas (using PlotManager)
        plot_widget = self.plot_manager.get_plot_widget()
        plot_widget.setMinimumHeight(250)
        # plot_layout.addWidget(QLabel("Plot Canvas"))
        plot_layout.addWidget(plot_widget)
        
        content_splitter.addWidget(plot_widget_container)
        
        # Set initial splitter sizes: Tables(50%), Plot(50%)
        content_splitter.setSizes([500, 500])
        
        # Add splitter to main layout
        main_content_layout.addWidget(content_splitter)
        main_splitter.addWidget(main_content_panel)

        # Right panel: Console Output (spans full height)
        right_panel = QWidget()
        right_panel.setMaximumWidth(300)
        right_panel.setMinimumWidth(250)
        console_layout = QVBoxLayout(right_panel)
        # console_label = QLabel("Console Output")
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # console_layout.addWidget(console_label)
        console_layout.addWidget(self.console)

        main_splitter.addWidget(right_panel)
        
        # Set initial splitter sizes: Main Content(expand), Console(300px)
        main_splitter.setSizes([900, 300])

        # Apply main layout
        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)
    
    def connect_signals(self):
        """Connect all signal handlers"""
     
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
        
        # Connect plot manager signals
        self.plot_manager.legend_item_gear_clicked.connect(self.handle_legend_gear_clicked)
    
    def handle_files_dropped(self, file_paths):
        """Handle files dropped into the drop zone"""
        self.log_message(f"Processing {len(file_paths)} dropped file(s)...", "processing")
        
        successful_files = 0
        total_channels = 0
        newly_loaded_file_id = None  # Track the most recently loaded file
        
        for file_path in file_paths:
            try:
                # Check if file already exists
                if self.file_manager.has_file_path(file_path):
                    self.log_message(f"File already loaded: {file_path.name}", "warning")
                    continue
                
                # Check file size and warn if large
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb >= self.large_file_threshold_mb and self.show_large_file_warnings:
                    if not self._show_large_file_warning(file_path, file_size_mb):
                        continue  # User cancelled loading
                
                # Parse the file
                self.log_message(f"Parsing: {file_path.name}", "processing")
                file_obj, channels = self.auto_parser.parse_file(file_path)
                
                # Add to managers
                self.file_manager.add_file(file_obj)
                
                if channels:
                    added_count = self.channel_manager.add_channels(channels)
                    successful_files += 1
                    total_channels += added_count
                    newly_loaded_file_id = file_obj.file_id  # Update to most recent successful file
                else:
                    self.log_message(f"Failed: {file_path.name} → {file_obj.state.error_message or 'No channels created'}", "error")
                    self.log_message(f"Auto-parsing failed. Please try manual parsing using the ✂️ icon in the file table to customize parsing settings", "suggestion")
                
            except Exception as e:
                self.log_message(f"Error processing {file_path.name}: {str(e)}", "error")
                traceback.print_exc()
        
        # Update UI
        self.refresh_file_table()
        
        # Auto-select the most recently loaded file
        if newly_loaded_file_id is not None:
            file_obj = self.file_manager.get_file(newly_loaded_file_id)
            if file_obj:
                self.selected_file_id = newly_loaded_file_id
                self.selected_file_label.setText(f"📁 {file_obj.filename}")
                
                # Select the file in the table
                files = self.file_manager.get_files_in_order()
                for row, current_file in enumerate(files):
                    if current_file.file_id == newly_loaded_file_id:
                        self.file_table.selectRow(row)
                        break
        
        # Refresh channel table and plot (this will use the newly selected file)
        self.refresh_channel_table()
        
        # Summary message
        if successful_files > 0:
            self.log_message(f"Completed: {successful_files} files parsed, {total_channels} channels created", "success")
            self.log_message(f"If results need adjustment, use the ✂️ icon for manual parsing with custom settings", "suggestion")
        else:
            self.log_message("No files were successfully processed", "warning")
    
    def refresh_file_table(self):
        """Refresh the file table with current data"""
        files = self.file_manager.get_files_in_order()
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
            info_button.setMaximumWidth(25)
            info_button.setMaximumHeight(25)
            info_button.setToolTip("Show file information")
            info_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_file_info(f_id))
            actions_layout.addWidget(info_button)
            
            # Preview button (magnifying glass)
            preview_button = QPushButton("🔍")
            preview_button.setMaximumWidth(25)
            preview_button.setMaximumHeight(25)
            preview_button.setToolTip("Preview raw file content before parsing")
            preview_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_raw_file_preview(f_id))
            actions_layout.addWidget(preview_button)
            
            # Scissor button (manual parse)
            scissor_button = QPushButton("✂️")
            scissor_button.setMaximumWidth(25)
            scissor_button.setMaximumHeight(25)
            scissor_button.setToolTip("Manual parse wizard - customize parsing settings or fix autoparsing errors")
            scissor_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._show_parse_wizard(f_id))
            actions_layout.addWidget(scissor_button)
            
            # Refresh button
            refresh_button = QPushButton("🔄")
            refresh_button.setMaximumWidth(25)
            refresh_button.setMaximumHeight(25)
            refresh_button.setToolTip("Refresh file to original state")
            refresh_button.clicked.connect(lambda checked=False, f_id=file_obj.file_id: self._refresh_file_to_original(f_id))
            actions_layout.addWidget(refresh_button)
            
            # Delete button
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(25)
            delete_button.setMaximumHeight(25)
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
            files = self.file_manager.get_files_in_order()
            if selected_row < len(files):
                selected_file = files[selected_row]
                self.selected_file_id = selected_file.file_id
                self.selected_file_label.setText(f"📁 {selected_file.filename}")
            else:
                self.selected_file_id = None
                self.selected_file_label.setText("No file selected")
        else:
            self.selected_file_id = None
            self.selected_file_label.setText("No file selected")
        
        # Refresh channel table to show channels from selected file
        self.refresh_channel_table()
        
        # Update plot to show channels from selected file
        self.plot_manager.refresh_plot_for_file(self.selected_file_id)
    
    def refresh_channel_table(self):
        """Refresh the channel table with current data - only shows channels from selected file"""
        if self.selected_file_id is None:
            # No file selected, show empty table
            self.channel_table.setRowCount(0)
            self.channel_count_label.setText("Channels: 0 (0 visible)")
            return
        
        # Get filtered channels from selected file
        channels = self._get_filtered_channels()
        all_channels = self.channel_manager.get_channels_by_file(self.selected_file_id)
        
        self.channel_table.setRowCount(len(channels))
        
        for row, channel in enumerate(channels):
            # Column 0: Show (checkbox for visibility toggle)
            show_checkbox = QCheckBox()
            show_checkbox.setChecked(channel.show)
            
            # Check if this channel should have disabled Show checkbox
            should_disable_show = self._should_disable_show_checkbox(channel)
            
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
            
            # Column 3: Shape (data shape/length)
            if channel.xdata is not None and channel.ydata is not None:
                shape_str = f"({len(channel.xdata)}, 2)"
            elif channel.ydata is not None:
                shape_str = f"({len(channel.ydata)},)"
            else:
                shape_str = "No data"
            self.channel_table.setItem(row, 3, QTableWidgetItem(shape_str))
            
            # Column 4: Type (channel source type)
            type_str = channel.type.value.upper() if hasattr(channel.type, 'value') else str(channel.type).upper()
            self.channel_table.setItem(row, 4, QTableWidgetItem(type_str))
            
            # Column 5: Actions (info, inspect, styling, transform, delete)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Info button (channel information)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(25)
            info_button.setMaximumHeight(25)
            info_button.setToolTip("Channel information and metadata")
            info_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._show_channel_info(ch_id))
            actions_layout.addWidget(info_button)
            
            # Magnifying glass button (inspect data)
            zoom_button = QPushButton("🔍")
            zoom_button.setMaximumWidth(25)
            zoom_button.setMaximumHeight(25)
            zoom_button.setToolTip("Inspect and edit channel data")
            zoom_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._inspect_channel_data(ch_id))
            actions_layout.addWidget(zoom_button)
            
            # Paint brush button (styling)
            style_button = QPushButton("🎨")
            style_button.setMaximumWidth(25)
            style_button.setMaximumHeight(25)
            style_button.setToolTip("Channel styling and appearance settings")
            style_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self.handle_legend_gear_clicked(ch_id))
            actions_layout.addWidget(style_button)
            
            # Tool button (transform data)
            tool_button = QPushButton("🔨")
            tool_button.setMaximumWidth(25)
            tool_button.setMaximumHeight(25)
            tool_button.setToolTip("Transform channel data with math expressions")
            tool_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._transform_channel_data(ch_id))
            actions_layout.addWidget(tool_button)
            
            # Trash button (delete) - always last
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(25)
            delete_button.setMaximumHeight(25)
            delete_button.setToolTip("Delete channel")
            delete_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._delete_channel(ch_id))
            actions_layout.addWidget(delete_button)
            
            self.channel_table.setCellWidget(row, 5, actions_widget)
        
        # Update count label with filtering information
        visible_channels = [ch for ch in channels if ch.show]
        show_all_types = self.show_all_types_checkbox.isChecked()
        
        if show_all_types:
            self.channel_count_label.setText(f"Channels: {len(channels)} ({len(visible_channels)} visible)")
        else:
            self.channel_count_label.setText(f"Channels: {len(channels)} RAW / {len(all_channels)} total ({len(visible_channels)} visible)")
        
        # Update plot
        self.plot_manager.update_plot(channels)
        
        # Ensure column widths are maintained after table refresh
        self.channel_table.setColumnWidth(0, 60)   # Show checkbox
        self.channel_table.setColumnWidth(1, 80)   # Style preview
        self.channel_table.setColumnWidth(3, 80)   # Shape column
        self.channel_table.setColumnWidth(4, 100)  # Type column
        self.channel_table.setColumnWidth(5, 180)  # Actions buttons
    
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
        if file_id is None:
            file_id = self.selected_file_id
        
        if not file_id:
            return []
        
        # Get all channels from the file
        all_channels = self.channel_manager.get_channels_by_file(file_id)
        
        # Apply filter based on "Show all data types" toggle
        show_all_types = self.show_all_types_checkbox.isChecked()
        if show_all_types:
            return all_channels
        else:
            # Show only RAW channels by default
            from channel import SourceType
            return [ch for ch in all_channels if ch.type == SourceType.RAW]

    def _on_channel_filter_changed(self):
        """Handle change in channel type filter"""
        self.refresh_channel_table()
        show_all = self.show_all_types_checkbox.isChecked()

        
        # Warning when showing all types about plot preview limitations
        if show_all:
            self.log_message("Note: Plot preview is disabled for SPECTROGRAM and STATISTICAL channel types", "warning")

    def _should_disable_show_checkbox(self, channel):
        """Determine if the Show checkbox should be disabled for this channel"""
        from channel import SourceType
        
        # Disable for SPECTROGRAM channels
        if channel.type == SourceType.SPECTROGRAM:
            return True
        
        # Disable for channels with spectrogram-related tags
        spectrogram_tags = ['spectrogram', 'spectrum', 'stft', 'cwt', 'welch', 'fft']
        if any(tag.lower() in spectrogram_tags for tag in channel.tags):
            return True
        
        # Disable for channels created by comparison wizard
        # Look for comparison-related tags or descriptions
        comparison_tags = ['comparison', 'difference', 'residual', 'correlation', 'bland-altman']
        if any(tag.lower() in comparison_tags for tag in channel.tags):
            return True
        
        # Check if description contains comparison wizard indicators
        description = channel.description.lower() if channel.description else ""
        comparison_indicators = ['comparison wizard', 'comparison result', 'residual analysis']
        if any(indicator in description for indicator in comparison_indicators):
            return True
        
        return False

    def _toggle_channel_visibility(self, channel_id: str):
        """Toggle visibility of a channel and update UI"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Toggle visibility
            new_state = not channel.show
            self.channel_manager.set_channel_visibility(channel_id, new_state)

            
            # Refresh the channel table to update button appearance
            self.refresh_channel_table()
    
    def _show_channel_info(self, channel_id: str):
        """Show detailed information about a channel using the metadata wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the comprehensive metadata wizard
            wizard = MetadataWizard(channel, self)
            wizard.exec()

    
    def _inspect_channel_data(self, channel_id: str):
        """Open the data inspection wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data inspection wizard
            wizard = InspectionWizard(channel, self)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()

    
    def handle_channel_data_updated(self, channel_id: str):
        """Handle when channel data is updated via inspection/transform wizards"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel and self.selected_file_id:
            # Get filtered channels respecting the current filter toggle
            channels = self._get_filtered_channels()
            
            # Refresh all UI components immediately
            self.refresh_channel_table()  # Update channel table with new statistics
            self.plot_manager.update_plot(channels)  # Update plot canvas with filtered channels
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
            

    
    def _transform_channel_data(self, channel_id: str):
        """Open the data transformation wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data transformation wizard
            wizard = TransformWizard(channel, self)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()

    
    def _delete_channel(self, channel_id: str):
        """Delete a channel with confirmation"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            reply = QMessageBox.question(
                self, 
                "Delete Channel", 
                f"Delete channel '{channel.ylabel}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.channel_manager.remove_channel(channel_id)
                self.refresh_channel_table()

    
    def _show_file_info(self, file_id: str):
        """Show detailed information about a file"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            channels = self.channel_manager.get_channels_by_file(file_id)
            
            # Calculate file statistics
            size_mb = file_obj.filesize / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_obj.filesize} B"
            
            # Get parse result information if available
            parse_result = file_obj.parse_config.parse_result if file_obj.parse_config else None
            
            # Format last modified time
            import datetime
            last_modified_str = datetime.datetime.fromtimestamp(file_obj.last_modified).strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate values for display
            processing_time = f"{(parse_result.parse_time_ms/1000):.3f}s" if parse_result and parse_result.parse_time_ms else 'Unknown'
            parse_method = parse_result.strategy_used.value if parse_result and parse_result.strategy_used else 'Unknown'
            data_rows = parse_result.rows_parsed if parse_result else 'Unknown'
            metadata_lines = parse_result.metadata_lines_skipped if parse_result else 'Unknown'
            encoding = parse_result.encoding_detected if parse_result else 'Unknown'
            
            info_text = f"""
File Information:
• Name: {file_obj.filename}
• File ID: {file_id}
• Size: {size_str} ({file_obj.filesize:,} bytes)
• Status: {file_obj.state.status.value}
• Channels: {len(channels)}
• Last Modified: {last_modified_str}
• Path: {file_obj.filepath}

Parse Statistics:
• Processing Time: {processing_time}
• Parse Method: {parse_method}
• Data Rows: {data_rows}
• Metadata Lines Skipped: {metadata_lines}
• Encoding: {encoding}

Channel Summary:
• Total Channels: {len(channels)}
• Visible Channels: {len([ch for ch in channels if ch.show])}
• Modified Channels: {len([ch for ch in channels if hasattr(ch, 'modified_at') and ch.modified_at])}

Error Information:
{file_obj.state.error_message or 'No errors'}
            """
            
            QMessageBox.information(self, f"File Info - {file_obj.filename}", info_text.strip())

    
    def _show_raw_file_preview(self, file_id: str):
        """Show a preview of the raw file content before parsing"""
        file_obj = self.file_manager.get_file(file_id)
        if not file_obj:
            return
            
        try:
            # Create preview dialog
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLabel, QDialogButtonBox
            from PySide6.QtGui import QFont
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Raw File Preview - {file_obj.filename}")
            dialog.setMinimumSize(800, 600)
            dialog.resize(1000, 700)
            
            layout = QVBoxLayout(dialog)
            
            # File info header
            file_size_mb = file_obj.filesize / (1024 * 1024)
            size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_obj.filesize:,} bytes"
            
            info_label = QLabel(f"File: {file_obj.filename} | Size: {size_str}")
            info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f0f0f0;")
            layout.addWidget(info_label)
            
            # Read file content with encoding detection
            file_content, detected_encoding = self._read_file_safely(file_obj.filepath)
            
            # Encoding info
            encoding_label = QLabel(f"Detected Encoding: {detected_encoding} | Showing first 500 lines")
            encoding_label.setStyleSheet("color: #666; font-size: 11px; padding: 2px;")
            layout.addWidget(encoding_label)
            
            # Text area for file content
            text_area = QTextEdit()
            text_area.setReadOnly(True)
            text_area.setPlainText(file_content)
            
            # Use monospace font for better formatting
            font = QFont("Consolas", 10)
            if not font.exactMatch():
                font = QFont("Courier New", 10)
            text_area.setFont(font)
            
            layout.addWidget(text_area)
            
            # Dialog buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Close)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # Show dialog
            dialog.exec()

            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Could not preview file:\n{str(e)}")
            self.log_message(f"Error previewing {file_obj.filename}: {str(e)}", "error")
    
    def _read_file_safely(self, file_path, max_lines=500):
        """Read file content safely with encoding detection"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            lines.append(f"\n... (file continues, showing first {max_lines} lines only) ...")
                            break
                        lines.append(line.rstrip('\n\r'))
                    
                    content = '\n'.join(lines)
                    return content, encoding
                    
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Final fallback - read as binary and decode with replace
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                # Only read first portion if file is very large
                if len(raw_content) > 1024 * 1024:  # 1MB limit
                    raw_content = raw_content[:1024 * 1024]
                    
                content = raw_content.decode('utf-8', errors='replace')
                lines = content.split('\n')[:max_lines]
                if len(content.split('\n')) > max_lines:
                    lines.append(f"\n... (large file truncated, showing first {max_lines} lines only) ...")
                    
                return '\n'.join(lines), 'utf-8 (with fallback)'
                
        except Exception as e:
            return f"Error reading file: {str(e)}", 'unknown'
    
    def _show_large_file_warning(self, file_path: Path, file_size_mb: float) -> bool:
        """Show warning dialog for large files and suggest downsampling"""
        try:
            from PySide6.QtWidgets import QMessageBox, QCheckBox
            
            # Create custom dialog with checkbox
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Large File Warning")
            dialog.setIcon(QMessageBox.Warning)
            
            # Main warning text
            warning_text = f"""⚠️ Large File Detected: {file_path.name}

File Size: {file_size_mb:.1f} MB
Expected slow loading and processing times.

This large file may cause:
• Slow parsing and loading
• High memory usage
• Delayed response times during processing

Recommendations:
• Consider using downsampling to reduce file size
• Use Manual Parse (✂️ icon) with downsample option
• Process in smaller chunks if possible"""
            
            dialog.setText(warning_text)
            dialog.setInformativeText("Would you like to continue loading this file?")
            
            # Add buttons
            continue_button = dialog.addButton("Continue Loading", QMessageBox.AcceptRole)
            downsample_button = dialog.addButton("Use Manual Parse (Downsample)", QMessageBox.ActionRole)
            cancel_button = dialog.addButton("Cancel", QMessageBox.RejectRole)
            
            # Create checkbox for disabling warnings
            checkbox = QCheckBox("Don't show this warning again (can be re-enabled in Manual Parse)")
            dialog.setCheckBox(checkbox)
            
            # Show dialog and handle response
            dialog.exec()
            clicked_button = dialog.clickedButton()
            
            # Handle checkbox state
            if checkbox.isChecked():
                self.show_large_file_warnings = False
                self.log_message("Large file warnings disabled. Can be re-enabled in Manual Parse settings.", "info")
            
            if clicked_button == continue_button:
                self.log_message(f"User chose to continue loading large file: {file_path.name}", "info")
                return True
            elif clicked_button == downsample_button:
                self.log_message(f"User chose Manual Parse with downsample for: {file_path.name}", "info")
                # Create a temporary file object to open Manual Parse
                from file import File
                temp_file = File(file_path)
                self.file_manager.add_file(temp_file)
                self._show_parse_wizard(temp_file.file_id)
                return False
            else:
                self.log_message(f"User cancelled loading large file: {file_path.name}", "info")
                return False
                
        except Exception as e:
            self.log_message(f"Error showing large file warning: {str(e)}", "error")
            return True  # Default to continue if error occurs
    
    def _refresh_file_to_original(self, file_id: str):
        """Refresh all channels in a file back to their original state"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            reply = QMessageBox.question(
                self, 
                "Refresh File", 
                f"Refresh file '{file_obj.filename}' and reset all channels to original state?\n\n"
                "This will undo all transformations and edits for this file's channels.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    # Re-parse the file to get original data
                    self.log_message(f"Re-parsing: {file_obj.filename}", "processing")
                    new_file_obj, new_channels = self.auto_parser.parse_file(file_obj.filepath)
                    
                    if new_channels:
                        # Remove old channels
                        removed_count = self.channel_manager.remove_channels_by_file(file_id)
                        
                        # Add new channels with original data
                        added_count = self.channel_manager.add_channels(new_channels)
                        
                        # Update file object
                        file_obj.state = new_file_obj.state
                        file_obj.modified_at = datetime.now()
                        
                        # Refresh UI
                        self.refresh_file_table()
                        self.refresh_channel_table()
                        
                        self.log_message(f"Refreshed: {file_obj.filename} → {added_count} channels restored", "success")
                        
                        QMessageBox.information(
                            self, 
                            "File Refreshed", 
                            f"Successfully refreshed '{file_obj.filename}':\n"
                            f"• Removed {removed_count} modified channels\n"
                            f"• Restored {added_count} original channels"
                        )
                    else:
                        self.log_message(f"Failed to refresh: {file_obj.filename}", "error")
                        QMessageBox.warning(
                            self, 
                            "Refresh Failed", 
                            f"Failed to re-parse file '{file_obj.filename}':\n{new_file_obj.state.error_message or 'Unknown error'}"
                        )
                        
                except Exception as e:
                    self.log_message(f"Error refreshing {file_obj.filename}: {str(e)}", "error")
                    QMessageBox.critical(
                        self, 
                        "Refresh Error", 
                        f"An error occurred while refreshing the file:\n{str(e)}"
                    )
    
    def _delete_file(self, file_id: str):
        """Delete a file and all its channels with confirmation"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            channels = self.channel_manager.get_channels_by_file(file_id)
            reply = QMessageBox.question(
                self, 
                "Delete File", 
                f"Delete file '{file_obj.filename}' and all its {len(channels)} channels?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                # Remove channels first
                removed_channels = self.channel_manager.remove_channels_by_file(file_id)
                # Remove file
                self.file_manager.remove_file(file_id)
                
                # If this was the selected file, clear selection
                if self.selected_file_id == file_id:
                    self.selected_file_id = None
                    self.selected_file_label.setText("No file selected")
                
                # Refresh UI
                self.refresh_file_table()
                self.refresh_channel_table()
                self.log_message(f"Deleted: {file_obj.filename} ({removed_channels} channels)", "success")
    
    def handle_legend_gear_clicked(self, channel_id: str):
        """Handle gear button clicks in legend table"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open line wizard
            wizard = LineWizard(channel, self)
            wizard.channel_updated.connect(self.handle_channel_updated)
            wizard.exec()
    
    def handle_channel_updated(self, channel_id: str):
        """Handle when a channel is updated via line wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel and self.selected_file_id:
            # Get filtered channels respecting the current filter toggle
            channels = self._get_filtered_channels()
            
            # Refresh all UI components immediately
            self.refresh_channel_table()  # Update channel table with new properties
            self.plot_manager.update_plot(channels)  # Update plot canvas with filtered channels
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
            


    def show_mix_wizard(self):
        """Show the Mix Wizard dialog."""
        try:
            # Check if we have data to work with
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the mixer wizard.")
                return
            
            # Track channels before opening wizard
            self._track_channels_before_wizard_open()
            
            # Import and create the signal mixer wizard window directly
            # (This wizard has different architecture - window creates manager internally)
            from signal_mixer_wizard_window import SignalMixerWizardWindow
            
            self.mixer_wizard_window = SignalMixerWizardWindow(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self
            )
            
            # Connect to wizard close signal
            self.mixer_wizard_window.wizard_closed.connect(
                lambda: self._on_mixer_wizard_closed()
            )
            
            # Show the wizard window
            self.mixer_wizard_window.show()
            
        except Exception as e:
            self.log_message(f"Error opening Mix Wizard: {str(e)}", "error")
            traceback.print_exc()

    def show_process_wizard(self):
        """Show the process wizard for channel processing"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the process wizard.")
                return
            
            # Track channels before opening wizard
            self._track_channels_before_wizard_open()
            
            # Import the window class directly since ProcessWizard has different structure
            from process_wizard_window import ProcessWizardWindow
            
            # Create the process wizard window (it creates its own manager internally)
            # Pass the currently selected file ID from the main window as the default
            self.process_wizard_window = ProcessWizardWindow(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self,
                default_file_id=self.selected_file_id  # Pass the currently selected file
            )
            
            # Connect to wizard close signal
            self.process_wizard_window.wizard_closed.connect(
                lambda: self._on_process_wizard_closed()
            )
            
            # Show the wizard window
            self.process_wizard_window.show()
            
        except Exception as e:
            self.log_message(f"Error opening Process Wizard: {str(e)}", "error")
            traceback.print_exc()
        
    def show_comparison_wizard(self):
        """Show the comparison wizard for data analysis"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the comparison wizard.")
                return
            
            # Create the comparison wizard manager
            self.comparison_wizard_manager = ComparisonWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self
            )
            
            # Show the wizard window
            self.comparison_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error opening Comparison Wizard: {str(e)}", "error")
            traceback.print_exc()
        
    def show_plot_wizard(self):
        """Show the plot wizard for custom plotting"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the plot wizard.")
                return
            
            # Create and show the plot wizard manager
            self.plot_wizard_manager = PlotWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Could be added later for inter-wizard communication
                parent=self
            )
            
            # Show the wizard window
            self.plot_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error launching Plot Wizard: {str(e)}", "error")
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to launch Plot Wizard: {str(e)}")
    
    def show_load_file_dialog(self):
        """Show file dialog to load data files"""
        from PySide6.QtWidgets import QFileDialog
        
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
            self.log_message(f"Loading {len(paths)} file(s) via file dialog...", "processing")
            
            # Use existing file handling logic
            self.handle_files_dropped(paths)
        else:
            self.log_message("File loading cancelled")
    
    def show_export_wizard(self):
        """Show the export wizard for saving data"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the export wizard.")
                return
            
            # Create the export wizard manager
            self.export_wizard_manager = ExportWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self
            )
            
            # Connect signals for export completion
            self.export_wizard_manager.export_complete.connect(self._on_export_completed)
            
            # Show the wizard window
            self.export_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error opening Export Wizard: {str(e)}", "error")
            traceback.print_exc()
    
    def _show_parse_wizard(self, file_id: str):
        """Show the parse wizard for manual parsing"""
        try:
            # Get the file object
            file_obj = self.file_manager.get_file(file_id)
            if not file_obj:
                QMessageBox.warning(self, "File Not Found", "Could not find the selected file.")
                return
            
            # Create the parse wizard manager
            self.parse_wizard_manager = ParseWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                parent=self
            )
            
            # Connect signals
            self.parse_wizard_manager.file_parsed.connect(self._on_file_parsed)
            self.parse_wizard_manager.parsing_complete.connect(self._on_parsing_complete)
            
            # Show the wizard with the file path
            self.parse_wizard_manager.show(file_obj.filepath)
            
            # Log success
            self.log_message(f"Parse Wizard opened for: {file_obj.filename}")
            
        except Exception as e:
            self.log_message(f"Error opening Parse Wizard: {str(e)}", "error")
            traceback.print_exc()
    
    def _on_file_parsed(self, file_id: str):
        """Handle when a file is parsed via parse wizard"""
        try:
            # Refresh the file table to show the new file
            self.refresh_file_table()
            
            # Select the newly parsed file
            files = self.file_manager.get_files_in_order()
            for row, file_obj in enumerate(files):
                if file_obj.file_id == file_id:
                    self.file_table.selectRow(row)
                    break
            
            # Refresh channel table
            self.refresh_channel_table()
            
            # Log success
            file_obj = self.file_manager.get_file(file_id)
            if file_obj:
                self.log_message(f"File parsed successfully: {file_obj.filename}", "success")
            
        except Exception as e:
            self.log_message(f"Error handling parsed file: {str(e)}", "error")
    
    def _on_parsing_complete(self, result: dict):
        """Handle when parsing is complete"""
        try:
            channels_count = result.get('channels_created', 0)
            rows_count = result.get('rows_parsed', 0)
            reparsed = result.get('reparsed', False)
            
            # Log completion with different message for re-parsing
            if reparsed:
                self.log_message(f"Re-parse complete: {channels_count} channels from {rows_count} rows (replaced existing data)", "success")
            else:
                self.log_message(f"Parse complete: {channels_count} channels from {rows_count} rows", "success")
            
        except Exception as e:
            self.log_message(f"Error handling parsing completion: {str(e)}", "error")

    def _on_export_completed(self, file_path: str):
        """Handle when export is completed"""
        self.log_message(f"Export completed: {file_path}", "success")
    
    def log_message(self, message, msg_type="info"):
        """Add a timestamped, color-coded message to the console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Define colors for different message types
        colors = {
            "success": "#228B22",      # Dark Green
            "suggestion": "#000000",   # Black  
            "error": "#FFA500",        # Dark Orange
            "warning": "#FFA500",      # Dark Orange
            "processing": "#696969"    # Grey
        }
        
        color = colors.get(msg_type, "#000000")  # Default to black
        
        # Format with HTML for color and bold
        formatted_message = f'<span style="color: {color}; font-weight: bold;">[{timestamp}] {message}</span>'
        self.console.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def get_system_status(self):
        """Get current system status for debugging"""
        return {
            'files': self.file_manager.get_stats(),
            'channels': self.channel_manager.get_stats(),
            'parser': self.auto_parser.get_stats()
        }
    
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
                self.log_message("Only CSV, TXT, TSV, and DAT files are supported", "warning")
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
                    self.log_message(f"Unsupported file type: {file_path.name} (only CSV, TXT, TSV, DAT supported)", "warning")
                else:
                    self.log_message(f"File not found: {file_path.name}", "error")
        
        if files:
            self.handle_files_dropped(files)
            event.acceptProposedAction()
        else:
            self.log_message("No valid files to process", "error")
            event.ignore()
    
    def _track_channels_before_wizard_open(self):
        """Track channels before wizard opens to calculate creation statistics"""
        # Get all current channels by file and type with detailed information
        self._pre_wizard_channels = {}
        for file_obj in self.file_manager.get_all_files():
            channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
            channel_details = []
            for ch in channels:
                channel_details.append({
                    'id': ch.channel_id,
                    'name': ch.ylabel or ch.legend_label or ch.channel_id,
                    'type': ch.type,
                    'dimensions': getattr(ch, 'ydata', None).shape if hasattr(ch, 'ydata') and ch.ydata is not None else None,
                    'step': getattr(ch, 'step', 0)
                })
            self._pre_wizard_channels[file_obj.file_id] = {
                'filename': file_obj.filename,
                'channels': channel_details
            }
    
    def _on_process_wizard_closed(self):
        """Handle when Process Wizard is closed"""
        if not hasattr(self, '_pre_wizard_channels'):
            return
        
        # Calculate new channels created with detailed information
        new_channels = []
        for file_obj in self.file_manager.get_all_files():
            current_channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
            
            # Get pre-wizard channel info for this file
            pre_wizard_channels = {}
            if file_obj.file_id in self._pre_wizard_channels:
                for ch_info in self._pre_wizard_channels[file_obj.file_id]['channels']:
                    pre_wizard_channels[ch_info['id']] = ch_info
            
            # Find new channels and their parent information
            for channel in current_channels:
                if channel.channel_id not in pre_wizard_channels:
                    # Find the parent channel (look in current channel manager, not just pre-wizard)
                    parent_info = None
                    if hasattr(channel, 'parent_ids') and channel.parent_ids:
                        # Get the most recent parent
                        for parent_id in channel.parent_ids:
                            parent_channel = self.channel_manager.get_channel(parent_id)
                            if parent_channel:
                                parent_info = {
                                    'name': parent_channel.ylabel or parent_channel.legend_label or parent_channel.channel_id
                                }
                                break
                    
                    # Get dimensions
                    dimensions = None
                    if hasattr(channel, 'ydata') and channel.ydata is not None:
                        if hasattr(channel.ydata, 'shape'):
                            dimensions = channel.ydata.shape
                        else:
                            dimensions = (len(channel.ydata),)
                    
                    new_channels.append({
                        'name': channel.ylabel or channel.legend_label or channel.channel_id,
                        'type': channel.type.value if hasattr(channel.type, 'value') else str(channel.type),
                        'filename': file_obj.filename,
                        'parent_name': parent_info['name'] if parent_info else 'Unknown',
                        'dimensions': dimensions,
                        'step': getattr(channel, 'step', 0)
                    })
        
        # Report statistics
        if new_channels:
            # Overall summary
            self.log_message(f"Process Wizard created {len(new_channels)} new channels:", "success")
            
            # Build tree structure by grouping channels by lineage and sorting by step
            trees = self._build_channel_tree(new_channels)
            
            # Display each tree
            for tree in trees:
                self._display_channel_tree(tree, 0)
        else:
            self.log_message("Process Wizard closed - no new channels created", "info")
        
        # Clean up
        del self._pre_wizard_channels
        
        # Refresh main window UI to show new channels
        self.refresh_channel_table()
        self.refresh_file_table()
        
        # Update plot with current channels
        if self.selected_file_id:
            channels = self._get_filtered_channels(self.selected_file_id)
            self.plot_manager.update_plot(channels)
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
        
        # Refresh main window UI to show new channels
        self.refresh_channel_table()
        self.refresh_file_table()
        
        # Update plot with current channels
        if self.selected_file_id:
            channels = self._get_filtered_channels(self.selected_file_id)
            self.plot_manager.update_plot(channels)
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
    
    def _build_channel_tree(self, new_channels):
        """Build tree structure from new channels showing processing lineage"""
        # Group channels by their root lineage and build hierarchy
        trees = {}
        
        for ch in new_channels:
            # Find the root of this processing chain
            root_name = self._find_root_channel_name(ch)
            
            if root_name not in trees:
                trees[root_name] = {
                    'name': root_name,
                    'filename': ch['filename'],
                    'children': [],
                    'step': 0,
                    'is_new': False  # Root is not new, it existed before
                }
            
            # Add this channel to the tree structure
            self._add_channel_to_tree(trees[root_name], ch)
        
        return list(trees.values())
    
    def _find_root_channel_name(self, channel_info):
        """Find the root (original) channel name for a processing chain"""
        # Walk up the parent chain until we find the root
        current_name = channel_info['name']
        seen_names = set([current_name])
        
        # Try to find the parent through the channel manager
        try:
            # Get the actual channel object to trace lineage
            for ch in self.channel_manager.get_all_channels():
                if (ch.ylabel or ch.legend_label or ch.channel_id) == current_name:
                    # Walk up the parent chain
                    while hasattr(ch, 'parent_ids') and ch.parent_ids:
                        parent_id = ch.parent_ids[0]  # Take first parent
                        parent_ch = self.channel_manager.get_channel(parent_id)
                        if parent_ch:
                            parent_name = parent_ch.ylabel or parent_ch.legend_label or parent_ch.channel_id
                            if parent_name in seen_names:  # Avoid circular references
                                break
                            seen_names.add(parent_name)
                            current_name = parent_name
                            ch = parent_ch
                        else:
                            break
                    break
        except:
            # Fallback: try to extract from the name pattern
            if ' - ' in channel_info['name']:
                # Assume the first part before " - " is the root
                current_name = channel_info['name'].split(' - ')[0]
        
        return current_name
    
    def _add_channel_to_tree(self, tree_node, channel_info):
        """Recursively add a channel to the tree structure"""
        # If this channel is a direct child of the current node
        if channel_info['parent_name'] == tree_node['name']:
            tree_node['children'].append({
                'name': channel_info['name'],
                'filename': channel_info['filename'],
                'parent_name': channel_info['parent_name'],
                'dimensions': channel_info['dimensions'],
                'step': channel_info['step'],
                'children': [],
                'is_new': True
            })
        else:
            # Check if it belongs to any of the children
            for child in tree_node['children']:
                self._add_channel_to_tree(child, channel_info)
    
    def _display_channel_tree(self, tree_node, indent_level):
        """Display a channel tree with proper indentation"""
        indent = "  " * indent_level
        
        if tree_node['is_new']:
            dim_str = f"{tree_node['dimensions']}" if tree_node['dimensions'] else "(?)"
            self.log_message(f"{indent}└─ {tree_node['name']} {dim_str}", "info")
        else:
            # This is the root node, show the filename context
            if indent_level == 0:
                self.log_message(f"{indent}{tree_node['filename']} / {tree_node['name']}:", "info")
        
        # Display children
        for child in tree_node['children']:
            self._display_channel_tree(child, indent_level + 1)
    
    def _on_mixer_wizard_closed(self):
        """Handle when Mix Wizard is closed"""
        if not hasattr(self, '_pre_wizard_channels'):
            return
        
        # Calculate new channels created
        new_channels = []
        for file_obj in self.file_manager.get_all_files():
            current_channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
            
            # Get pre-wizard channel info for this file
            pre_wizard_channels = {}
            if file_obj.file_id in self._pre_wizard_channels:
                for ch_info in self._pre_wizard_channels[file_obj.file_id]['channels']:
                    pre_wizard_channels[ch_info['id']] = ch_info
            
            # Find new channels and their parent information
            for channel in current_channels:
                if channel.channel_id not in pre_wizard_channels:
                    # Find the parent channel (look in current channel manager)
                    parent_info = None
                    if hasattr(channel, 'parent_ids') and channel.parent_ids:
                        # Get the most recent parent
                        for parent_id in channel.parent_ids:
                            parent_channel = self.channel_manager.get_channel(parent_id)
                            if parent_channel:
                                parent_info = {
                                    'name': parent_channel.ylabel or parent_channel.legend_label or parent_channel.channel_id
                                }
                                break
                    
                    # Get dimensions
                    dimensions = None
                    if hasattr(channel, 'ydata') and channel.ydata is not None:
                        if hasattr(channel.ydata, 'shape'):
                            dimensions = channel.ydata.shape
                        else:
                            dimensions = (len(channel.ydata),)
                    
                    new_channels.append({
                        'name': channel.ylabel or channel.legend_label or channel.channel_id,
                        'type': channel.type.value if hasattr(channel.type, 'value') else str(channel.type),
                        'filename': file_obj.filename,
                        'parent_name': parent_info['name'] if parent_info else 'Unknown',
                        'dimensions': dimensions,
                        'step': getattr(channel, 'step', 0)
                    })
        
        # Report statistics
        if new_channels:
            # Overall summary
            self.log_message(f"Mix Wizard created {len(new_channels)} new channels:", "success")
            
            # Build tree structure by grouping channels by lineage and sorting by step
            trees = self._build_channel_tree(new_channels)
            
            # Display each tree
            for tree in trees:
                self._display_channel_tree(tree, 0)
        else:
            self.log_message("Mix Wizard closed - no new channels created", "info")
        
        # Clean up
        del self._pre_wizard_channels
