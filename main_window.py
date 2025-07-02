# main_window.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QFrame, QCheckBox, QSplitter, QTextEdit, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPalette
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
        
        # Setup UI
        self.init_ui()
        
        # Connect signals
        self.connect_signals()
        
        # Initialize UI state
        self.log_message("System initialized. Ready for file processing.")
    
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
        
        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["File Name", "Status", "Actions"])
        self.file_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        channel_manager_layout.addWidget(channel_manager_label)
        channel_manager_layout.addWidget(self.selected_file_label)
        channel_manager_layout.addStretch()
        channel_manager_layout.addWidget(self.channel_count_label)
        tables_layout.addLayout(channel_manager_layout)
        
        self.channel_table = QTableWidget(0, 3)
        self.channel_table.setHorizontalHeaderLabels(["Channel Name", "Style", "Actions"])
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
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
        self.log_message(f"Processing {len(file_paths)} dropped file(s)...")
        
        successful_files = 0
        total_channels = 0
        newly_loaded_file_id = None  # Track the most recently loaded file
        
        for file_path in file_paths:
            try:
                # Check if file already exists
                if self.file_manager.has_file_path(file_path):
                    self.log_message(f"⚠️ File already loaded: {file_path.name}")
                    continue
                
                # Parse the file
                self.log_message(f"📖 Parsing: {file_path.name}")
                file_obj, channels = self.auto_parser.parse_file(file_path)
                
                # Add to managers
                self.file_manager.add_file(file_obj)
                
                if channels:
                    added_count = self.channel_manager.add_channels(channels)
                    successful_files += 1
                    total_channels += added_count
                    newly_loaded_file_id = file_obj.file_id  # Update to most recent successful file
                    self.log_message(f"✅ Success: {file_path.name} → {added_count} channels")
                else:
                    self.log_message(f"❌ Failed: {file_path.name} → {file_obj.state.error_message or 'No channels created'}")
                
            except Exception as e:
                self.log_message(f"❌ Error processing {file_path.name}: {str(e)}")
                traceback.print_exc()
        
        # Update UI
        self.refresh_file_table()
        
        # Auto-select the most recently loaded file
        if newly_loaded_file_id is not None:
            file_obj = self.file_manager.get_file(newly_loaded_file_id)
            if file_obj:
                self.selected_file_id = newly_loaded_file_id
                self.selected_file_label.setText(f"📁 {file_obj.filename}")
                self.log_message(f"📂 Auto-selected newly loaded file: {file_obj.filename}")
                
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
            self.log_message(f"🎉 Completed: {successful_files} files parsed, {total_channels} channels created")
        else:
            self.log_message("⚠️ No files were successfully processed")
    
    def refresh_file_table(self):
        """Refresh the file table with current data"""
        files = self.file_manager.get_files_in_order()
        self.file_table.setRowCount(len(files))
        
        for row, file_obj in enumerate(files):
            # File name
            self.file_table.setItem(row, 0, QTableWidgetItem(file_obj.filename))
            
            # Status with color coding
            status_item = QTableWidgetItem(file_obj.state.status.value)
            if file_obj.state.status == FileStatus.PARSED:
                status_item.setBackground(Qt.green)
            elif file_obj.state.status == FileStatus.ERROR:
                status_item.setBackground(Qt.red)
            elif file_obj.state.status == FileStatus.PROCESSED:
                status_item.setBackground(Qt.blue)
            else:
                status_item.setBackground(Qt.yellow)
            self.file_table.setItem(row, 1, status_item)
            
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
            
            self.file_table.setCellWidget(row, 2, actions_widget)
        
        # Update count label
        self.file_count_label.setText(f"Files: {len(files)}")
    
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
                self.log_message(f"📂 Selected file: {selected_file.filename}")
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
        
        # Get channels only from selected file
        channels = self.channel_manager.get_channels_by_file(self.selected_file_id)
        self.channel_table.setRowCount(len(channels))
        
        for row, channel in enumerate(channels):
            # Column 0: Channel Name (legend label)
            channel_name = channel.legend_label or channel.ylabel or "Unnamed"
            self.channel_table.setItem(row, 0, QTableWidgetItem(channel_name))
            
            # Column 1: Style (visual preview widget)
            style_widget = StylePreviewWidget(
                color=channel.color or '#1f77b4',
                style=channel.style or '-',
                marker=channel.marker if channel.marker != "None" else None
            )
            self.channel_table.setCellWidget(row, 1, style_widget)
            
            # Column 2: Actions (plot, gear, exclamation, data inspector, transform, trashcan)
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 2, 2, 2)
            actions_layout.setSpacing(2)
            
            # Plot toggle button (colored/greyed based on visibility)
            plot_button = QPushButton("📊")
            plot_button.setMaximumWidth(25)
            plot_button.setMaximumHeight(25)
            if channel.show:
                plot_button.setStyleSheet("QPushButton { background-color: #e8f5e8; }")  # Green background when visible
            else:
                plot_button.setStyleSheet("QPushButton { background-color: #f5f5f5; color: #aaa; }")  # Greyed when hidden
            plot_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._toggle_channel_visibility(ch_id))
            actions_layout.addWidget(plot_button)
            
            # Gear button (settings)
            gear_button = QPushButton("⚙")
            gear_button.setMaximumWidth(25)
            gear_button.setMaximumHeight(25)
            gear_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self.handle_legend_gear_clicked(ch_id))
            actions_layout.addWidget(gear_button)
            
            # Exclamation button (warnings/info)
            info_button = QPushButton("❗")
            info_button.setMaximumWidth(25)
            info_button.setMaximumHeight(25)
            info_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._show_channel_info(ch_id))
            actions_layout.addWidget(info_button)
            
            # Magnifying glass button (inspect data)
            zoom_button = QPushButton("🔍")
            zoom_button.setMaximumWidth(25)
            zoom_button.setMaximumHeight(25)
            zoom_button.setToolTip("Inspect and edit channel data")
            zoom_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._inspect_channel_data(ch_id))
            actions_layout.addWidget(zoom_button)
            
            # Tool button (transform data)
            tool_button = QPushButton("🔨")
            tool_button.setMaximumWidth(25)
            tool_button.setMaximumHeight(25)
            tool_button.setToolTip("Transform channel data with math expressions")
            tool_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._transform_channel_data(ch_id))
            actions_layout.addWidget(tool_button)
            
            # Trash button (delete)
            delete_button = QPushButton("🗑️")
            delete_button.setMaximumWidth(25)
            delete_button.setMaximumHeight(25)
            delete_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._delete_channel(ch_id))
            actions_layout.addWidget(delete_button)
            
            self.channel_table.setCellWidget(row, 2, actions_widget)
        
        # Update count label - only count visible channels from this file
        visible_channels = [ch for ch in channels if ch.show]
        self.channel_count_label.setText(f"Channels: {len(channels)} ({len(visible_channels)} visible)")
        
        # Update plot
        self.plot_manager.update_plot(channels)
    
    def handle_file_table_click(self, row, column):
        """Handle clicks in the file table (now handled by action buttons)"""
        # File actions are now handled by individual buttons in the actions column
        pass
    
    def handle_channel_table_click(self, row, column):
        """Handle clicks in the channel table (now mostly handled by action buttons)"""
        # Most actions are now handled by individual buttons in the actions column
        pass
    
    def _toggle_channel_visibility(self, channel_id: str):
        """Toggle visibility of a channel and update UI"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Toggle visibility
            new_state = not channel.show
            self.channel_manager.set_channel_visibility(channel_id, new_state)
            self.log_message(f"📊 {'Showing' if new_state else 'Hiding'}: {channel.ylabel}")
            
            # Refresh the channel table to update button appearance
            self.refresh_channel_table()
    
    def _show_channel_info(self, channel_id: str):
        """Show detailed information about a channel using the metadata wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the comprehensive metadata wizard
            wizard = MetadataWizard(channel, self)
            wizard.exec()
            self.log_message(f"ℹ️ Viewed metadata for: {channel.ylabel}")
    
    def _inspect_channel_data(self, channel_id: str):
        """Open the data inspection wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data inspection wizard
            wizard = InspectionWizard(channel, self)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()
            self.log_message(f"🔍 Inspected data: {channel.ylabel}")
    
    def handle_channel_data_updated(self, channel_id: str):
        """Handle when channel data is updated via inspection/transform wizards"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel and self.selected_file_id:
            channels = self.channel_manager.get_channels_by_file(self.selected_file_id)
            
            # Refresh all UI components immediately
            self.refresh_channel_table()  # Update channel table with new statistics
            self.plot_manager.update_plot(channels)  # Update plot canvas
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
            
            self.log_message(f"📊 Updated channel data: {channel.ylabel}")
    
    def _transform_channel_data(self, channel_id: str):
        """Open the data transformation wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data transformation wizard
            wizard = TransformWizard(channel, self)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()
            self.log_message(f"🔨 Transformed data: {channel.ylabel}")
    
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
                self.log_message(f"🗑️ Deleted channel: {channel.ylabel}")
    
    def _show_file_info(self, file_id: str):
        """Show detailed information about a file"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            channels = self.channel_manager.get_channels_by_file(file_id)
            
            # Calculate file statistics
            size_mb = file_obj.filesize / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1 else f"{file_obj.filesize} B"
            
            info_text = f"""
File Information:
• Name: {file_obj.filename}
• File ID: {file_id}
• Size: {size_str} ({file_obj.filesize:,} bytes)
• Status: {file_obj.state.status.value}
• Channels: {len(channels)}
• Created: {file_obj.created_at.strftime("%Y-%m-%d %H:%M:%S") if file_obj.created_at else 'Unknown'}
• Modified: {file_obj.modified_at.strftime("%Y-%m-%d %H:%M:%S") if file_obj.modified_at else 'Unknown'}
• Path: {file_obj.filepath}

Parse Statistics:
• Processing Time: {file_obj.state.parsing_time:.3f}s
• Parse Method: {file_obj.state.parse_method or 'Unknown'}
• Data Rows: {file_obj.state.data_rows or 'Unknown'}
• Header Rows: {file_obj.state.header_rows or 'Unknown'}

Channel Summary:
• Total Channels: {len(channels)}
• Visible Channels: {len([ch for ch in channels if ch.show])}
• Modified Channels: {len([ch for ch in channels if ch.modified_at and ch.modified_at > ch.created_at])}

Error Information:
{file_obj.state.error_message or 'No errors'}
            """
            
            QMessageBox.information(self, f"File Info - {file_obj.filename}", info_text.strip())
            self.log_message(f"ℹ️ Viewed file info: {file_obj.filename}")
    
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
                    self.log_message(f"🔄 Re-parsing: {file_obj.filename}")
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
                        
                        self.log_message(f"✅ Refreshed: {file_obj.filename} → {added_count} channels restored")
                        
                        QMessageBox.information(
                            self, 
                            "File Refreshed", 
                            f"Successfully refreshed '{file_obj.filename}':\n"
                            f"• Removed {removed_count} modified channels\n"
                            f"• Restored {added_count} original channels"
                        )
                    else:
                        self.log_message(f"❌ Failed to refresh: {file_obj.filename}")
                        QMessageBox.warning(
                            self, 
                            "Refresh Failed", 
                            f"Failed to re-parse file '{file_obj.filename}':\n{new_file_obj.state.error_message or 'Unknown error'}"
                        )
                        
                except Exception as e:
                    self.log_message(f"❌ Error refreshing {file_obj.filename}: {str(e)}")
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
                self.log_message(f"🗑️ Deleted: {file_obj.filename} ({removed_channels} channels)")
    
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
            channels = self.channel_manager.get_channels_by_file(self.selected_file_id)
            
            # Refresh all UI components immediately
            self.refresh_channel_table()  # Update channel table with new properties
            self.plot_manager.update_plot(channels)  # Update plot canvas
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
            
            self.log_message(f"🎨 Updated line properties: {channel.ylabel}")

    def show_mix_wizard(self):
        """Show the Mix Wizard dialog."""
        try:
            # Check if we have data to work with
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the mixer wizard.")
                return
            
            # Import and create the signal mixer wizard window directly
            # (This wizard has different architecture - window creates manager internally)
            from signal_mixer_wizard_window import SignalMixerWizardWindow
            
            self.mixer_wizard_window = SignalMixerWizardWindow(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self
            )
            
            # Show the wizard window
            self.mixer_wizard_window.show()
            
            # Log success
            self.log_message("🎛️ Mix Wizard opened")
            
        except Exception as e:
            self.log_message(f"❌ Error opening Mix Wizard: {str(e)}")
            traceback.print_exc()

    def show_process_wizard(self):
        """Show the process wizard for channel processing"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(self, "No Data", "Please load some files first before using the process wizard.")
                return
            
            # Import the window class directly since ProcessWizard has different structure
            from process_wizard_window import ProcessWizardWindow
            
            # Create the process wizard window (it creates its own manager internally)
            self.process_wizard_window = ProcessWizardWindow(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=self
            )
            
            # Show the wizard window
            self.process_wizard_window.show()
            
            # Log success
            self.log_message("📊 Process Wizard opened")
            
        except Exception as e:
            self.log_message(f"❌ Error opening Process Wizard: {str(e)}")
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
            
            # Log success
            self.log_message("📈 Comparison Wizard opened")
            
        except Exception as e:
            self.log_message(f"❌ Error opening Comparison Wizard: {str(e)}")
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
            
            # Log the successful launch
            self.log_message("📊 Plot Wizard launched successfully")
            
        except Exception as e:
            self.log_message(f"❌ Error launching Plot Wizard: {str(e)}")
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
            self.log_message(f"📁 Loading {len(paths)} file(s) via file dialog...")
            
            # Use existing file handling logic
            self.handle_files_dropped(paths)
        else:
            self.log_message("📁 File loading cancelled")
    
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
            
            # Log success
            self.log_message("💾 Export Wizard opened")
            
        except Exception as e:
            self.log_message(f"❌ Error opening Export Wizard: {str(e)}")
            traceback.print_exc()

    def _on_export_completed(self, file_path: str):
        """Handle when export is completed"""
        self.log_message(f"✅ Export completed: {file_path}")
    
    def log_message(self, message):
        """Add a timestamped message to the console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
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
                self.log_message("📥 Drop files here to load them...")
            else:
                event.ignore()
                # Visual feedback for unsupported files
                self.file_table.setStyleSheet("""
                    QTableWidget { 
                        border: 3px dashed #f44336; 
                        background-color: rgba(244, 67, 54, 0.1);
                    }
                """)
                self.log_message("⚠️ Only CSV, TXT, TSV, and DAT files are supported")
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
                    self.log_message(f"⚠️ Unsupported file type: {file_path.name} (only CSV, TXT, TSV, DAT supported)")
                else:
                    self.log_message(f"❌ File not found: {file_path.name}")
        
        if files:
            self.handle_files_dropped(files)
            event.acceptProposedAction()
        else:
            self.log_message("❌ No valid files to process")
            event.ignore()
