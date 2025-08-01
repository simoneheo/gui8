# main_window_manager.py
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from pathlib import Path
import traceback
from datetime import datetime

# Import our managers
from auto_parser import AutoParser
from file_manager import FileManager
from channel_manager import ChannelManager
from file import FileStatus
from plot_manager import PlotManager
from line_wizard import LineWizard
from metadata_wizard import MetadataWizard
from inspection_wizard import InspectionWizard
from transform_wizard import TransformWizard
from comparison_wizard_manager import ComparisonWizardManager
from export_wizard_manager import ExportWizardManager
from signal_mixer_wizard_manager import SignalMixerWizardManager
from process_wizard_manager import ProcessWizardManager
from plot_wizard_manager import PlotWizardManager
from parse_wizard_manager import ParseWizardManager
from console import ConsoleManager


class MainWindowManager(QObject):
    """Business logic manager for the main window"""
    
    # Signals for UI updates
    file_table_updated = Signal()
    channel_table_updated = Signal()
    plot_updated = Signal()
    console_message = Signal(str, str, str)  # message, msg_type, category
    file_selection_changed = Signal(str)  # file_id
    channel_visibility_toggled = Signal(str, bool)  # channel_id, visible
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Initialize managers
        self.auto_parser = AutoParser()
        self.file_manager = FileManager()
        self.channel_manager = ChannelManager()
        self.plot_manager = PlotManager(self.channel_manager)
        self.console_manager = ConsoleManager()
        
        # Track selected file for channel filtering
        self.selected_file_id = None
        
        # File size warning settings
        self.large_file_threshold_mb = 50  # MB threshold for large file warning
        self.show_large_file_warnings = True  # Can be disabled via Manual Parse
        
        # Wizard instances
        self.mixer_wizard_window = None
        self.process_wizard_window = None
        self.comparison_wizard_manager = None
        self.plot_wizard_manager = None
        self.export_wizard_manager = None
        self.parse_wizard_manager = None
        
        # Channel tracking for wizard statistics
        self._pre_wizard_channels = {}
    
    def get_plot_manager(self):
        """Get the plot manager instance"""
        return self.plot_manager
    
    def get_file_manager(self):
        """Get the file manager instance"""
        return self.file_manager
    
    def get_channel_manager(self):
        """Get the channel manager instance"""
        return self.channel_manager
    
    def get_auto_parser(self):
        """Get the auto parser instance"""
        return self.auto_parser
    
    def get_console_manager(self):
        """Get the console manager instance"""
        return self.console_manager
    
    def set_selected_file_id(self, file_id):
        """Set the selected file ID"""
        self.selected_file_id = file_id
        self.file_selection_changed.emit(file_id)
    
    def get_selected_file_id(self):
        """Get the selected file ID"""
        return self.selected_file_id
    
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
                    # Emit signal to refresh channel table with correct style information
                    self.channel_table_updated.emit()
                else:
                    self.log_message(f"Failed: {file_path.name} → {file_obj.state.error_message or 'No channels created'}", "error")
                    self.log_message(f"Auto-parsing failed. Please try manual parsing using the ✂️ icon in the file table to customize parsing settings", "suggestion")
                
            except Exception as e:
                self.log_message(f"Error processing {file_path.name}: {str(e)}", "error")
                traceback.print_exc()
        
        # Auto-select the most recently loaded file
        if newly_loaded_file_id is not None:
            self.set_selected_file_id(newly_loaded_file_id)
        
        # Summary message
        if successful_files > 0:
            self.log_message(f"Completed: {successful_files} files parsed, {total_channels} channels created", "success")
            self.log_message(f"If results need adjustment, use the ✂️ icon for manual parsing with custom settings", "suggestion")
            
            # Check for channels with N/A sampling rates and warn user
            self._check_and_warn_na_sampling_rates()
        else:
            self.log_message("No files were successfully processed", "warning")
    
    def get_files_in_order(self):
        """Get files in order for display"""
        return self.file_manager.get_files_in_order()
    
    def get_filtered_channels(self, file_id: str = None):
        """Get channels for the selected file with current filter applied"""
        from channel import SourceType
        
        if file_id is None:
            file_id = self.selected_file_id
        
        if not file_id:
            return []
        
        # Get all channels from the file
        all_channels = self.channel_manager.get_channels_by_file(file_id)
        
        # ALWAYS exclude comparison channels from main window plot
        # Comparison channels should only appear in the comparison wizard
        valid_channels = [ch for ch in all_channels 
        if not ch.type == SourceType.COMPARISON and not ch.type == SourceType.SPECTROGRAM]
        
        # Apply filter based on "Show all data types" toggle
        # Note: This will be controlled by the UI checkbox state
        # For now, return all valid channels - filtering will be applied in UI
        return valid_channels
    
    def should_disable_show_checkbox(self, channel):
        """Determine if the Show checkbox should be disabled for this channel"""
        from channel import SourceType
        
        # Disable for SPECTROGRAM channels
        if channel.type == SourceType.SPECTROGRAM:
            return True
        
        # Disable for channels with spectrogram-related tags
        spectrogram_tags = ['spectrogram', 'spectrum', 'stft', 'cwt', 'welch', 'fft']
        if any(tag.lower() in spectrogram_tags for tag in channel.tags):
            return True
        
        # Disable for comparison channels - they should be controlled from comparison wizard
        if channel.type == SourceType.COMPARISON:
            return True
        
        return False
    
    def toggle_channel_visibility(self, channel_id: str):
        """Toggle visibility of a channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Toggle visibility
            new_state = not channel.show
            self.channel_manager.set_channel_visibility(channel_id, new_state)
            self.channel_visibility_toggled.emit(channel_id, new_state)
    
    def show_channel_info(self, channel_id: str, parent_widget):
        """Show detailed information about a channel using the metadata wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the comprehensive metadata wizard
            wizard = MetadataWizard(channel, parent_widget, self.file_manager)
            wizard.exec()
    
    def inspect_channel_data(self, channel_id: str, parent_widget):
        """Open the data inspection wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data inspection wizard
            wizard = InspectionWizard(channel, parent_widget)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()
    
    def handle_channel_data_updated(self, channel_id: str):
        """Handle when channel data is updated via inspection/transform wizards"""
        print(f"Manager received data_updated signal for channel: {channel_id}")
        channel = self.channel_manager.get_channel(channel_id)
        if channel and self.selected_file_id:
            print(f"Updating plot for channel: {channel_id}")
            # Get filtered channels
            channels = self.get_filtered_channels()
            
            # Force a complete plot refresh by clearing and rebuilding
            self.plot_manager.plot_canvas.clear_plot()
            self.plot_manager.currently_plotted.clear()
            
            # Update plot
            self.plot_manager.update_plot(channels)
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
            
            print(f"Plot update completed for channel: {channel_id}")
    
    def transform_channel_data(self, channel_id: str, parent_widget):
        """Open the data transformation wizard for this channel"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open the data transformation wizard
            wizard = TransformWizard(channel, parent_widget)
            wizard.data_updated.connect(self.handle_channel_data_updated)
            wizard.exec()
    
    def delete_channel(self, channel_id: str, parent_widget):
        """Delete a channel with confirmation"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            reply = QMessageBox.question(
                parent_widget, 
                "Delete Channel", 
                f"Delete channel '{channel.ylabel}'?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.channel_manager.remove_channel(channel_id)
                self.channel_table_updated.emit()
    
    def show_file_info(self, file_id: str, parent_widget):
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
            last_modified_str = datetime.fromtimestamp(file_obj.last_modified).strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate values for display
            processing_time = f"{(parse_result.parse_time_ms/1000):.3f}s" if parse_result and parse_result.parse_time_ms else 'Unknown'
            parse_method = parse_result.strategy_used.value if parse_result and parse_result.strategy_used else 'Unknown'
            data_rows = parse_result.rows_parsed if parse_result else 'Unknown'
            metadata_lines = parse_result.metadata_lines_skipped if parse_result else 'Unknown'
            encoding = parse_result.encoding_detected if parse_result else 'Unknown'
            
            info_text = f"""
File Information:
• Name: {file_obj.filename}
• Size: {size_str} ({file_obj.filesize:,} bytes)
• Last Modified: {last_modified_str}
• Path: {file_obj.filepath}

Parse Statistics:
• Processing Time: {processing_time}
• Data Rows: {data_rows}
• Encoding: {encoding}
            """
            
            # Create custom message box without icon
            msg_box = QMessageBox(parent_widget)
            msg_box.setWindowTitle(f"File Info - {file_obj.filename}")
            msg_box.setText(info_text.strip())
            msg_box.setIcon(QMessageBox.NoIcon)  # Remove the rocket icon
            msg_box.exec()
    
    def show_raw_file_preview(self, file_id: str, parent_widget):
        """Show a preview of the raw file content before parsing"""
        file_obj = self.file_manager.get_file(file_id)
        if not file_obj:
            return
            
        try:
            # Create preview dialog
            from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLabel, QDialogButtonBox
            from PySide6.QtGui import QFont
            
            dialog = QDialog(parent_widget)
            dialog.setWindowTitle(f"Raw File Preview - {file_obj.filename}")
            dialog.setMinimumSize(800, 600)
            dialog.resize(1000, 700)
            
            layout = QVBoxLayout(dialog)
            
            # File info header
            file_size_mb = file_obj.filesize / (1024 * 1024)
            size_str = f"{file_size_mb:.2f} MB" if file_size_mb >= 1 else f"{file_obj.filesize:,} bytes"
            
            info_label = QLabel(f"File: {file_obj.filename} | Size: {size_str}")
            info_label.setStyleSheet("font-weight: bold; padding: 5px; background-color: #f0f0f0; color: #333333;")
            layout.addWidget(info_label)
            
            # Read file content with encoding detection
            file_content, detected_encoding = self._read_file_safely(file_obj.filepath)
            
            # Encoding info
            encoding_label = QLabel(f"Detected Encoding: {detected_encoding} | Showing first 500 lines")
            encoding_label.setStyleSheet("color: #444444; font-size: 11px; padding: 2px;")
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
            QMessageBox.critical(parent_widget, "Preview Error", f"Could not preview file:\n{str(e)}")
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
            from PySide6.QtWidgets import QCheckBox
            
            # Create custom dialog with checkbox
            dialog = QMessageBox()
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
                try:
                    # Create a temporary file object to open Manual Parse
                    from file import File
                    temp_file = File(file_path)
                    self.file_manager.add_file(temp_file)
                    self.show_parse_wizard(temp_file.file_id)
                    return False
                except Exception as e:
                    self.log_message(f"Error creating manual parse for large file: {str(e)}", "error")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                self.log_message(f"User cancelled loading large file: {file_path.name}", "info")
                return False
                
        except Exception as e:
            self.log_message(f"Error showing large file warning: {str(e)}", "error")
            return True  # Default to continue if error occurs
    
    def refresh_file_to_original(self, file_id: str, parent_widget):
        """Refresh all channels in a file back to their original state"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            reply = QMessageBox.question(
                parent_widget, 
                "Refresh File", 
                f"Refresh file '{file_obj.filename}' and reset all channels to original state?\n\n"
                "This will undo all transformations and edits for this file's channels.",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                try:
                    # Get existing channels to preserve styling information
                    existing_channels = self.channel_manager.get_channels_by_file(file_id)
                    styling_map = {}
                    
                    # Create a mapping of channel styling by ylabel (which should be consistent)
                    for channel in existing_channels:
                        if channel.ylabel:
                            styling_map[channel.ylabel] = {
                                'color': channel.color,
                                'style': channel.style,
                                'marker': channel.marker,
                                'show': channel.show,
                                'yaxis': channel.yaxis,
                                'legend_label': channel.legend_label,
                                'z_order': getattr(channel, 'z_order', 0)
                            }
                    
                    # Re-parse the file to get original data
                    self.log_message(f"Re-parsing: {file_obj.filename}", "processing")
                    new_file_obj, new_channels = self.auto_parser.parse_file(file_obj.filepath)
                    
                    if new_channels:
                        # Apply preserved styling to new channels
                        for channel in new_channels:
                            if channel.ylabel and channel.ylabel in styling_map:
                                preserved_style = styling_map[channel.ylabel]
                                channel.color = preserved_style['color']
                                channel.style = preserved_style['style']
                                channel.marker = preserved_style['marker']
                                channel.show = preserved_style['show']
                                channel.yaxis = preserved_style['yaxis']
                                channel.legend_label = preserved_style['legend_label']
                                channel.z_order = preserved_style['z_order']
                        
                        # Remove old channels
                        removed_count = self.channel_manager.remove_channels_by_file(file_id)
                        
                        # Add new channels with original data and preserved styling
                        added_count = self.channel_manager.add_channels(new_channels)
                        
                        # Update file object
                        file_obj.state = new_file_obj.state
                        file_obj.modified_at = datetime.now()
                        
                        # Emit signal to update channel table
                        self.channel_table_updated.emit()
                        
                        self.log_message(f"Successfully refreshed '{file_obj.filename}': Removed {removed_count} modified channels, restored {added_count} original channels, preserved styling for {len(styling_map)} channels", "success")
                        
                        # Check for channels with N/A sampling rates and warn user
                        self._check_and_warn_na_sampling_rates()
                    else:
                        self.log_message(f"Failed to refresh: {file_obj.filename}", "error")
                        QMessageBox.warning(
                            parent_widget, 
                            "Refresh Failed", 
                            f"Failed to re-parse file '{file_obj.filename}':\n{new_file_obj.state.error_message or 'Unknown error'}"
                        )
                        
                except Exception as e:
                    self.log_message(f"Error refreshing {file_obj.filename}: {str(e)}", "error")
                    QMessageBox.critical(
                        parent_widget, 
                        "Refresh Error", 
                        f"An error occurred while refreshing the file:\n{str(e)}"
                    )
    
    def delete_file(self, file_id: str, parent_widget):
        """Delete a file and all its channels with confirmation"""
        file_obj = self.file_manager.get_file(file_id)
        if file_obj:
            channels = self.channel_manager.get_channels_by_file(file_id)
            reply = QMessageBox.question(
                parent_widget, 
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
                    self.set_selected_file_id(None)
                
                self.log_message(f"Deleted: {file_obj.filename} ({removed_channels} channels)", "success")
    
    def handle_legend_gear_clicked(self, channel_id: str, parent_widget):
        """Handle gear button clicks in legend table"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel:
            # Open line wizard
            wizard = LineWizard(channel, parent_widget)
            wizard.channel_updated.connect(self.handle_channel_updated)
            wizard.exec()
    
    def handle_channel_updated(self, channel_id: str):
        """Handle when a channel is updated via line wizard"""
        channel = self.channel_manager.get_channel(channel_id)
        if channel and self.selected_file_id:
            # Get filtered channels
            channels = self.get_filtered_channels()
            
            # Update plot canvas with filtered channels
            self.plot_manager.update_plot(channels)
            
            # Force plot canvas to redraw immediately
            self.plot_manager.plot_canvas.fig.canvas.draw()
            self.plot_manager.plot_canvas.fig.canvas.flush_events()
    
    def show_mix_wizard(self, parent_widget):
        """Show the Mix Wizard dialog."""
        try:
            # Check if we have data to work with
            if not self.file_manager.get_all_files():
                QMessageBox.information(parent_widget, "No Data", "Please load some files first before using the mixer wizard.")
                return
            
            # Track channels before opening wizard
            self._track_channels_before_wizard_open()
            
            # Import and create the signal mixer wizard window directly
            from signal_mixer_wizard_window import SignalMixerWizardWindow
            
            self.mixer_wizard_window = SignalMixerWizardWindow(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=parent_widget
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

    def show_process_wizard(self, parent_widget):
        """Show the process wizard for channel processing"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(parent_widget, "No Data", "Please load some files first before using the process wizard.")
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
                parent=parent_widget,
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
        
    def show_comparison_wizard(self, parent_widget):
        """Show the comparison wizard for data analysis"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(parent_widget, "No Data", "Please load some files first before using the comparison wizard.")
                return
            
            # Create the comparison wizard manager
            self.comparison_wizard_manager = ComparisonWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=parent_widget,
                selected_file_id=self.selected_file_id  # Pass the currently selected file
            )
            
            # Connect signals from the comparison wizard manager
            self.comparison_wizard_manager.comparison_completed.connect(self._on_comparison_completed)
            self.comparison_wizard_manager.wizard_closed.connect(self._on_comparison_wizard_closed)
            
            # Show the wizard window
            self.comparison_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error opening Comparison Wizard: {str(e)}", "error")
            traceback.print_exc()
        
    def show_plot_wizard(self, parent_widget):
        """Show the plot wizard for custom plotting"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(parent_widget, "No Data", "Please load some files first before using the plot wizard.")
                return
            
            # Create and show the plot wizard manager
            self.plot_wizard_manager = PlotWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Could be added later for inter-wizard communication
                parent=parent_widget
            )
            
            # Show the wizard window
            self.plot_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error launching Plot Wizard: {str(e)}", "error")
            traceback.print_exc()
            QMessageBox.critical(parent_widget, "Error", f"Failed to launch Plot Wizard: {str(e)}")
    
    def show_export_wizard(self, parent_widget):
        """Show the export wizard for saving data"""
        try:
            if not self.file_manager.get_all_files():
                QMessageBox.information(parent_widget, "No Data", "Please load some files first before using the export wizard.")
                return
            
            # Create the export wizard manager
            self.export_wizard_manager = ExportWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                signal_bus=None,  # Add signal bus if available
                parent=parent_widget
            )
            
            # Connect signals for export completion
            self.export_wizard_manager.export_complete.connect(self._on_export_completed)
            
            # Show the wizard window
            self.export_wizard_manager.show()
            
        except Exception as e:
            self.log_message(f"Error opening Export Wizard: {str(e)}", "error")
            traceback.print_exc()
    
    def show_parse_wizard(self, file_id: str, parent_widget):
        """Show the parse wizard for manual parsing"""
        try:
            # Get the file object
            file_obj = self.file_manager.get_file(file_id)
            if not file_obj:
                QMessageBox.warning(parent_widget, "File Not Found", "Could not find the selected file.")
                return
            
            # Create the parse wizard manager
            self.parse_wizard_manager = ParseWizardManager(
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                parent=parent_widget
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
            # Select the newly parsed file
            self.set_selected_file_id(file_id)
            
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
            
            # Emit signal to update channel table
            self.channel_table_updated.emit()
            
            # Check for channels with N/A sampling rates and warn user
            self._check_and_warn_na_sampling_rates()
            
        except Exception as e:
            self.log_message(f"Error handling parsing completion: {str(e)}", "error")

    def _check_and_warn_na_sampling_rates(self):
        """Check for channels with N/A sampling rates and warn user"""
        try:
            na_channels = []
            
            # Check all files and their channels
            for file_obj in self.file_manager.get_all_files():
                channels = self.channel_manager.get_channels_by_file(file_obj.file_id)
                
                for channel in channels:
                    # Get sampling rate description
                    sampling_rate = channel.get_sampling_rate_description()
                    
                    # Check if sampling rate is N/A
                    if sampling_rate == "N/A":
                        na_channels.append({
                            'file': file_obj.filename,
                            'channel': channel.legend_label or channel.ylabel or channel.channel_id
                        })
            
            # If we found channels with N/A sampling rates, warn the user
            if na_channels:
                if len(na_channels) == 1:
                    channel_info = na_channels[0]
                    self.log_message(
                        f"Detected sampling rate is N/A for file '{channel_info['file']}' / channel '{channel_info['channel']}'. "
                        f"Try resampling using Process Wizard.", 
                        "warning"
                    )
                else:
                    # Multiple channels with N/A sampling rates
                    file_names = list(set(ch['file'] for ch in na_channels))
                    if len(file_names) == 1:
                        self.log_message(
                            f"Detected sampling rate is N/A for {len(na_channels)} channels in file '{file_names[0]}'. "
                            f"Try resampling using Process Wizard.", 
                            "warning"
                        )
                    else:
                        self.log_message(
                            f"Detected sampling rate is N/A for {len(na_channels)} channels across {len(file_names)} files. "
                            f"Try resampling using Process Wizard.", 
                            "warning"
                        )
                        
        except Exception as e:
            # Don't let this error break the parsing completion
            self.log_message(f"Error checking sampling rates: {str(e)}", "error")

    def _on_export_completed(self, file_path: str):
        """Handle when export is completed"""
        self.log_message(f"Export completed: {file_path}", "success")
    
    def log_message(self, message, msg_type="info", category="MAIN"):
        """Add a timestamped, color-coded message to the console"""
        # Log to console manager
        self.console_manager.log_message(message, msg_type, category)
        # Also emit signal for backward compatibility
        self.console_message.emit(message, msg_type, category)
    
    def get_system_status(self):
        """Get current system status for debugging"""
        return {
            'files': self.file_manager.get_stats(),
            'channels': self.channel_manager.get_stats(),
            'parser': self.auto_parser.get_stats()
        }
    
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
            self.log_message(f"Process Wizard created {len(new_channels)} new channels", "success")
            # Emit signal to refresh channel table with new channels
            self.channel_table_updated.emit()
        else:
            self.log_message("Process Wizard closed - no new channels created", "info")
        
        # Clean up
        del self._pre_wizard_channels
    
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
            self.log_message(f"Mix Wizard created {len(new_channels)} new channels", "success")
            # Emit signal to refresh channel table with new channels
            self.channel_table_updated.emit()
        else:
            self.log_message("Mix Wizard closed - no new channels created", "info")
        
        # Clean up
        del self._pre_wizard_channels

    def _on_comparison_completed(self, result):
        """Handle when comparison wizard completes an action"""
        try:
            result_type = result.get('type', 'unknown')
            
            if result_type == 'plot_generated':
                plot_data = result.get('data', {})
                self.log_message("Comparison plot generated successfully", "success")
                
        except Exception as e:
            self.log_message(f"Error handling comparison completion: {str(e)}", "error")
    
    def _on_comparison_wizard_closed(self):
        """Handle when comparison wizard is closed"""
        try:
            self.log_message("Comparison Wizard closed - no new channels created", "info")
            
            # Clean up the manager reference
            if hasattr(self, 'comparison_wizard_manager'):
                del self.comparison_wizard_manager
                
        except Exception as e:
            self.log_message(f"Error handling comparison wizard close: {str(e)}", "error") 