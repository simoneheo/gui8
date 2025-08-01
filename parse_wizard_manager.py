from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox
from parse_wizard_window import ParseWizardWindow
from auto_parser import AutoParser
from pathlib import Path
import time
from typing import Dict, Optional, Any


class ParseWizardManager(QObject):
    """
    Manager for the parse wizard that handles:
    - Creating and managing the parse wizard window
    - Coordinating between file parsing and channel creation
    - Managing parse wizard state and results
    """
    
    parsing_complete = Signal(dict)
    state_changed = Signal(str)
    file_parsed = Signal(str)  # file_id
    
    def __init__(self, file_manager=None, channel_manager=None, parent=None):
        super().__init__(parent)
        
        # Store managers
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.parent_window = parent
        
        # Initialize autoparser for detection
        self.autoparser = AutoParser()
        
        # Initialize state
        self.window = None
        self.current_file_path = None
        
        # Parse statistics
        self._stats = {
            'total_manual_parses': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'channels_created': 0,
            'session_start': time.time()
        }
        
        # Validate managers
        if not self._validate_managers():
            raise ValueError("Required managers not available for ParseWizardManager")
    
    def _validate_managers(self) -> bool:
        """Validate that required managers are available"""
        if self.file_manager is None:
            print("[ParseWizardManager] Error: No file manager available")
            return False
        
        if self.channel_manager is None:
            print("[ParseWizardManager] Error: No channel manager available")
            return False
        
        return True
    
    def show(self, file_path: Optional[str] = None):
        """Show the parse wizard window"""
        try:
            # Close existing window if open
            if self.window:
                self.window.close()
            
            # Create new window
            self.window = ParseWizardWindow(
                file_path=file_path,
                file_manager=self.file_manager,
                channel_manager=self.channel_manager,
                parent=self.parent_window
            )
            
            # Connect signals
            self._connect_signals()
            
            # Apply autoparser detection if file is provided
            if file_path:
                self._apply_detected_settings(file_path)
            
            # Show window
            self.window.show()
            
            # Update state
            self.current_file_path = file_path
            self.state_changed.emit(f"Parse wizard opened for: {file_path or 'no file'}")
            
        except Exception as e:
            QMessageBox.critical(
                self.parent_window,
                "Parse Wizard Error",
                f"Failed to open parse wizard:\n{str(e)}"
            )
            self.state_changed.emit(f"Parse wizard error: {str(e)}")
    
    def close(self):
        """Close the parse wizard window"""
        if self.window:
            self.window.close()
            self.window = None
            self.current_file_path = None
            self.state_changed.emit("Parse wizard closed")
    
    def _connect_signals(self):
        """Connect signals from the parse wizard window"""
        if self.window:
            self.window.file_parsed.connect(self._on_file_parsed)
            self.window.parsing_complete.connect(self._on_parsing_complete)
    
    def _on_file_parsed(self, file_id: str):
        """Handle when a file is successfully parsed"""
        try:
            # Update statistics
            self._stats['successful_parses'] += 1
            self._stats['total_manual_parses'] += 1
            
            # Get file info
            file_obj = self.file_manager.get_file(file_id)
            if file_obj:
                channels = self.channel_manager.get_channels_by_file(file_id)
                self._stats['channels_created'] += len(channels)
                
                # Log success
                self._log_state_change(
                    f"Successfully parsed file: {file_obj.filename} "
                    f"({len(channels)} channels created)"
                )
                
                # Emit signal
                self.file_parsed.emit(file_id)
            else:
                self._log_state_change(f"Warning: Could not find parsed file with ID: {file_id}")
                
        except Exception as e:
            self._log_state_change(f"Error handling parsed file: {str(e)}")
    
    def _on_parsing_complete(self, result: Dict[str, Any]):
        """Handle when parsing is complete"""
        try:
            # Update statistics
            file_id = result.get('file_id')
            channels_count = result.get('channels', 0)
            rows_count = result.get('rows', 0)
            columns_count = result.get('columns', 0)
            
            # Create result summary
            result_summary = {
                'file_id': file_id,
                'channels_created': channels_count,
                'rows_parsed': rows_count,
                'columns_parsed': columns_count,
                'parse_method': 'manual',
                'timestamp': time.time()
            }
            
            # Log completion
            self._log_state_change(
                f"Manual parsing complete: {channels_count} channels, "
                f"{rows_count} rows, {columns_count} columns"
            )
            
            # Emit completion signal
            self.parsing_complete.emit(result_summary)
            
        except Exception as e:
            self._log_state_change(f"Error handling parsing completion: {str(e)}")
    
    def _log_state_change(self, message: str):
        """Log state changes"""
        print(f"[ParseWizardManager] {message}")
        self.state_changed.emit(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parse wizard statistics"""
        return {
            'total_manual_parses': self._stats['total_manual_parses'],
            'successful_parses': self._stats['successful_parses'],
            'failed_parses': self._stats['failed_parses'],
            'channels_created': self._stats['channels_created'],
            'success_rate': (
                self._stats['successful_parses'] / self._stats['total_manual_parses'] * 100
                if self._stats['total_manual_parses'] > 0 else 0
            ),
            'session_duration': time.time() - self._stats['session_start']
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self._stats = {
            'total_manual_parses': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'channels_created': 0,
            'session_start': time.time()
        }
        self._log_state_change("Statistics reset")
    
    def is_window_open(self) -> bool:
        """Check if the parse wizard window is currently open"""
        return self.window is not None and self.window.isVisible()
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the currently loaded file path"""
        return self.current_file_path
    
    def set_file_path(self, file_path: str):
        """Set the file path for parsing"""
        if self.window:
            self.window.set_file_path(file_path)
            self.current_file_path = file_path
            self._log_state_change(f"File path set to: {file_path}")
        else:
            self._log_state_change("Cannot set file path - no window open")
    
    def _detect_parsing_settings(self, file_path: str) -> Dict[str, Any]:
        """
        Use autoparser to detect delimiter and header row for a file.
        Returns detection results or defaults.
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                print(f"[ParseWizardManager] File not found: {file_path}")
                return self._get_default_settings()
            
            # Read file with encoding detection
            lines, encoding = self.autoparser._read_file_with_encoding(file_path_obj)
            if not lines:
                print(f"[ParseWizardManager] Could not read file: {file_path}")
                return self._get_default_settings()
            
            # Detect and skip metadata lines
            metadata_skip = self.autoparser._detect_metadata_lines(lines)
            data_lines = lines[metadata_skip:]
            
            if not data_lines:
                print(f"[ParseWizardManager] No data lines found after metadata: {file_path}")
                return self._get_default_settings()
            
            # Detect structure using autoparser
            structure_info = self.autoparser._detect_structure(data_lines)
            if not structure_info:
                print(f"[ParseWizardManager] Could not detect structure: {file_path}")
                return self._get_default_settings()
            
            # Extract detection results
            delimiter = structure_info.get('delimiter', ',')
            header_info = structure_info.get('header_info', {})
            has_header = header_info.get('has_header', False)
            data_start_row = header_info.get('data_start_row', 0)
            
            # Map delimiter to UI format
            delimiter_map = {
                ',': 'Comma (,)',
                '\t': 'Tab (\\t)',
                ';': 'Semicolon (;)',
                '|': 'Pipe (|)',
                ' ': 'Space',
                '': 'None'
            }
            ui_delimiter = delimiter_map.get(delimiter, 'Comma (,)')
            
            # Calculate header row position
            # data_start_row is the row AFTER the header in the data_lines
            # We need to convert this back to the actual header row position
            if has_header:
                # If data_start_row is 1, header was at row 0 in data_lines
                # If data_start_row is 2, header was at row 1 in data_lines
                header_row_in_data = data_start_row - 1
                # Add metadata skip to get the actual file row
                header_row = metadata_skip + header_row_in_data
            else:
                header_row = -1  # No header
            
            detection_result = {
                'delimiter': ui_delimiter,
                'header_row': header_row,
                'detection_success': True,
                'structure_score': structure_info.get('score', 0),
                'num_columns': structure_info.get('num_cols', 0),
                'consistency': structure_info.get('consistency', 0)
            }
            
            print(f"[ParseWizardManager] Detection successful: delimiter='{ui_delimiter}', header_row={header_row}")
            return detection_result
            
        except Exception as e:
            print(f"[ParseWizardManager] Detection failed: {str(e)}")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default parsing settings when detection fails"""
        return {
            'delimiter': 'Comma (,)',
            'header_row': 0,
            'detection_success': False,
            'structure_score': 0,
            'num_columns': 0,
            'consistency': 0
        }
    
    def _apply_detected_settings(self, file_path: str):
        """Apply detected parsing settings to the parse wizard window"""
        try:
            # Detect parsing settings using autoparser
            detection_result = self._detect_parsing_settings(file_path)
            
            if not self.window:
                print("[ParseWizardManager] No window available to apply settings")
                return
            
            # Apply delimiter setting
            if hasattr(self.window, 'delimiter_combo'):
                self.window.delimiter_combo.setCurrentText(detection_result['delimiter'])
                print(f"[ParseWizardManager] Applied delimiter: {detection_result['delimiter']}")
            
            # Apply header row setting
            if hasattr(self.window, 'header_row_spin'):
                self.window.header_row_spin.setValue(detection_result['header_row'])
                print(f"[ParseWizardManager] Applied header row: {detection_result['header_row']}")
            
            # Mark settings as applied to prevent auto-detection override
            self.window.mark_settings_applied()
            
            # Log detection results
            if detection_result['detection_success']:
                self._log_state_change(
                    f"Auto-detected settings: delimiter='{detection_result['delimiter']}', "
                    f"header_row={detection_result['header_row']}, "
                    f"score={detection_result['structure_score']:.3f}"
                )
            else:
                self._log_state_change(
                    f"Using default settings: delimiter='{detection_result['delimiter']}', "
                    f"header_row={detection_result['header_row']}"
                )
                
        except Exception as e:
            print(f"[ParseWizardManager] Error applying detected settings: {str(e)}")
            self._log_state_change(f"Error applying auto-detected settings: {str(e)}")