from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox
from parse_wizard_window import ParseWizardWindow
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