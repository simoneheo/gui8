"""
Consolidated Floating Console System

This module provides a unified console interface that replaces all embedded
consoles throughout the application with a single floating/dockable console.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Set
from enum import Enum

from PySide6.QtCore import (
    QObject, Signal, Qt
)
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTextEdit, QApplication
)
from PySide6.QtGui import QFont, QTextCursor


class MessageLevel(Enum):
    """Message severity levels"""
    DEBUG = "debug"
    INFO = "info" 
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class MessageCategory(Enum):
    """Message source categories"""
    MAIN = "MAIN"
    PROCESS = "PROCESS"
    MIXER = "MIXER"
    COMPARE = "COMPARE"
    PLOT = "PLOT"
    EXPORT = "EXPORT"
    PARSE = "PARSE"
    SYSTEM = "SYSTEM"


class ConsoleMessage:
    """Structured console message"""
    
    def __init__(self, text: str, level: MessageLevel = MessageLevel.INFO, 
                 category: MessageCategory = MessageCategory.SYSTEM, timestamp: datetime = None):
        self.text = text
        self.level = level
        self.category = category
        self.timestamp = timestamp or datetime.now()
        
    def to_html(self) -> str:
        """Convert message to styled HTML"""
        # Color scheme for categories
        category_colors = {
            MessageCategory.MAIN: "#2E7D32",      # Green
            MessageCategory.PROCESS: "#1976D2",   # Blue  
            MessageCategory.MIXER: "#7B1FA2",     # Purple
            MessageCategory.COMPARE: "#F57C00",   # Orange
            MessageCategory.PLOT: "#C2185B",      # Pink
            MessageCategory.EXPORT: "#5D4037",    # Brown
            MessageCategory.PARSE: "#455A64",     # Blue Grey
            MessageCategory.SYSTEM: "#424242",    # Grey
        }
        
        # Color scheme for levels
        level_colors = {
            MessageLevel.DEBUG: "#9E9E9E",        # Grey
            MessageLevel.INFO: "#212529",         # Dark
            MessageLevel.WARNING: "#F57C00",      # Orange
            MessageLevel.ERROR: "#D32F2F",        # Red
            MessageLevel.SUCCESS: "#388E3C",      # Green (unused - SUCCESS uses category color)
        }
        
        timestamp_str = self.timestamp.strftime("%H:%M:%S")
        category_color = category_colors.get(self.category, "#424242")
        
        # For SUCCESS messages, use the category color to match the tag
        if self.level == MessageLevel.SUCCESS:
            text_color = category_color
        else:
            text_color = level_colors.get(self.level, "#212529")
        
        # Build HTML with proper escaping
        html = f'''<span style="color: #666666;">[{timestamp_str}]</span> '''
        html += f'''<span style="color: {category_color}; font-weight: bold;">[{self.category.value}]</span> '''
        html += f'''<span style="color: {text_color};">{self.text}</span>'''
        
        return html


class ConsoleManager(QObject):
    """Singleton console manager for routing and formatting messages"""
    
    _instance = None
    message_received = Signal(ConsoleMessage)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        super().__init__()
        self._initialized = True
        
        self._message_history: List[ConsoleMessage] = []
        self._max_history = 10000  # Prevent memory bloat
        
    def log_message(self, text: str, level: str = "info", category: str = "SYSTEM"):
        """
        Log a message to the console
        
        Args:
            text: Message text
            level: Message level (debug, info, warning, error, success)
            category: Message category (MAIN, PROCESS, MIXER, etc.)
        """
        try:
            msg_level = MessageLevel(level.lower())
        except ValueError:
            msg_level = MessageLevel.INFO
            
        try:
            msg_category = MessageCategory(category.upper())
        except ValueError:
            msg_category = MessageCategory.SYSTEM
            
        message = ConsoleMessage(text, msg_level, msg_category)
        
        # Add to history (with size limit)
        self._message_history.append(message)
        if len(self._message_history) > self._max_history:
            self._message_history = self._message_history[-self._max_history//2:]
            
        # Emit to console
        self.message_received.emit(message)
    
    def get_message_history(self) -> List[ConsoleMessage]:
        """Get complete message history"""
        return self._message_history.copy()
    
    def clear_history(self):
        """Clear message history"""
        self._message_history.clear()


class FloatingConsole(QMainWindow):
    """Main floating console window"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Console")
        self.setMinimumSize(400, 300)  # Much smaller window
        
        # Position window on right side of screen
        self._position_window_right()
        
        # Console manager
        self.console_manager = ConsoleManager()
        
        # UI setup (simplified)
        self._setup_ui()
        self._connect_signals()
        
        # Track activity for show_and_focus method
        self._last_activity = datetime.now()
        
    def _setup_ui(self):
        """Setup simplified main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Simple layout with just the console text area
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)  # Smaller margins
        
        # Console text area (simplified)
        self.console_display = QTextEdit()
        self.console_display.setReadOnly(True)
        self.console_display.setFont(QFont("Monaco", 10))  # Smaller font, better default
        self.console_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 2px;
                padding: 4px;
                color: #212529;
            }
        """)
        
        layout.addWidget(self.console_display)
        
    def _connect_signals(self):
        """Connect simplified signals"""
        self.console_manager.message_received.connect(self._on_message_received)
        
    def _on_message_received(self, message: ConsoleMessage):
        """Handle new console message (simplified - always show all messages)"""
        self._last_activity = datetime.now()
        
        # Always add message to display
        self._add_message_to_display(message)
        
    def _add_message_to_display(self, message: ConsoleMessage):
        """Add message to console display (simplified with always auto-scroll)"""
        html = message.to_html()
        
        # Insert at end
        cursor = self.console_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html + "<br>")
        
        # Always auto-scroll to bottom
        scrollbar = self.console_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def closeEvent(self, event):
        """Handle close event (simplified)"""
        event.accept()
        
    def show_and_focus(self):
        """Show console and bring to focus"""
        # Reposition to right side (in case user moved it)
        self._position_window_right()
        
        # Make sure window is shown first
        self.show()
        
        # Force window to front with multiple methods for cross-platform compatibility
        self.raise_()
        self.activateWindow()
        
        # Additional methods to ensure it comes to front
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        
        # Process events to ensure UI updates
        QApplication.processEvents()
        
        # Final raise and activate after processing
        self.raise_()
        self.activateWindow()
        
        self._last_activity = datetime.now()
    
    def _position_window_right(self):
        """Position console window on the right side of the screen (smaller size)"""
        # Get primary screen geometry
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Calculate dimensions for right half but smaller height
        window_width = screen_geometry.width() // 5
        window_height = min(500, screen_geometry.height() // 2)  # Smaller height
        
        # Position on right side, top area
        self.setGeometry(
            screen_geometry.x() + window_width,  # x position (right half)
            screen_geometry.y(),                 # y position (top edge)  
            window_width,                        # width (half screen)
            window_height                        # height (smaller)
        )


# Global console instance
_console_instance: Optional[FloatingConsole] = None

def get_console() -> FloatingConsole:
    """Get or create global console instance"""
    global _console_instance
    if _console_instance is None:
        _console_instance = FloatingConsole()
    return _console_instance

def log_message(text: str, level: str = "info", category: str = "SYSTEM"):
    """Convenience function for logging messages"""
    manager = ConsoleManager()
    manager.log_message(text, level, category)

def show_console():
    """Show the floating console"""
    console = get_console()
    console.show_and_focus()

def hide_console():
    """Hide the floating console"""
    console = get_console()
    console.hide()


if __name__ == "__main__":
    # Test the console
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Create and show console
    console = get_console()
    console.show()
    
    # Test messages
    log_message("Application started", "info", "MAIN")
    log_message("Loading configuration...", "info", "SYSTEM")
    log_message("File processed successfully", "success", "PROCESS")
    log_message("Warning: Large file detected", "warning", "MAIN")
    log_message("Mixing operation completed", "info", "MIXER")
    log_message("Error: Invalid parameter", "error", "COMPARE")
    
    sys.exit(app.exec_())