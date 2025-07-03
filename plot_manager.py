import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
from typing import List, Dict, Optional
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QLabel
from PySide6.QtCore import Signal, Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QPixmap, QColor, QPolygon
import itertools

from channel import Channel
from channel_manager import ChannelManager


class StylePreviewWidget(QLabel):
    """Widget that shows a visual preview of line style, color, and marker"""
    
    def __init__(self, color='#1f77b4', style='-', marker=None, parent=None):
        super().__init__(parent)
        self.line_color = color
        self.line_style = style
        self.line_marker = marker
        self.setFixedSize(80, 20)
        self.setStyleSheet("border: 1px solid lightgray;")
        
    def update_style(self, color, style, marker):
        """Update the style and repaint"""
        self.line_color = color
        self.line_style = style
        self.line_marker = marker
        self.update()
        
    def paintEvent(self, event):
        """Custom paint event to draw the line preview"""
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Convert hex color to QColor
        color = QColor(self.line_color)
        
        # Create pen with style
        pen = QPen(color, 2)
        
        # Map matplotlib line styles to Qt pen styles
        style_map = {
            '-': Qt.PenStyle.SolidLine,
            '--': Qt.PenStyle.DashLine,
            '-.': Qt.PenStyle.DashDotLine,
            ':': Qt.PenStyle.DotLine,
            'None': Qt.PenStyle.NoPen,
        }
        pen.setStyle(style_map.get(self.line_style, Qt.PenStyle.SolidLine))
        
        painter.setPen(pen)
        
        # Draw the line across the widget (only if not "None")
        if self.line_style != "None":
            y_center = self.height() // 2
            margin = 5
            painter.drawLine(margin, y_center, self.width() - margin, y_center)
        
        # Draw marker if specified
        if self.line_marker and self.line_marker != "None":
            # Draw marker at center
            x_center = self.width() // 2
            marker_size = 4
            
            painter.setPen(QPen(color, 1))
            painter.setBrush(color)
            
            if self.line_marker == 'o':  # Circle
                painter.drawEllipse(x_center - marker_size//2, y_center - marker_size//2, marker_size, marker_size)
            elif self.line_marker == 's':  # Square
                painter.drawRect(x_center - marker_size//2, y_center - marker_size//2, marker_size, marker_size)
            elif self.line_marker == '^':  # Triangle
                points = [
                    (x_center, y_center - marker_size//2),
                    (x_center - marker_size//2, y_center + marker_size//2),
                    (x_center + marker_size//2, y_center + marker_size//2)
                ]
                triangle = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(triangle)
            elif self.line_marker == 'D':  # Diamond
                points = [
                    (x_center, y_center - marker_size//2),
                    (x_center + marker_size//2, y_center),
                    (x_center, y_center + marker_size//2),
                    (x_center - marker_size//2, y_center)
                ]
                diamond = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(diamond)
            elif self.line_marker == '+':  # Plus
                painter.drawLine(x_center - marker_size//2, y_center, x_center + marker_size//2, y_center)
                painter.drawLine(x_center, y_center - marker_size//2, x_center, y_center + marker_size//2)
            elif self.line_marker == 'x':  # X
                painter.drawLine(x_center - marker_size//2, y_center - marker_size//2, 
                               x_center + marker_size//2, y_center + marker_size//2)
                painter.drawLine(x_center - marker_size//2, y_center + marker_size//2, 
                               x_center + marker_size//2, y_center - marker_size//2)
            elif self.line_marker == '*':  # Star
                # Draw a simple star (plus + x)
                painter.drawLine(x_center - marker_size//2, y_center, x_center + marker_size//2, y_center)
                painter.drawLine(x_center, y_center - marker_size//2, x_center, y_center + marker_size//2)
                painter.drawLine(x_center - marker_size//2, y_center - marker_size//2, 
                               x_center + marker_size//2, y_center + marker_size//2)
                painter.drawLine(x_center - marker_size//2, y_center + marker_size//2, 
                               x_center + marker_size//2, y_center - marker_size//2)


class PlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for plotting channels"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create primary and secondary axes
        self.ax_left = self.fig.add_subplot(111)
        self.ax_right = self.ax_left.twinx()
        
        # Configure axes (no labels for preview)
        # Labels removed for cleaner preview interface
        
        # Track plotted lines
        self.plotted_lines = {}  # channel_id -> line object
        self.channel_axes = {}   # channel_id -> axis (left/right)
        
        # Default colors and styles
        self.default_colors = itertools.cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
        self.default_styles = itertools.cycle(['-', '--', '-.', ':'])
        
        self.fig.canvas.draw()
    
    def clear_plot(self):
        """Clear all plotted data"""
        self.ax_left.clear()
        self.ax_right.clear()
        self.plotted_lines.clear()
        self.channel_axes.clear()
        
        # No axis labels for preview
        
        self.fig.canvas.draw()
    
    def plot_channel(self, channel: Channel):
        """Plot a single channel"""
        if channel.xdata is None or channel.ydata is None:
            return None
        
        # Determine which axis to use
        axis = self.ax_left if channel.yaxis == "y-left" else self.ax_right
        
        # Get line properties
        color = channel.color or next(self.default_colors)
        style = channel.style if channel.style and channel.style != "None" else next(self.default_styles)
        marker = channel.marker if channel.marker != "None" else None
        label = channel.legend_label or channel.ylabel or "Unnamed"
        
        # Handle case where both line and marker are "None"
        if channel.style == "None" and (channel.marker == "None" or not channel.marker):
            # If both are None, use a default style to ensure visibility
            style = next(self.default_styles)
            marker = None
        
        # Plot the data
        line, = axis.plot(
            channel.xdata, 
            channel.ydata,
            color=color,
            linestyle=style,
            marker=marker,
            label=label,
            linewidth=1.5,
            markersize=4
        )
        
        # Set z_order if specified
        z_order = getattr(channel, 'z_order', 0)
        if z_order > 0:
            line.set_zorder(z_order)
        
        # Store references
        self.plotted_lines[channel.channel_id] = line
        self.channel_axes[channel.channel_id] = channel.yaxis
        
        return line
    
    def remove_channel(self, channel_id: str):
        """Remove a channel from the plot"""
        if channel_id in self.plotted_lines:
            line = self.plotted_lines[channel_id]
            line.remove()
            del self.plotted_lines[channel_id]
            del self.channel_axes[channel_id]
    
    def update_channel_style(self, channel: Channel):
        """Update the visual style of a plotted channel"""
        if channel.channel_id not in self.plotted_lines:
            return
        
        line = self.plotted_lines[channel.channel_id]
        
        # Update line properties
        if channel.color:
            line.set_color(channel.color)
        if channel.style and channel.style != "None":
            line.set_linestyle(channel.style)
        else:
            line.set_linestyle('')  # No line
        if channel.marker and channel.marker != "None":
            line.set_marker(channel.marker)
        else:
            line.set_marker('')
        
        # Update label
        label = channel.legend_label or channel.ylabel or "Unnamed"
        line.set_label(label)
        
        # Handle axis change
        old_axis = self.channel_axes.get(channel.channel_id)
        new_axis = channel.yaxis
        
        if old_axis != new_axis:
            # Remove from old axis and re-plot on new axis
            self.remove_channel(channel.channel_id)
            self.plot_channel(channel)
        
        # Handle z_order change - bring line to front if needed
        z_order = getattr(channel, 'z_order', 0)
        if z_order > 0:
            line.set_zorder(z_order)
        else:
            line.set_zorder(0)
    
    def update_plot(self):
        """Refresh the plot display"""
        # Auto-scale axes
        self.ax_left.relim()
        self.ax_left.autoscale()
        self.ax_right.relim()
        self.ax_right.autoscale()
        
        # Add grid
        self.ax_left.grid(True, alpha=0.3)
        
        self.fig.canvas.draw()


class PlotManager(QWidget):
    """
    Manages plotting of channels and legend table updates
    """
    
    # Signals
    legend_item_gear_clicked = Signal(str)  # channel_id
    
    def __init__(self, channel_manager: ChannelManager):
        super().__init__()
        self.channel_manager = channel_manager
        
        # Create plot canvas
        self.plot_canvas = PlotCanvas()
        
        # Create navigation toolbar
        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        
        # Track plot state
        self.currently_plotted = set()  # channel_ids currently plotted
        
        # Setup layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)  # Add toolbar first
        layout.addWidget(self.plot_canvas)
        
        # Initialize with default colors for new channels
        self.color_cycle = itertools.cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
        self.style_cycle = itertools.cycle(['-', '--', '-.', ':'])
    
    def update_plot(self, channels: List[Channel]):
        """Update the plot with visible channels"""
        # Get visible channels
        visible_channels = [ch for ch in channels if ch.show]
        visible_ids = {ch.channel_id for ch in visible_channels}
        
        # Remove channels that are no longer visible
        for channel_id in list(self.currently_plotted):
            if channel_id not in visible_ids:
                self.plot_canvas.remove_channel(channel_id)
                self.currently_plotted.remove(channel_id)
        
        # Sort channels by z_order (higher z_order plotted last = on top)
        visible_channels.sort(key=lambda ch: getattr(ch, 'z_order', 0))
        
        # Add or update visible channels
        for channel in visible_channels:
            if channel.channel_id not in self.currently_plotted:
                # New channel - assign default style if not set
                if not channel.color:
                    channel.color = next(self.color_cycle)
                if not channel.style:
                    channel.style = next(self.style_cycle)
                
                # Plot the channel
                self.plot_canvas.plot_channel(channel)
                self.currently_plotted.add(channel.channel_id)
            else:
                # Update existing channel style
                self.plot_canvas.update_channel_style(channel)
        
        # Refresh the display
        self.plot_canvas.update_plot()
    
    def _handle_gear_button_clicked(self, channel_id):
        """Handle gear button clicks from legend table"""
        self.legend_item_gear_clicked.emit(channel_id)
    
    def update_legend_table(self, legend_table: QTableWidget, channels: List[Channel]):
        """Update the legend table with visible channels"""
        visible_channels = [ch for ch in channels if ch.show]
        legend_table.setRowCount(len(visible_channels))
        
        for row, channel in enumerate(visible_channels):
            # Legend Label
            label = channel.legend_label or channel.ylabel or "Unnamed"
            legend_table.setItem(row, 0, QTableWidgetItem(label))
            
            # Style - Use visual preview widget
            style_widget = StylePreviewWidget(
                color=channel.color or '#1f77b4',
                style=channel.style if channel.style and channel.style != "None" else '-',
                marker=channel.marker if channel.marker != "None" else None
            )
            legend_table.setCellWidget(row, 1, style_widget)
            
            # Axis
            axis_text = "Left" if channel.yaxis == "y-left" else "Right"
            legend_table.setItem(row, 2, QTableWidgetItem(axis_text))
            
            # Gear button
            gear_button = QPushButton("⚙")
            gear_button.setMaximumWidth(30)
            # Use default parameter to properly capture channel_id value (not reference)
            gear_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._handle_gear_button_clicked(ch_id))
            legend_table.setCellWidget(row, 3, gear_button)
    
    def refresh_plot_for_file(self, file_id: str):
        """Refresh plot showing only channels from specified file"""
        if file_id:
            channels = self.channel_manager.get_channels_by_file(file_id)
            self.update_plot(channels)
        else:
            # No file selected, clear plot
            self.plot_canvas.clear_plot()
            self.currently_plotted.clear()
    
    def clear_plot(self):
        """Clear the entire plot"""
        self.plot_canvas.clear_plot()
        self.currently_plotted.clear()
    
    def export_plot(self, filename: str, dpi: int = 300):
        """Export the current plot to file"""
        self.plot_canvas.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def get_plot_widget(self) -> QWidget:
        """Get the plot widget for embedding in main window"""
        return self
    
    def channel_updated(self, channel: Channel):
        """Called when a channel's properties are updated"""
        if channel.channel_id in self.currently_plotted:
            self.plot_canvas.update_channel_style(channel)
            self.plot_canvas.update_plot()


# Predefined style options for the line wizard
PLOT_STYLES = {
    'Line Styles': {
        'None': 'None',
        'Solid': '-',
        'Dashed': '--', 
        'Dash-dot': '-.',
        'Dotted': ':'
    },
    'Markers': {
        'None': 'None',
        'Circle': 'o',
        'Square': 's',
        'Triangle': '^',
        'Diamond': 'D',
        'Plus': '+',
        'X': 'x',
        'Star': '*'
    },
    'Colors': {
        'Blue': '#1f77b4',
        'Orange': '#ff7f0e', 
        'Green': '#2ca02c',
        'Red': '#d62728',
        'Purple': '#9467bd',
        'Brown': '#8c564b',
        'Pink': '#e377c2',
        'Gray': '#7f7f7f',
        'Olive': '#bcbd22',
        'Cyan': '#17becf',
        'Black': '#000000'
    }
} 