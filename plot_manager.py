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
        
        # Map matplotlib line styles to Qt pen styles and handle custom styles
        style_map = {
            '-': Qt.PenStyle.SolidLine,
            '--': Qt.PenStyle.DashLine,
            '-.': Qt.PenStyle.DashDotLine,
            ':': Qt.PenStyle.DotLine,
            'None': Qt.PenStyle.NoPen,
        }
        
        # Handle custom line styles
        if self.line_style == "Solid (thick)":
            pen.setStyle(Qt.PenStyle.SolidLine)
            pen.setWidth(3)
        elif self.line_style == "Dashed (long)":
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
        elif self.line_style == "Dotted (sparse)":
            pen.setStyle(Qt.PenStyle.DotLine)
            pen.setWidth(1)
        elif self.line_style == "Dash-dot-dot":
            pen.setStyle(Qt.PenStyle.DashDotLine)
            pen.setWidth(2)
        elif self.line_style == "Dash-dash-dot":
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setWidth(2)
        else:
            pen.setStyle(style_map.get(self.line_style, Qt.PenStyle.SolidLine))
        
        painter.setPen(pen)
        
        # Calculate center position (always needed for markers)
        y_center = self.height() // 2
        
        # Draw the line across the widget (only if not "None")
        if self.line_style != "None":
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
            elif self.line_marker == 'v':  # Inverted Triangle
                points = [
                    (x_center, y_center + marker_size//2),
                    (x_center - marker_size//2, y_center - marker_size//2),
                    (x_center + marker_size//2, y_center - marker_size//2)
                ]
                triangle = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(triangle)
            elif self.line_marker == '<':  # Left Triangle
                points = [
                    (x_center - marker_size//2, y_center),
                    (x_center + marker_size//2, y_center - marker_size//2),
                    (x_center + marker_size//2, y_center + marker_size//2)
                ]
                triangle = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(triangle)
            elif self.line_marker == '>':  # Right Triangle
                points = [
                    (x_center + marker_size//2, y_center),
                    (x_center - marker_size//2, y_center - marker_size//2),
                    (x_center - marker_size//2, y_center + marker_size//2)
                ]
                triangle = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(triangle)
            elif self.line_marker == 'p':  # Pentagon
                # Draw a simple pentagon approximation (5-sided polygon)
                import math
                points = []
                for i in range(5):
                    angle = 2 * math.pi * i / 5 - math.pi / 2  # Start from top
                    x = x_center + marker_size//2 * math.cos(angle)
                    y = y_center + marker_size//2 * math.sin(angle)
                    points.append((int(x), int(y)))
                pentagon = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(pentagon)
            elif self.line_marker == 'h':  # Hexagon
                # Draw a simple hexagon (6-sided polygon)
                import math
                points = []
                for i in range(6):
                    angle = 2 * math.pi * i / 6
                    x = x_center + marker_size//2 * math.cos(angle)
                    y = y_center + marker_size//2 * math.sin(angle)
                    points.append((int(x), int(y)))
                hexagon = QPolygon([QPoint(x, y) for x, y in points])
                painter.drawPolygon(hexagon)


class PlotCanvas(FigureCanvas):
    """Custom matplotlib canvas for plotting channels"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Create only primary axis (no secondary axis)
        self.ax = self.fig.add_subplot(111)
        
        # Configure axes (no labels for preview)
        # Labels removed for cleaner preview interface
        
        # Track plotted lines
        self.plotted_lines = {}  # channel_id -> line object
        
        # Default colors and styles
        self.default_colors = itertools.cycle([
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ])
        self.default_styles = itertools.cycle(['-', '--', '-.', ':'])
        
        self.fig.canvas.draw()
    
    def clear_plot(self):
        """Clear all plotted data"""
        self.ax.clear()
        self.plotted_lines.clear()
        
        # No axis labels for preview
        
        self.fig.canvas.draw()
    
    def plot_channel(self, channel: Channel):
        """Plot a single channel"""
        if channel.xdata is None or channel.ydata is None:
            return None
        
        print(f"PlotCanvas.plot_channel called for {channel.channel_id}")
        print(f"  Data points: {len(channel.ydata) if channel.ydata is not None else 0}")
        print(f"  Y range: {np.min(channel.ydata):.6g} to {np.max(channel.ydata):.6g}" if channel.ydata is not None else "No data")
        
        # Always use the primary axis (no secondary axis)
        axis = self.ax
        
        # Get line properties
        color = channel.color or next(self.default_colors)
        style = channel.style if channel.style and channel.style != "None" else next(self.default_styles)
        # Convert 'None' to 'none' for matplotlib compatibility
        if style == "None":
            style = "none"
        marker = channel.marker if channel.marker != "None" else None
        label = channel.legend_label or channel.ylabel or "Unnamed"
        
        # Handle case where both line and marker are "None"
        if channel.style == "None" and (channel.marker == "None" or not channel.marker):
            # If both are None, use a default style to ensure visibility
            style = next(self.default_styles)
            marker = None
        
        # Determine linewidth based on style
        linewidth = 1.5  # Default
        if style == "Solid (thick)":
            linewidth = 3.0
            style = "-"
        elif style == "Dashed (long)":
            linewidth = 2.0
            style = "--"
        elif style == "Dotted (sparse)":
            linewidth = 1.5
            style = ":"
        elif style == "Dash-dot-dot":
            linewidth = 1.5
            style = "-."
        elif style == "Dash-dash-dot":
            linewidth = 1.5
            style = "--"
        
        # Plot the data
        line, = axis.plot(
            channel.xdata, 
            channel.ydata,
            color=color,
            linestyle=style,
            marker=marker,
            label=label,
            linewidth=linewidth,
            markersize=4
        )
        
        # Set z_order if specified
        z_order = getattr(channel, 'z_order', 0)
        if z_order > 0:
            line.set_zorder(z_order)
        
        # Store references
        self.plotted_lines[channel.channel_id] = line
        
        return line
    
    def remove_channel(self, channel_id: str):
        """Remove a channel from the plot"""
        if channel_id in self.plotted_lines:
            line = self.plotted_lines[channel_id]
            line.remove()
            del self.plotted_lines[channel_id]
    
    def update_channel_style(self, channel: Channel):
        """Update the visual style of a plotted channel"""
        if channel.channel_id not in self.plotted_lines:
            return
        
        line = self.plotted_lines[channel.channel_id]
        
        # Update line properties
        if channel.color:
            line.set_color(channel.color)
        
        # Handle line style and width
        if channel.style and channel.style not in ["None", "none"]:
            # Determine linewidth based on style
            linewidth = 1.5  # Default
            style = channel.style
            
            if style == "Solid (thick)":
                linewidth = 3.0
                style = "-"
            elif style == "Dashed (long)":
                linewidth = 2.0
                style = "--"
            elif style == "Dotted (sparse)":
                linewidth = 1.5
                style = ":"
            elif style == "Dash-dot-dot":
                linewidth = 1.5
                style = "-."
            elif style == "Dash-dash-dot":
                linewidth = 1.5
                style = "--"
            
            line.set_linestyle(style)
            line.set_linewidth(linewidth)
        else:
            line.set_linestyle('none')  # No line (matplotlib expects 'none')
        
        if channel.marker and channel.marker != "None":
            line.set_marker(channel.marker)
        else:
            line.set_marker('')
        
        # Update label
        label = channel.legend_label or channel.ylabel or "Unnamed"
        line.set_label(label)
        
        # Handle z_order change - bring line to front if needed
        z_order = getattr(channel, 'z_order', 0)
        if z_order > 0:
            line.set_zorder(z_order)
        else:
            line.set_zorder(0)
    
    def update_plot(self):
        """Refresh the plot display"""
        # Auto-scale axis
        self.ax.relim()
        self.ax.autoscale()
        
        # Add grid
        self.ax.grid(True, alpha=0.3)
        
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
        print(f"PlotManager.update_plot called with {len(channels)} channels")
        
        # Get visible channels
        visible_channels = [ch for ch in channels if ch.show]
        visible_ids = {ch.channel_id for ch in visible_channels}
        
        print(f"Visible channels: {len(visible_channels)}, IDs: {list(visible_ids)}")
        print(f"Currently plotted: {list(self.currently_plotted)}")
        
        # Debug: Print data ranges for visible channels
        for ch in visible_channels:
            if ch.ydata is not None:
                print(f"  Channel {ch.channel_id}: Y range {np.min(ch.ydata):.6g} to {np.max(ch.ydata):.6g}")
        
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
                print(f"Adding new channel {channel.channel_id} to plot")
                # New channel - assign default style if not set
                if not channel.color:
                    channel.color = next(self.color_cycle)
                if not channel.style:
                    channel.style = next(self.style_cycle)
                
                # Plot the channel
                self.plot_canvas.plot_channel(channel)
                self.currently_plotted.add(channel.channel_id)
            else:
                print(f"Replotting existing channel {channel.channel_id} (data may have changed)")
                # Update existing channel - need to replot if data changed
                self.plot_canvas.remove_channel(channel.channel_id)
                self.plot_canvas.plot_channel(channel)
        
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
            
            # Gear button
            gear_button = QPushButton("⚙")
            gear_button.setMaximumWidth(30)
            # Use default parameter to properly capture channel_id value (not reference)
            gear_button.clicked.connect(lambda checked=False, ch_id=channel.channel_id: self._handle_gear_button_clicked(ch_id))
            legend_table.setCellWidget(row, 2, gear_button)
    
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
        'Dotted': ':',
        'Solid (thick)': '-',
        'Dashed (long)': '--',
        'Dash-dot-dot': '-.',
        'Dotted (sparse)': ':',
        'Dash-dash-dot': '--'
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