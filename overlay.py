from typing import Optional, Dict, Any, List, Union

class Overlay:
    # Default style settings for each overlay type
    DEFAULT_STYLES = {
        'line': {
            'color': '#808080',  # Grey
            'linewidth': 2,
            'linestyle': ':',  # Dotted
            'alpha': 0.8,
            'label': None
        },
        
        'hline': {
            'color': '#808080',  # Grey
            'linewidth': 2,
            'linestyle': ':',  # Dotted
            'alpha': 0.8,
            'label': None
        },
        
        'vline': {
            'color': '#808080',  # Grey
            'linewidth': 2,
            'linestyle': ':',  # Dotted
            'alpha': 0.8,
            'label': None
        },

        'text': {
            'color': '#000000',  # Black
            'fontsize': 12,
            'alpha': 1.0,
            'position': (0.02, 0.98),  # (x, y) coordinates in axes fraction (0-1)
            'bbox': {'boxstyle': 'round,pad=0.5', 'facecolor': 'white', 'alpha': 0.8},
            'verticalalignment': 'top',
            'horizontalalignment': 'left',
            'transform': 'axes',  # 'axes' for fraction coordinates, 'data' for data coordinates
            'label': None
        },
        'fill': {
            'color': '#ff69b4',  # Hot pink (darker)
            'alpha': 0.1,  # More transparent
            'edgecolor': '#ff69b4',
            'linewidth': 1,
            'label': None
        },
        'marker': {
            'color': '#1f77b4',  # Blue
            'marker': 'o',  # Circle
            'markersize': 6,
            'alpha': 0.8,
            'label': None
        },
        'shading': {
            'color': '#ffc0cb',  # Pink
            'alpha': 0.1,  # More transparent
            'hatch': None,
            'label': None
        }
    }

    def __init__(self,
                 id: str,
                 name: str,
                 type: str,
                 data: Optional[Dict[str, Any]] = None,
                 style: Optional[Dict[str, Any]] = None,
                 channel: Union[str, Any] = None,
                 show: bool = True,
                 tags: Optional[List[str]] = None):
        self.id = id
        self.name = name
        self.type = type  # e.g., 'line', 'fill', 'text', 'marker'
        self.data = data or {}  # Store overlay data (x, y, text, y_upper, y_lower, etc.)
        
        # Merge provided style with defaults for this type
        default_style = self.get_default_style(type)
        if style:
            # Merge provided style with defaults, with provided values taking precedence
            merged_style = default_style.copy()
            merged_style.update(style)
            self.style = merged_style
        else:
            self.style = default_style.copy()
            
        self.channel = channel  # Channel name/id or Channel object
        self.show = show
        self.tags = tags if tags is not None else []
        
        # Custom text tracking for text overlays
        self._original_text = None
        self._custom_text = None
        self._is_custom_text = False
        
    @property
    def display_text(self):
        """Get the text that should be displayed."""
        if self.type != 'text':
            return None
            
        if self._is_custom_text and self._custom_text is not None:
            return self._custom_text
        return self._get_current_auto_text()
    
    @property
    def is_custom_text(self):
        """Check if the overlay is using custom text."""
        return self._is_custom_text
    
    @property
    def original_text(self):
        """Get the original auto-generated text."""
        return self._original_text
    
    def set_custom_text(self, text: str):
        """Set user-edited text for this overlay."""
        if self.type != 'text':
            print(f"[Overlay] Warning: Cannot set custom text for non-text overlay type: {self.type}")
            return
            
        # Store original text if not already stored
        if self._original_text is None:
            self._original_text = self._get_current_auto_text()
        
        # Set custom text
        self._custom_text = text
        self._is_custom_text = True
        
        # Update the data dict for rendering
        self.data['text'] = text
        self.data['text_lines'] = text.split('\n') if text else []
        
        print(f"[Overlay] Set custom text for {self.id}: {len(text)} characters")
    
    def reset_to_auto_text(self):
        """Reset to auto-generated text."""
        if self.type != 'text':
            print(f"[Overlay] Warning: Cannot reset text for non-text overlay type: {self.type}")
            return
            
        self._is_custom_text = False
        self._custom_text = None
        
        # Restore original data if available
        if self._original_text:
            self.data['text'] = self._original_text
            self.data['text_lines'] = self._original_text.split('\n') if self._original_text else []
            print(f"[Overlay] Reset {self.id} to auto-generated text")
        else:
            print(f"[Overlay] Warning: No original text stored for {self.id}")
    
    def _get_current_auto_text(self):
        """Get current auto-generated text from data."""
        if 'text' in self.data and self.data['text']:
            return self.data['text']
        elif 'text_lines' in self.data and self.data['text_lines']:
            return '\n'.join(str(line) for line in self.data['text_lines'] if line)
        return ""

    def apply_to_plot(self, ax):
        """Draw the overlay on a matplotlib axes based on type."""
        if not self.show:
            return
            
        if self.type == 'line':
            self._render_line(ax)
        elif self.type == 'text':
            self._render_text(ax)
        elif self.type == 'fill':
            self._render_fill(ax)
        elif self.type == 'marker':
            self._render_marker(ax)
        elif self.type == 'shading':
            self._render_shading(ax)
        else:
            print(f"[Overlay] Unknown overlay type: {self.type}")
    
    def _render_line(self, ax):
        """Render a line overlay."""
        x_data = self.data.get('x')
        y_data = self.data.get('y')
        
        if x_data is not None and y_data is not None:
            ax.plot(x_data, y_data, label=self.name, **self.style)
    
    def _render_text(self, ax):
        """Render a text overlay."""
        text = self.display_text
        
        # Get position from style, with fallback to data or defaults
        position = self.style.get('position', (0.02, 0.98))
        if isinstance(position, str):
            # Handle string positions like 'top-right', 'bottom-left', etc.
            position = self._parse_position_string(position)
        
        x, y = position
        
        # Get transform from style or data
        transform_str = self.style.get('transform', self.data.get('transform', 'axes'))
        if transform_str == 'axes':
            transform = ax.transAxes
        elif transform_str == 'data':
            transform = ax.transData
        else:
            transform = ax.transAxes  # Default
        
        if text:
            # Extract text-specific style properties
            text_style = {k: v for k, v in self.style.items() 
                         if k in ['color', 'fontsize', 'alpha', 'bbox', 'verticalalignment', 'horizontalalignment']}
            ax.text(x, y, text, label=self.name, transform=transform, **text_style)
    
    def _parse_position_string(self, position_str: str) -> tuple:
        """Parse string position descriptions into (x, y) coordinates."""
        position_map = {
            'top-left': (0.02, 0.98),
            'top-right': (0.98, 0.98),
            'bottom-left': (0.02, 0.02),
            'bottom-right': (0.98, 0.02),
            'center': (0.5, 0.5),
            'top-center': (0.5, 0.98),
            'bottom-center': (0.5, 0.02),
            'left-center': (0.02, 0.5),
            'right-center': (0.98, 0.5)
        }
        return position_map.get(position_str.lower(), (0.02, 0.98))
    
    def _render_fill(self, ax):
        """Render a fill overlay (confidence bands, etc.)."""
        x_data = self.data.get('x')
        y_lower = self.data.get('y_lower')
        y_upper = self.data.get('y_upper')
        
        if x_data is not None and y_lower is not None and y_upper is not None:
            ax.fill_between(x_data, y_lower, y_upper, label=self.name, **self.style)
    
    def _render_marker(self, ax):
        """Render a marker overlay."""
        x_data = self.data.get('x')
        y_data = self.data.get('y')
        
        if x_data is not None and y_data is not None:
            ax.scatter(x_data, y_data, label=self.name, **self.style)
    
    def _render_shading(self, ax):
        """Render a shading overlay."""
        x_data = self.data.get('x')
        y_data = self.data.get('y')
        
        if x_data is not None and y_data is not None:
            ax.fill_between(x_data, y_data, alpha=self.style.get('alpha', 0.1), 
                          color=self.style.get('color', '#ffc0cb'), **self.style)

    def update_style(self, new_style: Dict[str, Any]):
        """Update the style dictionary."""
        self.style.update(new_style)

    def toggle_show(self):
        """Toggle the visibility of the overlay."""
        self.show = not self.show

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the overlay to a dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'data': self.data,
            'style': self.style,
            'channel': self.channel,
            'show': self.show,
            'tags': self.tags,
            # Custom text state
            '_original_text': self._original_text,
            '_custom_text': self._custom_text,
            '_is_custom_text': self._is_custom_text
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Overlay':
        """Deserialize an overlay from a dictionary."""
        overlay = cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            data=data.get('data', {}),
            style=data['style'],
            channel=data['channel'],
            show=data.get('show', True),
            tags=data.get('tags', [])
        )
        
        # Restore custom text state if present
        if '_original_text' in data:
            overlay._original_text = data['_original_text']
        if '_custom_text' in data:
            overlay._custom_text = data['_custom_text']
        if '_is_custom_text' in data:
            overlay._is_custom_text = data['_is_custom_text']
            
        return overlay

    def is_for_channel(self, channel: Union[str, Any]) -> bool:
        """Check if the overlay is tied to a given channel."""
        return self.channel == channel 

    @classmethod
    def get_default_style(cls, overlay_type: str) -> Dict[str, Any]:
        """Get default style settings for a given overlay type."""
        return cls.DEFAULT_STYLES.get(overlay_type, cls.DEFAULT_STYLES['line']).copy()

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available overlay types."""
        return list(cls.DEFAULT_STYLES.keys())
    
    @classmethod
    def create_text_overlay(cls, id: str, name: str, text: str, position: Union[str, tuple] = 'top-left', 
                           **kwargs) -> 'Overlay':
        """
        Convenience method to create a text overlay with common positioning.
        
        Args:
            id: Unique identifier for the overlay
            name: Display name for the overlay
            text: Text content to display
            position: Position as string ('top-left', 'top-right', etc.) or tuple (x, y)
            **kwargs: Additional style properties to override defaults
        """
        style = {'position': position}
        style.update(kwargs)
        
        return cls(
            id=id,
            name=name,
            type='text',
            data={'text': text},
            style=style
        ) 