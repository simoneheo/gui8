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
        'text': {
            'color': '#000000',  # Black
            'fontsize': 12,
            'alpha': 1.0,
            'position': 'top-right',
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
                 style: Optional[Dict[str, Any]] = None,
                 channel: Union[str, Any] = None,
                 show: bool = True,
                 tags: Optional[List[str]] = None):
        self.id = id
        self.name = name
        self.type = type  # e.g., 'line', 'shading', 'text', 'marker'
        
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

    def apply_to_plot(self, ax):
        """Draw the overlay on a matplotlib axes. Implementation depends on type."""
        # This is a stub. Actual drawing logic should be implemented per overlay type.
        pass

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
            'style': self.style,
            'channel': self.channel,
            'show': self.show,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Overlay':
        """Deserialize an overlay from a dictionary."""
        return cls(
            id=data['id'],
            name=data['name'],
            type=data['type'],
            style=data['style'],
            channel=data['channel'],
            show=data.get('show', True),
            tags=data.get('tags', [])
        )

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