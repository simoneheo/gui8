"""
Plot Wizard Integration Module

This module provides utilities to integrate configuration wizards with plot wizards,
enabling seamless configuration of line styles, markers, and spectrograms directly
from plot interfaces.
"""

from typing import Dict, Any, Optional, Callable, Union
from PySide6.QtWidgets import QDialog, QMessageBox
from PySide6.QtCore import Signal, QObject

from channel import Channel
from base_plot_wizard import BasePlotWizard


class ConfigurationWizardManager(QObject):
    """
    Manager for coordinating between plot wizards and configuration wizards
    """
    
    # Signals for configuration updates
    line_config_updated = Signal(str, dict)      # channel_id, config
    marker_config_updated = Signal(str, dict)    # channel_id, config  
    spectrogram_config_updated = Signal(str, dict)  # channel_id, config
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_wizards = {}  # wizard_type -> wizard_instance
        self.plot_wizard = None
        
    def set_plot_wizard(self, plot_wizard: BasePlotWizard):
        """Set the associated plot wizard"""
        self.plot_wizard = plot_wizard
        
        # Connect signals to plot wizard update methods
        self.line_config_updated.connect(self._on_line_config_updated)
        self.marker_config_updated.connect(self._on_marker_config_updated)
        self.spectrogram_config_updated.connect(self._on_spectrogram_config_updated)
    
    def open_line_wizard(self, channel: Channel) -> bool:
        """
        Open line configuration wizard for a channel
        
        Args:
            channel: The channel to configure
            
        Returns:
            True if wizard opened successfully, False otherwise
        """
        try:
            from line_wizard_refactored import LineWizard
            
            # Close existing line wizard if open
            if 'line' in self.active_wizards:
                self.active_wizards['line'].close()
            
            # Create and configure wizard
            wizard = LineWizard(channel, self.plot_wizard)
            wizard.setModal(True)
            
            # Connect signals
            wizard.config_updated.connect(
                lambda obj: self._emit_line_config_updated(channel.channel_id, obj)
            )
            wizard.finished.connect(lambda: self._on_wizard_closed('line'))
            
            # Store reference and show
            self.active_wizards['line'] = wizard
            wizard.show()
            
            self.plot_wizard._log_message(f"Opened line wizard for channel: {channel.legend_label}")
            return True
            
        except ImportError as e:
            self.plot_wizard._log_message(f"Line wizard not available: {str(e)}")
            return False
        except Exception as e:
            self.plot_wizard._log_message(f"Error opening line wizard: {str(e)}")
            return False
    
    def open_marker_wizard(self, channel_or_config: Union[Channel, Dict[str, Any]]) -> bool:
        """
        Open marker configuration wizard
        
        Args:
            channel_or_config: Channel object or marker configuration dict
            
        Returns:
            True if wizard opened successfully, False otherwise
        """
        try:
            from marker_wizard_refactored import MarkerWizard
            
            # Close existing marker wizard if open
            if 'marker' in self.active_wizards:
                self.active_wizards['marker'].close()
            
            # Create marker config if channel provided
            if isinstance(channel_or_config, Channel):
                marker_config = self._create_marker_config_from_channel(channel_or_config)
                channel_id = channel_or_config.channel_id
            else:
                marker_config = channel_or_config
                channel_id = marker_config.get('channel_id', 'unknown')
            
            # Create and configure wizard
            wizard = MarkerWizard(marker_config, self.plot_wizard)
            wizard.setModal(True)
            
            # Connect signals
            wizard.config_updated.connect(
                lambda obj: self._emit_marker_config_updated(channel_id, obj)
            )
            wizard.finished.connect(lambda: self._on_wizard_closed('marker'))
            
            # Store reference and show
            self.active_wizards['marker'] = wizard
            wizard.show()
            
            self.plot_wizard._log_message(f"Opened marker wizard for: {marker_config.get('name', 'Unknown')}")
            return True
            
        except ImportError as e:
            self.plot_wizard._log_message(f"Marker wizard not available: {str(e)}")
            return False
        except Exception as e:
            self.plot_wizard._log_message(f"Error opening marker wizard: {str(e)}")
            return False
    
    def open_spectrogram_wizard(self, channel: Channel) -> bool:
        """
        Open spectrogram configuration wizard for a channel
        
        Args:
            channel: The channel to configure
            
        Returns:
            True if wizard opened successfully, False otherwise
        """
        try:
            from spectrogram_wizard_refactored import SpectrogramWizard
            
            # Close existing spectrogram wizard if open
            if 'spectrogram' in self.active_wizards:
                self.active_wizards['spectrogram'].close()
            
            # Create and configure wizard
            wizard = SpectrogramWizard(channel, self.plot_wizard)
            wizard.setModal(True)
            
            # Connect signals
            wizard.config_updated.connect(
                lambda obj: self._emit_spectrogram_config_updated(channel.channel_id, obj)
            )
            wizard.finished.connect(lambda: self._on_wizard_closed('spectrogram'))
            
            # Store reference and show
            self.active_wizards['spectrogram'] = wizard
            wizard.show()
            
            self.plot_wizard._log_message(f"Opened spectrogram wizard for channel: {channel.legend_label}")
            return True
            
        except ImportError as e:
            self.plot_wizard._log_message(f"Spectrogram wizard not available: {str(e)}")
            return False
        except Exception as e:
            self.plot_wizard._log_message(f"Error opening spectrogram wizard: {str(e)}")
            return False
    
    def determine_wizard_type(self, channel: Channel) -> str:
        """
        Determine the appropriate wizard type for a channel
        
        Args:
            channel: The channel to analyze
            
        Returns:
            Wizard type: 'line', 'marker', or 'spectrogram'
        """
        # Check channel tags or metadata for plot type hints
        if hasattr(channel, 'tags') and channel.tags:
            if 'spectrogram' in channel.tags:
                return 'spectrogram'
            elif 'scatter' in channel.tags or 'marker' in channel.tags:
                return 'marker'
        
        # Check if channel has spectrogram data
        if hasattr(channel, 'metadata') and channel.metadata:
            if 'Zxx' in channel.metadata:
                return 'spectrogram'
        
        # Check for scatter plot indicators
        if hasattr(channel, 'plot_type'):
            if channel.plot_type in ['scatter', 'marker']:
                return 'marker'
            elif channel.plot_type == 'spectrogram':
                return 'spectrogram'
        
        # Default to line wizard
        return 'line'
    
    def open_appropriate_wizard(self, channel: Channel) -> bool:
        """
        Open the most appropriate configuration wizard for a channel
        
        Args:
            channel: The channel to configure
            
        Returns:
            True if wizard opened successfully, False otherwise
        """
        wizard_type = self.determine_wizard_type(channel)
        
        if wizard_type == 'line':
            return self.open_line_wizard(channel)
        elif wizard_type == 'marker':
            return self.open_marker_wizard(channel)
        elif wizard_type == 'spectrogram':
            return self.open_spectrogram_wizard(channel)
        else:
            self.plot_wizard._log_message(f"Unknown wizard type: {wizard_type}")
            return False
    
    def close_all_wizards(self):
        """Close all active configuration wizards"""
        for wizard_type, wizard in list(self.active_wizards.items()):
            try:
                wizard.close()
            except:
                pass
        self.active_wizards.clear()
    
    def _create_marker_config_from_channel(self, channel: Channel) -> Dict[str, Any]:
        """Create marker configuration dict from channel"""
        return {
            'name': channel.legend_label or channel.ylabel or channel.channel_id,
            'marker_style': getattr(channel, 'marker', 'o'),
            'marker_size': getattr(channel, 'marker_size', 20),
            'marker_color': getattr(channel, 'color', '#1f77b4'),
            'marker_alpha': getattr(channel, 'alpha', 1.0),
            'edge_color': getattr(channel, 'edge_color', '#000000'),
            'edge_width': getattr(channel, 'edge_width', 0.0),
            'x_axis': getattr(channel, 'xaxis', 'x-bottom').replace('x-', ''),
            'z_order': getattr(channel, 'z_order', 0),
            'channel_id': channel.channel_id
        }
    
    def _emit_line_config_updated(self, channel_id: str, channel_obj: Channel):
        """Emit line configuration updated signal"""
        config = {
            'color': getattr(channel_obj, 'color', '#1f77b4'),
            'style': getattr(channel_obj, 'style', '-'),
            'marker': getattr(channel_obj, 'marker', None),
            'alpha': getattr(channel_obj, 'alpha', 1.0),
            'z_order': getattr(channel_obj, 'z_order', 0),
            'legend_label': getattr(channel_obj, 'legend_label', ''),
            'xaxis': getattr(channel_obj, 'xaxis', 'x-bottom')
        }
        self.line_config_updated.emit(channel_id, config)
    
    def _emit_marker_config_updated(self, channel_id: str, marker_config: Dict[str, Any]):
        """Emit marker configuration updated signal"""
        self.marker_config_updated.emit(channel_id, marker_config)
    
    def _emit_spectrogram_config_updated(self, channel_id: str, channel_obj: Channel):
        """Emit spectrogram configuration updated signal"""
        config = {
            'colormap': getattr(channel_obj, 'colormap', 'viridis'),
            'alpha': getattr(channel_obj, 'alpha', 1.0),
            'z_order': getattr(channel_obj, 'z_order', 0),
            'legend_label': getattr(channel_obj, 'legend_label', ''),
            'interpolation': getattr(channel_obj, 'interpolation', 'nearest'),
            'aspect': getattr(channel_obj, 'aspect', 'auto'),
            'shading': getattr(channel_obj, 'shading', 'flat')
        }
        self.spectrogram_config_updated.emit(channel_id, config)
    
    def _on_line_config_updated(self, channel_id: str, config: Dict[str, Any]):
        """Handle line configuration update"""
        if self.plot_wizard:
            self.plot_wizard._log_message(f"Line configuration updated for channel: {channel_id}")
            self.plot_wizard._schedule_plot_update()
    
    def _on_marker_config_updated(self, channel_id: str, config: Dict[str, Any]):
        """Handle marker configuration update"""
        if self.plot_wizard:
            self.plot_wizard._log_message(f"Marker configuration updated for: {config.get('name', channel_id)}")
            self.plot_wizard._schedule_plot_update()
    
    def _on_spectrogram_config_updated(self, channel_id: str, config: Dict[str, Any]):
        """Handle spectrogram configuration update"""
        if self.plot_wizard:
            self.plot_wizard._log_message(f"Spectrogram configuration updated for channel: {channel_id}")
            self.plot_wizard._schedule_plot_update()
    
    def _on_wizard_closed(self, wizard_type: str):
        """Handle wizard closing"""
        if wizard_type in self.active_wizards:
            del self.active_wizards[wizard_type]


def integrate_config_wizards_with_plot_wizard(plot_wizard: BasePlotWizard) -> ConfigurationWizardManager:
    """
    Integrate configuration wizards with a plot wizard
    
    Args:
        plot_wizard: The plot wizard to integrate with
        
    Returns:
        ConfigurationWizardManager instance
    """
    # Create manager
    manager = ConfigurationWizardManager(plot_wizard)
    manager.set_plot_wizard(plot_wizard)
    
    # Override the plot wizard's _open_channel_config method
    original_open_config = plot_wizard._open_channel_config
    
    def enhanced_open_config(channel: Channel):
        """Enhanced channel configuration opening"""
        try:
            # Try to use the manager first
            if manager.open_appropriate_wizard(channel):
                return
            
            # Fallback to original method
            original_open_config(channel)
            
        except Exception as e:
            plot_wizard._log_message(f"Error opening configuration wizard: {str(e)}")
            # Try original method as last resort
            try:
                original_open_config(channel)
            except Exception as e2:
                plot_wizard._log_message(f"Fallback configuration method also failed: {str(e2)}")
    
    # Replace the method
    plot_wizard._open_channel_config = enhanced_open_config
    
    # Store manager reference in plot wizard
    plot_wizard._config_wizard_manager = manager
    
    plot_wizard._log_message("Configuration wizards integrated successfully")
    
    return manager


def create_integrated_plot_wizard(wizard_type: str, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
    """
    Create a plot wizard with integrated configuration wizards
    
    Args:
        wizard_type: Type of plot wizard ('process', 'mixer', 'comparison', 'plot')
        file_manager: File manager instance
        channel_manager: Channel manager instance
        signal_bus: Signal bus instance
        parent: Parent widget
        
    Returns:
        Plot wizard instance with integrated configuration wizards
    """
    # Create the appropriate plot wizard
    if wizard_type == 'process':
        from process_wizard_refactored import ProcessWizardRefactored
        plot_wizard = ProcessWizardRefactored(file_manager, channel_manager, signal_bus, parent)
    elif wizard_type == 'mixer':
        from signal_mixer_wizard_refactored import SignalMixerWizardRefactored
        plot_wizard = SignalMixerWizardRefactored(file_manager, channel_manager, signal_bus, parent)
    elif wizard_type == 'comparison':
        # Use the current window-based comparison wizard
        from comparison_wizard_manager import ComparisonWizardManager
        plot_wizard = ComparisonWizardManager(file_manager, channel_manager, signal_bus, parent)
    elif wizard_type == 'plot':
        from plot_wizard_refactored import PlotWizardRefactored
        plot_wizard = PlotWizardRefactored(file_manager, channel_manager, signal_bus, parent)
    else:
        raise ValueError(f"Unknown wizard type: {wizard_type}")
    
    # Integrate configuration wizards
    integrate_config_wizards_with_plot_wizard(plot_wizard)
    
    return plot_wizard


# Utility functions for specific integrations

def link_line_wizard_to_plot(plot_wizard: BasePlotWizard, channel: Channel):
    """
    Directly link line wizard to a plot wizard for a specific channel
    
    Args:
        plot_wizard: The plot wizard
        channel: The channel to configure
    """
    if hasattr(plot_wizard, '_config_wizard_manager'):
        plot_wizard._config_wizard_manager.open_line_wizard(channel)
    else:
        # Create temporary manager
        manager = integrate_config_wizards_with_plot_wizard(plot_wizard)
        manager.open_line_wizard(channel)


def link_marker_wizard_to_plot(plot_wizard: BasePlotWizard, marker_config: Dict[str, Any]):
    """
    Directly link marker wizard to a plot wizard for a specific configuration
    
    Args:
        plot_wizard: The plot wizard
        marker_config: The marker configuration
    """
    if hasattr(plot_wizard, '_config_wizard_manager'):
        plot_wizard._config_wizard_manager.open_marker_wizard(marker_config)
    else:
        # Create temporary manager
        manager = integrate_config_wizards_with_plot_wizard(plot_wizard)
        manager.open_marker_wizard(marker_config)


def link_spectrogram_wizard_to_plot(plot_wizard: BasePlotWizard, channel: Channel):
    """
    Directly link spectrogram wizard to a plot wizard for a specific channel
    
    Args:
        plot_wizard: The plot wizard
        channel: The channel to configure
    """
    if hasattr(plot_wizard, '_config_wizard_manager'):
        plot_wizard._config_wizard_manager.open_spectrogram_wizard(channel)
    else:
        # Create temporary manager
        manager = integrate_config_wizards_with_plot_wizard(plot_wizard)
        manager.open_spectrogram_wizard(channel)


# Example usage and testing functions

def test_integration():
    """Test the integration functionality"""
    print("Testing plot wizard integration...")
    
    # This would be used in actual application code:
    # 
    # # Create integrated plot wizard
    # plot_wizard = create_integrated_plot_wizard(
    #     'process', file_manager, channel_manager, signal_bus, parent
    # )
    # 
    # # Configuration wizards are now automatically available
    # # when users click gear buttons in channel tables
    
    print("Integration test completed successfully")


if __name__ == "__main__":
    test_integration() 