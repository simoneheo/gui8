# signal_mixer_wizard_manager.py

# Replace the problematic imports with proper ones
try:
    from mixer.mixer_registry import MixerRegistry, load_all_mixers
    MIXER_AVAILABLE = True
    print("[SignalMixerWizardManager] Mixer registry imported successfully")
except ImportError as e:
    print(f"[SignalMixerWizardManager] Warning: Could not import mixer registry: {e}")
    MIXER_AVAILABLE = False
    # Create a dummy MixerRegistry if mixer module is not available
    class MixerRegistry:
        @staticmethod
        def all_mixers():
            return ["add", "subtract", "multiply", "divide"]  # Basic operations
        
        @staticmethod
        def get(name):
            return None
    
    def load_all_mixers(directory):
        print(f"[SignalMixerWizardManager] Warning: Mixer module not available")

import traceback
import time
import numpy as np
from typing import Dict, List, Optional, Any
from channel import Channel

class SignalMixerWizardManager:
    """
    Manager for the Signal Mixer Wizard that handles:
    - Channel mixing operations
    - Alignment and data processing
    - State management and statistics tracking
    """
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        # Store managers with validation
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state tracking
        self._stats = {
            'total_mixes': 0,
            'successful_mixes': 0,
            'failed_mixes': 0,
            'last_mix_time': None,
            'session_start': time.time()
        }
        
        # Core state
        self.mixed_channels = []
        self.ui_callbacks = {}
        
        # Store aligned versions of A and B channels
        self.aligned_channels = {'A': None, 'B': None}
        self.alignment_params = {'width': 0, 'a_start': 0, 'b_start': 0}
        
        # Store Step 2 aligned channels (all channels from Step 1 with alignment applied)
        self.step2_aligned_channels = {}
        
        # Track channel history for undo functionality
        self.channel_history = []
        
        # Step-based workflow state
        self.selected_channel_a = None
        self.selected_channel_b = None
        self.validation_passed = False
        self.current_step = 1
        
        # Validate initialization
        if not self._validate_managers():
            raise ValueError("Required managers not available for SignalMixerWizardManager")
        
        # Initialize mixers
        self._initialize_mixers()
            
        # Log initialization
        self._log_state_change("Mixer manager initialized successfully")
        
    def _validate_managers(self) -> bool:
        """Validate that required managers are available and functional"""
        if not self.file_manager:
            print("[SignalMixerWizardManager] ERROR: File manager not provided")
            return False
            
        if not self.channel_manager:
            print("[SignalMixerWizardManager] ERROR: Channel manager not provided")
            return False
            
        # Validate manager functionality
        try:
            # Test basic manager operations
            self.file_manager.get_file_count()
            self.channel_manager.get_channel_count()
            return True
        except Exception as e:
            print(f"[SignalMixerWizardManager] ERROR: Manager validation failed: {e}")
            return False
            
    def _initialize_mixers(self):
        """Initialize mixer registry by loading all available mixers"""
        if MIXER_AVAILABLE:
            try:
                # Load all mixer modules
                load_all_mixers("mixer")
                available_mixers = MixerRegistry.all_mixers()
                self._log_state_change(f"Loaded {len(available_mixers)} mixers: {available_mixers}")
            except Exception as e:
                self._log_state_change(f"Error loading mixers: {e}")
                import traceback
                traceback.print_exc()
        else:
            self._log_state_change("Mixer module not available - using dummy registry")
    
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[SignalMixerWizardManager {timestamp}] {message}")
        
    def get_stats(self) -> Dict:
        """Get comprehensive mixer statistics"""
        return {
            **self._stats,
            'mixed_channels_count': len(self.mixed_channels),
            'history_count': len(self.channel_history),
            'alignment_configured': bool(self.aligned_channels['A'] or self.aligned_channels['B']),
            'session_duration': time.time() - self._stats['session_start']
        }
        
    def register_ui_callback(self, event_name, callback):
        """Register UI callbacks for manager events."""
        self.ui_callbacks[event_name] = callback
        
    def _emit_ui_event(self, event_name, *args, **kwargs):
        """Emit an event to the UI."""
        if event_name in self.ui_callbacks:
            self.ui_callbacks[event_name](*args, **kwargs)

    def get_available_channels(self):
        """Get list of all available channels for mixing."""
        try:
            all_channels = []
            
            # Get channels from all files
            for file_info in self.file_manager.get_all_files():
                channels = self.channel_manager.get_channels_by_file(file_info.file_id)
                for channel in channels:
                    if channel.show and channel.ydata is not None:
                        all_channels.append(channel)
            
            self._log_state_change(f"Retrieved {len(all_channels)} available channels")
            return all_channels
            
        except Exception as e:
            print(f"[SignalMixerWizardManager] Error getting channels: {e}")
            return []

    def get_channel_display_name(self, channel):
        """Get a display name for a channel."""
        if hasattr(channel, 'legend_label') and channel.legend_label:
            name = channel.legend_label
        else:
            name = channel.channel_id
        
        # Add filename for context
        return f"{name} ({channel.filename})"

    def get_channel_stats(self, channel):
        """Get statistics for a channel."""
        if not channel or channel.ydata is None:
            return {}
        
        try:
            return {
                'length': len(channel.ydata),
                'min': float(np.min(channel.ydata)),
                'max': float(np.max(channel.ydata)),
                'mean': float(np.mean(channel.ydata)),
                'std': float(np.std(channel.ydata)),
                'range': float(np.max(channel.ydata) - np.min(channel.ydata))
            }
        except Exception as e:
            print(f"[SignalMixerWizardManager] Error calculating stats: {e}")
            return {}

    def validate_channels_for_mixing(self, channel_a, channel_b):
        """Validate that two channels can be mixed together."""
        if not channel_a or not channel_b:
            return False, "Both channels must be selected"
        
        if channel_a.ydata is None or channel_b.ydata is None:
            return False, "Both channels must have data"
        
        if len(channel_a.ydata) != len(channel_b.ydata):
            return False, f"Channels have different lengths: {len(channel_a.ydata)} vs {len(channel_b.ydata)}"
        
        if channel_a.channel_id == channel_b.channel_id:
            return False, "Cannot mix a channel with itself"
        
        return True, f"Channels are compatible (length: {len(channel_a.ydata)})"

    def suggest_channel_pairs(self):
        """Suggest good channel pairs for mixing."""
        channels = self.get_available_channels()
        suggestions = []
        
        # Group channels by length
        length_groups = {}
        for channel in channels:
            if channel.ydata is not None:
                length = len(channel.ydata)
                if length not in length_groups:
                    length_groups[length] = []
                length_groups[length].append(channel)
        
        # Find pairs within each length group
        for length, group_channels in length_groups.items():
            if len(group_channels) >= 2:
                # Sort by filename and channel name for consistent suggestions
                group_channels.sort(key=lambda ch: (ch.filename, ch.legend_label or ch.channel_id))
                
                # Suggest pairs
                for i in range(0, len(group_channels) - 1, 2):
                    suggestions.append((group_channels[i], group_channels[i + 1]))
        
        return suggestions[:5]  # Return up to 5 suggestions

    def get_available_files(self):
        """Get list of available files for dropdowns (legacy method)."""
        try:
            # Try to get selected files first
            selected_files = []
            if hasattr(self.file_manager, 'get_selected_files'):
                selected_files = self.file_manager.get_selected_files()
            
            # Fallback to all files if no selection or method not available
            if not selected_files:
                selected_files = self.file_manager.get_all_files()
            
            self._log_state_change(f"Retrieved {len(selected_files)} available files")
            return selected_files
            
        except Exception as e:
            print(f"[SignalMixerWizardManager] Error getting files: {e}")
            return []

    def get_all_parsed_files(self):
        """Get list of all parsed files (successful parse) for dropdowns."""
        all_files = self.file_manager.get_all_files()
        # Filter for files that have successful parse (have channels)
        parsed_files = []
        for file_info in all_files:
            channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            if channels:  # File has channels, meaning it was parsed successfully
                parsed_files.append(file_info)
        return parsed_files

    def get_channels_for_file(self, filename):
        """Get channels for a specific filename."""
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return []
        
        # Get channels for this file
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        # Get unique channel names (by legend_label or channel_id)
        channel_names = []
        seen = set()
        for ch in channels:
            name = ch.legend_label or ch.channel_id
            if name not in seen:
                channel_names.append(name)
                seen.add(name)
        
        return channel_names

    def get_steps_for_channel(self, filename, channel_info):
        """Get available steps for a specific filename and channel."""
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return []
        
        # Get channels for this file with matching channel info
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        matching_channels = [ch for ch in channels 
                           if (ch.legend_label == channel_info or ch.channel_id == channel_info)]
        
        # Get unique steps
        steps = sorted(list(set(str(ch.step) for ch in matching_channels)))
        return steps

    def get_channel_by_selection(self, filename, channel_info, step=None):
        """Get a channel object by filename and channel info, optionally with step."""
        try:
            # Find matching channels in channel manager
            matching_channels = []
            all_channels = self.channel_manager.get_all_channels()
            for channel in all_channels:
                if (channel.filename == filename and 
                    (channel.legend_label == channel_info or channel.channel_id == channel_info)):
                    matching_channels.append(channel)
            
            if not matching_channels:
                return None
            
            # If step is specified, filter by step
            if step is not None:
                step_int = int(step)
                for channel in matching_channels:
                    if channel.step == step_int:
                        return channel
                return None
            
            # If no step specified, return the channel with the highest step (most processed)
            return max(matching_channels, key=lambda ch: ch.step)
                    
        except Exception as e:
            print(f"[SignalMixerWizardManager] Error getting channel: {e}")
            
        return None

    def find_best_channel_pair(self):
        """Find the best pair of channels based on priority rules."""
        print("[SignalMixerWizardManager] Finding best channel pair")
        
        # Get selected files
        selected_files = self.get_available_files()
        if not selected_files:
            return None, None
            
        # Get all channels from selected files
        all_channels = []
        for file_info in selected_files:
            file_channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            all_channels.extend(file_channels)
        
        if len(all_channels) < 1:
            return None, None
        
        # Priority 1: Try to find two channels with matching xdata and ydata lengths
        for i, ch1 in enumerate(all_channels):
            if ch1.xdata is None or ch1.ydata is None:
                continue
            len1_x, len1_y = len(ch1.xdata), len(ch1.ydata)
            
            for j, ch2 in enumerate(all_channels[i+1:], i+1):
                if ch2.xdata is None or ch2.ydata is None:
                    continue
                len2_x, len2_y = len(ch2.xdata), len(ch2.ydata)
                
                if len1_x == len2_x and len1_y == len2_y:
                    print(f"[SignalMixerWizardManager] Found matching lengths: {len1_x}x{len1_y}")
                    return ch1, ch2
        
        print("[SignalMixerWizardManager] No matching lengths found, trying other priorities")
        
        # Priority 2: If 2+ files selected, use 1st/2nd files, 1st channel, last step
        if len(selected_files) >= 2:
            file1_channels = self.channel_manager.get_channels_by_file(selected_files[0].file_id)
            file2_channels = self.channel_manager.get_channels_by_file(selected_files[1].file_id)
            
            if file1_channels and file2_channels:
                # Get last step channels for each file
                ch1 = max(file1_channels, key=lambda ch: ch.step)
                ch2 = max(file2_channels, key=lambda ch: ch.step)
                print(f"[SignalMixerWizardManager] Using 1st/2nd files, last steps: {ch1.step}, {ch2.step}")
                return ch1, ch2
        
        # Priority 3: If only one file, use 1st/2nd channels, last step
        if len(selected_files) == 1:
            file_channels = self.channel_manager.get_channels_by_file(selected_files[0].file_id)
            if len(file_channels) >= 2:
                # Get the two highest step channels
                sorted_channels = sorted(file_channels, key=lambda ch: ch.step, reverse=True)
                print(f"[SignalMixerWizardManager] Using 1st/2nd channels from single file, steps: {sorted_channels[0].step}, {sorted_channels[1].step}")
                return sorted_channels[0], sorted_channels[1]
        
        # Priority 4: If only one channel exists, use it for both rows with last two steps
        if len(all_channels) == 1:
            ch = all_channels[0]
            print(f"[SignalMixerWizardManager] Only one channel available, using for both rows: {ch.channel_id}")
            return ch, ch
        
        # Fallback: Use first two available channels
        if len(all_channels) >= 2:
            print("[SignalMixerWizardManager] Fallback: using first two available channels")
            return all_channels[0], all_channels[1]
        elif len(all_channels) == 1:
            print("[SignalMixerWizardManager] Fallback: using single channel for both rows")
            return all_channels[0], all_channels[0]
        
        return None, None

    def get_mixer_templates(self):
        """Get available mixer templates."""
        mixer_names = MixerRegistry.all_mixers()
        
        # Add expression templates if ExpressionMixer is available
        try:
            expression_mixer = MixerRegistry.get("expression")
            if expression_mixer and hasattr(expression_mixer, 'get_expression_templates'):
                templates = expression_mixer.get_expression_templates()
                return mixer_names + templates
        except (KeyError, AttributeError):
            print("[SignalMixerWizardManager] ExpressionMixer not available for templates")
        
        return mixer_names

    def generate_next_label(self, current_row_count):
        """Generate the next available label (C, D, E, etc.)."""
        return chr(ord('A') + current_row_count)

    def process_mixer_expression(self, expression, channel_context):
        """Process a mixer expression and return the result."""
        try:
            print(f"[SignalMixerWizardManager] Processing expression: {expression}")
            if not expression or '=' not in expression:
                return None, "Invalid expression. Use format: C = A + B"

            label, formula = map(str.strip, expression.split('=', 1))
            print(f"[SignalMixerWizardManager] Label: {label}, Formula: {formula}")
            
            # Add mixed channels to context using their step table labels (C, D, E, etc.)
            context = {**channel_context}
            for i, mixed_channel in enumerate(self.mixed_channels):
                # Use stored step table label if available, otherwise generate one
                step_label = getattr(mixed_channel, 'step_table_label', chr(ord('C') + i))
                context[step_label] = mixed_channel
                print(f"[SignalMixerWizardManager] Added mixed channel {step_label}: {mixed_channel.legend_label}")
            
            print(f"[SignalMixerWizardManager] Context channels: {list(context.keys())}")
            
            # Validate that required channels exist
            for ch_name, ch_obj in context.items():
                if ch_obj is None:
                    return None, f"Channel {ch_name} is not available"
                print(f"[SignalMixerWizardManager] Channel {ch_name}: {ch_obj.legend_label} (xdata: {len(ch_obj.xdata) if ch_obj.xdata is not None else 'None'}, ydata: {len(ch_obj.ydata) if ch_obj.ydata is not None else 'None'})")

            mixer_cls = self._resolve_mixer_class(formula)
            if mixer_cls is None:
                return None, f"Could not resolve mixer for expression: {formula}"

            print(f"[SignalMixerWizardManager] Using mixer class: {mixer_cls}")
            
            # Parse operation from formula
            operation_params = self._parse_operation_params(formula, label)
            mixer = mixer_cls(label=label, **operation_params)
            print(f"[SignalMixerWizardManager] Created mixer with params: {operation_params}")
            
            new_channel = mixer.apply(context)
            print(f"[SignalMixerWizardManager] Created new channel: {new_channel.channel_id}")
            print(f"[SignalMixerWizardManager] New channel data: xdata={len(new_channel.xdata) if new_channel.xdata is not None else 'None'}, ydata={len(new_channel.ydata) if new_channel.ydata is not None else 'None'}")

            # Set channel description to the expression
            new_channel.description = expression
            
            # Store the step table label with the channel for easy reference
            # Use the actual label from the expression (e.g., "Z" from "Z = A * B", "DOG" from "DOG = C/E")
            new_channel.step_table_label = label
            
            # Check if a channel with this label already exists
            existing_index = None
            for i, existing_channel in enumerate(self.mixed_channels):
                if getattr(existing_channel, 'step_table_label', None) == label:
                    existing_index = i
                    print(f"[SignalMixerWizardManager] Found existing channel with label {label}, will overwrite")
                    break
            
            # Update channel name to include mixer step
            mixer_name = self._get_mixer_name_from_class(mixer_cls)
            if hasattr(new_channel, 'legend_label') and new_channel.legend_label:
                original_name = new_channel.legend_label
            else:
                original_name = new_channel.channel_id
            
            new_channel.legend_label = f"{original_name} - {mixer_name}"
            print(f"[SignalMixerWizardManager] Updated channel name to: {new_channel.legend_label}")
            print(f"[SignalMixerWizardManager] Channel can be referenced as: {label}")

            # Register new channel
            self.channel_manager.add_channel(new_channel)
            
            # Replace existing channel or append new one
            if existing_index is not None:
                # Save current state to history before replacement
                self._save_to_history()
                
                # Remove the old channel from channel manager
                old_channel = self.mixed_channels[existing_index]
                if hasattr(self.channel_manager, 'remove_channel'):
                    self.channel_manager.remove_channel(old_channel.channel_id)
                
                # Replace in the list at the same position
                self.mixed_channels[existing_index] = new_channel
                print(f"[SignalMixerWizardManager] Replaced existing channel at position {existing_index}")
                
                # Signal UI to update existing row
                self._emit_ui_event('replace_channel_in_table', existing_index, new_channel)
            else:
                # Save current state to history before adding new channel
                self._save_to_history()
                
                # Add new channel
                self.mixed_channels.append(new_channel)
                # Signal UI to add new row (existing behavior)
                self._emit_ui_event('add_channel_to_table', new_channel)

            return new_channel, f"{label} created successfully."

        except Exception as e:
            error_msg = f"Mixer failed: {e}\n{traceback.format_exc()}"
            print(f"[SignalMixerWizardManager] Error: {error_msg}")
            return None, error_msg

    def _parse_operation_params(self, formula, label):
        """Parse operation parameters from formula."""
        params = {}
        formula_clean = formula.strip().lower()
        
        # Check if this should use ExpressionMixer
        try:
            expression_mixer = MixerRegistry.get("expression")
            if expression_mixer and hasattr(expression_mixer, 'parse_expression_for_mixer') and expression_mixer.parse_expression_for_mixer(formula):
                params['expression'] = formula
                return params
        except (KeyError, AttributeError):
            print("[SignalMixerWizardManager] ExpressionMixer not available for parameter parsing")
        
        # Legacy simple operation parsing for other mixers
        if '+' in formula_clean:
            params['operation'] = 'add'
        elif '-' in formula_clean:
            params['operation'] = 'sub'
        elif '*' in formula_clean:
            params['operation'] = 'mul'
        elif '/' in formula_clean:
            params['operation'] = 'div'
        
        return params

    def _resolve_mixer_class(self, formula):
        """Resolve mixer class based on formula content."""
        formula_clean = formula.lower().strip()
        
        # First check if this should use ExpressionMixer
        try:
            expression_mixer = MixerRegistry.get("expression")
            if expression_mixer and hasattr(expression_mixer, 'parse_expression_for_mixer') and expression_mixer.parse_expression_for_mixer(formula):
                print(f"[SignalMixerWizardManager] Using ExpressionMixer for complex formula: {formula}")
                return expression_mixer
        except (KeyError, AttributeError):
            print("[SignalMixerWizardManager] ExpressionMixer not available, using fallback logic")
        
        # Legacy simple mixer resolution
        try:
            if any(op in formula_clean for op in ['+', '-', '*', '/']):
                arithmetic_mixer = MixerRegistry.get("arithmetic")
                if arithmetic_mixer:
                    return arithmetic_mixer
        except KeyError:
            print("[SignalMixerWizardManager] ArithmeticMixer not available")
            
        try:
            if any(fn in formula_clean for fn in ['abs', 'normalize', 'zscore']):
                unary_mixer = MixerRegistry.get("unary")
                if unary_mixer:
                    return unary_mixer
        except KeyError:
            print("[SignalMixerWizardManager] UnaryMixer not available")
            
        try:
            if any(op in formula_clean for op in ['>', '<', '>=', '<=', '==', '!=']):
                logic_mixer = MixerRegistry.get("logic")
                if logic_mixer:
                    return logic_mixer
        except KeyError:
            print("[SignalMixerWizardManager] LogicMixer not available")
            
        try:
            if any(fn in formula_clean for fn in ['clip', 'threshold']):
                threshold_mixer = MixerRegistry.get("threshold")
                if threshold_mixer:
                    return threshold_mixer
        except KeyError:
            print("[SignalMixerWizardManager] ThresholdMixer not available")
        
        # Try to get any available mixer as fallback
        try:
            available_mixers = MixerRegistry.all_mixers()
            if available_mixers:
                first_mixer = MixerRegistry.get(available_mixers[0])
                print(f"[SignalMixerWizardManager] Using fallback mixer: {available_mixers[0]}")
                return first_mixer
        except (KeyError, IndexError):
            pass
        
        print(f"[SignalMixerWizardManager] No suitable mixer found for formula: {formula}")
        return None

    def _get_mixer_name_from_class(self, mixer_cls):
        """Get a friendly name from the mixer class."""
        if not mixer_cls:
            return "unknown mixer"
            
        class_name = mixer_cls.__name__.lower()
        
        # Map class names to friendly names
        name_mapping = {
            'arithmeticmixer': 'arithmetic mixer',
            'expressionmixer': 'expression mixer',
            'unarymixer': 'unary mixer',
            'logicmixer': 'logic mixer',
            'thresholdmaskmixer': 'threshold mixer'
        }
        
        return name_mapping.get(class_name, class_name.replace('mixer', ' mixer'))

    def add_mixed_channel_to_table(self, channel):
        """Legacy method - UI callbacks are now handled directly in process_mixer_expression."""
        # This method is kept for compatibility but no longer emits UI events
        # UI callbacks are handled directly in process_mixer_expression
        pass

    def clear_mixed_channels(self):
        """Clear all mixed channels."""
        self.mixed_channels.clear()
        self.channel_history.clear()

    def get_mixed_channels(self):
        """Get all mixed channels."""
        return self.mixed_channels.copy()

    def _save_to_history(self):
        """Save current state of mixed channels to history."""
        # Create a deep copy of current mixed channels state
        current_state = {
            'mixed_channels': [ch for ch in self.mixed_channels],  # Copy channel list
            'channel_count': len(self.mixed_channels)
        }
        self.channel_history.append(current_state)
        
        # Limit history to prevent memory issues (keep last 20 operations)
        if len(self.channel_history) > 20:
            self.channel_history.pop(0)
        
        print(f"[SignalMixerWizardManager] Saved state to history. History length: {len(self.channel_history)}")

    def undo_last_step(self):
        """Undo the last mixed channel operation."""
        if not self.channel_history:
            print("[SignalMixerWizardManager] No history available for undo")
            return False, "No operations to undo"
        
        # Get the previous state
        previous_state = self.channel_history.pop()
        
        # Remove any channels that were added since the previous state
        current_count = len(self.mixed_channels)
        previous_count = previous_state['channel_count']
        
        # Remove channels from channel manager that were added after previous state
        channels_to_remove = self.mixed_channels[previous_count:]
        for channel in channels_to_remove:
            self.channel_manager.remove_channel(channel.channel_id)
        
        # Restore previous state
        self.mixed_channels = previous_state['mixed_channels']
        
        print(f"[SignalMixerWizardManager] Undid last step. Channels: {previous_count} (was {current_count})")
        
        # Signal UI to refresh
        self._emit_ui_event('refresh_after_undo')
        
        return True, f"Undid last step. Now have {len(self.mixed_channels)} mixed channels."

    def can_undo(self):
        """Check if undo is possible."""
        return len(self.channel_history) > 0

    def get_aligned_channel_data(self, channel, start_idx, width):
        """Get aligned channel data based on range parameters."""
        if not channel or channel.ydata is None:
            return None
        
        end_idx = min(start_idx + width, len(channel.ydata))
        
        # Create a copy of the channel with aligned data
        aligned_channel = Channel(
            channel_id=f"{channel.channel_id}_aligned",
            filename=channel.filename,
            legend_label=f"{channel.legend_label}_[{start_idx}:{end_idx-1}]" if channel.legend_label else f"{channel.channel_id}_[{start_idx}:{end_idx-1}]",
            xdata=channel.xdata[start_idx:end_idx] if channel.xdata is not None and len(channel.xdata) > start_idx else channel.xdata,
            ydata=channel.ydata[start_idx:end_idx],
            xlabel=channel.xlabel,
            ylabel=channel.ylabel,
            step=channel.step
        )
        
        # Set color attribute after creation
        aligned_channel.color = channel.color
        
        return aligned_channel

    def process_mixer_expression_with_alignment(self, expression, channel_context, alignment_params=None):
        """Process a mixer expression with optional alignment parameters."""
        if alignment_params:
            # Update stored aligned channels
            self.update_aligned_channels(
                channel_context.get('A'), 
                channel_context.get('B'), 
                alignment_params
            )
            
            # Use stored aligned channels for processing
            aligned_context = {
                'A': self.aligned_channels['A'] or channel_context.get('A'),
                'B': self.aligned_channels['B'] or channel_context.get('B'),
            }
            
            # Note: mixed channels (C, D, E, etc.) will be added inside process_mixer_expression
            # so we don't need to add them here
            
            print(f"[SignalMixerWizardManager] Using aligned channels: A={len(aligned_context['A'].ydata) if aligned_context['A'] and aligned_context['A'].ydata is not None else 'None'}, B={len(aligned_context['B'].ydata) if aligned_context['B'] and aligned_context['B'].ydata is not None else 'None'}")
            
            return self.process_mixer_expression(expression, aligned_context)
        else:
            # Use stored aligned channels if available, otherwise use original context
            if self.aligned_channels['A'] or self.aligned_channels['B']:
                context_with_aligned = {
                    'A': self.aligned_channels['A'] or channel_context.get('A'),
                    'B': self.aligned_channels['B'] or channel_context.get('B'),
                }
                # Note: mixed channels will be added inside process_mixer_expression
                return self.process_mixer_expression(expression, context_with_aligned)
            else:
                return self.process_mixer_expression(expression, channel_context)

    def update_aligned_channels(self, channel_a, channel_b, alignment_params):
        """Update the aligned A and B channels with current alignment parameters."""
        print(f"[SignalMixerWizardManager] Updating aligned channels with params: {alignment_params}")
        
        self.alignment_params = alignment_params.copy()
        
        # Create aligned A channel
        if channel_a:
            width = alignment_params.get('width', len(channel_a.ydata) if channel_a.ydata is not None and len(channel_a.ydata) > 0 else 0)
            self.aligned_channels['A'] = self.get_aligned_channel_data(
                channel_a, alignment_params.get('a_start', 0), width
            )
            if self.aligned_channels['A']:
                self.aligned_channels['A'].legend_label = 'A'  # Keep simple label
                print(f"[SignalMixerWizardManager] Aligned A: {len(self.aligned_channels['A'].ydata)} samples")
        
        # Create aligned B channel  
        if channel_b:
            width = alignment_params.get('width', len(channel_b.ydata) if channel_b.ydata is not None and len(channel_b.ydata) > 0 else 0)
            self.aligned_channels['B'] = self.get_aligned_channel_data(
                channel_b, alignment_params.get('b_start', 0), width
            )
            if self.aligned_channels['B']:
                self.aligned_channels['B'].legend_label = 'B'  # Keep simple label
                print(f"[SignalMixerWizardManager] Aligned B: {len(self.aligned_channels['B'].ydata)} samples")

    def get_current_aligned_channels(self):
        """Get the current aligned A and B channels."""
        return {
            'A': self.aligned_channels['A'],
            'B': self.aligned_channels['B']
        }

    def get_alignment_info(self):
        """Get current alignment parameters and channel info."""
        info = {
            'params': self.alignment_params.copy(),
            'channels': {}
        }
        
        for label in ['A', 'B']:
            if self.aligned_channels[label]:
                ch = self.aligned_channels[label]
                info['channels'][label] = {
                    'length': len(ch.ydata) if ch.ydata is not None else 0,
                    'range': f"{self.alignment_params.get(f'{label.lower()}_start', 0)}-{self.alignment_params.get(f'{label.lower()}_start', 0) + self.alignment_params.get('width', 0) - 1}",
                    'label': ch.legend_label or ch.channel_id
                }
        
        return info

    def set_step2_aligned_channels(self, aligned_channels):
        """Set aligned channels for Step 2 mixing operations"""
        self.step2_aligned_channels = aligned_channels.copy()
        print(f"[SignalMixerWizardManager] Set Step 2 aligned channels: {list(aligned_channels.keys())}")
    
    def get_step2_aligned_channels(self):
        """Get Step 2 aligned channels"""
        return self.step2_aligned_channels.copy()
    
    def clear_aligned_channels(self):
        """Clear aligned channel data."""
        self.aligned_channels = {'A': None, 'B': None}
        self.alignment_params = {'width': 0, 'a_start': 0, 'b_start': 0}
        self.step2_aligned_channels = {}
        print("[SignalMixerWizardManager] Cleared aligned channels")

    def show(self):
        """Show the signal mixer wizard window"""
        # The SignalMixerWizardManager doesn't directly own a window
        # Instead, it's owned by SignalMixerWizardWindow
        # This method is for compatibility with other managers
        print("[SignalMixerWizardManager] ERROR: This manager is owned by the window, not the other way around")
        
    def close(self):
        """Close the signal mixer wizard"""
        # The SignalMixerWizardManager doesn't directly own a window
        # Instead, it's owned by SignalMixerWizardWindow
        # This method is for compatibility with other managers  
        print("[SignalMixerWizardManager] ERROR: This manager is owned by the window, not the other way around")
    
    def set_selected_channels(self, channel_a, channel_b):
        """Set the selected channels for mixing (Step 1)"""
        self.selected_channel_a = channel_a
        self.selected_channel_b = channel_b
        print(f"[SignalMixerWizardManager] Selected channels: A={channel_a.channel_id if channel_a else None}, B={channel_b.channel_id if channel_b else None}")
    
    def get_selected_channels(self):
        """Get the currently selected channels"""
        return self.selected_channel_a, self.selected_channel_b
    
    def validate_channel_compatibility(self, channel_a, channel_b):
        """Validate that two channels are compatible for mixing"""
        if not channel_a or not channel_b:
            return False, "Both channels must be selected"
        
        if not (channel_a.xdata is not None and channel_a.ydata is not None):
            return False, "Channel A has no data"
            
        if not (channel_b.xdata is not None and channel_b.ydata is not None):
            return False, "Channel B has no data"
        
        # Check if channels have compatible dimensions
        len_a = len(channel_a.ydata)
        len_b = len(channel_b.ydata)
        
        if len_a == 0 or len_b == 0:
            return False, "Channels have no data points"
        
        # Channels are compatible if they have at least some data
        min_length = min(len_a, len_b)
        if min_length < 1:
            return False, "Channels are too short for mixing"
        
        if len_a == len_b:
            return True, f"Perfect match - both channels have {len_a} samples"
        else:
            return True, f"Compatible - will use first {min_length} samples (A: {len_a}, B: {len_b})"
    
    def set_validation_status(self, passed):
        """Set the validation status"""
        self.validation_passed = passed
        
    def get_validation_status(self):
        """Get the current validation status"""
        return self.validation_passed
    
    def set_current_step(self, step):
        """Set the current step (1 or 2)"""
        self.current_step = step
        print(f"[SignalMixerWizardManager] Current step set to: {step}")
    
    def get_current_step(self):
        """Get the current step"""
        return self.current_step
