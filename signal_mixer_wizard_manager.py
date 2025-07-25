# signal_mixer_wizard_manager.py

# Replace the problematic imports with proper ones
try:
    from mixer.mixer_registry import MixerRegistry, load_all_mixers
    MIXER_AVAILABLE = True
except ImportError as e:
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
        pass

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
            return False
            
        if not self.channel_manager:
            return False
            
        # Validate manager functionality
        try:
            # Test basic manager operations
            self.file_manager.get_file_count()
            self.channel_manager.get_channel_count()
            return True
        except Exception as e:
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
        # Debug logging disabled
        
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
    
    def validate_channels_for_mixing_with_alignment(self, channel_a, channel_b, alignment_config):
        """Validate that two channels can be mixed together with alignment options using DataAligner."""
        if not channel_a or not channel_b:
            return False, "Both channels must be selected", {}
        
        if channel_a.ydata is None or channel_b.ydata is None:
            return False, "Both channels must have data", {}
        
        if channel_a.channel_id == channel_b.channel_id:
            return False, "Cannot mix a channel with itself", {}
        
        # Use DataAligner for enhanced validation
        from data_aligner import DataAligner
        
        try:
            data_aligner = DataAligner()
            
            # Validate channel data quality
            ref_validation = data_aligner.data_validator.validate_channel_data(channel_a)
            test_validation = data_aligner.data_validator.validate_channel_data(channel_b)
            
            # Check for critical issues
            if not ref_validation.is_valid:
                return False, f"Channel A validation failed: {'; '.join(ref_validation.issues)}", {}
            
            if not test_validation.is_valid:
                return False, f"Channel B validation failed: {'; '.join(test_validation.issues)}", {}
            
            # Validate alignment parameters
            param_validation = data_aligner.param_validator.validate_alignment_params(alignment_config)
            if not param_validation.is_valid:
                return False, f"Alignment parameters invalid: {'; '.join(param_validation.issues)}", {}
            
            # Check basic dimensions
            len_a = len(channel_a.ydata)
            len_b = len(channel_b.ydata)
            
            # Same length - no alignment needed
            if len_a == len_b:
                return True, f"Channels are compatible (length: {len_a})", {'needs_alignment': False}
            
            # Different lengths - check if alignment can handle this
            alignment_method = alignment_config.get('alignment_method', 'time')
            
            if alignment_method == 'index':
                # Index-based alignment
                mode = alignment_config.get('mode', 'truncate')
                if mode == 'truncate':
                    min_length = min(len_a, len_b)
                    return True, f"Channels compatible with alignment (will truncate to {min_length})", {
                        'needs_alignment': True,
                        'alignment_message': f"Will truncate to {min_length} samples"
                    }
                else:  # custom
                    start_idx = alignment_config.get('start_index', 0)
                    end_idx = alignment_config.get('end_index', 500)
                    max_possible = min(len_a, len_b) - 1
                    
                    if start_idx >= max_possible or end_idx >= max_possible:
                        return False, f"Index range exceeds data bounds (max: {max_possible})", {}
                    
                    if start_idx >= end_idx:
                        return False, "Invalid index range: start >= end", {}
                    
                    aligned_length = end_idx - start_idx + 1
                    return True, f"Channels compatible with alignment (will use {aligned_length} samples)", {
                        'needs_alignment': True,
                        'alignment_message': f"Will align to {aligned_length} samples"
                    }
            
            else:  # time-based
                # Check if channels have time data
                has_time_a = hasattr(channel_a, 'xdata') and channel_a.xdata is not None
                has_time_b = hasattr(channel_b, 'xdata') and channel_b.xdata is not None
                
                if not has_time_a or not has_time_b:
                    # Try to create time data
                    if self._can_create_time_data(channel_a, channel_b):
                        return True, "Channels compatible with alignment (will create time data)", {
                            'needs_alignment': True,
                            'alignment_message': "Will create time data and align"
                        }
                    else:
                        return False, "Time-based alignment requires time data or sampling rate", {}
                
                # Both have time data - check if alignment is possible
                mode = alignment_config.get('mode', 'overlap')
                if mode == 'overlap':
                    # Check for overlapping time ranges
                    time_a_range = (channel_a.xdata.min(), channel_a.xdata.max())
                    time_b_range = (channel_b.xdata.min(), channel_b.xdata.max())
                    
                    overlap_start = max(time_a_range[0], time_b_range[0])
                    overlap_end = min(time_a_range[1], time_b_range[1])
                    
                    if overlap_start >= overlap_end:
                        return False, "No overlapping time range found", {}
                    
                    return True, f"Channels compatible with alignment (overlap {overlap_start:.3f}s to {overlap_end:.3f}s)", {
                        'needs_alignment': True,
                        'alignment_message': f"Will align to overlap region ({overlap_start:.3f}s to {overlap_end:.3f}s)"
                    }
                else:  # custom
                    start_time = alignment_config.get('start_time', 0.0)
                    end_time = alignment_config.get('end_time', 10.0)
                    
                    if start_time >= end_time:
                        return False, "Invalid time range: start >= end", {}
                    
                    return True, f"Channels compatible with alignment (custom time range)", {
                        'needs_alignment': True,
                        'alignment_message': f"Will align to custom time range ({start_time:.3f}s to {end_time:.3f}s)"
                    }
        
        except Exception as e:
            # Fallback to basic validation if DataAligner fails
            return self.validate_channels_for_mixing(channel_a, channel_b)
    
    def _can_create_time_data(self, channel_a, channel_b):
        """Check if time data can be created for channels that don't have it."""
        for channel in [channel_a, channel_b]:
            if not hasattr(channel, 'xdata') or channel.xdata is None:
                # Check if channel has sampling rate
                if hasattr(channel, 'sampling_rate') and channel.sampling_rate is not None and channel.sampling_rate > 0:
                    continue
                else:
                    return False
        return True

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
            pass
            
        return None

    def find_best_channel_pair(self):
        """Find the best pair of channels based on smart matching priority rules."""
        
        # Get selected files
        selected_files = self.get_available_files()
        if not selected_files:
            return None, None
            
        # Get all channels from selected files
        all_channels = []
        for file_info in selected_files:
            file_channels = self.channel_manager.get_channels_by_file(file_info.file_id)
            # Only include channels with valid data
            for ch in file_channels:
                if ch.show and ch.xdata is not None and ch.ydata is not None and len(ch.ydata) > 0:
                    all_channels.append(ch)
        
        if len(all_channels) < 1:
            return None, None
        
        # PRIORITY 1: Smart match - Two DIFFERENT channels with SAME SIZE from DIFFERENT files
        for i, ch1 in enumerate(all_channels):
            len1_x, len1_y = len(ch1.xdata), len(ch1.ydata)
            
            for j, ch2 in enumerate(all_channels[i+1:], i+1):
                len2_x, len2_y = len(ch2.xdata), len(ch2.ydata)
                
                # Must be different channels with same size from different files
                if (ch1.channel_id != ch2.channel_id and 
                    ch1.file_id != ch2.file_id and
                    len1_x == len2_x and len1_y == len2_y):
                    return ch1, ch2
        
        # PRIORITY 2: Smart match - Two DIFFERENT channels with SAME SIZE from SAME file
        for i, ch1 in enumerate(all_channels):
            len1_x, len1_y = len(ch1.xdata), len(ch1.ydata)
            
            for j, ch2 in enumerate(all_channels[i+1:], i+1):
                len2_x, len2_y = len(ch2.xdata), len(ch2.ydata)
                
                # Must be different channels with same size (can be same file)
                if (ch1.channel_id != ch2.channel_id and 
                    len1_x == len2_x and len1_y == len2_y):
                    return ch1, ch2
        
        # PRIORITY 3: If 2+ files selected, use 1st/2nd files, 1st channel, last step
        if len(selected_files) >= 2:
            file1_channels = [ch for ch in all_channels if ch.file_id == selected_files[0].file_id]
            file2_channels = [ch for ch in all_channels if ch.file_id == selected_files[1].file_id]
            
            if file1_channels and file2_channels:
                # Get last step channels for each file
                ch1 = max(file1_channels, key=lambda ch: ch.step)
                ch2 = max(file2_channels, key=lambda ch: ch.step)
                return ch1, ch2
        
        # PRIORITY 4: If only one file, use 1st/2nd channels, last step
        if len(selected_files) == 1:
            file_channels = [ch for ch in all_channels if ch.file_id == selected_files[0].file_id]
            if len(file_channels) >= 2:
                # Get the two highest step channels that are different
                sorted_channels = sorted(file_channels, key=lambda ch: ch.step, reverse=True)
                # Ensure we pick different channels
                for i, ch1 in enumerate(sorted_channels):
                    for ch2 in sorted_channels[i+1:]:
                        if ch1.channel_id != ch2.channel_id:
                            return ch1, ch2
                
                # Fallback if all channels are the same - use top 2 steps
                if len(sorted_channels) >= 2:
                    return sorted_channels[0], sorted_channels[1]
        
        # PRIORITY 5: Fallback - Use first two available channels if they're different
        if len(all_channels) >= 2:
            for i, ch1 in enumerate(all_channels):
                for ch2 in all_channels[i+1:]:
                    if ch1.channel_id != ch2.channel_id:
                        return ch1, ch2
            
            # If no different channels found, use first two
            return all_channels[0], all_channels[1]
        elif len(all_channels) == 1:
            return all_channels[0], all_channels[0]
        
        return None, None

    def get_mixer_templates(self):
        """Get available mixer templates from the template registry."""
        try:
            # Get templates from the centralized template registry
            templates = MixerRegistry.get_all_templates()
            return templates
        except Exception as e:
            # Fallback to mixer names if template registry fails
            try:
                mixer_names = MixerRegistry.all_mixers()
                return [(name, "mixed") for name in mixer_names]
            except Exception as e2:
                return []

    def generate_next_label(self, current_row_count):
        """Generate the next available label (C, D, E, etc.)."""
        return chr(ord('A') + current_row_count)

    def process_mixer_expression(self, expression, channel_context, channel_name=None):
        """Process a mixer expression and return the result."""
        try:
            if not expression or '=' not in expression:
                return None, "Invalid expression. Use format: C = A + B"

            label, formula = map(str.strip, expression.split('=', 1))
            
            # Add mixed channels to context using their step table labels (C, D, E, etc.)
            context = {**channel_context}
            for i, mixed_channel in enumerate(self.mixed_channels):
                # Use stored step table label if available, otherwise generate one
                step_label = getattr(mixed_channel, 'step_table_label', chr(ord('C') + i))
                context[step_label] = mixed_channel
            
            # Validate that required channels exist
            for ch_name, ch_obj in context.items():
                if ch_obj is None:
                    return None, f"Channel {ch_name} is not available"

            mixer_cls = self._resolve_mixer_class(formula)
            if mixer_cls is None:
                return None, f"Could not resolve mixer for expression: {formula}"
            
            # Parse operation from formula
            operation_params = self._parse_operation_params(formula, label)
            mixer = mixer_cls(label=label, **operation_params)
            
            new_channel = mixer.apply(context)

            # Check if channels A and B have identical x-axis data and share it with the mixed channel
            self._share_identical_xdata_if_possible(new_channel, context)

            # Set channel description to the expression
            new_channel.description = expression
            
            # Store the step table label with the channel for easy reference
            # Use the actual label from the expression (e.g., "Z" from "Z = A * B", "DOG" from "DOG = C/E")
            new_channel.step_table_label = label
            
            # Set the channel name separately from the expression label
            if channel_name:
                # Use the provided channel name for display purposes
                new_channel.legend_label = channel_name
            else:
                # Fallback to generating a name based on the mixer operation
                mixer_name = self._get_mixer_name_from_class(mixer_cls)
                if hasattr(new_channel, 'legend_label') and new_channel.legend_label:
                    original_name = new_channel.legend_label
                else:
                    original_name = new_channel.channel_id
                
                new_channel.legend_label = f"{original_name} - {mixer_name}"
            
            # Check if a channel with this label already exists
            existing_index = None
            for i, existing_channel in enumerate(self.mixed_channels):
                if getattr(existing_channel, 'step_table_label', None) == label:
                    existing_index = i
                    break

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
            pass
        
        # Legacy simple operation parsing for other mixers
        if '%' in formula_clean:
            params['operation'] = 'mod'
        elif '+' in formula_clean:
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
            if expression_mixer:
                # For any formula containing functions or complex expressions, use ExpressionMixer
                if (hasattr(expression_mixer, 'parse_expression_for_mixer') and 
                    expression_mixer.parse_expression_for_mixer(formula)) or \
                   any(func in formula_clean for func in ['sqrt', 'sin', 'cos', 'abs', 'exp', 'log', 'mean', 'sum', 'np.', '**', '(', ')']):
                    return expression_mixer
        except (KeyError, AttributeError):
            pass
        
        # Legacy simple mixer resolution
        try:
            if any(op in formula_clean for op in ['+', '-', '*', '/']):
                arithmetic_mixer = MixerRegistry.get("arithmetic")
                if arithmetic_mixer:
                    return arithmetic_mixer
        except KeyError:
            pass
            
        try:
            if any(fn in formula_clean for fn in ['abs', 'normalize', 'zscore']):
                unary_mixer = MixerRegistry.get("unary")
                if unary_mixer:
                    return unary_mixer
        except KeyError:
            pass
            
        try:
            if any(op in formula_clean for op in ['>', '<', '>=', '<=', '==', '!=']):
                logic_mixer = MixerRegistry.get("logic")
                if logic_mixer:
                    return logic_mixer
        except KeyError:
            pass
            
        try:
            if any(fn in formula_clean for fn in ['clip', 'threshold']):
                threshold_mixer = MixerRegistry.get("threshold")
                if threshold_mixer:
                    return threshold_mixer
        except KeyError:
            pass
        
        # Try to get any available mixer as fallback
        try:
            available_mixers = MixerRegistry.all_mixers()
            if available_mixers:
                first_mixer = MixerRegistry.get(available_mixers[0])
                return first_mixer
        except (KeyError, IndexError):
            pass
        
        return None

    def _share_identical_xdata_if_possible(self, new_channel, context):
        """Check if channels A and B have identical x-axis data and share it with the mixed channel."""
        try:
            # Get channels A and B from context
            channel_a = context.get('A')
            channel_b = context.get('B')
            
            # Only proceed if we have both channels
            if not channel_a or not channel_b:
                return
            
            # Check if both channels have x-axis data
            if (not hasattr(channel_a, 'xdata') or channel_a.xdata is None or 
                not hasattr(channel_b, 'xdata') or channel_b.xdata is None):
                return
            
            # Check if x-axis data lengths match
            if len(channel_a.xdata) != len(channel_b.xdata):
                return
            
            # Check if x-axis data values are identical
            import numpy as np
            if np.allclose(channel_a.xdata, channel_b.xdata, rtol=1e-10, atol=1e-10):
                # X-axis data is identical - share it with the mixed channel
                if new_channel.ydata is not None and len(new_channel.ydata) == len(channel_a.xdata):
                    new_channel.xdata = channel_a.xdata.copy()
        except Exception as e:
            pass

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

    def process_mixer_expression_with_alignment(self, expression, channel_context, alignment_params=None, channel_name=None):
        """Process mixer expression with alignment applied to channels with different dimensions."""
        try:
            # Log the expression processing
            self._log_state_change(f"Processing mixer expression with alignment: {expression}")
            
            # Get channels A and B from context
            channel_a = channel_context.get('A')
            channel_b = channel_context.get('B')
            
            if not channel_a or not channel_b:
                return None, "Channels A and B must be available in context"
            
            # Apply alignment if different dimensions or if alignment is explicitly requested
            len_a = len(channel_a.ydata)
            len_b = len(channel_b.ydata)
            
            if len_a != len_b or (alignment_params and alignment_params.get('alignment_method')):
                if alignment_params:
                    # Use DataAligner with enhanced validation and fallback strategies
                    from data_aligner import DataAligner
                    
                    data_aligner = DataAligner()
                    
                    # Use the DataAligner's align_from_wizard_params method directly
                    alignment_result = data_aligner.align_from_wizard_params(channel_a, channel_b, alignment_params)
                    
                    if not alignment_result.success:
                        return None, f"Failed to align channels: {alignment_result.error_message}"
                    
                    # DEBUG: Print range of x that are being returned from aligner (ref and test)
                    print(f"[DEBUG] Alignment result - ref_data length: {len(alignment_result.ref_data)}")
                    print(f"[DEBUG] Alignment result - test_data length: {len(alignment_result.test_data)}")
                    if hasattr(alignment_result, 'time_data') and alignment_result.time_data is not None:
                        print(f"[DEBUG] Alignment result - time_data range: {alignment_result.time_data[0]:.6f} to {alignment_result.time_data[-1]:.6f}")
                        print(f"[DEBUG] Alignment result - time_data length: {len(alignment_result.time_data)}")
                    else:
                        print(f"[DEBUG] Alignment result - no time_data returned")
                    
                    # Get alignment stats for logging
                    alignment_stats = data_aligner.get_alignment_stats()
                    
                    # Create new channel objects with aligned data
                    aligned_a = self._create_aligned_channel(channel_a, alignment_result.ref_data, alignment_result.time_data, channel_a, channel_b)
                    aligned_b = self._create_aligned_channel(channel_b, alignment_result.test_data, alignment_result.time_data, channel_a, channel_b)
                    
                    # DEBUG: Print downstream x-data information for aligned channels
                    print(f"[DEBUG] Downstream - aligned_a xdata range: {aligned_a.xdata[0]:.6f} to {aligned_a.xdata[-1]:.6f}")
                    print(f"[DEBUG] Downstream - aligned_a xdata length: {len(aligned_a.xdata)}")
                    print(f"[DEBUG] Downstream - aligned_b xdata range: {aligned_b.xdata[0]:.6f} to {aligned_b.xdata[-1]:.6f}")
                    print(f"[DEBUG] Downstream - aligned_b xdata length: {len(aligned_b.xdata)}")
                    
                    # Update context with aligned channels
                    channel_context['A'] = aligned_a
                    channel_context['B'] = aligned_b
                    
                    # Log detailed alignment information
                    self._log_state_change(f"✅ DataAligner: A={len(aligned_a.ydata)}, B={len(aligned_b.ydata)}")
                    if alignment_result.warnings:
                        for warning in alignment_result.warnings:
                            self._log_state_change(f"⚠️ DataAligner warning: {warning}")
                    
                    # Log alignment statistics
                    if alignment_stats.get('datetime_conversions', 0) > 0:
                        self._log_state_change(f"🕐 Applied datetime conversion")
                    if alignment_stats.get('fallback_usage', 0) > 0:
                        self._log_state_change(f"⚠️ Used fallback alignment strategies")
                    
                    # Log quality metrics if available
                    if alignment_result.quality_metrics:
                        retention_ref = alignment_result.quality_metrics.get('data_retention_ref', 0) * 100
                        retention_test = alignment_result.quality_metrics.get('data_retention_test', 0) * 100
                        self._log_state_change(f"📊 Data retention: ref={retention_ref:.1f}%, test={retention_test:.1f}%")
                else:
                    return None, "Channels have different dimensions but no alignment parameters provided"
            
            # Process the expression with (potentially aligned) channels
            return self.process_mixer_expression(expression, channel_context, channel_name)
            
        except Exception as e:
            error_msg = f"Error processing mixer expression with alignment: {str(e)}"
            self._log_state_change(error_msg)
            import traceback
            traceback.print_exc()
            return None, error_msg
    
    def _align_channels(self, ref_channel, test_channel, config):
        """Align channels based on configuration (adapted from comparison wizard)."""
        alignment_method = config.get('alignment_method', 'time')
        
        if alignment_method == 'index':
            return self._align_by_index(ref_channel, test_channel, config)
        else:  # time-based
            return self._align_by_time(ref_channel, test_channel, config)
    
    def _align_by_index(self, ref_channel, test_channel, config):
        """Align channels by index (adapted from comparison wizard)."""
        try:
            # Validate input data
            if not hasattr(ref_channel, 'ydata') or not hasattr(test_channel, 'ydata'):
                raise ValueError("Channels missing data arrays")
            
            ref_data = ref_channel.ydata
            test_data = test_channel.ydata
            
            if ref_data is None or test_data is None:
                raise ValueError("Channel data is None")
            
            if len(ref_data) == 0 or len(test_data) == 0:
                raise ValueError("Channel data is empty")
            
            # Check if both channels have the same xdata for index-based alignment
            shared_xdata = None
            if (hasattr(ref_channel, 'xdata') and ref_channel.xdata is not None and
                hasattr(test_channel, 'xdata') and test_channel.xdata is not None):
                
                # Check if xdata arrays are the same length and have identical values
                if len(ref_channel.xdata) == len(test_channel.xdata):
                    try:
                        if np.allclose(ref_channel.xdata, test_channel.xdata, rtol=1e-10, atol=1e-10):
                            print("[SignalMixerWizardManager] Index-based alignment: channels have identical xdata, preserving for mixed channel")
                            shared_xdata = ref_channel.xdata.copy()
                        else:
                            print("[SignalMixerWizardManager] Index-based alignment: channels have different xdata values")
                    except Exception as e:
                        print(f"[SignalMixerWizardManager] Error comparing xdata: {e}")
                else:
                    print("[SignalMixerWizardManager] Index-based alignment: channels have different xdata lengths")
            else:
                print("[SignalMixerWizardManager] Index-based alignment: one or both channels missing xdata")
            
            # Get configuration parameters
            mode = config.get('mode', 'truncate')
            
            if mode == 'truncate':
                # Truncate to shortest length
                min_length = min(len(ref_data), len(test_data))
                if min_length == 0:
                    raise ValueError("No data points available for alignment")
                
                ref_aligned = ref_data[:min_length].copy()
                test_aligned = test_data[:min_length].copy()
                actual_range = (0, min_length - 1)
                
                # If we have shared xdata, truncate it to match the aligned data
                aligned_xdata = None
                if shared_xdata is not None:
                    aligned_xdata = shared_xdata[:min_length].copy()
                
            else:  # custom
                # Custom range with validation
                start_idx = config.get('start_index', 0)
                end_idx = config.get('end_index', 500)
                
                # Ensure end_idx doesn't exceed data length
                max_idx = min(len(ref_data), len(test_data)) - 1
                if end_idx > max_idx:
                    end_idx = max_idx
                
                # Validate indices
                if start_idx < 0:
                    start_idx = 0
                if end_idx < 0:
                    end_idx = 0
                    
                if start_idx > max_idx:
                    raise ValueError(f"Start index {start_idx} exceeds data length")
                
                if start_idx >= end_idx:
                    raise ValueError(f"Invalid index range: start ({start_idx}) >= end ({end_idx})")
                
                ref_aligned = ref_data[start_idx:end_idx+1].copy()
                test_aligned = test_data[start_idx:end_idx+1].copy()
                actual_range = (start_idx, end_idx)
                
                # If we have shared xdata, slice it to match the custom range
                aligned_xdata = None
                if shared_xdata is not None:
                    # Make sure we don't exceed the xdata bounds
                    if end_idx < len(shared_xdata):
                        aligned_xdata = shared_xdata[start_idx:end_idx+1].copy()
                    else:
                        print(f"[SignalMixerWizardManager] Warning: custom range exceeds xdata bounds, xdata will not be preserved")
                        aligned_xdata = None
                
            # Apply offset if specified
            offset = config.get('offset', 0)
            if offset != 0:
                if offset > 0:
                    # Positive offset: shift test data forward, truncate ref data
                    if offset >= len(test_aligned):
                        raise ValueError(f"Positive offset ({offset}) too large for test data length ({len(test_aligned)})")
                    test_aligned = test_aligned[offset:]
                    ref_aligned = ref_aligned[:len(test_aligned)]
                    
                    # Adjust xdata if we have it
                    if aligned_xdata is not None:
                        if offset < len(aligned_xdata):
                            aligned_xdata = aligned_xdata[:len(test_aligned)]
                        else:
                            print(f"[SignalMixerWizardManager] Warning: offset exceeds xdata bounds, xdata will not be preserved")
                            aligned_xdata = None
                else:
                    # Negative offset: shift ref data forward, truncate test data
                    offset_abs = abs(offset)
                    if offset_abs >= len(ref_aligned):
                        raise ValueError(f"Negative offset magnitude ({offset_abs}) too large for ref data length ({len(ref_aligned)})")
                    ref_aligned = ref_aligned[offset_abs:]
                    test_aligned = test_aligned[:len(ref_aligned)]
                    
                    # Adjust xdata if we have it
                    if aligned_xdata is not None:
                        if offset_abs < len(aligned_xdata):
                            aligned_xdata = aligned_xdata[offset_abs:offset_abs+len(ref_aligned)]
                        else:
                            print(f"[SignalMixerWizardManager] Warning: negative offset exceeds xdata bounds, xdata will not be preserved")
                            aligned_xdata = None
            else:
                # No offset, keep aligned_xdata as is
                pass
            
            # Final validation
            if len(ref_aligned) != len(test_aligned):
                raise ValueError("Aligned data arrays have different lengths")
            
            if len(ref_aligned) == 0:
                raise ValueError("No data points remaining after alignment")
            
            # Prepare result with time_data if we have shared xdata
            result = {
                'ref_data': ref_aligned,
                'test_data': test_aligned,
                'alignment_method': 'index',
                'index_range': actual_range,
                'n_points': len(ref_aligned),
                'offset_applied': offset
            }
            
            # Add time_data if we have preserved xdata
            if aligned_xdata is not None:
                result['time_data'] = aligned_xdata
                print(f"[SignalMixerWizardManager] Preserved xdata for mixed channel ({len(aligned_xdata)} points)")
            else:
                print("[SignalMixerWizardManager] No shared xdata to preserve, mixed channel will use indices")
            
            return result
            
        except Exception as e:
            error_msg = f"Index alignment failed: {str(e)}"
            self._log_state_change(error_msg)
            raise
    
    def _align_by_time(self, ref_channel, test_channel, config):
        """Align channels by time (adapted from comparison wizard)."""
        try:
            # Validate and create time data if needed
            if not self._validate_and_create_time_data(ref_channel, test_channel):
                raise ValueError("Could not create or validate time data")
            
            ref_x = ref_channel.xdata.copy()
            ref_y = ref_channel.ydata.copy()
            test_x = test_channel.xdata.copy()
            test_y = test_channel.ydata.copy()
            
            # Clean and sort data by time
            ref_x, ref_y = self._clean_and_sort_time_data(ref_x, ref_y)
            test_x, test_y = self._clean_and_sort_time_data(test_x, test_y)
            
            # Apply time offset if specified
            offset = config.get('offset', 0.0)
            if offset != 0.0:
                test_x = test_x + offset
                
            # Determine time range
            mode = config.get('mode', 'overlap')
            
            if mode == 'overlap':
                # Find overlapping time range
                start_time = max(ref_x.min(), test_x.min())
                end_time = min(ref_x.max(), test_x.max())
                
                if start_time >= end_time:
                    raise ValueError("No overlapping time range found between channels")
            else:  # custom
                start_time = config.get('start_time', 0.0)
                end_time = config.get('end_time', 10.0)
                
            # Create time grid
            round_to = config.get('round_to', 0.01)
            if round_to <= 0:
                raise ValueError("Round-to value must be positive")
            
            time_grid = np.arange(start_time, end_time + round_to/2, round_to)
            
            if len(time_grid) == 0:
                raise ValueError("Generated time grid is empty")
            
            # Interpolate both channels to common time grid
            interp_method = config.get('interpolation', 'linear')
            
            ref_interp = self._interpolate_channel(ref_x, ref_y, time_grid, interp_method, 'reference')
            test_interp = self._interpolate_channel(test_x, test_y, time_grid, interp_method, 'test')
            
            # Final validation
            if len(ref_interp) != len(test_interp) or len(ref_interp) != len(time_grid):
                raise ValueError("Interpolated data length mismatch")
            
            # Check for excessive NaN values
            valid_mask = ~(np.isnan(ref_interp) | np.isnan(test_interp))
            valid_ratio = np.sum(valid_mask) / len(valid_mask)
            
            if valid_ratio < 0.1:
                raise ValueError(f"Too many invalid values after interpolation ({valid_ratio*100:.1f}% valid)")
            
            return {
                'ref_data': ref_interp,
                'test_data': test_interp,
                'time_data': time_grid,
                'alignment_method': 'time',
                'valid_ratio': valid_ratio,
                'time_range': (start_time, end_time),
                'n_points': len(time_grid),
                'round_to_used': round_to
            }
            
        except Exception as e:
            error_msg = f"Time alignment failed: {str(e)}"
            self._log_state_change(error_msg)
            raise
    
    def _validate_and_create_time_data(self, ref_channel, test_channel):
        """Validate and create time data if needed."""
        try:
            for channel in [ref_channel, test_channel]:
                if not hasattr(channel, 'xdata') or channel.xdata is None:
                    # Try to create time data
                    if hasattr(channel, 'sampling_rate') and channel.sampling_rate is not None and channel.sampling_rate > 0:
                        n_samples = len(channel.ydata)
                        dt = 1.0 / channel.sampling_rate
                        channel.xdata = np.arange(n_samples) * dt
                    else:
                        # Fallback to indices as time
                        channel.xdata = np.arange(len(channel.ydata))
            return True
        except Exception as e:
            self._log_state_change(f"Error creating time data: {e}")
            return False
    
    def _clean_and_sort_time_data(self, x_data, y_data):
        """Clean and sort time data."""
        # Remove NaN values
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        # Sort by time
        sort_indices = np.argsort(x_clean)
        return x_clean[sort_indices], y_clean[sort_indices]
    
    def _interpolate_channel(self, x_data, y_data, time_grid, method, channel_name):
        """Interpolate channel data to time grid."""
        try:
            from scipy.interpolate import interp1d
            
            if method == 'linear':
                f = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value=np.nan)
            elif method == 'nearest':
                f = interp1d(x_data, y_data, kind='nearest', bounds_error=False, fill_value=np.nan)
            elif method == 'cubic':
                if len(x_data) >= 4:  # cubic requires at least 4 points
                    f = interp1d(x_data, y_data, kind='cubic', bounds_error=False, fill_value=np.nan)
                else:
                    # Fallback to linear if not enough points
                    f = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value=np.nan)
            else:
                raise ValueError(f"Unsupported interpolation method: {method}")
            
            return f(time_grid)
            
        except Exception as e:
            self._log_state_change(f"Error interpolating {channel_name} channel: {e}")
            raise
    
    def _create_aligned_channel(self, original_channel, aligned_data, time_data=None, channel_a=None, channel_b=None):
        """Create a new channel with aligned data."""
        from channel import Channel
        
        # Determine xdata to use
        xdata_to_use = None
        
        if time_data is not None:
            # Use provided time data (from DataAligner)
            xdata_to_use = time_data
            print(f"[DEBUG] _create_aligned_channel - using provided time_data with {len(time_data)} points")
        elif channel_a is not None and channel_b is not None:
            # Check if channels A and B have identical xdata
            if (hasattr(channel_a, 'xdata') and channel_a.xdata is not None and 
                hasattr(channel_b, 'xdata') and channel_b.xdata is not None):
                
                # Check if xdata lengths match and values are identical
                if len(channel_a.xdata) == len(channel_b.xdata):
                    try:
                        xdata_identical = np.allclose(channel_a.xdata, channel_b.xdata, rtol=1e-10, atol=1e-10)
                        
                        if xdata_identical:
                            # Channels have identical xdata - use it directly with the aligned data
                            xdata_to_use = channel_a.xdata.copy()
                            print(f"[DEBUG] _create_aligned_channel - using shared xdata from channels A and B")
                        else:
                            # Different xdata values - use indices
                            xdata_to_use = np.arange(len(aligned_data))
                            print(f"[DEBUG] _create_aligned_channel - using indices (different xdata values)")
                    except Exception as e:
                        # Error comparing xdata - use indices
                        xdata_to_use = np.arange(len(aligned_data))
                        print(f"[DEBUG] _create_aligned_channel - using indices (error comparing xdata: {e})")
                else:
                    # Different xdata lengths - use indices
                    xdata_to_use = np.arange(len(aligned_data))
                    print(f"[DEBUG] _create_aligned_channel - using indices (different xdata lengths)")
            else:
                # Missing xdata - use indices
                xdata_to_use = np.arange(len(aligned_data))
                print(f"[DEBUG] _create_aligned_channel - using indices (missing xdata)")
        else:
            # No channels provided for comparison - use indices
            xdata_to_use = np.arange(len(aligned_data))
        
        # Create new channel with aligned data
        aligned_channel = Channel(
            filename=original_channel.filename,
            file_id=original_channel.file_id,
            channel_id=original_channel.channel_id,
            xlabel=original_channel.xlabel,
            ylabel=original_channel.ylabel,
            step=original_channel.step,
            xdata=xdata_to_use,
            ydata=aligned_data,
            legend_label=original_channel.legend_label,
            description=original_channel.description,
            tags=original_channel.tags,
            metadata=original_channel.metadata
        )
        
        # Set show property after creation (it defaults to True in constructor)
        aligned_channel.show = original_channel.show
        
        return aligned_channel

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
