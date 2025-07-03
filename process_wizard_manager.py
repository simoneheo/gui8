from PySide6.QtWidgets import QTableWidgetItem, QComboBox
from PySide6.QtCore import Qt
import time
import itertools
from typing import Dict, Optional, Any, Callable

class ProcessWizardManager:
    """
    Manager for the Process Wizard that handles:
    - Processing step application
    - Parameter management
    - State tracking and statistics
    """
    
    def __init__(self, ui=None, registry=None, channel_lookup=None, signal_bus=None):
        # Store references
        self.ui = ui  # ProcessWizardWindow instance
        self.registry = registry  # ProcessRegistry instance
        self.channel_lookup = channel_lookup  # function to return current channel
        self.signal_bus = signal_bus
        
        # Initialize state tracking
        self._stats = {
            'total_steps_applied': 0,
            'successful_steps': 0,
            'failed_steps': 0,
            'last_step_time': None,
            'session_start': time.time()
        }
        
        # Processing state
        self.pending_step = None
        
        # Undo state tracking
        self.undo_stack = []  # List of (step_name, parent_channel_id, new_channel_ids) tuples
        self.max_undo_steps = 10  # Maximum number of undo steps to remember
        
        # Validate initialization
        if not self._validate_initialization():
            raise ValueError("Invalid initialization parameters for ProcessWizardManager")
        
        # Connect UI signals
        self._connect_ui()
        
        # Log initialization
        self._log_state_change("Process manager initialized successfully")

    def _validate_initialization(self) -> bool:
        """Validate initialization parameters"""
        if not self.ui:
            print("[ProcessWizardManager] ERROR: UI reference not provided")
            return False
            
        if not self.registry:
            print("[ProcessWizardManager] ERROR: Registry not provided")
            return False
            
        if not self.channel_lookup:
            print("[ProcessWizardManager] ERROR: Channel lookup function not provided")
            return False
            
        return True
        
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[ProcessWizardManager {timestamp}] {message}")
        
    def get_stats(self) -> Dict:
        """Get comprehensive processing statistics"""
        return {
            **self._stats,
            'pending_step': self.pending_step.name if self.pending_step else None,
            'success_rate': (
                self._stats['successful_steps'] / max(1, self._stats['total_steps_applied']) * 100
            ),
            'session_duration': time.time() - self._stats['session_start']
        }

    def _connect_ui(self):
        self.ui.filter_list.itemClicked.connect(self._on_filter_selected)
        # Note: The apply step button is handled by the window's _on_console_input method
        # No need to connect add_filter_btn here as it's handled in the window

    def _on_filter_selected(self, item=None):
        """Handle filter selection from the list or programmatically."""
        # Allow calling manually with no item (get from UI)
        if item is None:
            item = self.ui.filter_list.currentItem()
            if not item:
                return

        step_name = item.text()
        step_cls = self.registry.get(step_name)
        if not step_cls:
            self.ui.console_output.setPlainText(f"Error: Step '{step_name}' not found.")
            return

        self.pending_step = step_cls
        prompt = step_cls.get_prompt()
        print(f"[ProcessWizardManager] Prompt: {prompt}")
        info = prompt.get("info", "")
        params = prompt.get("params", [])

        # Display only the step description (parameters are shown in the table)
        self.ui.console_output.setPlainText(info)
        

        # Update channel name entry with default name
        self._update_channel_name_entry(step_name)

        # Populate parameter table with dynamic values for special parameters
        current_channel = self.channel_lookup()
        
        # Clear existing table
        self.ui.param_table.setRowCount(0)
        
        for i, p in enumerate(params):
            param_name = p["name"]
            default_value = p.get("default", "")
            
            # Dynamic prefilling for special parameters
            if param_name == "fs" and current_channel and hasattr(current_channel, 'fs_median') and current_channel.fs_median:
                # Use the actual sampling frequency from the selected channel, rounded to 3 decimal places
                rounded_fs = round(current_channel.fs_median, 3)
                param_value = str(rounded_fs)
                print(f"[ProcessWizardManager] Dynamic prefill: {param_name}={rounded_fs} (from channel.fs_median)")
            else:
                # Use static default value
                param_value = str(default_value)
            
            # Add row to table
            self.ui.param_table.insertRow(i)
            
            # Create parameter name item with tooltip (uneditable)
            param_name_item = QTableWidgetItem(param_name)
            param_name_item.setFlags(param_name_item.flags() & ~Qt.ItemIsEditable)  # Make uneditable
            help_text = p.get("help", "")
            if help_text:
                tooltip = f"{param_name}: {help_text}"
            else:
                tooltip = f"Parameter: {param_name}"
            param_name_item.setToolTip(tooltip)
            self.ui.param_table.setItem(i, 0, param_name_item)
            
            # Check if parameter has predefined options for dropdown
            if "options" in p and p["options"]:
                # Create dropdown combo box
                combo = QComboBox()
                combo.addItems([str(option) for option in p["options"]])
                
                # Set default selection
                if param_value in p["options"]:
                    combo.setCurrentText(str(param_value))
                elif str(param_value) in [str(option) for option in p["options"]]:
                    combo.setCurrentText(str(param_value))
                
                combo.setToolTip(tooltip)
                self.ui.param_table.setCellWidget(i, 1, combo)
            else:
                # Create regular text item
                param_value_item = QTableWidgetItem(param_value)
                param_value_item.setToolTip(tooltip)
                self.ui.param_table.setItem(i, 1, param_value_item)
        
        print(f"[ProcessWizardManager] Populated parameter table with {len(params)} parameters")
    
    def _update_channel_name_entry(self, step_name: str):
        """Update the channel name entry with a default name based on current channel and step."""
        try:
            # Get the current input channel
            current_channel = self.channel_lookup()
            if not current_channel:
                # Fallback to generic name if no channel selected
                default_name = f"New {step_name}"
            else:
                # Generate name based on current channel and step
                base_name = current_channel.legend_label or current_channel.ylabel or "Signal"
                # Clean up the step name (remove underscores, capitalize)
                clean_step = step_name.replace('_', ' ').title()
                default_name = f"{base_name} - {clean_step}"
            
            # Set the default name in the UI entry field
            if hasattr(self.ui, 'channel_name_entry'):
                self.ui.channel_name_entry.setText(default_name)
                print(f"[ProcessWizardManager] Set default channel name: {default_name}")
            
        except Exception as e:
            print(f"[ProcessWizardManager] Error updating channel name entry: {e}")
    
    def on_input_submitted(self, user_input_dict: dict):
        """Handle input submission with parameters from table."""
        if not self.pending_step:
            print("[ProcessWizardManager] No pending step to apply")
            return

        print(f"[ProcessWizardManager] Received params from table: {user_input_dict}")
        
        try:
            params = self.pending_step.parse_input(user_input_dict)
            print(f"[ProcessWizardManager] Parsed parameters: {params}")
        except Exception as e:
            self.ui.console_output.setPlainText(f"Parameter parsing error: {e}")
            return

        # Get the parent channel
        parent_channel = self.channel_lookup()
        if not parent_channel:
            print("[ProcessWizardManager] No parent channel selected")
            return

        print(f"[ProcessWizardManager] Selected channel: {parent_channel.channel_id} (step {parent_channel.step})")

        # Get all channels in the lineage
        lineage_dict = self.ui.channel_manager.get_channels_by_lineage(parent_channel.lineage_id)
        print(f"[ProcessWizardManager] Found lineage dict for {parent_channel.lineage_id}")
        
        # Collect all channels from the lineage (parents, children, siblings)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        print(f"[ProcessWizardManager] Found {len(all_lineage_channels)} channels in lineage {parent_channel.lineage_id}")

        # Use the most recent channel in the lineage as the parent, or fall back to the original
        if all_lineage_channels:
            parent_channel = max(all_lineage_channels, key=lambda ch: ch.step)
            print(f"[ProcessWizardManager] Using parent channel {parent_channel.channel_id} (step {parent_channel.step})")
        else:
            print(f"[ProcessWizardManager] No lineage channels found, using original parent {parent_channel.channel_id}")

        print(f"[ProcessWizardManager] Applying step to parent channel {parent_channel.channel_id}")
        
        # Update statistics
        self._stats['total_steps_applied'] += 1
        self._stats['last_step_time'] = time.time()
        
        try:
            new_channels = self.pending_step.apply(parent_channel, params)
            # Handle both single channel and list of channels
            if not isinstance(new_channels, list):
                new_channels = [new_channels]
            
            # Check for empty channels and filter them out
            valid_channels = []
            for new_channel in new_channels:
                # Check if channel has empty or invalid data
                if (hasattr(new_channel, 'ydata') and hasattr(new_channel, 'xdata') and
                    new_channel.ydata is not None and new_channel.xdata is not None and
                    len(new_channel.ydata) > 0 and len(new_channel.xdata) > 0):
                    valid_channels.append(new_channel)
            
            # If no valid channels were created, inform the user
            if not valid_channels:
                step_name = self.pending_step.name if self.pending_step else "Unknown step"
                self.ui.console_output.setPlainText(
                    f"Step '{step_name}' completed but produced no results.\n"
                    f"This may happen when:\n"
                    f"• No events/features match the specified criteria\n"
                    f"• Parameter values are too restrictive\n"
                    f"• Input signal doesn't contain the target patterns\n\n"
                    f"Try adjusting the parameters and running again."
                )
                self.pending_step = None
                self.ui.param_table.setRowCount(0)
                return
            
            # Process valid channels
            for new_channel in valid_channels:
                new_channel.step = parent_channel.step + 1
                # Ensure lineage ID is inherited from parent
                new_channel.lineage_id = parent_channel.lineage_id
                # Set parent ID
                new_channel.parent_ids = [parent_channel.channel_id]
                
                # Apply custom channel name from entry field if provided
                if (hasattr(self.ui, 'channel_name_entry') and 
                    self.ui.channel_name_entry.text().strip()):
                    custom_name = self.ui.channel_name_entry.text().strip()
                    new_channel.legend_label = custom_name
                    new_channel.ylabel = custom_name
                    print(f"[ProcessWizardManager] Applied custom name: {custom_name}")
                
                # Assign a unique color to the new channel based on existing channels in the file
                if not hasattr(new_channel, 'color') or new_channel.color is None:
                    # Get all channels from the same file to count existing colors
                    file_channels = self.ui.channel_manager.get_channels_by_file(parent_channel.file_id)
                    existing_colors = set()
                    for ch in file_channels:
                        if hasattr(ch, 'color') and ch.color is not None:
                            existing_colors.add(ch.color)
                    
                    # Define color palette
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Find the first available color not used by other channels in the file
                    available_colors = [c for c in colors if c not in existing_colors]
                    if available_colors:
                        new_channel.color = available_colors[0]
                    else:
                        # If all colors are used, cycle through them based on total channel count
                        color_index = len(file_channels) % len(colors)
                        new_channel.color = colors[color_index]
                
                print(f"[ProcessWizardManager] Created new channel {new_channel.channel_id} (step {new_channel.step}, lineage {parent_channel.lineage_id}, color {new_channel.color})")
                
                # Add the new channel to the manager
                self.ui.channel_manager.add_channel(new_channel)
            
            # Update new_channels to only include valid ones
            new_channels = valid_channels
            
            # Set the most recent new channel as the input for next step
            if new_channels:
                self.ui.input_ch = new_channels[-1]  # Use the last created channel
                print(f"[ProcessWizardManager] Set input_ch to new channel: {new_channels[-1].channel_id}")
            
            # Store step name before clearing pending_step
            step_name = self.pending_step.name if self.pending_step else "Unknown step"
            
            # Clear the pending step and channel name entry
            self.pending_step = None
            self.ui.param_table.setRowCount(0)
            self.ui.console_output.clear()
            
            # Clear the channel name entry for next use
            if hasattr(self.ui, 'channel_name_entry'):
                self.ui.channel_name_entry.clear()
            
        except Exception as e:
            # Update failure statistics
            self._stats['failed_steps'] += 1
            self._log_state_change(f"Step execution failed: {str(e)}")
            
            self.ui.console_output.setPlainText(f"Step execution failed: {e}")
            return

        self.ui.console_output.setPlainText(
            f"Step '{step_name}' applied successfully.\n"
            f"New channel: {new_channels[0].channel_id}\n"
            f"Description: {new_channels[0].description}\n"
            f"Step: {new_channels[0].step}\n"
            f"Parent: {parent_channel.channel_id}"
        )

        # Update UI
        self.ui._update_file_selector()
        self.ui._update_channel_selector()
        self.ui._update_step_table()
        self.ui._update_plot()

        print(f"[ProcessWizardManager] Created {len(new_channels)} new channel(s)")
        
        if len(new_channels) == 1:
            self.ui.console_output.setPlainText(
                f"Step applied successfully.\nNew channel: {new_channels[0].channel_id}"
            )
        else:
            channel_list = ", ".join([ch.channel_id for ch in new_channels])
            self.ui.console_output.setPlainText(
                f"Step applied successfully.\nCreated {len(new_channels)} channels: {channel_list}"
            )
        
        # Clear the channel name entry for next use
        if hasattr(self.ui, 'channel_name_entry'):
            self.ui.channel_name_entry.clear()
        
        # Clear the parameter table and pending step
        self.ui.param_table.setRowCount(0)
        self.pending_step = None
        
        return new_channels[-1] if new_channels else None  # Return the last channel for UI consistency

    def apply_pending_step(self):
        if not self.pending_step:
            self.ui.console_output.setPlainText("No step selected.")
            return None

        # Collect parameters from table
        user_input_dict = {}
        for row in range(self.ui.param_table.rowCount()):
            key_item = self.ui.param_table.item(row, 0)
            if key_item:
                key = key_item.text().strip()
                if key:
                    # Check if the value cell contains a widget (dropdown) or text item
                    widget = self.ui.param_table.cellWidget(row, 1)
                    if widget:
                        # It's a QComboBox dropdown
                        from PySide6.QtWidgets import QComboBox
                        if isinstance(widget, QComboBox):
                            val = widget.currentText().strip()
                        else:
                            val = ""
                    else:
                        # It's a regular text item
                        val_item = self.ui.param_table.item(row, 1)
                        val = val_item.text().strip() if val_item else ""
                    
                    if val:
                        # Try to convert to appropriate type
                        try:
                            if '.' in val:
                                user_input_dict[key] = float(val)
                            else:
                                user_input_dict[key] = int(val)
                        except ValueError:
                            user_input_dict[key] = val  # Keep as string if conversion fails
        
        try:
            print(f"[ProcessWizardManager] User input dict from table: {user_input_dict}")
            params = self.pending_step.parse_input(user_input_dict)
            print(f"[ProcessWizardManager] Final parsed parameters: {params}")
            
            # Debug: Check what the step class expects vs what we're providing
            step_params = self.pending_step.get_prompt().get("params", [])
            expected_param_names = [p["name"] for p in step_params if p["name"] != "fs"]
            print(f"[ProcessWizardManager] Step expects parameters: {expected_param_names}")
            print(f"[ProcessWizardManager] We provided parameters: {list(user_input_dict.keys())}")
        except Exception as e:
            self.ui.console_output.setPlainText(f"Input error: {e}")
            return None

        # Get the exact channel selected by radio button (don't override it)
        selected_channel = self.channel_lookup()
        if not selected_channel:
            self.ui.console_output.setPlainText("No active channel.")
            return None
            
        print(f"[ProcessWizardManager] Using radio-button-selected channel: {selected_channel.channel_id} (step {selected_channel.step})")

        # Use the radio-button-selected channel directly as parent (don't find "most recent")
        parent_channel = selected_channel
        print(f"[ProcessWizardManager] Applying step to parent channel {parent_channel.channel_id} (step {parent_channel.step})")

        try:
            new_channels = self.pending_step.apply(parent_channel, params)
            # Handle both single channel and list of channels
            if not isinstance(new_channels, list):
                new_channels = [new_channels]
            
            # Check for empty channels and filter them out
            valid_channels = []
            empty_channels = []
            
            for new_channel in new_channels:
                # Check if channel has empty or invalid data
                if (hasattr(new_channel, 'ydata') and hasattr(new_channel, 'xdata') and
                    new_channel.ydata is not None and new_channel.xdata is not None and
                    len(new_channel.ydata) > 0 and len(new_channel.xdata) > 0):
                    valid_channels.append(new_channel)
                else:
                    empty_channels.append(new_channel)
            
            # If no valid channels were created, inform the user
            if not valid_channels:
                step_name = self.pending_step.name if self.pending_step else "Unknown step"
                self.ui.console_output.setPlainText(
                    f"Step '{step_name}' completed but produced no results.\n"
                    f"This may happen when:\n"
                    f"• No events/features match the specified criteria\n"
                    f"• Parameter values are too restrictive\n"
                    f"• Input signal doesn't contain the target patterns\n\n"
                    f"Try adjusting the parameters and running again."
                )
                self.ui.param_table.setRowCount(0)
                self.pending_step = None
                return None
            
            # Process valid channels
            new_channel_ids = []  # Track created channel IDs for undo
            for new_channel in valid_channels:
                new_channel.step = parent_channel.step + 1
                
                # Apply custom channel name from entry field if provided
                if (hasattr(self.ui, 'channel_name_entry') and 
                    self.ui.channel_name_entry.text().strip()):
                    custom_name = self.ui.channel_name_entry.text().strip()
                    new_channel.legend_label = custom_name
                    new_channel.ylabel = custom_name
                    print(f"[ProcessWizardManager] Applied custom name: {custom_name}")
                
                # Assign a unique color to the new channel based on existing channels in the file
                if not hasattr(new_channel, 'color') or new_channel.color is None:
                    # Get all channels from the same file to count existing colors
                    file_channels = self.ui.channel_manager.get_channels_by_file(parent_channel.file_id)
                    existing_colors = set()
                    for ch in file_channels:
                        if hasattr(ch, 'color') and ch.color is not None:
                            existing_colors.add(ch.color)
                    
                    # Define color palette
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    
                    # Find the first available color not used by other channels in the file
                    available_colors = [c for c in colors if c not in existing_colors]
                    if available_colors:
                        new_channel.color = available_colors[0]
                    else:
                        # If all colors are used, cycle through them based on total channel count
                        color_index = len(file_channels) % len(colors)
                        new_channel.color = colors[color_index]
                
                print(f"[ProcessWizardManager] Created new channel {new_channel.channel_id} (step {new_channel.step}, lineage {parent_channel.lineage_id}, color {new_channel.color})")
                
                # Add the new channel to the manager
                self.ui.channel_manager.add_channel(new_channel)
                new_channel_ids.append(new_channel.channel_id)
            
            # Update new_channels to only include valid ones
            new_channels = valid_channels
            
            # Add operation to undo stack
            step_name = self.pending_step.name if self.pending_step else "Unknown step"
            self._add_to_undo_stack(step_name, parent_channel.channel_id, new_channel_ids)
            
            # Inform user if some channels were empty
            if empty_channels:
                print(f"[ProcessWizardManager] Filtered out {len(empty_channels)} empty channel(s)")
                
        except Exception as e:
            self.ui.console_output.setPlainText(f"Step execution failed: {e}")
            return None

        # Set the most recent new channel as the input for next step
        if new_channels:
            self.ui.input_ch = new_channels[-1]  # Use the last created channel
            print(f"[ProcessWizardManager] Set input_ch to new channel: {new_channels[-1].channel_id}")
        
        print(f"[ProcessWizardManager] Created {len(new_channels)} new channel(s)")
        
        # Get the step name for the console message
        step_name = self.pending_step.name if self.pending_step else "Unknown step"
        # Clean up the step name for display
        display_step_name = step_name.replace('_', ' ').title()
        
        if len(new_channels) == 1:
            # Get the channel name for display
            channel_name = new_channels[0].legend_label or new_channels[0].ylabel or f"Channel {new_channels[0].channel_id}"
            self.ui.console_output.setPlainText(
                f"Filter Applied: {display_step_name}\n"
                f"New Channel: {channel_name}"
            )
        else:
            # For multiple channels, show names if available
            channel_names = []
            for ch in new_channels:
                name = ch.legend_label or ch.ylabel or f"Channel {ch.channel_id}"
                channel_names.append(name)
            channel_list = ", ".join(channel_names)
            self.ui.console_output.setPlainText(
                f"Filter Applied: {display_step_name}\n"
                f"Created {len(new_channels)} channels: {channel_list}"
            )
        
        # Clear the channel name entry for next use
        if hasattr(self.ui, 'channel_name_entry'):
            self.ui.channel_name_entry.clear()
        
        # Clear the parameter table and pending step
        self.ui.param_table.setRowCount(0)
        self.pending_step = None
        
        return new_channels[-1] if new_channels else None  # Return the last channel for UI consistency

    def can_undo(self) -> bool:
        """Check if undo operation is available."""
        return len(self.undo_stack) > 0

    def get_undo_info(self) -> Optional[Dict]:
        """Get information about the last operation that can be undone."""
        if not self.can_undo():
            return None
        
        last_operation = self.undo_stack[-1]
        return {
            'step_name': last_operation[0],
            'parent_channel_id': last_operation[1],
            'new_channel_ids': last_operation[2]
        }

    def undo_last_step(self) -> bool:
        """Undo the last processing step."""
        if not self.can_undo():
            return False
        
        try:
            # Get the last operation from the stack
            step_name, parent_channel_id, new_channel_ids = self.undo_stack.pop()
            
            # Remove the channels that were created by this step
            removed_channels = []
            for channel_id in new_channel_ids:
                channel = self.ui.channel_manager.get_channel(channel_id)
                if channel:
                    self.ui.channel_manager.remove_channel(channel_id)
                    removed_channels.append(channel)
            
            # Log the undo operation
            self._log_state_change(f"Undid step '{step_name}': removed {len(removed_channels)} channels")
            
            # Update statistics
            self._stats['total_steps_applied'] = max(0, self._stats['total_steps_applied'] - 1)
            self._stats['successful_steps'] = max(0, self._stats['successful_steps'] - 1)
            
            return True
            
        except Exception as e:
            self._log_state_change(f"Undo operation failed: {str(e)}")
            return False

    def _add_to_undo_stack(self, step_name: str, parent_channel_id: str, new_channel_ids: list):
        """Add an operation to the undo stack."""
        self.undo_stack.append((step_name, parent_channel_id, new_channel_ids))
        
        # Limit the size of the undo stack
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)  # Remove oldest operation

