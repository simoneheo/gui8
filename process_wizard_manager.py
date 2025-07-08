from PySide6.QtWidgets import QTableWidgetItem, QComboBox, QTextEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QWidget, QHBoxLayout
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
            return False
            
        if not self.registry:
            return False
            
        if not self.channel_lookup:
            return False
            
        return True
        
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        pass
        
    def _calculate_smart_target_fs(self, original_fs: float) -> float:
        """Calculate a smart default target sampling frequency based on the original fs"""
        if original_fs <= 0:
            return 10.0  # Fallback default
        
        # Common downsampling ratios and their typical use cases
        downsample_options = [
            (1.0, "Same rate"),      # No downsampling
            (0.5, "Half rate"),      # 2x downsampling
            (0.25, "Quarter rate"),  # 4x downsampling
            (0.1, "Tenth rate"),     # 10x downsampling
            (0.05, "Twentieth rate"), # 20x downsampling
            (0.01, "Hundredth rate") # 100x downsampling
        ]
        
        # Choose based on original sampling frequency
        if original_fs >= 1000:
            # High frequency signals - suggest 10x or 20x downsampling
            target_ratio = 0.1 if original_fs >= 5000 else 0.05
        elif original_fs >= 100:
            # Medium frequency signals - suggest 4x or 10x downsampling
            target_ratio = 0.25 if original_fs >= 500 else 0.1
        elif original_fs >= 10:
            # Low frequency signals - suggest 2x or 4x downsampling
            target_ratio = 0.5 if original_fs >= 50 else 0.25
        else:
            # Very low frequency - minimal downsampling
            target_ratio = 0.5
        
        # Calculate target frequency
        target_fs = original_fs * target_ratio
        
        # Round to nice numbers
        if target_fs >= 100:
            # Round to nearest 10
            target_fs = round(target_fs / 10) * 10
        elif target_fs >= 10:
            # Round to nearest 5
            target_fs = round(target_fs / 5) * 5
        elif target_fs >= 1:
            # Round to nearest 1
            target_fs = round(target_fs)
        else:
            # Round to nearest 0.1
            target_fs = round(target_fs, 1)
        
        # Ensure minimum reasonable value
        if target_fs < 0.1:
            target_fs = 0.1
        
        return target_fs
        
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
        print(f"[ProcessWizardManager] === FILTER SELECTION STARTED ===")
        
        # Removed stack trace for cleaner output after debugging
        
        # Check if UI is still initializing
        if hasattr(self.ui, '_initializing') and self.ui._initializing:
            print(f"[ProcessWizardManager] UI still initializing, skipping filter selection")
            return
        
        # Basic try-catch around the entire method
        try:
            # Step 1: Basic item validation
            if item is None:
                item = self.ui.filter_list.currentItem()
                if not item:
                    print(f"[ProcessWizardManager] No item selected")
                    return

            step_name = item.text()
            print(f"[ProcessWizardManager] Selected step: {step_name}")
            
            # Step 2: Set basic console output 
            try:
                print(f"[ProcessWizardManager] Setting console output...")
                self.ui.console_output.setPlainText(f"Loading step: {step_name}...")
                print(f"[ProcessWizardManager] Console output set successfully")
            except Exception as console_e:
                print(f"[ProcessWizardManager] ERROR setting console output: {console_e}")
                return
            
            # Step 3: Clear parameter table
            try:
                print(f"[ProcessWizardManager] Clearing parameter table...")
                self.ui.param_table.setRowCount(0)
                print(f"[ProcessWizardManager] Parameter table cleared successfully")
            except Exception as table_e:
                print(f"[ProcessWizardManager] ERROR clearing parameter table: {table_e}")
                return
                
            # Step 4: If we get here, try minimal registry access
            if not self.registry:
                print(f"[ProcessWizardManager] Registry is None!")
                self.ui.console_output.setPlainText(f"Error: Registry not available")
                return
                
            print(f"[ProcessWizardManager] Registry exists, type: {type(self.registry)}")
            
            # Step 5: Try very basic registry access
            try:
                # Just check if the registry has the method, don't call it yet
                has_get = hasattr(self.registry, 'get')
                print(f"[ProcessWizardManager] Registry has get method: {has_get}")
                
                if not has_get:
                    self.ui.console_output.setPlainText(f"Error: Registry missing 'get' method")
                    return
                    
                # Now try to actually call the method
                print(f"[ProcessWizardManager] About to call registry.get('{step_name}')...")
                step_cls = self.registry.get(step_name)
                print(f"[ProcessWizardManager] SUCCESS: Registry.get() returned: {step_cls}")
                
            except Exception as reg_e:
                print(f"[ProcessWizardManager] ERROR in registry access: {reg_e}")
                import traceback
                traceback.print_exc()
                self.ui.console_output.setPlainText(f"Error accessing step '{step_name}': {reg_e}")
                return
            
            # Step 6: Basic validation of step class
            if not step_cls:
                print(f"[ProcessWizardManager] Step class is None for {step_name}")
                self.ui.console_output.setPlainText(f"Error: Step '{step_name}' not found.")
                return

            print(f"[ProcessWizardManager] Step class loaded successfully: {step_cls}")
            self.pending_step = step_cls
            
            # Step 7: Get step prompt and info
            try:
                prompt = step_cls.get_prompt()
                info = prompt.get("info", "")
                params = prompt.get("params", [])
                print(f"[ProcessWizardManager] Step prompt loaded, {len(params)} parameters")
            except Exception as prompt_e:
                print(f"[ProcessWizardManager] Error getting step prompt: {prompt_e}")
                self.ui.console_output.setPlainText(f"Error getting step information: {prompt_e}")
                return

            # Step 8: Get current channel for intelligent defaults
            current_channel = None
            try:
                current_channel = self.channel_lookup()
                print(f"[ProcessWizardManager] Current channel: {current_channel}")
            except Exception as channel_e:
                print(f"[ProcessWizardManager] Error getting current channel: {channel_e}")
            
            # Step 9: Try to get intelligent defaults with robust error handling
            intelligent_defaults = None
            try:
                print(f"[ProcessWizardManager] Attempting to get intelligent defaults...")
                from steps.default_config import get_intelligent_defaults, format_intelligent_default_info
                
                # Only try intelligent defaults if we have a valid channel
                if current_channel:
                    intelligent_defaults = get_intelligent_defaults(step_name, current_channel)
                    if intelligent_defaults:
                        print(f"[ProcessWizardManager] Using intelligent defaults for {step_name}: {intelligent_defaults}")
                        # Add intelligent defaults info to the console output
                        try:
                            intelligent_info = format_intelligent_default_info(step_name, current_channel, intelligent_defaults)
                            if intelligent_info:
                                info = f"{info}\n\n{intelligent_info}"
                        except Exception as format_e:
                            print(f"[ProcessWizardManager] Error formatting intelligent defaults info: {format_e}")
                    else:
                        print(f"[ProcessWizardManager] No intelligent defaults available for {step_name}")
                else:
                    print(f"[ProcessWizardManager] No current channel, skipping intelligent defaults")
            except ImportError as import_e:
                print(f"[ProcessWizardManager] Could not import intelligent defaults: {import_e}")
            except Exception as e:
                print(f"[ProcessWizardManager] Error with intelligent defaults: {e}")
                import traceback
                traceback.print_exc()

            # Step 10: Display step description and intelligent defaults info
            print(f"[ProcessWizardManager] Setting console output...")
            self.ui.console_output.setPlainText(info)
            
            # Step 11: Update channel name entry with default name
            print(f"[ProcessWizardManager] Updating channel name entry...")
            try:
                self._update_channel_name_entry(step_name)
            except Exception as name_e:
                print(f"[ProcessWizardManager] Error updating channel name: {name_e}")

            # Step 12: Populate parameter table
            print(f"[ProcessWizardManager] Populating parameter table with {len(params)} parameters...")
            try:
                for i, p in enumerate(params):
                    try:
                        print(f"[ProcessWizardManager] Processing parameter {i}: {p}")
                        param_name = p["name"]
                        default_value = p.get("default", "")
                        
                        # Use intelligent default if available, otherwise use legacy dynamic logic or hard-coded default
                        if intelligent_defaults and param_name in intelligent_defaults:
                            param_value = intelligent_defaults[param_name]
                            print(f"[ProcessWizardManager] Using intelligent default for {param_name}: {param_value}")
                        elif param_name == "fs" and current_channel and hasattr(current_channel, 'fs_median') and current_channel.fs_median:
                            # Legacy dynamic prefilling for sampling frequency
                            try:
                                rounded_fs = round(current_channel.fs_median, 3)
                                param_value = str(rounded_fs)
                                print(f"[ProcessWizardManager] Using channel fs for {param_name}: {param_value}")
                            except Exception as fs_e:
                                print(f"[ProcessWizardManager] Error calculating fs: {fs_e}")
                                param_value = str(default_value)
                        elif param_name == "target_fs" and current_channel and hasattr(current_channel, 'fs_median') and current_channel.fs_median:
                            # Legacy smart default for target sampling frequency
                            try:
                                smart_target_fs = self._calculate_smart_target_fs(current_channel.fs_median)
                                param_value = str(smart_target_fs)
                                print(f"[ProcessWizardManager] Using smart target_fs for {param_name}: {param_value}")
                            except Exception as target_fs_e:
                                print(f"[ProcessWizardManager] Error calculating target_fs: {target_fs_e}")
                                param_value = str(default_value)
                        else:
                            # Use static default value as fallback
                            param_value = str(default_value)
                            print(f"[ProcessWizardManager] Using default value for {param_name}: {param_value}")
                        
                        # Add row to table
                        print(f"[ProcessWizardManager] Adding row {i} to parameter table")
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
                        
                        # Check parameter type and create appropriate control
                        param_type = p.get("type", "string")
                        print(f"[ProcessWizardManager] Creating control for {param_name}, type: {param_type}")
                        
                        if "options" in p and p["options"]:
                            # Create dropdown combo box for parameters with predefined options
                            try:
                                print(f"[ProcessWizardManager] Creating combo box for {param_name}")
                                combo = QComboBox()
                                combo.addItems([str(option) for option in p["options"]])
                                
                                # Set default selection
                                if param_value in p["options"]:
                                    combo.setCurrentText(str(param_value))
                                elif str(param_value) in [str(option) for option in p["options"]]:
                                    combo.setCurrentText(str(param_value))
                                
                                combo.setToolTip(tooltip)
                                self.ui.param_table.setCellWidget(i, 1, combo)
                                print(f"[ProcessWizardManager] Combo box created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating combo box for {param_name}: {e}")
                                # Fallback to regular text item
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                                
                        elif param_type == "bool" or param_type == "boolean":
                            # Create checkbox for boolean parameters
                            try:
                                print(f"[ProcessWizardManager] Creating checkbox for {param_name}")
                                from PySide6.QtWidgets import QCheckBox, QWidget, QHBoxLayout
                                
                                # Convert param_value to boolean
                                bool_value = False
                                if isinstance(param_value, bool):
                                    bool_value = param_value
                                elif isinstance(param_value, str):
                                    bool_value = param_value.lower() in ['true', '1', 'yes', 'on']
                                else:
                                    bool_value = bool(param_value)
                                
                                checkbox = QCheckBox()
                                checkbox.setChecked(bool_value)
                                checkbox.setToolTip(tooltip)
                                
                                # Center the checkbox in the cell
                                checkbox_widget = QWidget()
                                checkbox_layout = QHBoxLayout(checkbox_widget)
                                checkbox_layout.addWidget(checkbox)
                                checkbox_layout.setAlignment(Qt.AlignCenter)
                                checkbox_layout.setContentsMargins(0, 0, 0, 0)
                                
                                self.ui.param_table.setCellWidget(i, 1, checkbox_widget)
                                print(f"[ProcessWizardManager] Checkbox created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating checkbox for {param_name}: {e}")
                                # Fallback to regular text item
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                            
                        elif param_type == "multiline":
                            # Create multiline text editor for code/long text
                            try:
                                print(f"[ProcessWizardManager] Creating text edit for {param_name}")
                                text_edit = QTextEdit()
                                text_edit.setPlainText(str(param_value))
                                text_edit.setToolTip(tooltip)
                                text_edit.setMaximumHeight(150)  # Limit height to fit in table
                                text_edit.setMinimumHeight(80)   # Ensure readable height
                                
                                # Set font for better code readability
                                from PySide6.QtGui import QFont
                                font = QFont("Consolas", 9)
                                if not font.exactMatch():
                                    font = QFont("Courier New", 9)
                                text_edit.setFont(font)
                                
                                self.ui.param_table.setCellWidget(i, 1, text_edit)
                                
                                # Increase row height for multiline parameters
                                self.ui.param_table.setRowHeight(i, 100)
                                print(f"[ProcessWizardManager] Text edit created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating text edit for {param_name}: {e}")
                                # Fallback to regular text item
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                                
                        elif param_type == "int":
                            # Create spinbox for integer parameters
                            try:
                                print(f"[ProcessWizardManager] Creating spinbox for {param_name}")
                                from PySide6.QtWidgets import QSpinBox
                                
                                spinbox = QSpinBox()
                                spinbox.setRange(p.get("min", -999999), p.get("max", 999999))
                                spinbox.setValue(int(param_value) if param_value else 0)
                                spinbox.setToolTip(tooltip)
                                
                                self.ui.param_table.setCellWidget(i, 1, spinbox)
                                print(f"[ProcessWizardManager] Spinbox created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating spinbox for {param_name}: {e}")
                                # Fallback to regular text item
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                                
                        elif param_type == "float":
                            # Create double spinbox for float parameters
                            try:
                                print(f"[ProcessWizardManager] Creating double spinbox for {param_name}")
                                from PySide6.QtWidgets import QDoubleSpinBox
                                
                                spinbox = QDoubleSpinBox()
                                spinbox.setRange(p.get("min", -999999.0), p.get("max", 999999.0))
                                spinbox.setDecimals(4)
                                spinbox.setValue(float(param_value) if param_value else 0.0)
                                spinbox.setToolTip(tooltip)
                                
                                self.ui.param_table.setCellWidget(i, 1, spinbox)
                                print(f"[ProcessWizardManager] Double spinbox created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating double spinbox for {param_name}: {e}")
                                # Fallback to regular text item
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                            
                        else:
                            # Create regular text item for string and other parameters
                            try:
                                print(f"[ProcessWizardManager] Creating text item for {param_name}")
                                param_value_item = QTableWidgetItem(str(param_value))
                                param_value_item.setToolTip(tooltip)
                                self.ui.param_table.setItem(i, 1, param_value_item)
                                print(f"[ProcessWizardManager] Text item created successfully for {param_name}")
                            except Exception as e:
                                print(f"[ProcessWizardManager] Error creating text item for {param_name}: {e}")
                                # Create basic fallback item
                                try:
                                    fallback_item = QTableWidgetItem(str(default_value))
                                    self.ui.param_table.setItem(i, 1, fallback_item)
                                except:
                                    pass
                        
                    except Exception as param_e:
                        print(f"[ProcessWizardManager] ERROR processing parameter {i}: {param_e}")
                        import traceback
                        traceback.print_exc()
                        continue
            except Exception as table_e:
                print(f"[ProcessWizardManager] ERROR populating parameter table: {table_e}")
                import traceback
                traceback.print_exc()

            print(f"[ProcessWizardManager] === FILTER SELECTION COMPLETED SUCCESSFULLY ===")
            return

            # Update script tab if it exists and if not on the script tab currently
            try:
                print(f"[ProcessWizardManager] Checking for script sync capability...")
                if (hasattr(self.ui, '_sync_script_from_params') and 
                    hasattr(self.ui, 'param_tab_widget') and 
                    self.ui.param_tab_widget.currentIndex() == 1):  # Script tab is active
                    print(f"[ProcessWizardManager] Script tab is active, syncing script...")
                    self.ui._sync_script_from_params()
                    print(f"[ProcessWizardManager] Script sync completed")
                else:
                    print(f"[ProcessWizardManager] Script sync not needed (not on script tab)")
            except Exception as sync_e:
                print(f"[ProcessWizardManager] Error with script sync: {sync_e}")
                
        except Exception as e:
            print(f"[ProcessWizardManager] CRITICAL ERROR in _on_filter_selected: {e}")
            import traceback
            traceback.print_exc()
            # Try to set an error message in the console
            try:
                self.ui.console_output.setPlainText(f"Error loading filter: {e}")
            except:
                pass

    def _validate_parameters(self, user_input_dict: dict) -> list:
        """Validate parameters before processing. Returns list of error messages."""
        errors = []
        
        if not self.pending_step:
            return ["No step selected"]
        
        try:
            # Get step parameter definitions
            step_params = getattr(self.pending_step, 'params', [])
            
            for param in step_params:
                param_name = param.get('name', '')
                param_type = param.get('type', 'str')
                param_value = user_input_dict.get(param_name)
                
                # Skip fs parameter as it's injected automatically
                if param_name == 'fs':
                    continue
                
                # Check for required parameters that are missing or empty
                if param_value is None or (isinstance(param_value, str) and not param_value.strip()):
                    if param.get('required', False):
                        errors.append(f"Required parameter '{param_name}' is missing")
                    continue
                
                # Type-specific validation
                try:
                    if param_type == 'float':
                        val = float(param_value)
                        if not np.isfinite(val):
                            errors.append(f"Parameter '{param_name}' must be a finite number")
                        # Check min/max bounds if specified
                        if 'min' in param and val < param['min']:
                            errors.append(f"Parameter '{param_name}' must be >= {param['min']}")
                        if 'max' in param and val > param['max']:
                            errors.append(f"Parameter '{param_name}' must be <= {param['max']}")
                    
                    elif param_type == 'int':
                        val = int(param_value)
                        # Check min/max bounds if specified
                        if 'min' in param and val < param['min']:
                            errors.append(f"Parameter '{param_name}' must be >= {param['min']}")
                        if 'max' in param and val > param['max']:
                            errors.append(f"Parameter '{param_name}' must be <= {param['max']}")
                    
                    elif param_type in ['bool', 'boolean']:
                        # Boolean validation is lenient
                        pass
                    
                    elif 'options' in param:
                        # Validate that value is in allowed options
                        if str(param_value) not in [str(opt) for opt in param['options']]:
                            errors.append(f"Parameter '{param_name}' must be one of: {param['options']}")
                
                except (ValueError, TypeError) as e:
                    errors.append(f"Parameter '{param_name}' has invalid value '{param_value}' for type {param_type}")
        
        except Exception as e:
            errors.append(f"Parameter validation failed: {e}")
        
        return errors
    
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
            
        except Exception as e:
            pass
    
    def on_input_submitted(self, user_input_dict: dict):
        """Handle input submission with parameters from table."""
        if not self.pending_step:
            return

        # Check if we should use script execution instead of parameter-based execution
        if self._should_use_script_execution():
            return self._execute_from_script(user_input_dict)

        try:
            # Basic parameter validation before parsing
            validation_errors = self._validate_parameters(user_input_dict)
            if validation_errors:
                error_msg = "Parameter validation errors:\n" + "\n".join([f"• {err}" for err in validation_errors])
                self.ui.console_output.setPlainText(error_msg)
                return
                
            params = self.pending_step.parse_input(user_input_dict)
        except Exception as e:
            self.ui.console_output.setPlainText(f"Parameter parsing error: {e}")
            return

        # Get the parent channel
        parent_channel = self.channel_lookup()
        if not parent_channel:
            return

        # Get all channels in the lineage
        lineage_dict = self.ui.channel_manager.get_channels_by_lineage(parent_channel.lineage_id)
        
        # Collect all channels from the lineage (parents, children, siblings)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Use the most recent channel in the lineage as the parent, or fall back to the original
        if all_lineage_channels:
            parent_channel = max(all_lineage_channels, key=lambda ch: ch.step)

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
                
                # Add the new channel to the manager
                self.ui.channel_manager.add_channel(new_channel)
            
            # Update new_channels to only include valid ones
            new_channels = valid_channels
            
            # Set the most recent new channel as the input for next step
            if new_channels:
                self.ui.input_ch = new_channels[-1]  # Use the last created channel
            
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

        # Update UI - force refresh of cached lineage to include new channels
        self.ui._cached_lineage = []  # Clear cache to force rebuild
        self.ui._update_file_selector()
        self.ui._update_channel_selector()
        self.ui._update_step_table()
        
        # Rebuild lineage for the active channel to include new channels
        active_channel = self.ui.get_active_channel_info()
        if active_channel:
            self.ui._build_lineage_for_channel(active_channel)
        
        # Update input channel combobox to include new channels
        self.ui._update_input_channel_combobox()
        
        # Update plot with refreshed data
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
                    # Check if the value cell contains a widget or text item
                    widget = self.ui.param_table.cellWidget(row, 1)
                    if widget:
                        # Handle different widget types
                        if isinstance(widget, QComboBox):
                            val = widget.currentText().strip()
                        elif isinstance(widget, QTextEdit):
                            val = widget.toPlainText().strip()
                        elif hasattr(widget, 'findChild'):
                            # This is a container widget (like for checkboxes)
                            from PySide6.QtWidgets import QCheckBox, QSpinBox, QDoubleSpinBox
                            checkbox = widget.findChild(QCheckBox)
                            if checkbox:
                                val = checkbox.isChecked()
                            else:
                                val = ""
                        elif isinstance(widget, QSpinBox):
                            val = widget.value()
                        elif isinstance(widget, QDoubleSpinBox):
                            val = widget.value()
                        else:
                            val = ""
                    else:
                        # It's a regular text item
                        val_item = self.ui.param_table.item(row, 1)
                        val = val_item.text().strip() if val_item else ""
                    
                    # Store the value with appropriate type conversion
                    if isinstance(val, bool):
                        user_input_dict[key] = val
                    elif isinstance(val, (int, float)):
                        user_input_dict[key] = val
                    elif val:
                        # Try to convert string values to appropriate types
                        try:
                            if '.' in str(val) and str(val).replace('.', '').replace('-', '').isdigit():
                                user_input_dict[key] = float(val)
                            elif str(val).replace('-', '').isdigit():
                                user_input_dict[key] = int(val)
                            else:
                                user_input_dict[key] = str(val)  # Keep as string
                        except ValueError:
                            user_input_dict[key] = str(val)  # Keep as string if conversion fails
        
        try:
            params = self.pending_step.parse_input(user_input_dict)
            
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
    
    def _should_use_script_execution(self):
        """Check if we should use script execution instead of parameter-based execution"""
        # Check if UI has script tab and user is on script tab with read-only disabled
        if (hasattr(self.ui, 'param_tab_widget') and 
            hasattr(self.ui, 'script_readonly_checkbox') and
            self.ui.param_tab_widget.currentIndex() == 1 and  # Script tab is active
            not self.ui.script_readonly_checkbox.isChecked()):  # Read-only is disabled
            return True
        return False
    
    def _execute_from_script(self, fallback_params):
        """Execute processing using the script, falling back to parameters if script has errors"""
        if not hasattr(self.ui, 'script_editor'):
            return self._execute_from_parameters_direct(fallback_params)
            
        script_text = self.ui.script_editor.toPlainText()
        
        # Try to validate the script syntax
        try:
            # Basic syntax check
            compile(script_text, '<script>', 'exec')
        except SyntaxError as e:
            # Script has syntax errors, fall back to parameter-based execution
            self.ui.console_output.setPlainText(
                f"Script syntax error at line {e.lineno}: {e.msg}\n"
                f"Falling back to parameter-based execution..."
            )
            return self._execute_from_parameters_direct(fallback_params)
        except Exception as e:
            # Other compilation errors, fall back to parameter-based execution
            self.ui.console_output.setPlainText(
                f"Script error: {str(e)}\n"
                f"Falling back to parameter-based execution..."
            )
            return self._execute_from_parameters_direct(fallback_params)
        
        # Execute the validated script
        try:
            return self._execute_script_safely(script_text, fallback_params)
        except Exception as e:
            # Script execution failed, fall back to parameter-based execution
            self.ui.console_output.setPlainText(
                f"Script execution error: {str(e)}\n"
                f"Falling back to parameter-based execution..."
            )
            return self._execute_from_parameters_direct(fallback_params)
    
    def _execute_from_parameters_direct(self, user_input_dict):
        """Execute processing using parameters directly, bypassing script checks"""
        if not self.pending_step:
            return

        try:
            params = self.pending_step.parse_input(user_input_dict)
        except Exception as e:
            self.ui.console_output.setPlainText(f"Parameter parsing error: {e}")
            return

        # Get the parent channel
        parent_channel = self.channel_lookup()
        if not parent_channel:
            return

        # Get all channels in the lineage
        lineage_dict = self.ui.channel_manager.get_channels_by_lineage(parent_channel.lineage_id)
        
        # Collect all channels from the lineage (parents, children, siblings)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Use the most recent channel in the lineage as the parent, or fall back to the original
        if all_lineage_channels:
            parent_channel = max(all_lineage_channels, key=lambda ch: ch.step)

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
                
                # Add the new channel to the manager
                self.ui.channel_manager.add_channel(new_channel)
            
            # Update new_channels to only include valid ones
            new_channels = valid_channels
            
            # Set the most recent new channel as the input for next step
            if new_channels:
                self.ui.input_ch = new_channels[-1]  # Use the last created channel
            
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

    def _execute_script_safely(self, script_text, fallback_params):
        """Execute the user's script in a controlled environment"""
        import numpy as np
        import scipy.signal
        import copy
        
        # Get the parent channel
        parent_channel = self.channel_lookup()
        if not parent_channel:
            return
        
        # Get all channels in the lineage for context
        lineage_dict = self.ui.channel_manager.get_channels_by_lineage(parent_channel.lineage_id)
        all_lineage_channels = []
        all_lineage_channels.extend(lineage_dict.get('parents', []))
        all_lineage_channels.extend(lineage_dict.get('children', []))
        all_lineage_channels.extend(lineage_dict.get('siblings', []))
        
        # Use the most recent channel in the lineage as the parent
        if all_lineage_channels:
            parent_channel = max(all_lineage_channels, key=lambda ch: ch.step)
        
        # Create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'print': print,
            },
            'np': np,
            'scipy': scipy,
            'copy': copy,
            'parent_channel': parent_channel,
            'channel_manager': self.ui.channel_manager,
            'registry': self.registry,
        }
        
        # Create local variables for the script
        safe_locals = {}
        
        # Execute the script
        exec(script_text, safe_globals, safe_locals)
        
        # The script should define a variable called 'result_channel' or 'result_channels'
        new_channels = None
        if 'result_channels' in safe_locals:
            new_channels = safe_locals['result_channels']
        elif 'result_channel' in safe_locals:
            new_channels = [safe_locals['result_channel']]
        else:
            raise ValueError("Script must define 'result_channel' or 'result_channels' variable")
        
        # Ensure we have a list
        if not isinstance(new_channels, list):
            new_channels = [new_channels]
        
        # Validate and process the result channels
        valid_channels = []
        for new_channel in new_channels:
            # Basic validation
            if not hasattr(new_channel, 'ydata') or not hasattr(new_channel, 'xdata'):
                raise ValueError("Result channel must have 'ydata' and 'xdata' attributes")
            
            if new_channel.ydata is None or new_channel.xdata is None:
                raise ValueError("Result channel data cannot be None")
            
            if len(new_channel.ydata) == 0 or len(new_channel.xdata) == 0:
                raise ValueError("Result channel data cannot be empty")
            
            # Set up channel metadata
            new_channel.step = parent_channel.step + 1
            new_channel.lineage_id = parent_channel.lineage_id
            new_channel.parent_ids = [parent_channel.channel_id]
            
            # Apply custom channel name if provided
            if (hasattr(self.ui, 'channel_name_entry') and 
                self.ui.channel_name_entry.text().strip()):
                custom_name = self.ui.channel_name_entry.text().strip()
                new_channel.legend_label = custom_name
                new_channel.ylabel = custom_name
            
                            # Assign color if not already set
                if not hasattr(new_channel, 'color') or new_channel.color is None:
                    file_channels = self.ui.channel_manager.get_channels_by_file(parent_channel.file_id)
                    existing_colors = set()
                    for ch in file_channels:
                        if hasattr(ch, 'color') and ch.color is not None:
                            existing_colors.add(ch.color)
                    
                    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                    available_colors = [c for c in colors if c not in existing_colors]
                    if available_colors:
                        new_channel.color = available_colors[0]
                    else:
                        color_index = len(file_channels) % len(colors)
                        new_channel.color = colors[color_index]
                
                # Ensure new channel is visible by default
                if not hasattr(new_channel, 'show'):
                    new_channel.show = True
                
                # Add to channel manager
                self.ui.channel_manager.add_channel(new_channel)
            valid_channels.append(new_channel)
        
        # Update UI state
        if valid_channels:
            self.ui.input_ch = valid_channels[-1]
        
        # Clear UI
        self.pending_step = None
        self.ui.param_table.setRowCount(0)
        self.ui.console_output.clear()
        
        if hasattr(self.ui, 'channel_name_entry'):
            self.ui.channel_name_entry.clear()
        
        # Update UI components - force refresh of cached lineage to include new channels
        self.ui._cached_lineage = []  # Clear cache to force rebuild
        self.ui._update_file_selector()
        self.ui._update_channel_selector()
        self.ui._update_step_table()
        
        # Rebuild lineage for the active channel to include new channels
        active_channel = self.ui.get_active_channel_info()
        if active_channel:
            self.ui._build_lineage_for_channel(active_channel)
        
        # Update input channel combobox to include new channels
        self.ui._update_input_channel_combobox()
        
        # Update plot with refreshed data
        self.ui._update_plot()
        
        # Show success message
        if len(valid_channels) == 1:
            self.ui.console_output.setPlainText(
                f"Script executed successfully.\nNew channel: {valid_channels[0].channel_id}"
            )
        else:
            channel_list = ", ".join([ch.channel_id for ch in valid_channels])
            self.ui.console_output.setPlainText(
                f"Script executed successfully.\nCreated {len(valid_channels)} channels: {channel_list}"
            )
        
        return valid_channels[-1] if valid_channels else None

