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
        
        # Script tracking for customization detection
        self.original_script_content = ""  # Store original script content for comparison
        
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

            # Always update script tab when filter is selected (like parameters tab)
            try:
                print(f"[ProcessWizardManager] Updating script tab for filter selection...")
                if hasattr(self.ui, '_sync_script_from_params'):
                    self.ui._sync_script_from_params()
                    # Store the original script content for customization detection
                    if hasattr(self.ui, 'script_editor'):
                        self.original_script_content = self.ui.script_editor.toPlainText()
                        print(f"[ProcessWizardManager] Original script content stored for customization detection")
                    print(f"[ProcessWizardManager] Script tab updated successfully")
                else:
                    print(f"[ProcessWizardManager] Script sync method not available")
            except Exception as script_update_e:
                print(f"[ProcessWizardManager] Error updating script tab: {script_update_e}")

            print(f"[ProcessWizardManager] === FILTER SELECTION COMPLETED SUCCESSFULLY ===")
            return
                
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
    
    def _collect_parameters_from_table(self):
        """Unified method to collect parameters from the parameter table"""
        user_input_dict = {}
        
        for row in range(self.ui.param_table.rowCount()):
            key_item = self.ui.param_table.item(row, 0)
            if not key_item:
                continue
                
            key = key_item.text().strip()
            if not key:
                continue
            
            # Check if the value cell contains a widget or text item
            widget = self.ui.param_table.cellWidget(row, 1)
            
            if widget:
                # Handle different widget types in correct order
                if isinstance(widget, QComboBox):
                    val = widget.currentText().strip()
                elif isinstance(widget, QTextEdit):
                    val = widget.toPlainText().strip()
                elif isinstance(widget, QSpinBox):
                    val = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    val = widget.value()
                elif hasattr(widget, 'findChild'):
                    # This is a container widget (like for checkboxes)
                    checkbox = widget.findChild(QCheckBox)
                    if checkbox:
                        val = checkbox.isChecked()
                    else:
                        val = ""
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
            else:
                # Handle empty values - don't add them to the dict unless they're boolean False
                if isinstance(val, bool):
                    user_input_dict[key] = val
        
        return user_input_dict

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
            
            # Set the most appropriate new channel as input for next step
            if new_channels:
                # Prefer time-series channels for further processing over spectrogram/visualization channels
                time_series_channels = [ch for ch in new_channels if 'time-series' in getattr(ch, 'tags', [])]
                if time_series_channels:
                    self.ui.input_ch = time_series_channels[-1]  # Use the last time-series channel
                else:
                    self.ui.input_ch = new_channels[-1]  # Fallback to last created channel
            
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
        
        # Return the most appropriate channel for UI consistency (prefer time-series for further processing)
        if new_channels:
            time_series_channels = [ch for ch in new_channels if 'time-series' in getattr(ch, 'tags', [])]
            if time_series_channels:
                return time_series_channels[-1]  # Return the last time-series channel
            else:
                return new_channels[-1]  # Fallback to last created channel
        return None

    def apply_pending_step(self):
        if not self.pending_step:
            self.ui.console_output.setPlainText("No step selected.")
            return None

        # Use unified parameter collection method
        user_input_dict = self._collect_parameters_from_table()
        
        # Debug output to see what parameters were collected
        print(f"[ProcessWizardManager] Collected parameters: {user_input_dict}")
        
        # Check if we should use script execution
        if self._should_use_script_execution():
            # Use script execution with fallback to original step implementation
            return self._execute_from_script(user_input_dict)
        else:
            # Use parameter-based execution (original behavior)
            return self._execute_from_parameters_direct(user_input_dict)

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
        # Check if UI has script editor
        if not hasattr(self.ui, 'script_editor'):
            return False
        
        # Check if there's actual script content
        script_text = self.ui.script_editor.toPlainText().strip()
        if not script_text or script_text.startswith("# No filter selected"):
            return False
        
        # Check if script has been customized by user
        return self._is_script_customized()
    
    def _is_script_customized(self):
        """Check if the current script content is different from the original"""
        if not hasattr(self.ui, 'script_editor'):
            return False
        
        current_script = self.ui.script_editor.toPlainText().strip()
        original_script = self.original_script_content.strip()
        
        # Normalize whitespace for comparison
        current_normalized = '\n'.join(line.strip() for line in current_script.split('\n') if line.strip())
        original_normalized = '\n'.join(line.strip() for line in original_script.split('\n') if line.strip())
        
        # Return True if content is different (user has customized it)
        return current_normalized != original_normalized
    
    def _execute_from_script(self, fallback_params):
        """Execute processing using the script, falling back to original step implementation if script has errors"""
        if not hasattr(self.ui, 'script_editor'):
            return self._execute_from_parameters_direct(fallback_params)
            
        script_text = self.ui.script_editor.toPlainText()
        
        # Try to validate the script syntax
        try:
            # Basic syntax check
            compile(script_text, '<script>', 'exec')
        except SyntaxError as e:
            # Script has syntax errors, fall back to original step implementation
            self.ui.console_output.setPlainText(
                f"Script syntax error at line {e.lineno}: {e.msg}\n"
                f"Falling back to original step implementation..."
            )
            return self._execute_from_parameters_direct(fallback_params)
        except Exception as e:
            # Other compilation errors, fall back to original step implementation
            self.ui.console_output.setPlainText(
                f"Script error: {str(e)}\n"
                f"Falling back to original step implementation..."
            )
            return self._execute_from_parameters_direct(fallback_params)
        
        # Execute the validated script
        try:
            return self._execute_script_safely(script_text, fallback_params)
        except Exception as e:
            # Script execution failed, fall back to original step implementation
            self.ui.console_output.setPlainText(
                f"Script execution error: {str(e)}\n"
                f"Falling back to original step implementation..."
            )
            return self._execute_from_parameters_direct(fallback_params)
    
    def _execute_from_parameters_direct(self, user_input_dict):
        """Execute processing using parameters directly, bypassing script checks"""
        if not self.pending_step:
            return

        # Debug output to see what parameters are being passed
        print(f"[ProcessWizardManager] Executing step '{self.pending_step.name}' with parameters: {user_input_dict}")

        try:
            params = self.pending_step.parse_input(user_input_dict)
            print(f"[ProcessWizardManager] Parsed parameters: {params}")
        except Exception as e:
            print(f"[ProcessWizardManager] Parameter parsing error: {e}")
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
                self.pending_step = None
                self.ui.param_table.setRowCount(0)
                return
            
            # Process valid channels
            new_channel_ids = []  # Track created channel IDs for undo
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
                new_channel_ids.append(new_channel.channel_id)
            
            # Update new_channels to only include valid ones
            new_channels = valid_channels
            
            # Add operation to undo stack
            step_name = self.pending_step.name if self.pending_step else "Unknown step"
            self._add_to_undo_stack(step_name, parent_channel.channel_id, new_channel_ids)
            
            # Inform user if some channels were empty
            if empty_channels:
                print(f"[ProcessWizardManager] Filtered out {len(empty_channels)} empty channel(s)")
            
            # Set the most appropriate new channel as input for next step
            if new_channels:
                # Prefer time-series channels for further processing over spectrogram/visualization channels
                time_series_channels = [ch for ch in new_channels if 'time-series' in getattr(ch, 'tags', [])]
                if time_series_channels:
                    self.ui.input_ch = time_series_channels[-1]  # Use the last time-series channel
                else:
                    self.ui.input_ch = new_channels[-1]  # Fallback to last created channel
            
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
            
            print(f"[ProcessWizardManager] Step execution failed: {e}")
            self.ui.console_output.setPlainText(f"Step execution failed: {e}")
            return

        # Get the step name for the console message (use stored step_name)
        # Clean up the step name for display
        display_step_name = step_name.replace('_', ' ').title()
        
        # Check for data repair information in any of the created channels
        repair_info_messages = []
        for channel in new_channels:
            if (hasattr(channel, 'metadata') and channel.metadata is not None and 
                'data_repair_info' in channel.metadata):
                repair_info = channel.metadata['data_repair_info']
                if repair_info and repair_info != "No repairs needed":
                    repair_info_messages.append(repair_info)
        
        if len(new_channels) == 1:
            # Get the channel name for display
            channel_name = new_channels[0].legend_label or new_channels[0].ylabel or f"Channel {new_channels[0].channel_id}"
            console_message = f"Operation applied successfully!\nCreated channel: {channel_name}"
            
            # Add repair information if present
            if repair_info_messages:
                console_message += f"\n\nData Repair Applied:\n{repair_info_messages[0]}"
            
            self.ui.console_output.setPlainText(console_message)
        else:
            # For multiple channels, show names if available
            channel_names = []
            for ch in new_channels:
                name = ch.legend_label or ch.ylabel or f"Channel {ch.channel_id}"
                channel_names.append(name)
            channel_list = ", ".join(channel_names)
            
            console_message = f"Operation applied successfully!\nCreated {len(new_channels)} channels: {channel_list}"
            
            # Add repair information if present
            if repair_info_messages:
                console_message += f"\n\nData Repair Applied:\n" + "\n".join(repair_info_messages)
            
            self.ui.console_output.setPlainText(console_message)

        # Update UI
        self.ui._update_file_selector()
        self.ui._update_channel_selector()
        self.ui._update_step_table()
        self.ui._update_plot()

        # Return the most appropriate channel for UI consistency (prefer time-series for further processing)
        if new_channels:
            time_series_channels = [ch for ch in new_channels if 'time-series' in getattr(ch, 'tags', [])]
            if time_series_channels:
                return time_series_channels[-1]  # Return the last time-series channel
            else:
                return new_channels[-1]  # Fallback to last created channel
        return None

    def _execute_script_safely(self, script_text, fallback_params):
        """Execute the user's script in a controlled environment with fallback to original step"""
        import numpy as np
        import scipy.signal
        import copy
        
        # Get the parent channel
        parent_channel = self.channel_lookup()
        if not parent_channel:
            raise ValueError("No parent channel available")
        
        # Get all channels in the lineage for context
        lineage_dict = self.ui.channel_manager.get_channels_by_lineage(parent_channel.lineage_id)
        
        # Create safe global variables for the script
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
                'print': print,
                'isinstance': isinstance,
                'type': type,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                '__import__': __import__,  # Needed for import statements
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
            },
            'np': np,
            'numpy': np,
            'scipy': scipy,
            'copy': copy,
            'parent_channel': parent_channel,
            'x': parent_channel.xdata,  # Simple variable access for user scripts
            'y': parent_channel.ydata,  # Simple variable access for user scripts
            'fs': getattr(parent_channel, 'fs_median', None),  # Sampling frequency
            'params': fallback_params,  # Parameters from the UI
            'lineage': lineage_dict,
            'channel_manager': self.ui.channel_manager,
            'registry': self.registry,
        }
        
        # Create local variables for the script
        safe_locals = {}
        
        try:
            # Execute the script
            print(f"[ProcessWizardManager] Executing script...")
            exec(script_text, safe_globals, safe_locals)
            print(f"[ProcessWizardManager] Script executed successfully")
            print(f"[ProcessWizardManager] Script locals: {list(safe_locals.keys())}")
        except Exception as e:
            # Script execution failed, raise to trigger fallback
            raise ValueError(f"Script execution failed: {str(e)}")
        
        # Handle different script output formats
        new_channels = None
        print(f"[ProcessWizardManager] Looking for result variables...")
        
        # Format 1: NEW - Script returns result_channels_data (new structured format)
        if 'result_channels_data' in safe_locals:
            channels_data = safe_locals['result_channels_data']
            print(f"[ProcessWizardManager] Found result_channels_data: {type(channels_data)}, count: {len(channels_data) if hasattr(channels_data, '__len__') else 'unknown'}")
            
            # Process the new structured format using base step logic
            new_channels = self._process_channels_data(channels_data, parent_channel, fallback_params)
            
        # Format 2: LEGACY - Script returns result_channel or result_channels (complex format)
        elif 'result_channels' in safe_locals:
            new_channels = safe_locals['result_channels']
            print(f"[ProcessWizardManager] Found result_channels (legacy): {type(new_channels)}, count: {len(new_channels) if hasattr(new_channels, '__len__') else 'unknown'}")
        elif 'result_channel' in safe_locals:
            new_channels = [safe_locals['result_channel']]
            print(f"[ProcessWizardManager] Found result_channel (legacy): {type(safe_locals['result_channel'])}")
        # Format 3: LEGACY - Script returns y_new directly (simple format)
        elif 'y_new' in safe_locals:
            print(f"[ProcessWizardManager] Found y_new: {type(safe_locals['y_new'])}")
            # Create a new channel from y_new
            y_new = safe_locals['y_new']
            
            # Get x_new if provided, otherwise use original x
            x_new = safe_locals.get('x_new', parent_channel.xdata)
            
            # Validate the output data
            if y_new is None:
                print(f"[ProcessWizardManager] ERROR: y_new is None")
                raise ValueError("Script output y_new cannot be None")
            
            # Convert to numpy arrays if needed
            y_new = np.array(y_new)
            x_new = np.array(x_new)
            
            print(f"[ProcessWizardManager] y_new shape: {y_new.shape}, x_new shape: {x_new.shape}")
            
            if len(y_new) == 0:
                print(f"[ProcessWizardManager] ERROR: y_new is empty")
                raise ValueError("Script output y_new cannot be empty")
            
            # Create new channel from the simple output
            print(f"[ProcessWizardManager] Creating new channel from y_new...")
            new_channel = copy.deepcopy(parent_channel)
            new_channel.ydata = y_new
            new_channel.xdata = x_new
            
            # Generate a new unique channel ID (critical fix!)
            import uuid
            new_channel.channel_id = str(uuid.uuid4())[:8]
            print(f"[ProcessWizardManager] Assigned new channel ID: {new_channel.channel_id}")
            
            # Update channel metadata
            if hasattr(self, 'pending_step') and self.pending_step:
                new_channel.description = parent_channel.description + f' -> {self.pending_step.name} (custom)'
            else:
                new_channel.description = parent_channel.description + ' -> custom_script'
            
            print(f"[ProcessWizardManager] Created new channel: {new_channel.channel_id}")
            new_channels = [new_channel]
        else:
            print(f"[ProcessWizardManager] ERROR: No result variables found in script locals")
            raise ValueError("Script must define 'result_channel', 'result_channels', or 'y_new' variable")
        
        # Ensure we have a list
        if not isinstance(new_channels, list):
            new_channels = [new_channels]
        
        # Validate and process the result channels
        print(f"[ProcessWizardManager] Validating {len(new_channels)} channel(s)...")
        valid_channels = []
        for i, new_channel in enumerate(new_channels):
            print(f"[ProcessWizardManager] Validating channel {i}: {new_channel.channel_id}")
            # Basic validation
            if not hasattr(new_channel, 'ydata') or not hasattr(new_channel, 'xdata'):
                print(f"[ProcessWizardManager] ERROR: Channel missing ydata or xdata attributes")
                raise ValueError("Result channel must have 'ydata' and 'xdata' attributes")
            
            if new_channel.ydata is None or new_channel.xdata is None:
                print(f"[ProcessWizardManager] ERROR: Channel data is None")
                raise ValueError("Result channel data cannot be None")
            
            if len(new_channel.ydata) == 0 or len(new_channel.xdata) == 0:
                print(f"[ProcessWizardManager] ERROR: Channel data is empty")
                raise ValueError("Result channel data cannot be empty")
            
            # Ensure unique channel ID (fix for result_channel/result_channels format)
            if new_channel.channel_id == parent_channel.channel_id:
                import uuid
                new_channel.channel_id = str(uuid.uuid4())[:8]
                print(f"[ProcessWizardManager] Assigned new unique channel ID: {new_channel.channel_id}")
            
            # Set up channel metadata
            new_channel.step = parent_channel.step + 1
            new_channel.lineage_id = parent_channel.lineage_id
            new_channel.parent_ids = [parent_channel.channel_id]
            print(f"[ProcessWizardManager] Set step: {new_channel.step}, lineage: {new_channel.lineage_id}, parent: {parent_channel.channel_id}")
            
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
            print(f"[ProcessWizardManager] Adding channel to manager: {new_channel.channel_id}")
            success = self.ui.channel_manager.add_channel(new_channel)
            print(f"[ProcessWizardManager] Channel added successfully: {success}")
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
        print(f"[ProcessWizardManager] Updating UI components...")
        self.ui._cached_lineage = []  # Clear cache to force rebuild
        print(f"[ProcessWizardManager] Updating file selector...")
        self.ui._update_file_selector()
        print(f"[ProcessWizardManager] Updating channel selector...")
        self.ui._update_channel_selector()
        print(f"[ProcessWizardManager] Updating step table...")
        self.ui._update_step_table()
        
        # Rebuild lineage for the active channel to include new channels
        print(f"[ProcessWizardManager] Rebuilding lineage...")
        active_channel = self.ui.get_active_channel_info()
        if active_channel:
            print(f"[ProcessWizardManager] Building lineage for active channel: {active_channel.channel_id}")
            self.ui._build_lineage_for_channel(active_channel)
        else:
            print(f"[ProcessWizardManager] No active channel found")
        
        # Update input channel combobox to include new channels
        print(f"[ProcessWizardManager] Updating input channel combobox...")
        self.ui._update_input_channel_combobox()
        
        # Update plot with refreshed data
        print(f"[ProcessWizardManager] Updating plot...")
        self.ui._update_plot()
        print(f"[ProcessWizardManager] UI updates completed")
        
        # Show success message
        if len(valid_channels) == 1:
            self.ui.console_output.setPlainText(
                f"Custom script executed successfully.\nNew channel: {valid_channels[0].channel_id}"
            )
        else:
            channel_list = ", ".join([ch.channel_id for ch in valid_channels])
            self.ui.console_output.setPlainText(
                f"Custom script executed successfully.\nCreated {len(valid_channels)} channels: {channel_list}"
            )
        
        return valid_channels[-1] if valid_channels else None

    def _process_channels_data(self, channels_data, parent_channel, fallback_params):
        """Process the new structured channels data format using base step logic"""
        try:
            print(f"[ProcessWizardManager] Processing {len(channels_data)} channel(s) with new structured format")
            
            # Validate the channels_data structure
            if not isinstance(channels_data, list):
                raise ValueError("result_channels_data must be a list")
            
            created_channels = []
            
            # Import the base step for validation and processing
            from steps.base_step import BaseStep
            
            for i, channel_info in enumerate(channels_data):
                print(f"[ProcessWizardManager] Processing channel {i+1}: {channel_info}")
                
                # Validate channel structure
                BaseStep.validate_channel_structure(channel_info, i)
                
                # Extract channel data
                tags = channel_info['tags']
                channel_type = tags[0] if tags else 'main'
                x_data = channel_info['x']
                y_data = channel_info['y']
                
                # Validate output data based on channel type
                allow_length_change = channel_type in ['time-series', 'reduced']
                BaseStep.validate_output_data(parent_channel.ydata, y_data, channel_type=channel_type, allow_length_change=allow_length_change)
                
                # Generate channel suffix
                suffix = f"CustomScript_{channel_type}" if len(channels_data) > 1 else "CustomScript"
                
                # Create the channel (all channels are now time-series)
                new_channel = BaseStep.create_new_channel(
                    parent=parent_channel,
                    xdata=x_data,
                    ydata=y_data,
                    params=fallback_params,
                    suffix=suffix,
                    channel_tags=tags
                )
                
                # Update channel metadata for custom script
                new_channel.description = parent_channel.description + f' -> custom_script_{channel_type}'
                
                created_channels.append(new_channel)
                print(f"[ProcessWizardManager] Created channel {i+1}: {new_channel.channel_id}")
            
            # Add channels to the channel manager
            for new_channel in created_channels:
                self.ui.channel_manager.add_channel(new_channel)
            
            print(f"[ProcessWizardManager] Successfully processed {len(created_channels)} channels with new format")
            return created_channels
            
        except Exception as e:
            print(f"[ProcessWizardManager] Error processing channels data: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to process channels data: {str(e)}")

