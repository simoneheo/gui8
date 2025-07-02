from PySide6.QtCore import QObject, Signal, QCoreApplication
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QColor, QFont
from comparison_wizard_window import ComparisonWizardWindow
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import warnings
import pandas as pd
import time
from typing import Dict, List, Optional

# Import comparison methods from the new comparison folder
try:
    from comparison.comparison_registry import ComparisonRegistry, load_all_comparisons
    from comparison.base_comparison import BaseComparison
    COMPARISON_AVAILABLE = True
    print("[ComparisonWizardManager] Comparison registry imported successfully")
except ImportError as e:
    print(f"[ComparisonWizardManager] Warning: Could not import comparison registry: {e}")
    COMPARISON_AVAILABLE = False
    
    # Create dummy classes if comparison module is not available
    class ComparisonRegistry:
        @staticmethod
        def all_comparisons():
            return ["correlation", "bland_altman", "residual", "statistical"]
        
        @staticmethod
        def get(name):
            return None
    
    def load_all_comparisons(directory):
        print(f"[ComparisonWizardManager] Warning: Comparison module not available")
    
    class BaseComparison:
        pass

class ComparisonWizardManager(QObject):
    """
    Manager for the comparison wizard that handles:
    - Data alignment between channels
    - Statistical calculations using comparison methods
    - Plot generation
    - State management and progress tracking
    """
    
    comparison_complete = Signal(dict)
    state_changed = Signal(str)  # Emit state changes
    
    def __init__(self, file_manager=None, channel_manager=None, signal_bus=None, parent=None):
        super().__init__(parent)
        
        # Store managers with validation
        self.file_manager = file_manager
        self.channel_manager = channel_manager
        self.signal_bus = signal_bus
        self.parent_window = parent
        
        # Initialize state tracking
        self._stats = {
            'total_comparisons': 0,
            'successful_alignments': 0,
            'failed_alignments': 0,
            'last_comparison': None,
            'session_start': time.time()
        }
        
        # Store aligned data for each pair with performance tracking
        self.pair_aligned_data = {}  # pair_name -> aligned_data
        self.pair_statistics = {}    # pair_name -> statistics
        self._access_counts = {}     # pair_name -> access_count
        
        # Initialize comparison methods
        self._initialize_comparison_methods()
        
        # Validate initialization
        if not self._validate_managers():
            raise ValueError("Required managers not available for ComparisonWizardManager")
        
        # Create window after validation
        self.window = ComparisonWizardWindow(
            file_manager=self.file_manager,
            channel_manager=self.channel_manager,
            signal_bus=self.signal_bus,
            parent=self.parent_window
        )
        
        # Set bidirectional reference
        self.window.comparison_manager = self
        
        # Connect signals
        self._connect_signals()
        
        # Log initialization
        self._log_state_change("Manager initialized successfully")
    
    def _initialize_comparison_methods(self):
        """Initialize comparison methods from the comparison folder"""
        try:
            if COMPARISON_AVAILABLE:
                load_all_comparisons("comparison")
                self.comparison_registry = ComparisonRegistry
                self._log_state_change("Comparison methods loaded successfully")
            else:
                self._log_state_change("Comparison methods not available - using basic calculations only")
        except Exception as e:
            print(f"[ComparisonWizardManager] Warning: Could not load comparison methods: {e}")
            self._log_state_change("Failed to load comparison methods - using fallback")
    
    def get_available_comparison_methods(self):
        """Get list of available comparison methods"""
        if COMPARISON_AVAILABLE:
            try:
                return self.comparison_registry.all_comparisons()
            except:
                pass
        return ["correlation", "bland_altman", "residual", "statistical"]
    
    def apply_comparison_method(self, method_name, ref_data, test_data, parameters=None):
        """Apply a specific comparison method to the data using the new comparison folder"""
        try:
            if not COMPARISON_AVAILABLE:
                return self._fallback_comparison(method_name, ref_data, test_data)
            
            # Map method names to comparison types
            method_mapping = {
                'Correlation Analysis': 'correlation',
                'Bland-Altman': 'bland_altman', 
                'Residual Analysis': 'residual',
                'Statistical Tests': 'statistical'
            }
            
            comparison_type = method_mapping.get(method_name, method_name.lower())
            
            # Get the comparison method class
            comparison_cls = self.comparison_registry.get(comparison_type)
            if not comparison_cls:
                print(f"[ComparisonWizardManager] Comparison method '{method_name}' ({comparison_type}) not found")
                return self._fallback_comparison(method_name, ref_data, test_data)
            
            # Convert and prepare parameters
            converted_params = {}
            if parameters:
                for key, value in parameters.items():
                    if key == "Confidence Level":
                        converted_params['confidence_level'] = float(value)
                    elif key == "Method":
                        converted_params['method'] = value.lower()
                    elif key == "Agreement Limits":
                        converted_params['agreement_limits'] = float(value)
                    elif key == "Show CI":
                        converted_params['show_ci'] = value.lower() == 'true'
                    elif key == "Alpha Level":
                        converted_params['alpha'] = float(value)
                    elif key == "Normality Test":
                        converted_params['normality_test'] = value.lower()
                    elif key == "Outlier Detection":
                        converted_params['outlier_method'] = value.lower()
                    elif key == "Test Type":
                        converted_params['test_type'] = value.lower()
            
            # Create instance with converted parameters
            comparison_instance = comparison_cls(**converted_params)
            
            # Apply the comparison
            result = comparison_instance.compare(ref_data, test_data)
            
            return result
            
        except Exception as e:
            print(f"[ComparisonWizardManager] Error applying comparison method '{method_name}': {e}")
            return self._fallback_comparison(method_name, ref_data, test_data)
    
    def _fallback_comparison(self, method_name, ref_data, test_data):
        """Fallback comparison calculations when comparison module is not available"""
        try:
            if method_name == "correlation":
                r, p = scipy_stats.pearsonr(ref_data, test_data)
                return {
                    'correlation': r,
                    'p_value': p,
                    'r_squared': r**2,
                    'method': 'pearson'
                }
            elif method_name == "bland_altman":
                differences = test_data - ref_data
                means = (ref_data + test_data) / 2
                mean_diff = np.mean(differences)
                std_diff = np.std(differences)
                return {
                    'mean_bias': mean_diff,
                    'std_bias': std_diff,
                    'limits_of_agreement': (mean_diff - 1.96*std_diff, mean_diff + 1.96*std_diff),
                    'means': means,
                    'differences': differences
                }
            elif method_name == "residual":
                residuals = test_data - ref_data
                return {
                    'residuals': residuals,
                    'mean_residual': np.mean(residuals),
                    'std_residual': np.std(residuals),
                    'rmse': np.sqrt(np.mean(residuals**2))
                }
            elif method_name == "statistical":
                differences = test_data - ref_data
                return {
                    'mean_difference': np.mean(differences),
                    'std_difference': np.std(differences),
                    'rmse': np.sqrt(np.mean(differences**2)),
                    'mae': np.mean(np.abs(differences)),
                    'sample_size': len(ref_data)
                }
            else:
                return {'error': f'Unknown comparison method: {method_name}'}
        except Exception as e:
            return {'error': f'Fallback comparison failed: {str(e)}'}

    def _validate_managers(self) -> bool:
        """Validate that required managers are available and functional"""
        if not self.file_manager:
            print("[ComparisonWizardManager] ERROR: File manager not provided")
            return False
            
        if not self.channel_manager:
            print("[ComparisonWizardManager] ERROR: Channel manager not provided")
            return False
            
        # Validate manager functionality
        try:
            self.file_manager.get_file_count()
            self.channel_manager.get_channel_count()
            return True
        except Exception as e:
            print(f"[ComparisonWizardManager] ERROR: Manager validation failed: {e}")
            return False
            
    def _log_state_change(self, message: str):
        """Log state changes for debugging and monitoring"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[ComparisonWizardManager {timestamp}] {message}")
        self.state_changed.emit(message)
        
    def _connect_signals(self):
        """Connect window signals to manager methods"""
        self.window.pair_added.connect(self._on_pair_added)
        self.window.pair_deleted.connect(self._on_pair_deleted)
        self.window.plot_generated.connect(self._on_plot_generated)
        
    def show(self):
        """Show the comparison wizard window"""
        self.window.show()
        
    def close(self):
        """Close the comparison wizard"""
        if self.window:
            self.window.close()
            
    def _on_pair_added(self, pair_config):
        """Handle when a new pair is added"""
        print(f"[ComparisonWizard] Pair added: {pair_config['name']}")
        
        # Get the channels for this pair
        ref_channel = self._get_channel(pair_config['ref_file'], pair_config['ref_channel'])
        test_channel = self._get_channel(pair_config['test_file'], pair_config['test_channel'])
        
        if ref_channel and test_channel:
            pair_name = pair_config['name']
            
            try:
                # STEP 1: Perform alignment for this specific pair
                print(f"[ComparisonWizard] Step 1: Aligning data for pair '{pair_name}'...")
                aligned_data = self._align_channels(ref_channel, test_channel, pair_config)
                
                # STEP 2: Calculate individual pair statistics
                print(f"[ComparisonWizard] Step 2: Calculating statistics for pair '{pair_name}'...")
                stats = self._calculate_statistics(aligned_data)
                
                # STEP 3: Store the aligned data and statistics
                print(f"[ComparisonWizard] Step 3: Storing data for pair '{pair_name}'...")
                self.pair_aligned_data[pair_name] = aligned_data
                self.pair_statistics[pair_name] = stats
                
                # STEP 4: Update individual pair statistics in the table FIRST
                print(f"[ComparisonWizard] Step 4: Updating table row for pair '{pair_name}'...")
                self._update_pair_statistics(pair_name, stats)
                
                # Force table refresh to ensure individual stats are visible
                self._force_table_refresh()
                
                # Log the individual pair results
                self._log_individual_pair_stats(pair_name, stats)
                
                # STEP 5: Show alignment summary if there are warnings
                if self._has_alignment_warnings(aligned_data, stats):
                    print(f"[ComparisonWizard] Step 5: Showing alignment summary for pair '{pair_name}'...")
                    self._show_alignment_summary(pair_name, aligned_data, stats)
                
            except Exception as e:
                print(f"[ComparisonWizard] ERROR processing pair '{pair_name}': {str(e)}")
                # Store error information
                error_stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': str(e)}
                self.pair_statistics[pair_name] = error_stats
                self._update_pair_statistics(pair_name, error_stats)
                
                # Show error to user
                if hasattr(self, 'window'):
                    QMessageBox.warning(
                        self.window,
                        "Pair Processing Error",
                        f"Error processing pair '{pair_name}':\n\n{str(e)}\n\n"
                        f"The pair will be shown but statistics may not be available."
                    )
            
            # STEP 6: Ensure ALL pairs have valid statistics (fix any missing ones)
            print(f"[ComparisonWizard] Step 6: Verifying all pair statistics...")
            self._verify_and_fix_missing_statistics()
            
            # STEP 7: Finally, update cumulative display (includes preview plot)
            print(f"[ComparisonWizard] Step 7: Updating cumulative statistics and plot...")
            self._update_cumulative_display()
            
            print(f"[ComparisonWizard] Pair '{pair_name}' processing complete!")
        else:
            print(f"[ComparisonWizard] Error: Could not find channels for pair '{pair_config['name']}'")
            
    def _verify_and_fix_missing_statistics(self):
        """Verify all pairs have statistics and fix any missing ones"""
        try:
            active_pairs = self.window.get_active_pairs()
            table = self.window.active_pair_table
            
            for pair in active_pairs:
                pair_name = pair['name']
                
                # Check if statistics are missing or invalid
                needs_recalculation = False
                
                # Check stored statistics
                if pair_name not in self.pair_statistics:
                    needs_recalculation = True
                    print(f"[ComparisonWizard] Missing stored statistics for '{pair_name}'")
                
                # Check table display
                for row in range(table.rowCount()):
                    name_item = table.item(row, 1)
                    if name_item and name_item.text() == pair_name:
                        r_item = table.item(row, 2)
                        if not r_item or r_item.text() == "--":
                            needs_recalculation = True
                            print(f"[ComparisonWizard] Missing table statistics for '{pair_name}' at row {row}")
                        break
                
                if needs_recalculation:
                    print(f"[ComparisonWizard] Recalculating statistics for '{pair_name}'...")
                    self._recalculate_pair_statistics(pair)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error verifying statistics: {str(e)}")
    
    def _recalculate_pair_statistics(self, pair_config):
        """Recalculate statistics for a specific pair"""
        try:
            pair_name = pair_config['name']
            
            # Check if we already have aligned data
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                print(f"[ComparisonWizard] Using cached aligned data for '{pair_name}'")
            else:
                # Recalculate alignment
                print(f"[ComparisonWizard] Recalculating alignment for '{pair_name}'...")
                ref_channel = self._get_channel(pair_config['ref_file'], pair_config['ref_channel'])
                test_channel = self._get_channel(pair_config['test_file'], pair_config['test_channel'])
                
                if not ref_channel or not test_channel:
                    print(f"[ComparisonWizard] Cannot find channels for '{pair_name}'")
                    return
                
                aligned_data = self._align_channels(ref_channel, test_channel, pair_config)
                self.pair_aligned_data[pair_name] = aligned_data
            
            # Recalculate statistics
            stats = self._calculate_statistics(aligned_data)
            self.pair_statistics[pair_name] = stats
            
            # Update table
            self._update_pair_statistics(pair_name, stats)
            
            print(f"[ComparisonWizard] Successfully recalculated statistics for '{pair_name}'")
            self._log_individual_pair_stats(pair_name, stats)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error recalculating statistics for '{pair_config['name']}': {str(e)}")
            # Store error stats
            error_stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': f'Recalculation failed: {str(e)}'}
            self.pair_statistics[pair_config['name']] = error_stats
            self._update_pair_statistics(pair_config['name'], error_stats)
        
    def _force_table_refresh(self):
        """Force the active pairs table to refresh and show updates immediately"""
        try:
            table = self.window.active_pair_table
            table.repaint()
            table.update()
            
            # Also update the application to ensure UI responsiveness
            if hasattr(self.window, 'repaint'):
                self.window.repaint()
            
            # Process any pending events to ensure UI updates are visible
            QCoreApplication.processEvents()
            
        except Exception as e:
            print(f"[ComparisonWizard] Table refresh error: {str(e)}")
    
    def _log_individual_pair_stats(self, pair_name, stats):
        """Log individual pair statistics for debugging"""
        try:
            r_val = stats.get('r', np.nan)
            rms_val = stats.get('rms', np.nan)
            n_val = stats.get('n', 0)
            
            r_str = f"{r_val:.3f}" if not np.isnan(r_val) else "N/A"
            rms_str = f"{rms_val:.3f}" if not np.isnan(rms_val) else "N/A"
            
            print(f"[ComparisonWizard] Individual pair '{pair_name}' stats:")
            print(f"  - Correlation (r): {r_str}")
            print(f"  - RMS difference: {rms_str}")
            print(f"  - Sample size (N): {n_val}")
            
            if 'error' in stats:
                print(f"  - Error: {stats['error']}")
            
            if 'valid_ratio' in stats:
                print(f"  - Valid data ratio: {stats['valid_ratio']*100:.1f}%")
                
        except Exception as e:
            print(f"[ComparisonWizard] Error logging pair stats: {str(e)}")
    
    def _has_alignment_warnings(self, aligned_data, stats):
        """Check if alignment has warnings that should be shown to user"""
        try:
            # Check for data quality issues
            valid_ratio = aligned_data.get('valid_ratio', stats.get('valid_ratio', 1.0))
            if valid_ratio < 0.8:
                return True
                
            # Check for statistical issues
            if 'error' in stats:
                return True
                
            # Check for low correlation
            r_val = stats.get('r', np.nan)
            if not np.isnan(r_val) and abs(r_val) < 0.1:
                return True
                
            # Check for large RMS relative to data scale
            rms_val = stats.get('rms', np.nan)
            ref_std = stats.get('ref_std', np.nan)
            test_std = stats.get('test_std', np.nan)
            if not np.isnan(rms_val) and not np.isnan(ref_std) and not np.isnan(test_std):
                typical_scale = (ref_std + test_std) / 2
                if typical_scale > 0 and rms_val > 2 * typical_scale:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"[ComparisonWizard] Error checking alignment warnings: {str(e)}")
            return False
        
    def _on_pair_deleted(self):
        """Handle when a pair is deleted"""
        print("[ComparisonWizard] Pair deleted")
        
        # Remove data for deleted pairs
        active_pair_names = {pair['name'] for pair in self.window.get_active_pairs()}
        
        # Remove data for pairs that no longer exist
        pairs_to_remove = []
        for pair_name in self.pair_aligned_data.keys():
            if pair_name not in active_pair_names:
                pairs_to_remove.append(pair_name)
        
        for pair_name in pairs_to_remove:
            self.pair_aligned_data.pop(pair_name, None)
            self.pair_statistics.pop(pair_name, None)
        
        # Update cumulative display
        self._update_cumulative_display()
        
    def _on_plot_generated(self, plot_config):
        """Handle plot generation request from step 2"""
        # Get checked pairs with marker types
        checked_pairs = plot_config.get('checked_pairs', [])
        
        if not checked_pairs:
            QMessageBox.warning(self.window, "No Pairs Selected", 
                              "Please select at least one pair to generate a plot.")
            return
        
        # Determine plot type from comparison method of first pair
        plot_type = self._determine_plot_type_from_pairs(checked_pairs)
        plot_config['plot_type'] = plot_type
        
        print(f"[ComparisonWizard] Generating {plot_type} plot for {len(checked_pairs)} pairs")
        
        # Generate the plot with multiple pairs
        self._generate_multi_pair_plot(checked_pairs, plot_config)
    
    def _determine_plot_type_from_pairs(self, checked_pairs):
        """Determine appropriate plot type based on comparison methods"""
        if not checked_pairs:
            return 'scatter'
        
        # Get the comparison method from the first pair
        first_pair = checked_pairs[0]
        comparison_method = first_pair.get('comparison_method', 'Correlation Analysis')
        
        # Map comparison methods to plot types
        method_to_plot_type = {
            'Correlation Analysis': 'scatter',
            'Bland-Altman': 'bland_altman',
            'Residual Analysis': 'residual',
            'Statistical Tests': 'scatter'
        }
        
        return method_to_plot_type.get(comparison_method, 'scatter')
        
    def _generate_multi_pair_plot(self, checked_pairs, plot_config):
        """Generate plot with multiple pairs using different markers"""
        if not hasattr(self.window, 'canvas') or not self.window.canvas:
            return
            
        fig = self.window.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        
        plot_type = plot_config['plot_type']
        
        # Marker mapping
        marker_map = {
            '○ Circle': 'o',
            '□ Square': 's', 
            '△ Triangle': '^',
            '◇ Diamond': 'D',
            '▽ Inverted Triangle': 'v',
            '◁ Left Triangle': '<',
            '▷ Right Triangle': '>',
            '⬟ Pentagon': 'p',
            '✦ Star': '*',
            '⬢ Hexagon': 'h'
        }
        
        # Color mapping
        color_map = {
            '🔵 Blue': 'blue',
            '🔴 Red': 'red',
            '🟢 Green': 'green',
            '🟣 Purple': 'purple',
            '🟠 Orange': 'orange',
            '🟤 Brown': 'brown',
            '🩷 Pink': 'pink',
            '⚫ Gray': 'gray',
            '🟡 Yellow': 'gold',
            '🔶 Cyan': 'cyan'
        }
        
        all_ref_data = []
        all_test_data = []
        
        # Collect data for all pairs first
        pair_data = []
        for i, pair in enumerate(checked_pairs):
            pair_name = pair['name']
            marker_text = pair.get('marker_type', '○ Circle')
            color_text = pair.get('marker_color', '🔵 Blue')
            
            if pair_name not in self.pair_aligned_data:
                print(f"[ComparisonWizard] No aligned data for pair '{pair_name}'")
                continue
                
            aligned_data = self.pair_aligned_data[pair_name]
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None or len(ref_data) == 0:
                continue
            
            # Filter valid data for plotting
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_plot = ref_data[valid_mask]
            test_plot = test_data[valid_mask]
            
            if len(ref_plot) == 0:
                continue
            
            # Downsample for performance if too many points
            if len(ref_plot) > 2000:
                indices = np.random.choice(len(ref_plot), 2000, replace=False)
                ref_plot = ref_plot[indices]
                test_plot = test_plot[indices]
            
            # Apply downsampling if requested in config
            downsample_limit = plot_config.get('downsample')
            if downsample_limit and len(ref_plot) > downsample_limit:
                indices = np.random.choice(len(ref_plot), downsample_limit, replace=False)
                ref_plot = ref_plot[indices]
                test_plot = test_plot[indices]
            
            # Store pair data
            pair_data.append({
                'name': pair_name,
                'ref_data': ref_plot,
                'test_data': test_plot,
                'marker': marker_map.get(marker_text, 'o'),
                'color': color_map.get(color_text, 'blue')
            })
            
            # Collect for overall statistics
            all_ref_data.extend(ref_plot)
            all_test_data.extend(test_plot)
        
        if not all_ref_data:
            ax.text(0.5, 0.5, 'No valid data for plotting', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Data Available')
            try:
                fig.tight_layout()
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
                try:
                    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
                except Exception:
                    pass  # If both fail, continue without layout adjustment
            self.window.canvas.draw()
            return
        
        # Convert to arrays for overall statistics
        all_ref_data = np.array(all_ref_data)
        all_test_data = np.array(all_test_data)
        
        # Generate plots based on density type and plot type
        density_type = plot_config.get('density_display', 'scatter')
        bin_size = plot_config.get('bin_size', 20)
        kde_bandwidth = plot_config.get('kde_bandwidth', 0.2)
        
        if density_type == 'hexbin':
            # For hexbin, combine all data into single density plot
            try:
                if plot_type == 'bland_altman':
                    all_means = (all_ref_data + all_test_data) / 2
                    all_diffs = all_test_data - all_ref_data
                    
                    # Check for zero range (would cause ZeroDivisionError in hexbin)
                    if np.ptp(all_means) == 0 or np.ptp(all_diffs) == 0:
                        # Fallback to scatter plot
                        ax.scatter(all_means, all_diffs, alpha=0.6, s=20, c='blue')
                    else:
                        hb = ax.hexbin(all_means, all_diffs, gridsize=bin_size, cmap='viridis', mincnt=1)
                        
                elif plot_type == 'scatter' or plot_type == 'pearson':
                    # Check for zero range
                    if np.ptp(all_ref_data) == 0 or np.ptp(all_test_data) == 0:
                        # Fallback to scatter plot
                        ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
                    else:
                        hb = ax.hexbin(all_ref_data, all_test_data, gridsize=bin_size, cmap='viridis', mincnt=1)
                        
                elif plot_type == 'residual':
                    all_residuals = all_test_data - all_ref_data
                    # Check for zero range
                    if np.ptp(all_ref_data) == 0 or np.ptp(all_residuals) == 0:
                        # Fallback to scatter plot
                        ax.scatter(all_ref_data, all_residuals, alpha=0.6, s=20, c='blue')
                    else:
                        hb = ax.hexbin(all_ref_data, all_residuals, gridsize=bin_size, cmap='viridis', mincnt=1)
                
                # Add colorbar only if hexbin was successful
                if 'hb' in locals():
                    try:
                        cbar = plt.colorbar(hb, ax=ax)
                        cbar.set_label('Point Density', rotation=270, labelpad=15)
                    except:
                        pass
                        
            except Exception as e:
                print(f"[ComparisonWizard] Hexbin plotting failed: {e}, falling back to scatter")
                # Fallback to scatter plotting
                if plot_type == 'bland_altman':
                    all_means = (all_ref_data + all_test_data) / 2
                    all_diffs = all_test_data - all_ref_data
                    ax.scatter(all_means, all_diffs, alpha=0.6, s=20, c='blue')
                elif plot_type == 'scatter' or plot_type == 'pearson':
                    ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
                elif plot_type == 'residual':
                    all_residuals = all_test_data - all_ref_data
                    ax.scatter(all_ref_data, all_residuals, alpha=0.6, s=20, c='blue')
            
        elif density_type == 'kde':
            # For KDE, combine all data into density visualization
            try:
                if plot_type == 'bland_altman':
                    all_means = (all_ref_data + all_test_data) / 2
                    all_diffs = all_test_data - all_ref_data
                    self._create_kde_plot(ax, all_means, all_diffs, kde_bandwidth)
                    
                elif plot_type == 'scatter' or plot_type == 'pearson':
                    self._create_kde_plot(ax, all_ref_data, all_test_data, kde_bandwidth)
                    
                elif plot_type == 'residual':
                    all_residuals = all_test_data - all_ref_data
                    self._create_kde_plot(ax, all_ref_data, all_residuals, kde_bandwidth)
                    
            except Exception as e:
                print(f"[ComparisonWizard] KDE plotting failed: {e}, falling back to scatter")
                # Fallback to scatter plotting
                if plot_type == 'bland_altman':
                    all_means = (all_ref_data + all_test_data) / 2
                    all_diffs = all_test_data - all_ref_data
                    ax.scatter(all_means, all_diffs, alpha=0.6, s=20, c='blue')
                elif plot_type == 'scatter' or plot_type == 'pearson':
                    ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
                elif plot_type == 'residual':
                    all_residuals = all_test_data - all_ref_data
                    ax.scatter(all_ref_data, all_residuals, alpha=0.6, s=20, c='blue')
            
        else:
            # For scatter plots, plot each pair separately
            for pair_info in pair_data:
                ref_plot = pair_info['ref_data']
                test_plot = pair_info['test_data']
                marker = pair_info['marker']
                color = pair_info['color']
                pair_name = pair_info['name']
                
                # Generate plot based on type
                if plot_type == 'bland_altman':
                    means = (ref_plot + test_plot) / 2
                    diffs = test_plot - ref_plot
                    ax.scatter(means, diffs, alpha=0.6, s=30, 
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
                    
                elif plot_type == 'scatter' or plot_type == 'pearson':
                    ax.scatter(ref_plot, test_plot, alpha=0.6, s=30,
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
                    
                elif plot_type == 'residual':
                    residuals = test_plot - ref_plot
                    ax.scatter(ref_plot, residuals, alpha=0.6, s=30,
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
        
        # Generate plot-specific content
        if plot_type == 'pearson':
            self._generate_pearson_plot_content(ax, all_ref_data, all_test_data, plot_config)
        elif plot_type == 'bland_altman':
            self._generate_bland_altman_plot_content(ax, all_ref_data, all_test_data, plot_config)
        elif plot_type == 'scatter':
            self._generate_scatter_plot_content(ax, all_ref_data, all_test_data, plot_config)
        elif plot_type == 'residual':
            self._generate_residual_plot_content(ax, all_ref_data, all_test_data, plot_config)
        
        # Apply common plot configuration
        self._apply_common_plot_config(ax, fig, plot_config, checked_pairs)
        
        print(f"[ComparisonWizard] Plot generated successfully with {len(checked_pairs)} pairs")
        
    def _create_kde_plot(self, ax, x_data, y_data, bandwidth):
        """Create KDE density plot for multi-pair visualization"""
        try:
            if len(x_data) < 10:
                # Fallback to scatter for insufficient data
                print(f"[ComparisonWizard] Insufficient data for KDE ({len(x_data)} points), using scatter fallback")
                ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                return
            
            # Create KDE
            data_stack = np.vstack([x_data, y_data])
            kde = gaussian_kde(data_stack, bw_method=bandwidth)
            
            # Create grid for evaluation
            x_min, x_max = np.min(x_data), np.max(x_data)
            y_min, y_max = np.min(y_data), np.max(y_data)
            
            # Add padding to make sure all data is covered
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_padding = 0.1 * x_range if x_range > 0 else 1.0
            y_padding = 0.1 * y_range if y_range > 0 else 1.0
            
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
            
            # Create evaluation grid (adaptive size based on data)
            grid_size = min(100, max(30, int(np.sqrt(len(x_data)) / 2)))
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(xi, yi)
            
            # Evaluate KDE on grid
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)
            
            # Create filled contour plot
            levels = 15
            cs = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
            
            # Add colorbar
            try:
                cbar = plt.colorbar(cs, ax=ax)
                cbar.set_label('Density', rotation=270, labelpad=15)
            except Exception as cbar_error:
                print(f"[ComparisonWizard] Could not add colorbar: {cbar_error}")
            
            # Overlay scatter points with transparency for reference
            # Downsample scatter points if too many
            n_scatter = min(1000, len(x_data))
            if n_scatter < len(x_data):
                indices = np.random.choice(len(x_data), n_scatter, replace=False)
                x_scatter = x_data[indices]
                y_scatter = y_data[indices]
            else:
                x_scatter = x_data
                y_scatter = y_data
                
            ax.scatter(x_scatter, y_scatter, alpha=0.3, s=8, c='white', 
                      edgecolors='black', linewidth=0.3)
            
            print(f"[ComparisonWizard] KDE plot created successfully with {len(x_data)} points")
            
        except Exception as e:
            print(f"[ComparisonWizard] KDE plot creation failed: {e}")
            # Ultimate fallback to simple scatter
            ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
        
    def _generate_pearson_plot_content(self, ax, all_ref_data, all_test_data, plot_config):
        """Generate Pearson correlation plot content"""
        # Add 1:1 line
        min_val = min(np.min(all_ref_data), np.min(all_test_data))
        max_val = max(np.max(all_ref_data), np.max(all_test_data))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
               linewidth=2, label='1:1 line')
        
        # Calculate correlation statistics
        try:
            r_value, p_value = scipy_stats.pearsonr(all_ref_data, all_test_data)
            r2_value = r_value ** 2
            
            # Add correlation text box
            stats_text = f'r = {r_value:.4f}\nR² = {r2_value:.4f}\np = {p_value:.2e}' if not np.isnan(p_value) else f'r = {r_value:.4f}\nR² = {r2_value:.4f}'
            
            # Position text box in upper left or wherever there's space
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=12)
                   
        except Exception as e:
            print(f"[ComparisonWizard] Error calculating Pearson correlation: {e}")
            ax.text(0.05, 0.95, 'Correlation: N/A', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top')
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', linewidth=2,
                          label=f'y = {custom_val}')
            except (ValueError, TypeError):
                pass
        
        ax.set_xlabel(plot_config.get('xlabel', 'Reference'))
        ax.set_ylabel(plot_config.get('ylabel', 'Test'))
        ax.set_title('Pearson Correlation Plot')
        
    def _generate_bland_altman_plot_content(self, ax, all_ref_data, all_test_data, plot_config):
        """Generate Bland-Altman plot content"""
        # Add Bland-Altman lines
        all_means = (all_ref_data + all_test_data) / 2
        all_diffs = all_test_data - all_ref_data
        
        mean_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=2, 
                  label=f'Mean difference: {mean_diff:.3f}')
        
        # Add confidence intervals if requested
        if plot_config.get('confidence_interval', False):
            ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', 
                      label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
            ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--',
                      label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', linewidth=2,
                          label=f'Custom: {custom_val}')
            except (ValueError, TypeError):
                pass
        
        ax.set_xlabel(plot_config.get('xlabel', 'Mean of Reference and Test'))
        ax.set_ylabel(plot_config.get('ylabel', 'Test - Reference'))
        ax.set_title('Bland-Altman Plot')
        
    def _generate_scatter_plot_content(self, ax, all_ref_data, all_test_data, plot_config):
        """Generate scatter plot content"""
        # Add 1:1 line
        min_val = min(np.min(all_ref_data), np.min(all_test_data))
        max_val = max(np.max(all_ref_data), np.max(all_test_data))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, 
               linewidth=2, label='1:1 line')
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', linewidth=2,
                          label=f'y = {custom_val}')
            except (ValueError, TypeError):
                pass
        
        ax.set_xlabel(plot_config.get('xlabel', 'Reference'))
        ax.set_ylabel(plot_config.get('ylabel', 'Test'))
        
        # Add correlation to title
        try:
            r_value, _ = scipy_stats.pearsonr(all_ref_data, all_test_data)
            r_str = f"r = {r_value:.3f}" if not np.isnan(r_value) else "r = N/A"
        except:
            r_str = "r = N/A"
        ax.set_title(f'Scatter Plot ({r_str})')
        
    def _generate_residual_plot_content(self, ax, all_ref_data, all_test_data, plot_config):
        """Generate residual plot content"""
        # Add zero line
        ax.axhline(0, color='red', linestyle='-', alpha=0.8, linewidth=2, 
                  label='Zero residual')
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', linewidth=2,
                          label=f'y = {custom_val}')
            except (ValueError, TypeError):
                pass
        
        ax.set_xlabel(plot_config.get('xlabel', 'Reference'))
        ax.set_ylabel(plot_config.get('ylabel', 'Residuals (Test - Reference)'))
        ax.set_title('Residual Plot')
        
    def _apply_common_plot_config(self, ax, fig, plot_config, checked_pairs):
        """Apply common plot configuration options"""
        # Apply plot configuration
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
            
        # Only show legend for scatter plots and if requested
        density_type = plot_config.get('density_display', 'scatter')
        if plot_config.get('show_legend', True) and density_type == 'scatter' and len(checked_pairs) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Apply axis ranges if specified
        x_range = plot_config.get('x_range', 'Auto')
        if x_range != 'Auto' and x_range.strip():
            try:
                x_min, x_max = map(float, x_range.split(','))
                ax.set_xlim(x_min, x_max)
            except:
                pass  # Invalid range format, use auto
                
        y_range = plot_config.get('y_range', 'Auto')
        if y_range != 'Auto' and y_range.strip():
            try:
                y_min, y_max = map(float, y_range.split(','))
                ax.set_ylim(y_min, y_max)
            except:
                pass  # Invalid range format, use auto
        
        # Apply tight layout with error handling
        try:
            fig.tight_layout()
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
            try:
                # Fallback to manual layout adjustment
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
            except Exception as fallback_error:
                print(f"[ComparisonWizard] Layout adjustment fallback also failed: {fallback_error}")
        
        self.window.canvas.draw()
    
    def _clear_preview_plot(self):
        """Clear the preview plot"""
        if not hasattr(self.window, 'canvas') or not self.window.canvas:
            return
            
        fig = self.window.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No pairs selected for preview', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Data Alignment Preview')
        try:
            fig.tight_layout()
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
            try:
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
            except Exception:
                pass  # If both fail, continue without layout adjustment
        self.window.canvas.draw()
    
    def _get_channel(self, filename, channel_name):
        """Get channel object by filename and channel name"""
        if not self.channel_manager or not self.file_manager:
            return None
            
        # Find file by filename
        file_info = None
        for f in self.file_manager.get_all_files():
            if f.filename == filename:
                file_info = f
                break
        
        if not file_info:
            return None
        
        # Get channels for this file using file_id
        channels = self.channel_manager.get_channels_by_file(file_info.file_id)
        
        # Find matching channel by legend_label or channel_id
        for channel in channels:
            name = channel.legend_label or channel.channel_id
            if name == channel_name:
                return channel
        return None
        
    def _get_pair_config(self, pair_name):
        """Get pair configuration by name"""
        for pair in self.window.get_active_pairs():
            if pair['name'] == pair_name:
                return pair
        return None
        
    def _align_channels(self, ref_channel, test_channel, pair_config):
        """Align two channels based on configuration"""
        alignment_mode = pair_config.get('alignment_mode', 'index')
        alignment_config = pair_config.get('alignment_config', {})
        
        if alignment_mode == 'index':
            return self._align_by_index(ref_channel, test_channel, alignment_config)
        else:
            return self._align_by_time(ref_channel, test_channel, alignment_config)
            
    def _align_by_index(self, ref_channel, test_channel, config):
        """Align channels by index with robust validation"""
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
            
            # Validate configuration - handle both old and new config formats
            mode = config.get('mode', 'custom')  # Default to custom for new format
            if mode not in ['truncate', 'custom']:
                mode = 'custom'  # Fallback to custom for compatibility
            
            # For new format, always use custom mode with provided parameters
            if 'start_index' in config or 'end_index' in config:
                mode = 'custom'
            
            if mode == 'truncate':
                # Truncate to shortest length
                min_length = min(len(ref_data), len(test_data))
                if min_length == 0:
                    raise ValueError("No data points available for alignment")
                
                ref_aligned = ref_data[:min_length].copy()
                test_aligned = test_data[:min_length].copy()
                actual_range = (0, min_length - 1)
                
            else:
                # Custom range with validation
                start_idx = config.get('start_index', 0)
                end_idx = config.get('end_index', 500)  # Use default from new window
                
                # Ensure end_idx doesn't exceed data length
                max_idx = min(len(ref_data), len(test_data)) - 1
                if end_idx > max_idx:
                    end_idx = max_idx
                
                # Validate indices
                max_ref_idx = len(ref_data) - 1
                max_test_idx = len(test_data) - 1
                
                if start_idx < 0:
                    start_idx = 0
                if end_idx < 0:
                    end_idx = 0
                
                if start_idx > max_ref_idx or start_idx > max_test_idx:
                    raise ValueError(f"Start index {start_idx} exceeds data length (ref: {len(ref_data)}, test: {len(test_data)})")
                
                if end_idx > max_ref_idx or end_idx > max_test_idx:
                    end_idx = min(max_ref_idx, max_test_idx)
                    if hasattr(self, 'window'):
                        QMessageBox.warning(self.window, "Index Range Adjusted", 
                                          f"End index adjusted to {end_idx} to fit within data bounds")
                
                if start_idx >= end_idx:
                    raise ValueError(f"Invalid index range: start ({start_idx}) >= end ({end_idx})")
                
                ref_aligned = ref_data[start_idx:end_idx+1].copy()
                test_aligned = test_data[start_idx:end_idx+1].copy()
                actual_range = (start_idx, end_idx)
                
            # Apply offset if specified
            offset = config.get('offset', 0)
            if offset != 0:
                if abs(offset) >= len(ref_aligned):
                    raise ValueError(f"Offset magnitude ({abs(offset)}) exceeds aligned data length ({len(ref_aligned)})")
                
                if offset > 0:
                    # Positive offset: shift test data forward, truncate ref data
                    if offset >= len(test_aligned):
                        raise ValueError(f"Positive offset ({offset}) too large for test data length ({len(test_aligned)})")
                    test_aligned = test_aligned[offset:]
                    ref_aligned = ref_aligned[:len(test_aligned)]
                else:
                    # Negative offset: shift ref data forward, truncate test data
                    offset_abs = abs(offset)
                    if offset_abs >= len(ref_aligned):
                        raise ValueError(f"Negative offset magnitude ({offset_abs}) too large for ref data length ({len(ref_aligned)})")
                    ref_aligned = ref_aligned[offset_abs:]
                    test_aligned = test_aligned[:len(ref_aligned)]
            
            # Final validation
            if len(ref_aligned) != len(test_aligned):
                raise ValueError("Aligned data arrays have different lengths")
            
            if len(ref_aligned) == 0:
                raise ValueError("No data points remaining after alignment")
            
            # Check for valid numeric data
            valid_mask = np.isfinite(ref_aligned) & np.isfinite(test_aligned)
            valid_ratio = np.sum(valid_mask) / len(valid_mask)
            
            if valid_ratio < 0.1:
                raise ValueError(f"Too many invalid values in aligned data ({valid_ratio*100:.1f}% valid)")
            elif valid_ratio < 0.8 and hasattr(self, 'window'):
                QMessageBox.warning(self.window, "Data Quality Warning", 
                                  f"Only {valid_ratio*100:.1f}% of aligned data is valid")
                
            return {
                'ref_data': ref_aligned,
                'test_data': test_aligned,
                'ref_label': ref_channel.legend_label or ref_channel.ylabel,
                'test_label': test_channel.legend_label or test_channel.ylabel,
                'alignment_method': 'index',
                'valid_ratio': valid_ratio,
                'index_range': actual_range,
                'n_points': len(ref_aligned),
                'offset_applied': offset
            }
            
        except Exception as e:
            error_msg = f"Index alignment failed: {str(e)}"
            print(f"[ComparisonWizard] {error_msg}")
            if hasattr(self, 'window'):
                QMessageBox.critical(self.window, "Index Alignment Error", error_msg)
            raise
        
    def _align_by_time(self, ref_channel, test_channel, config):
        """Align channels by time with robust interpolation"""
        try:
            # Validate input data
            validation_result = self._validate_time_data(ref_channel, test_channel)
            if not validation_result['valid']:
                # Try to create time data if missing
                print(f"[ComparisonWizard] Time validation failed: {validation_result['error']}")
                print("[ComparisonWizard] Attempting to create time data from indices...")
                
                # Check if we can create time data from indices
                if self._try_create_time_data(ref_channel, test_channel):
                    print("[ComparisonWizard] Successfully created time data, retrying validation...")
                    validation_result = self._validate_time_data(ref_channel, test_channel)
                    if not validation_result['valid']:
                        raise ValueError(f"Time alignment failed even after creating time data: {validation_result['error']}")
                else:
                    raise ValueError(f"Time alignment failed: {validation_result['error']}")
            
            print("[ComparisonWizard] Time data validation passed, proceeding with alignment...")
            
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
                
            # Determine time range - handle both old and new config formats
            mode = config.get('mode', 'custom')  # Default to custom if not specified
            
            if mode == 'overlap':
                # Find overlapping time range
                start_time = max(ref_x.min(), test_x.min())
                end_time = min(ref_x.max(), test_x.max())
                
                if start_time >= end_time:
                    raise ValueError("No overlapping time range found between channels")
            else:
                # Custom time window - use parameters from new config format
                start_time = config.get('start_time', 0.0)
                end_time = config.get('end_time', 10.0)  # Updated default to match new window
                
                # Validate custom time range
                ref_range = (ref_x.min(), ref_x.max())
                test_range = (test_x.min(), test_x.max())
                
                warnings_msg = []
                if start_time < ref_range[0] or end_time > ref_range[1]:
                    warnings_msg.append(f"Custom time range [{start_time:.3f}, {end_time:.3f}] extends beyond reference data range [{ref_range[0]:.3f}, {ref_range[1]:.3f}]")
                if start_time < test_range[0] or end_time > test_range[1]:
                    warnings_msg.append(f"Custom time range [{start_time:.3f}, {end_time:.3f}] extends beyond test data range [{test_range[0]:.3f}, {test_range[1]:.3f}]")
                
                if warnings_msg and hasattr(self, 'window'):
                    QMessageBox.warning(self.window, "Time Range Warning", 
                                      "Extrapolation required:\n" + "\n".join(warnings_msg))
                
            # Create time grid with smart round_to calculation
            round_to = config.get('round_to', 0.01)
            if round_to <= 0:
                raise ValueError("Round-to value must be positive")
            
            # Smart adjustment of round_to to prevent excessive grid size
            time_range = end_time - start_time
            estimated_points = time_range / round_to
            
            if estimated_points > 100000:
                # Auto-adjust round_to to keep grid manageable
                suggested_round_to = time_range / 50000  # Target ~50K points
                
                if hasattr(self, 'window'):
                    reply = QMessageBox.question(
                        self.window, 
                        "Large Time Grid Detected",
                        f"The current settings would create {estimated_points:.0f} time points.\n\n"
                        f"This may cause performance issues or errors.\n\n"
                        f"Current round-to: {round_to:.4f}s\n"
                        f"Suggested round-to: {suggested_round_to:.4f}s (≈{time_range/suggested_round_to:.0f} points)\n\n"
                        f"Use suggested value?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        round_to = suggested_round_to
                        print(f"[ComparisonWizard] Auto-adjusted round_to to {round_to:.4f}s")
                    else:
                        # User declined, but still enforce a maximum
                        max_round_to = time_range / 100000
                        if round_to < max_round_to:
                            round_to = max_round_to
                            print(f"[ComparisonWizard] Enforced minimum round_to of {round_to:.4f}s to prevent crash")
                else:
                    # No window available, auto-adjust
                    round_to = suggested_round_to
                    print(f"[ComparisonWizard] Auto-adjusted round_to to {round_to:.4f}s (was {config.get('round_to', 0.01):.4f}s)")
                
            time_grid = np.arange(start_time, end_time + round_to/2, round_to)
            
            if len(time_grid) == 0:
                raise ValueError("Generated time grid is empty")
            if len(time_grid) > 100000:
                raise ValueError(f"Time grid still too large ({len(time_grid)} points) after adjustment. Use larger round_to value.")
            
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
            elif valid_ratio < 0.5 and hasattr(self, 'window'):
                QMessageBox.warning(self.window, "Data Quality Warning", 
                                  f"Only {valid_ratio*100:.1f}% of interpolated data is valid. Consider adjusting time range or interpolation method.")
            
            return {
                'ref_data': ref_interp,
                'test_data': test_interp,
                'time_data': time_grid,
                'ref_label': ref_channel.legend_label or ref_channel.ylabel,
                'test_label': test_channel.legend_label or test_channel.ylabel,
                'alignment_method': 'time',
                'valid_ratio': valid_ratio,
                'time_range': (start_time, end_time),
                'n_points': len(time_grid),
                'round_to_used': round_to,
                'round_to_original': config.get('round_to', 0.01)
            }
            
        except Exception as e:
            error_msg = f"Time alignment failed: {str(e)}"
            print(f"[ComparisonWizard] {error_msg}")
            if hasattr(self, 'window'):
                QMessageBox.critical(self.window, "Time Alignment Error", error_msg)
            raise
    
    def _try_create_time_data(self, ref_channel, test_channel):
        """Try to create time data for channels that don't have it"""
        try:
            channels_updated = []
            
            for channel, name in [(ref_channel, 'reference'), (test_channel, 'test')]:
                needs_time_data = False
                
                # Check if channel has no xdata at all
                if not hasattr(channel, 'xdata') or channel.xdata is None:
                    needs_time_data = True
                    print(f"[ComparisonWizard] No xdata found for {name} channel")
                
                # Check if xdata exists but contains unconvertible data
                elif hasattr(channel, 'xdata') and channel.xdata is not None:
                    try:
                        xdata_array = np.asarray(channel.xdata)
                        if not np.issubdtype(xdata_array.dtype, np.number):
                            # Check if this is datetime data that we should convert
                            if xdata_array.dtype == object or xdata_array.dtype.kind in ['U', 'S']:
                                # Check if it looks like datetime strings
                                sample_values = xdata_array[:min(5, len(xdata_array))]
                                datetime_like = any(isinstance(val, str) and any(char in str(val) for char in ['-', ':', ' ']) for val in sample_values)
                                
                                if datetime_like:
                                    print(f"[ComparisonWizard] Found datetime-like strings in {name} channel xdata, will handle in validation")
                                    # Don't mark as needing new time data - let validation handle the conversion
                                    continue
                            
                            # If not datetime-like, we need to create new time data
                            needs_time_data = True
                            print(f"[ComparisonWizard] Non-numeric, non-datetime xdata found for {name} channel")
                    except Exception as check_error:
                        needs_time_data = True
                        print(f"[ComparisonWizard] Error checking xdata for {name} channel: {check_error}")
                
                if needs_time_data:
                    print(f"[ComparisonWizard] Creating time data for {name} channel...")
                    # Create time data based on sampling rate or indices
                    
                    # Check if channel has sampling rate information
                    if hasattr(channel, 'sampling_rate') and channel.sampling_rate is not None and channel.sampling_rate > 0:
                        # Create time axis based on sampling rate
                        n_samples = len(channel.ydata)
                        dt = 1.0 / channel.sampling_rate
                        time_axis = np.arange(n_samples) * dt
                        channel.xdata = time_axis
                        print(f"  - Created time axis using sampling rate {channel.sampling_rate} Hz")
                        channels_updated.append(name)
                    
                    elif hasattr(channel, 'sample_period') and channel.sample_period is not None and channel.sample_period > 0:
                        # Create time axis based on sample period
                        n_samples = len(channel.ydata)
                        time_axis = np.arange(n_samples) * channel.sample_period
                        channel.xdata = time_axis
                        print(f"  - Created time axis using sample period {channel.sample_period} s")
                        channels_updated.append(name)
                    
                    else:
                        # Fallback: create time axis assuming 1 Hz sampling
                        n_samples = len(channel.ydata)
                        time_axis = np.arange(n_samples, dtype=float)
                        channel.xdata = time_axis
                        print(f"  - Created time axis using indices (assuming 1 Hz sampling)")
                        channels_updated.append(name)
                        
                        # Show warning to user
                        if hasattr(self, 'window'):
                            QMessageBox.information(
                                self.window,
                                "Time Data Created",
                                f"No usable time data found for {name} channel.\n"
                                f"Created time axis using sample indices (assuming 1 Hz).\n"
                                f"For better results, ensure your data includes proper time information."
                            )
            
            return len(channels_updated) > 0
            
        except Exception as e:
            print(f"[ComparisonWizard] Failed to create time data: {str(e)}")
            return False
    
    def _validate_time_data(self, ref_channel, test_channel):
        """Validate time data before alignment"""
        try:
            # Check if channels exist and have data
            for channel, name in [(ref_channel, 'reference'), (test_channel, 'test')]:
                # Check for basic attributes
                if not hasattr(channel, 'xdata') or not hasattr(channel, 'ydata'):
                    return {'valid': False, 'error': f'{name} channel missing time or data arrays'}
                
                # Check for None data
                if channel.xdata is None or channel.ydata is None:
                    return {'valid': False, 'error': f'{name} channel has None data'}
                
                # Check for empty data
                if len(channel.xdata) == 0 or len(channel.ydata) == 0:
                    return {'valid': False, 'error': f'{name} channel has empty data'}
                
                # Check for length mismatch
                if len(channel.xdata) != len(channel.ydata):
                    return {'valid': False, 'error': f'{name} channel time and data arrays have different lengths'}
                
                # Detailed inspection of time data
                xdata = channel.xdata
                print(f"[ComparisonWizard] {name} channel xdata info:")
                print(f"  - Type: {type(xdata)}")
                print(f"  - Shape: {getattr(xdata, 'shape', 'N/A')}")
                print(f"  - Dtype: {getattr(xdata, 'dtype', 'N/A')}")
                print(f"  - First 5 values: {xdata[:5] if len(xdata) > 0 else 'Empty'}")
                
                # Try to convert to numpy array if it's not already
                try:
                    xdata_array = np.asarray(xdata)
                    print(f"  - Converted dtype: {xdata_array.dtype}")
                    print(f"  - Is numeric: {np.issubdtype(xdata_array.dtype, np.number)}")
                except Exception as conv_error:
                    return {'valid': False, 'error': f'{name} channel time data cannot be converted to array: {conv_error}'}
                
                # Check if the data can be treated as numeric
                if not np.issubdtype(xdata_array.dtype, np.number):
                    # Try to convert string/object data to numeric
                    if xdata_array.dtype == object or xdata_array.dtype.kind in ['U', 'S']:
                        try:
                            # First try direct numeric conversion
                            xdata_numeric = pd.to_numeric(xdata_array, errors='coerce')
                            if not np.all(np.isnan(xdata_numeric)):
                                print(f"  - Successfully converted string/object data to numeric")
                                channel.xdata = xdata_numeric
                            else:
                                # Try datetime conversion if numeric conversion failed
                                print(f"  - Numeric conversion failed, trying datetime conversion...")
                                try:
                                    # Convert to pandas datetime then to timestamps
                                    datetime_series = pd.to_datetime(xdata_array, errors='coerce')
                                    if not datetime_series.isna().all():
                                        # Convert to seconds since the first timestamp (relative time)
                                        valid_datetimes = datetime_series.dropna()
                                        if len(valid_datetimes) == 0:
                                            raise ValueError("No valid datetime values found")
                                        
                                        # Get first valid datetime - handle both Series and Index
                                        if hasattr(valid_datetimes, 'iloc'):
                                            first_time = valid_datetimes.iloc[0]
                                        else:
                                            first_time = valid_datetimes[0]
                                        
                                        # Convert to seconds since the first timestamp (relative time)
                                        relative_seconds = (datetime_series - first_time).total_seconds()
                                        
                                        # Handle any NaT values by converting to NaN
                                        if hasattr(relative_seconds, 'values'):
                                            xdata_numeric = relative_seconds.values
                                        else:
                                            xdata_numeric = np.array(relative_seconds)
                                        
                                        # Replace any inf or -inf with NaN
                                        xdata_numeric = np.where(np.isfinite(xdata_numeric), xdata_numeric, np.nan)
                                        
                                        print(f"    - Successfully converted datetime strings to relative seconds")
                                        print(f"    - Time range: {np.nanmin(xdata_numeric):.3f} to {np.nanmax(xdata_numeric):.3f} seconds")
                                        print(f"    - Reference time: {first_time}")
                                        
                                        # Update the channel's xdata with converted values
                                        channel.xdata = xdata_numeric
                                        
                                        # Show info to user about datetime conversion
                                        if hasattr(self, 'window'):
                                            QMessageBox.information(
                                                self.window,
                                                "DateTime Conversion",
                                                f"Converted datetime strings to relative time for {name} channel.\n\n"
                                                f"Reference time: {first_time}\n"
                                                f"Time range: {np.nanmin(xdata_numeric):.1f} to {np.nanmax(xdata_numeric):.1f} seconds"
                                            )
                                    else:
                                        raise ValueError("All datetime values are invalid")
                                        
                                except Exception as dt_error:
                                    print(f"    - Datetime conversion also failed: {dt_error}")
                                    print(f"    - Attempting fallback: create time data from indices...")
                                    
                                    # Fallback: create time axis from indices
                                    try:
                                        n_samples = len(xdata_array)
                                        time_axis = np.arange(n_samples, dtype=float)
                                        channel.xdata = time_axis
                                        print(f"    - Created fallback time axis using indices (0 to {n_samples-1})")
                                        
                                        if hasattr(self, 'window'):
                                            QMessageBox.warning(
                                                self.window,
                                                "DateTime Conversion Failed",
                                                f"Could not convert datetime strings for {name} channel.\n\n"
                                                f"Error: {dt_error}\n\n"
                                                f"Created time axis using sample indices as fallback.\n"
                                                f"Time alignment may not be accurate."
                                            )
                                    except Exception as fallback_error:
                                        return {'valid': False, 'error': f'{name} channel time data conversion failed completely: {fallback_error}'}
                        except Exception as convert_error:
                            return {'valid': False, 'error': f'{name} channel time data is not numeric and cannot be converted: {convert_error}'}
                    else:
                        return {'valid': False, 'error': f'{name} channel time data is not numeric (dtype: {xdata_array.dtype})'}
                
                # Check for valid numeric data
                xdata_final = np.asarray(channel.xdata)
                ydata_final = np.asarray(channel.ydata)
                
                time_valid = np.isfinite(xdata_final)
                data_valid = np.isfinite(ydata_final)
                
                if np.sum(time_valid) < 2:
                    return {'valid': False, 'error': f'{name} channel has insufficient valid time points ({np.sum(time_valid)} valid)'}
                
                if np.sum(data_valid) < 2:
                    return {'valid': False, 'error': f'{name} channel has insufficient valid data points ({np.sum(data_valid)} valid)'}
                
                # Check time range
                valid_times = xdata_final[time_valid]
                time_range = valid_times.max() - valid_times.min()
                if time_range <= 0:
                    return {'valid': False, 'error': f'{name} channel has zero or negative time range ({time_range})'}
                
                print(f"  - Valid time points: {np.sum(time_valid)}/{len(time_valid)}")
                print(f"  - Time range: {valid_times.min():.3f} to {valid_times.max():.3f} ({time_range:.3f})")
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    def _clean_and_sort_time_data(self, x_data, y_data):
        """Clean and sort time data, removing invalid values and duplicates"""
        # Remove NaN and infinite values
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 2:
            raise ValueError("Insufficient valid data points after cleaning")
        
        # Sort by time
        sort_indices = np.argsort(x_clean)
        x_sorted = x_clean[sort_indices]
        y_sorted = y_clean[sort_indices]
        
        # Handle duplicate time values by averaging
        if len(np.unique(x_sorted)) < len(x_sorted):
            # Find unique times and average corresponding y values
            unique_times, inverse_indices = np.unique(x_sorted, return_inverse=True)
            averaged_y = np.zeros_like(unique_times)
            
            for i in range(len(unique_times)):
                mask = inverse_indices == i
                averaged_y[i] = np.mean(y_sorted[mask])
            
            x_sorted = unique_times
            y_sorted = averaged_y
        
        return x_sorted, y_sorted
    
    def _interpolate_channel(self, x_data, y_data, time_grid, method, channel_name):
        """Robustly interpolate channel data to time grid"""
        try:
            if method == 'linear':
                # Use numpy's linear interpolation (fastest)
                return np.interp(time_grid, x_data, y_data)
            
            elif method == 'nearest':
                # Nearest neighbor interpolation
                f = interp1d(x_data, y_data, kind='nearest', 
                           bounds_error=False, fill_value=np.nan)
                return f(time_grid)
            
            elif method == 'cubic':
                # Cubic spline interpolation
                if len(x_data) < 4:
                    # Fall back to linear for insufficient points
                    warnings.warn(f"Not enough points for cubic interpolation in {channel_name} channel, using linear")
                    return np.interp(time_grid, x_data, y_data)
                
                f = interp1d(x_data, y_data, kind='cubic', 
                           bounds_error=False, fill_value=np.nan)
                return f(time_grid)
            
            else:
                # Default to linear for unknown methods
                warnings.warn(f"Unknown interpolation method '{method}', using linear")
                return np.interp(time_grid, x_data, y_data)
                
        except Exception as e:
            raise ValueError(f"Interpolation failed for {channel_name} channel: {str(e)}")
        
    def _calculate_statistics(self, aligned_data):
        """Calculate comparison statistics with robust error handling"""
        try:
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None:
                return {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'Missing data'}
            
            if len(ref_data) != len(test_data):
                return {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'Data length mismatch'}
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_clean = ref_data[valid_mask]
            test_clean = test_data[valid_mask]
            
            n_valid = len(ref_clean)
            n_total = len(ref_data)
            
            if n_valid == 0:
                return {'r': np.nan, 'rms': np.nan, 'n': 0, 'valid_ratio': 0.0, 'error': 'No valid data points'}
            
            if n_valid < 3:
                return {'r': np.nan, 'rms': np.nan, 'n': n_valid, 'valid_ratio': n_valid/n_total, 'error': 'Insufficient data for statistics'}
            
            # Calculate statistics
            stats_dict = {'n': n_valid, 'valid_ratio': n_valid/n_total}
            
            # Correlation coefficient
            try:
                # Check for constant values
                if np.var(ref_clean) == 0 or np.var(test_clean) == 0:
                    stats_dict['r'] = np.nan
                    stats_dict['r_error'] = 'Constant values'
                else:
                    correlation, p_value = scipy_stats.pearsonr(ref_clean, test_clean)
                    stats_dict['r'] = correlation
                    stats_dict['r_pvalue'] = p_value
            except Exception as e:
                stats_dict['r'] = np.nan
                stats_dict['r_error'] = str(e)
            
            # RMS difference (always calculable if we have data)
            try:
                differences = test_clean - ref_clean
                rms = np.sqrt(np.mean(differences ** 2))
                stats_dict['rms'] = rms
                
                # Additional statistics
                stats_dict['mean_diff'] = np.mean(differences)
                stats_dict['std_diff'] = np.std(differences)
                stats_dict['median_diff'] = np.median(differences)
                stats_dict['max_abs_diff'] = np.max(np.abs(differences))
                
            except Exception as e:
                stats_dict['rms'] = np.nan
                stats_dict['rms_error'] = str(e)
            
            # Mean and standard deviation of original data
            try:
                stats_dict['ref_mean'] = np.mean(ref_clean)
                stats_dict['ref_std'] = np.std(ref_clean)
                stats_dict['test_mean'] = np.mean(test_clean)
                stats_dict['test_std'] = np.std(test_clean)
            except Exception as e:
                stats_dict['stats_error'] = str(e)
            
            # Bland-Altman statistics
            try:
                means = (ref_clean + test_clean) / 2
                diffs = test_clean - ref_clean
                
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                
                stats_dict['bland_altman'] = {
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'upper_limit': mean_diff + 1.96 * std_diff,
                    'lower_limit': mean_diff - 1.96 * std_diff
                }
            except Exception as e:
                stats_dict['bland_altman_error'] = str(e)
            
            return stats_dict
            
        except Exception as e:
            print(f"[ComparisonWizard] Statistics calculation error: {str(e)}")
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': str(e)}
        
    def _update_pair_statistics(self, pair_name, statistics):
        """Update statistics in the active pairs table"""
        print(f"[ComparisonWizard] Updating table statistics for pair '{pair_name}'...")
        
        table = self.window.active_pair_table
        pair_row_found = False
        
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            if name_item and name_item.text() == pair_name:
                pair_row_found = True
                print(f"[ComparisonWizard] Found pair '{pair_name}' at table row {row}")
                
                # Set tooltip for pair name showing file/channel details
                self._set_pair_name_tooltip(row, pair_name)
                
                # Handle correlation coefficient
                r_val = statistics.get('r', np.nan)
                if np.isnan(r_val):
                    r_text = "N/A"
                    r_color = None
                else:
                    r_text = f"{r_val:.3f}"
                    # Color code correlation: good (green), moderate (orange), poor (red)
                    if abs(r_val) >= 0.7:
                        r_color = QColor(34, 139, 34)  # Forest green
                    elif abs(r_val) >= 0.3:
                        r_color = QColor(255, 140, 0)  # Dark orange  
                    else:
                        r_color = QColor(220, 20, 60)  # Crimson
                
                # Handle RMS
                rms_val = statistics.get('rms', np.nan)
                if np.isnan(rms_val):
                    rms_text = "N/A"
                    rms_color = None
                else:
                    rms_text = f"{rms_val:.3f}"
                    rms_color = QColor(70, 130, 180)  # Steel blue
                
                # Handle sample size
                n_val = statistics.get('n', 0)
                n_text = f"{n_val:,}"  # Add comma separators for readability
                n_color = QColor(105, 105, 105) if n_val > 0 else QColor(220, 20, 60)
                
                # Update table items with colors
                from PySide6.QtWidgets import QTableWidgetItem
                from PySide6.QtCore import Qt
                
                # Correlation item
                r_item = QTableWidgetItem(r_text)
                if r_color:
                    r_item.setForeground(r_color)
                    r_item.setData(Qt.FontRole, self._get_bold_font())
                table.setItem(row, 2, r_item)
                
                # RMS item  
                rms_item = QTableWidgetItem(rms_text)
                if rms_color:
                    rms_item.setForeground(rms_color)
                    rms_item.setData(Qt.FontRole, self._get_bold_font())
                table.setItem(row, 3, rms_item)
                
                # N item
                n_item = QTableWidgetItem(n_text)
                n_item.setForeground(n_color)
                n_item.setData(Qt.FontRole, self._get_bold_font())
                table.setItem(row, 4, n_item)
                
                # Set comprehensive tooltips
                self._set_detailed_tooltips(row, statistics)
                
                print(f"[ComparisonWizard] Updated row {row}: r={r_text}, RMS={rms_text}, N={n_text}")
                
                # Immediately refresh this specific row
                table.update(table.model().index(row, 2))
                table.update(table.model().index(row, 3))
                table.update(table.model().index(row, 4))
                
                break
        
        if not pair_row_found:
            print(f"[ComparisonWizard] WARNING: Could not find table row for pair '{pair_name}'")
            print(f"[ComparisonWizard] Available pairs in table:")
            for row in range(table.rowCount()):
                name_item = table.item(row, 1)
                if name_item:
                    print(f"  Row {row}: '{name_item.text()}'")
                else:
                    print(f"  Row {row}: No name item")
                    
    def _set_pair_name_tooltip(self, row, pair_name):
        """Set tooltip for pair name showing file and channel details"""
        try:
            table = self.window.active_pair_table
            name_item = table.item(row, 1)
            
            if not name_item:
                return
            
            # Find the pair configuration
            active_pairs = self.window.get_active_pairs()
            pair_config = None
            for pair in active_pairs:
                if pair['name'] == pair_name:
                    pair_config = pair
                    break
            
            if not pair_config:
                name_item.setToolTip(f"Pair: {pair_name}\n(Configuration not found)")
                return
            
            # Build detailed tooltip
            tooltip_lines = []
            tooltip_lines.append(f"Comparison Pair: {pair_name}")
            tooltip_lines.append("")  # Empty line for spacing
            
            # Reference channel info
            ref_file = pair_config.get('ref_file', 'Unknown')
            ref_channel = pair_config.get('ref_channel', 'Unknown')
            tooltip_lines.append(f"📊 Reference:")
            tooltip_lines.append(f"   File: {ref_file}")
            tooltip_lines.append(f"   Channel: {ref_channel}")
            
            # Test channel info
            test_file = pair_config.get('test_file', 'Unknown')
            test_channel = pair_config.get('test_channel', 'Unknown')
            tooltip_lines.append(f"📊 Test:")
            tooltip_lines.append(f"   File: {test_file}")
            tooltip_lines.append(f"   Channel: {test_channel}")
            
            # Alignment info
            alignment_mode = pair_config.get('alignment_mode', 'index')
            tooltip_lines.append("")
            tooltip_lines.append(f"⚙️ Alignment: {alignment_mode.title()}-based")
            
            # Additional alignment details if available
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                n_points = aligned_data.get('n_points', 0)
                if n_points > 0:
                    tooltip_lines.append(f"📈 Data points: {n_points:,}")
                
                if alignment_mode == 'time':
                    time_range = aligned_data.get('time_range')
                    if time_range:
                        tooltip_lines.append(f"⏱️ Time range: {time_range[0]:.3f} to {time_range[1]:.3f}s")
                elif alignment_mode == 'index':
                    index_range = aligned_data.get('index_range')
                    if index_range:
                        tooltip_lines.append(f"📍 Index range: {index_range[0]} to {index_range[1]}")
            
            # Join all lines and set tooltip
            tooltip_text = "\n".join(tooltip_lines)
            name_item.setToolTip(tooltip_text)
            
        except Exception as e:
            print(f"[ComparisonWizard] Error setting pair name tooltip: {str(e)}")
            # Fallback simple tooltip
            if name_item:
                name_item.setToolTip(f"Pair: {pair_name}")
    
    def _get_bold_font(self):
        """Get a bold font for table items"""
        try:
            font = QFont()
            font.setBold(True)
            return font
        except:
            return None
    
    def _set_detailed_tooltips(self, row, statistics):
        """Set detailed tooltips for table items"""
        try:
            table = self.window.active_pair_table
            
            # Build comprehensive tooltip information
            tooltip_parts = []
            
            # Basic statistics
            r_val = statistics.get('r', np.nan)
            rms_val = statistics.get('rms', np.nan)
            n_val = statistics.get('n', 0)
            
            if not np.isnan(r_val):
                tooltip_parts.append(f"Correlation: {r_val:.4f}")
                if 'r_pvalue' in statistics:
                    tooltip_parts.append(f"p-value: {statistics['r_pvalue']:.2e}")
            
            if not np.isnan(rms_val):
                tooltip_parts.append(f"RMS difference: {rms_val:.4f}")
            
            tooltip_parts.append(f"Sample size: {n_val:,}")
            
            # Data quality info
            if 'valid_ratio' in statistics:
                tooltip_parts.append(f"Valid data: {statistics['valid_ratio']*100:.1f}%")
            
            # Additional statistics
            if 'mean_diff' in statistics:
                tooltip_parts.append(f"Mean difference: {statistics['mean_diff']:.4f}")
            if 'std_diff' in statistics:
                tooltip_parts.append(f"Std difference: {statistics['std_diff']:.4f}")
            if 'median_diff' in statistics:
                tooltip_parts.append(f"Median difference: {statistics['median_diff']:.4f}")
            if 'max_abs_diff' in statistics:
                tooltip_parts.append(f"Max |difference|: {statistics['max_abs_diff']:.4f}")
            
            # Reference and test data info
            if 'ref_mean' in statistics and 'ref_std' in statistics:
                tooltip_parts.append(f"Reference: μ={statistics['ref_mean']:.3f}, σ={statistics['ref_std']:.3f}")
            if 'test_mean' in statistics and 'test_std' in statistics:
                tooltip_parts.append(f"Test: μ={statistics['test_mean']:.3f}, σ={statistics['test_std']:.3f}")
            
            # Error information
            if 'error' in statistics:
                tooltip_parts.append(f"⚠️ Error: {statistics['error']}")
            
            # Create tooltip text
            tooltip_text = "\n".join(tooltip_parts)
            
            # Apply tooltip to all statistics columns
            for col in [2, 3, 4]:  # r, RMS, N columns
                item = table.item(row, col)
                if item:
                    item.setToolTip(tooltip_text)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error setting tooltips: {str(e)}")
        
    def _show_alignment_summary(self, pair_name, aligned_data, stats):
        """Show alignment summary and data quality information"""
        try:
            summary_parts = []
            warning_parts = []
            
            # Alignment method info
            method = aligned_data.get('alignment_method', 'unknown')
            n_points = aligned_data.get('n_points', 0)
            summary_parts.append(f"Alignment: {method} method, {n_points} data points")
            
            # Data quality info
            valid_ratio = aligned_data.get('valid_ratio', stats.get('valid_ratio', 1.0))
            if valid_ratio < 1.0:
                summary_parts.append(f"Data quality: {valid_ratio*100:.1f}% valid points")
                if valid_ratio < 0.8:
                    warning_parts.append(f"Low data quality ({valid_ratio*100:.1f}% valid)")
            
            # Time range info for time alignment
            if method == 'time':
                time_range = aligned_data.get('time_range')
                if time_range:
                    summary_parts.append(f"Time range: {time_range[0]:.3f} to {time_range[1]:.3f}")
                
                # Show round_to adjustments if any
                round_to_used = aligned_data.get('round_to_used')
                round_to_original = aligned_data.get('round_to_original')
                if round_to_used and round_to_original and round_to_used != round_to_original:
                    summary_parts.append(f"Time resolution adjusted: {round_to_original:.4f}s → {round_to_used:.4f}s")
                elif round_to_used:
                    summary_parts.append(f"Time resolution: {round_to_used:.4f}s")
            
            # Index range info for index alignment
            elif method == 'index':
                index_range = aligned_data.get('index_range')
                if index_range:
                    summary_parts.append(f"Index range: {index_range[0]} to {index_range[1]}")
                
                offset = aligned_data.get('offset_applied', 0)
                if offset != 0:
                    summary_parts.append(f"Offset applied: {offset} samples")
            
            # Statistical warnings
            if 'error' in stats:
                warning_parts.append(f"Statistics error: {stats['error']}")
            else:
                r_val = stats.get('r', np.nan)
                if np.isnan(r_val) and 'r_error' in stats:
                    warning_parts.append(f"Correlation: {stats['r_error']}")
                elif not np.isnan(r_val) and abs(r_val) < 0.1:
                    warning_parts.append(f"Low correlation (r = {r_val:.3f})")
                
                rms_val = stats.get('rms', np.nan)
                if not np.isnan(rms_val):
                    # Check if RMS is very large compared to data range
                    ref_std = stats.get('ref_std', np.nan)
                    test_std = stats.get('test_std', np.nan)
                    if not np.isnan(ref_std) and not np.isnan(test_std):
                        typical_scale = (ref_std + test_std) / 2
                        if typical_scale > 0 and rms_val > 2 * typical_scale:
                            warning_parts.append(f"Large RMS error ({rms_val:.3f}) relative to data scale")
            
            # Display summary in console or status
            summary_text = f"[{pair_name}] " + "; ".join(summary_parts)
            print(f"[ComparisonWizard] {summary_text}")
            
            # Show warnings if any
            if warning_parts and hasattr(self, 'window'):
                warning_text = f"Pair '{pair_name}' alignment completed with warnings:\n\n" + "\n".join(f"• {w}" for w in warning_parts)
                warning_text += f"\n\nSummary: {'; '.join(summary_parts)}"
                
                QMessageBox.information(self.window, "Alignment Summary", warning_text)
                
        except Exception as e:
            print(f"[ComparisonWizard] Error showing alignment summary: {str(e)}")
                
    def _generate_step1_preview(self, aligned_data):
        """Generate preview plot for step 1 with error handling - DEPRECATED"""
        # This method is now replaced by _generate_cumulative_preview
        # Keep for backward compatibility but redirect to cumulative display
        self._update_cumulative_display()
        
    def _update_step1_preview(self):
        """Update step 1 preview after pair deletion - DEPRECATED"""
        # This method is now replaced by _update_cumulative_display
        # Keep for backward compatibility but redirect to cumulative display
        self._update_cumulative_display()
        
    def _generate_plot(self, aligned_data, plot_config):
        """Generate the requested plot type in step 2"""
        if not hasattr(self.window, 'canvas') or not self.window.canvas:
            return
            
        fig = self.window.canvas.figure
        fig.clear()
        
        plot_type = plot_config['plot_type']
        
        if plot_type == 'bland_altman':
            self._generate_bland_altman_plot(fig, aligned_data, plot_config)
        elif plot_type == 'scatter':
            self._generate_scatter_plot(fig, aligned_data, plot_config)
        elif plot_type == 'residual':
            self._generate_residual_plot(fig, aligned_data, plot_config)
            
        try:
            fig.tight_layout()
        except (np.linalg.LinAlgError, ValueError) as e:
            print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
            try:
                fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
            except Exception:
                pass  # If both fail, continue without layout adjustment
        self.window.canvas.draw()
    
    def _create_density_plot(self, ax, x_data, y_data, density_type, bin_size, kde_bandwidth):
        """Create density-based visualization with proper coloring"""
        try:
            if density_type.lower() == 'hexbin':
                # Hexbin plot with colorbar
                hb = ax.hexbin(x_data, y_data, gridsize=bin_size, cmap='viridis', mincnt=1)
                # Add colorbar with density information
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('Point Density', rotation=270, labelpad=15)
                return hb
                
            elif density_type.lower() == 'kde':
                # KDE density plot
                if len(x_data) < 10:
                    # Fallback to scatter for insufficient data
                    return ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                
                # Create KDE
                try:
                    # Stack data for KDE
                    data_stack = np.vstack([x_data, y_data])
                    kde = gaussian_kde(data_stack, bw_method=kde_bandwidth)
                    
                    # Create grid for evaluation
                    x_min, x_max = np.min(x_data), np.max(x_data)
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    
                    # Add padding
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    x_min -= 0.1 * x_range
                    x_max += 0.1 * x_range
                    y_min -= 0.1 * y_range
                    y_max += 0.1 * y_range
                    
                    # Create evaluation grid (limit size for performance)
                    grid_size = min(100, int(np.sqrt(len(x_data))))
                    xi = np.linspace(x_min, x_max, grid_size)
                    yi = np.linspace(y_min, y_max, grid_size)
                    X, Y = np.meshgrid(xi, yi)
                    
                    # Evaluate KDE on grid
                    positions = np.vstack([X.ravel(), Y.ravel()])
                    Z = kde(positions).reshape(X.shape)
                    
                    # Create contour plot
                    cs = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
                    
                    # Add colorbar
                    cbar = plt.colorbar(cs, ax=ax)
                    cbar.set_label('Density', rotation=270, labelpad=15)
                    
                    # Overlay scatter points with transparency
                    ax.scatter(x_data, y_data, alpha=0.3, s=10, c='white', edgecolors='black', linewidth=0.5)
                    
                    return cs
                    
                except Exception as kde_error:
                    print(f"[ComparisonWizard] KDE failed: {kde_error}, falling back to scatter")
                    return ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                    
            else:
                # Regular scatter plot
                return ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
                
        except Exception as e:
            print(f"[ComparisonWizard] Density plot error: {e}, using scatter fallback")
            return ax.scatter(x_data, y_data, alpha=0.6, s=20, c='blue')
        
    def _generate_bland_altman_plot(self, fig, aligned_data, plot_config):
        """Generate Bland-Altman plot"""
        ax = fig.add_subplot(111)
        
        ref_data = aligned_data['ref_data']
        test_data = aligned_data['test_data']
        
        # Calculate means and differences
        means = (ref_data + test_data) / 2
        diffs = test_data - ref_data
        
        # Apply downsampling if requested
        if plot_config.get('downsample'):
            n_points = min(plot_config['downsample'], len(means))
            indices = np.random.choice(len(means), n_points, replace=False)
            means = means[indices]
            diffs = diffs[indices]
            
        # Create density plot
        density = plot_config.get('density_display', 'scatter')
        bin_size = plot_config.get('bin_size', 20)
        kde_bandwidth = plot_config.get('kde_bandwidth', 0.2)
        
        # Filter out invalid values for plotting
        valid_mask = np.isfinite(means) & np.isfinite(diffs)
        means_clean = means[valid_mask]
        diffs_clean = diffs[valid_mask]
        
        if len(means_clean) > 0:
            plot_obj = self._create_density_plot(ax, means_clean, diffs_clean, density, bin_size, kde_bandwidth)
            
        # Statistics
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        
        # Add lines
        ax.axhline(mean_diff, color='red', linestyle='-', label=f'Mean diff: {mean_diff:.3f}')
        
        if plot_config.get('confidence_interval'):
            ax.axhline(mean_diff + 1.96*std_diff, color='red', linestyle='--', 
                      label=f'+1.96 SD: {mean_diff + 1.96*std_diff:.3f}')
            ax.axhline(mean_diff - 1.96*std_diff, color='red', linestyle='--',
                      label=f'-1.96 SD: {mean_diff - 1.96*std_diff:.3f}')
                      
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', label=f'Custom: {custom_val}')
            except ValueError:
                pass
                
        ax.set_xlabel(f'Mean of {aligned_data["ref_label"]} and {aligned_data["test_label"]}')
        ax.set_ylabel(f'{aligned_data["test_label"]} - {aligned_data["ref_label"]}')
        ax.set_title('Bland-Altman Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _generate_scatter_plot(self, fig, aligned_data, plot_config):
        """Generate scatter plot"""
        ax = fig.add_subplot(111)
        
        ref_data = aligned_data['ref_data']
        test_data = aligned_data['test_data']
        
        # Apply downsampling if requested
        if plot_config.get('downsample'):
            n_points = min(plot_config['downsample'], len(ref_data))
            indices = np.random.choice(len(ref_data), n_points, replace=False)
            ref_data = ref_data[indices]
            test_data = test_data[indices]
            
        # Create density plot
        density = plot_config.get('density_display', 'scatter')
        bin_size = plot_config.get('bin_size', 20)
        kde_bandwidth = plot_config.get('kde_bandwidth', 0.2)
        
        # Filter out invalid values for plotting
        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
        ref_clean = ref_data[valid_mask]
        test_clean = test_data[valid_mask]
        
        if len(ref_clean) > 0:
            plot_obj = self._create_density_plot(ax, ref_clean, test_clean, density, bin_size, kde_bandwidth)
            
        # Add 1:1 line
        if len(ref_clean) > 0 and len(test_clean) > 0:
            min_val = min(np.min(ref_clean), np.min(test_clean))
            max_val = max(np.max(ref_clean), np.max(test_clean))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='1:1 line', linewidth=2)
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', label=f'y = {custom_val}')
            except ValueError:
                pass
                
        ax.set_xlabel(aligned_data['ref_label'])
        ax.set_ylabel(aligned_data['test_label'])
        ax.set_title('Scatter Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _generate_residual_plot(self, fig, aligned_data, plot_config):
        """Generate residual plot"""
        ax = fig.add_subplot(111)
        
        ref_data = aligned_data['ref_data']
        test_data = aligned_data['test_data']
        
        # Calculate residuals
        residuals = test_data - ref_data
        
        # Use reference data as x-axis for residual plot
        x_data = ref_data
        
        # Apply downsampling if requested
        if plot_config.get('downsample'):
            n_points = min(plot_config['downsample'], len(x_data))
            indices = np.random.choice(len(x_data), n_points, replace=False)
            x_data = x_data[indices]
            residuals = residuals[indices]
            
        # Create density plot
        density = plot_config.get('density_display', 'scatter')
        bin_size = plot_config.get('bin_size', 20)
        kde_bandwidth = plot_config.get('kde_bandwidth', 0.2)
        
        # Filter out invalid values for plotting
        valid_mask = np.isfinite(x_data) & np.isfinite(residuals)
        x_clean = x_data[valid_mask]
        residuals_clean = residuals[valid_mask]
        
        if len(x_clean) > 0:
            plot_obj = self._create_density_plot(ax, x_clean, residuals_clean, density, bin_size, kde_bandwidth)
            
        # Add zero line
        ax.axhline(0, color='red', linestyle='-', alpha=0.8, label='Zero residual')
        
        # Add custom line if specified
        if plot_config.get('custom_line'):
            try:
                custom_val = float(plot_config['custom_line'])
                ax.axhline(custom_val, color='green', linestyle=':', label=f'y = {custom_val}')
            except ValueError:
                pass
                
        ax.set_xlabel(aligned_data['ref_label'])
        ax.set_ylabel(f'Residuals ({aligned_data["test_label"]} - {aligned_data["ref_label"]})')
        ax.set_title('Residual Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _update_cumulative_display(self):
        """Update cumulative statistics and preview plot"""
        # Get checked pairs
        checked_pairs = self.window.get_checked_pairs()
        
        if not checked_pairs:
            # No pairs checked
            self.window.update_cumulative_stats("Cumulative Stats: No pairs selected")
            self._clear_preview_plot()
            return
        
        # Calculate cumulative statistics
        cumulative_stats = self._calculate_cumulative_statistics(checked_pairs)
        
        # Update cumulative stats display
        stats_text = self._format_cumulative_stats(cumulative_stats, len(checked_pairs))
        self.window.update_cumulative_stats(stats_text)
        
        # Generate plot based on comparison method (not just cumulative preview)
        plot_type = self._determine_plot_type_from_pairs(checked_pairs)
        plot_config = {
            'plot_type': plot_type,
            'show_grid': True,
            'show_legend': True,
            'checked_pairs': checked_pairs
        }
        
        # Use the same plot generation as the Generate Plot button
        self._generate_multi_pair_plot(checked_pairs, plot_config)

    def _update_step2_cumulative_display(self):
        """Update cumulative statistics display for step 2"""
        # Get checked pairs from step 2 table
        checked_pairs = self.window.get_step2_checked_pairs()
        
        if not checked_pairs:
            self.window.update_step2_cumulative_stats("No pairs selected for analysis")
            return
        
        # Get current plot configuration to determine plot type
        plot_config = self.window.get_plot_config()
        plot_type = plot_config.get('plot_type', 'scatter')
        
        # Calculate comprehensive cumulative statistics
        cumulative_stats = self._calculate_comprehensive_statistics(checked_pairs, plot_type)
        
        # Format and display comprehensive statistics
        stats_text = self._format_comprehensive_stats(cumulative_stats, len(checked_pairs), plot_type)
        self.window.update_step2_cumulative_stats(stats_text)
        
    def _calculate_comprehensive_statistics(self, checked_pairs, plot_type='scatter'):
        """Calculate comprehensive statistics for step 2 display"""
        if not checked_pairs:
            return {'error': 'No pairs selected'}
        
        # Collect all data from checked pairs
        all_ref_data = []
        all_test_data = []
        pair_stats = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                ref_data = aligned_data['ref_data']
                test_data = aligned_data['test_data']
                
                if ref_data is not None and test_data is not None:
                    # Filter valid data
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    valid_ref = ref_data[valid_mask]
                    valid_test = test_data[valid_mask]
                    
                    if len(valid_ref) > 0:
                        all_ref_data.extend(valid_ref)
                        all_test_data.extend(valid_test)
                        
                        # Store individual pair stats with more details
                        pair_stats.append({
                            'name': pair_name,
                            'n': len(valid_ref),
                            'ref_range': (np.min(valid_ref), np.max(valid_ref)),
                            'test_range': (np.min(valid_test), np.max(valid_test)),
                            'ref_mean': np.mean(valid_ref),
                            'test_mean': np.mean(valid_test),
                            'diff_mean': np.mean(valid_test - valid_ref),
                            'diff_std': np.std(valid_test - valid_ref),
                            'individual_r': scipy_stats.pearsonr(valid_ref, valid_test)[0] if len(valid_ref) > 2 else np.nan
                        })
        
        if not all_ref_data or not all_test_data:
            return {'error': 'No valid data points found'}
        
        # Convert to numpy arrays
        all_ref_data = np.array(all_ref_data)
        all_test_data = np.array(all_test_data)
        
        stats = {}
        stats['plot_type'] = plot_type
        
        # Basic information
        stats['n_total'] = len(all_ref_data)
        stats['n_pairs'] = len(checked_pairs)
        stats['pair_stats'] = pair_stats
        
        # Pearson correlation
        try:
            if np.var(all_ref_data) > 0 and np.var(all_test_data) > 0:
                r_value, p_value = scipy_stats.pearsonr(all_ref_data, all_test_data)
                r2_value = r_value ** 2
                
                # Correlation interpretation
                if abs(r_value) >= 0.9:
                    corr_strength = "very strong"
                elif abs(r_value) >= 0.7:
                    corr_strength = "strong"
                elif abs(r_value) >= 0.5:
                    corr_strength = "moderate"
                elif abs(r_value) >= 0.3:
                    corr_strength = "weak"
                else:
                    corr_strength = "very weak"
                
                stats['pearson'] = {
                    'r': r_value,
                    'r2': r2_value,
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                    'strength': corr_strength,
                    'direction': 'positive' if r_value > 0 else 'negative'
                }
            else:
                stats['pearson'] = {'error': 'Constant values - correlation undefined'}
        except Exception as e:
            stats['pearson'] = {'error': f'Correlation calculation failed: {str(e)}'}
        
        # Descriptive statistics with enhanced metrics
        try:
            differences = all_test_data - all_ref_data
            means = (all_ref_data + all_test_data) / 2
            percent_differences = (differences / all_ref_data) * 100 if np.all(all_ref_data != 0) else np.full_like(differences, np.nan)
            
            # Calculate coefficient of variation
            ref_cv = (np.std(all_ref_data) / np.mean(all_ref_data)) * 100 if np.mean(all_ref_data) != 0 else np.nan
            test_cv = (np.std(all_test_data) / np.mean(all_test_data)) * 100 if np.mean(all_test_data) != 0 else np.nan
            
            # Calculate quartiles and percentiles
            ref_q25, ref_q50, ref_q75 = np.percentile(all_ref_data, [25, 50, 75])
            test_q25, test_q50, test_q75 = np.percentile(all_test_data, [25, 50, 75])
            diff_q25, diff_q50, diff_q75 = np.percentile(differences, [25, 50, 75])
            
            stats['descriptive'] = {
                'ref_mean': np.mean(all_ref_data),
                'ref_std': np.std(all_ref_data),
                'ref_cv': ref_cv,
                'ref_range': (np.min(all_ref_data), np.max(all_ref_data)),
                'ref_quartiles': (ref_q25, ref_q50, ref_q75),
                'ref_iqr': ref_q75 - ref_q25,
                'test_mean': np.mean(all_test_data),
                'test_std': np.std(all_test_data),
                'test_cv': test_cv,
                'test_range': (np.min(all_test_data), np.max(all_test_data)),
                'test_quartiles': (test_q25, test_q50, test_q75),  
                'test_iqr': test_q75 - test_q25,
                'diff_mean': np.mean(differences),
                'diff_std': np.std(differences),
                'diff_range': (np.min(differences), np.max(differences)),
                'diff_quartiles': (diff_q25, diff_q50, diff_q75),
                'diff_iqr': diff_q75 - diff_q25,
                'percent_diff_mean': np.nanmean(percent_differences),
                'percent_diff_std': np.nanstd(percent_differences),
                'data_spread_ratio': np.std(all_test_data) / np.std(all_ref_data) if np.std(all_ref_data) != 0 else np.nan
            }
        except Exception as e:
            stats['descriptive'] = {'error': f'Descriptive statistics failed: {str(e)}'}
        
        # Bland-Altman statistics with enhanced analysis
        try:
            differences = all_test_data - all_ref_data
            means = (all_ref_data + all_test_data) / 2
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            
            # Limits of agreement (95%)
            loa_upper = mean_diff + 1.96 * std_diff
            loa_lower = mean_diff - 1.96 * std_diff
            
            # Percentage of points within LoA
            within_loa = np.sum((differences >= loa_lower) & (differences <= loa_upper))
            percent_within_loa = (within_loa / len(differences)) * 100
            
            # Check for proportional bias (correlation between differences and means)
            prop_bias_r, prop_bias_p = scipy_stats.pearsonr(means, differences)
            has_proportional_bias = abs(prop_bias_r) > 0.1 and prop_bias_p < 0.05
            
            # Calculate relative LoA (as percentage of mean)
            overall_mean = np.mean(means)
            relative_loa_width = (loa_upper - loa_lower) / overall_mean * 100 if overall_mean != 0 else np.nan
            
            stats['bland_altman'] = {
                'mean_bias': mean_diff,
                'std_bias': std_diff,
                'loa_upper': loa_upper,
                'loa_lower': loa_lower,
                'percent_within_loa': percent_within_loa,
                'loa_width': loa_upper - loa_lower,
                'relative_loa_width': relative_loa_width,
                'proportional_bias_r': prop_bias_r,
                'proportional_bias_p': prop_bias_p,
                'has_proportional_bias': has_proportional_bias,
                'bias_relative_to_mean': (mean_diff / overall_mean * 100) if overall_mean != 0 else np.nan
            }
        except Exception as e:
            stats['bland_altman'] = {'error': f'Bland-Altman analysis failed: {str(e)}'}
        
        # Enhanced error metrics with plot-type specific calculations
        try:
            differences = all_test_data - all_ref_data
            abs_differences = np.abs(differences)
            squared_differences = differences ** 2
            
            # Basic error metrics
            mae = np.mean(abs_differences)
            rmse = np.sqrt(np.mean(squared_differences))
            mse = np.mean(squared_differences)
            
            # Percentage-based metrics
            mape = np.mean(np.abs(differences / all_ref_data)) * 100 if np.all(all_ref_data != 0) else np.nan
            smape = np.mean(2 * abs_differences / (np.abs(all_ref_data) + np.abs(all_test_data))) * 100
            
            # Normalized metrics
            nrmse_range = rmse / (np.max(all_ref_data) - np.min(all_ref_data)) if (np.max(all_ref_data) - np.min(all_ref_data)) != 0 else np.nan
            nrmse_mean = rmse / np.mean(all_ref_data) if np.mean(all_ref_data) != 0 else np.nan
            
            # Additional useful metrics
            mard = np.median(abs_differences)  # Median Absolute Relative Difference
            
            # Concordance correlation coefficient (CCC)
            ref_mean = np.mean(all_ref_data)
            test_mean = np.mean(all_test_data)
            ref_var = np.var(all_ref_data)
            test_var = np.var(all_test_data)
            covariance = np.cov(all_ref_data, all_test_data)[0, 1]
            
            ccc = (2 * covariance) / (ref_var + test_var + (ref_mean - test_mean)**2) if (ref_var + test_var + (ref_mean - test_mean)**2) != 0 else np.nan
            
            stats['error_metrics'] = {
                'mae': mae,
                'rmse': rmse,
                'mse': mse,
                'mape': mape,
                'smape': smape,
                'nrmse_range': nrmse_range,
                'nrmse_mean': nrmse_mean,
                'max_abs_error': np.max(abs_differences),
                'median_abs_error': np.median(abs_differences),
                'mard': mard,
                'ccc': ccc,
                'q95_abs_error': np.percentile(abs_differences, 95),
                'q99_abs_error': np.percentile(abs_differences, 99)
            }
        except Exception as e:
            stats['error_metrics'] = {'error': f'Error metrics calculation failed: {str(e)}'}
        
        # Plot-type specific metrics
        if plot_type == 'pearson':
            # Additional correlation analysis
            try:
                # Spearman correlation (rank-based)
                spearman_r, spearman_p = scipy_stats.spearmanr(all_ref_data, all_test_data)
                
                # Kendall's tau
                kendall_tau, kendall_p = scipy_stats.kendalltau(all_ref_data, all_test_data)
                
                stats['correlation_analysis'] = {
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'kendall_tau': kendall_tau,
                    'kendall_p': kendall_p,
                    'linearity_check': abs(stats['pearson']['r'] - spearman_r) < 0.1 if 'pearson' in stats and 'r' in stats['pearson'] else False
                }
            except Exception as e:
                stats['correlation_analysis'] = {'error': f'Additional correlation analysis failed: {str(e)}'}
        
        elif plot_type == 'residual':
            # Residual analysis
            try:
                residuals = all_test_data - all_ref_data
                
                # Test for heteroscedasticity (Breusch-Pagan test approximation)
                means = (all_ref_data + all_test_data) / 2
                residuals_squared = residuals ** 2
                bp_r, bp_p = scipy_stats.pearsonr(means, residuals_squared)
                
                # Durbin-Watson test for autocorrelation (simplified)
                diff_residuals = np.diff(residuals)
                dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2) if np.sum(residuals**2) != 0 else np.nan
                
                # Normality test on residuals
                try:
                    shapiro_stat, shapiro_p = scipy_stats.shapiro(residuals[:5000])  # Limit for performance
                except:
                    shapiro_stat, shapiro_p = np.nan, np.nan
                
                stats['residual_analysis'] = {
                    'heteroscedasticity_r': bp_r,
                    'heteroscedasticity_p': bp_p,
                    'has_heteroscedasticity': abs(bp_r) > 0.1 and bp_p < 0.05,
                    'durbin_watson': dw_stat,
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p': shapiro_p,
                    'residuals_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
                }
            except Exception as e:
                stats['residual_analysis'] = {'error': f'Residual analysis failed: {str(e)}'}
        
        return stats
        
    def _format_comprehensive_stats(self, stats, n_pairs, plot_type='scatter'):
        """Format comprehensive statistics for display in step 2"""
        if 'error' in stats:
            return f"Error: {stats['error']}"
        
        lines = []
        
        # Header with plot type
        plot_type_names = {
            'scatter': 'SCATTER PLOT',
            'bland_altman': 'BLAND-ALTMAN',
            'residual': 'RESIDUAL',
            'pearson': 'PEARSON CORRELATION'
        }
        plot_name = plot_type_names.get(plot_type, plot_type.upper())
        
        lines.append(f"{plot_name} ANALYSIS - {n_pairs} pairs, {stats['n_total']:,} total points")
        lines.append("=" * 70)
        
        # Pearson Correlation with enhanced information
        lines.append("CORRELATION ANALYSIS:")
        if 'error' in stats.get('pearson', {}):
            lines.append(f"  Error: {stats['pearson']['error']}")
        else:
            p = stats['pearson']
            sig_text = " ✓ significant" if p.get('significant', False) else " ✗ not significant"
            strength = p.get('strength', 'unknown')
            direction = p.get('direction', 'unknown')
            
            lines.append(f"  Pearson r = {p['r']:7.4f} ({strength} {direction} correlation)")
            lines.append(f"  R²        = {p['r2']:7.4f} ({p['r2']*100:5.1f}% of variance explained)")
            lines.append(f"  p-value   = {p['p_value']:.2e}{sig_text}")
            
            # Add correlation interpretation
            if not np.isnan(p['r']):
                if abs(p['r']) >= 0.7:
                    lines.append(f"  💡 Strong correlation indicates good agreement between datasets")
                elif abs(p['r']) >= 0.3:
                    lines.append(f"  ⚠️  Moderate correlation - consider examining data relationships")
                else:
                    lines.append(f"  ⚠️  Weak correlation - datasets may have different patterns")
        
        # Plot-type specific correlation analysis
        if plot_type == 'pearson' and 'correlation_analysis' in stats:
            if 'error' not in stats['correlation_analysis']:
                ca = stats['correlation_analysis']
                lines.append(f"  Spearman ρ = {ca['spearman_r']:7.4f} (rank correlation)")
                lines.append(f"  Kendall τ  = {ca['kendall_tau']:7.4f} (concordance)")
                linearity = "✓ linear" if ca.get('linearity_check', False) else "✗ potentially non-linear"
                lines.append(f"  Linearity: {linearity} relationship")
        
        # Enhanced Descriptive Statistics
        lines.append("\nDESCRIPTIVE STATISTICS:")
        if 'error' in stats.get('descriptive', {}):
            lines.append(f"  Error: {stats['descriptive']['error']}")
        else:
            d = stats['descriptive']
            
            # Reference data statistics
            lines.append(f"  Reference Dataset:")
            lines.append(f"    Mean ± SD:     {d['ref_mean']:8.3f} ± {d['ref_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['ref_quartiles'][1]:8.3f} ({d['ref_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['ref_range'][0]:8.3f}, {d['ref_range'][1]:8.3f}]")
            if not np.isnan(d['ref_cv']):
                lines.append(f"    CV:            {d['ref_cv']:8.1f}% (coefficient of variation)")
            
            # Test data statistics  
            lines.append(f"  Test Dataset:")
            lines.append(f"    Mean ± SD:     {d['test_mean']:8.3f} ± {d['test_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['test_quartiles'][1]:8.3f} ({d['test_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['test_range'][0]:8.3f}, {d['test_range'][1]:8.3f}]")
            if not np.isnan(d['test_cv']):
                lines.append(f"    CV:            {d['test_cv']:8.1f}% (coefficient of variation)")
            
            # Difference statistics
            lines.append(f"  Differences (Test - Reference):")
            lines.append(f"    Mean ± SD:     {d['diff_mean']:8.3f} ± {d['diff_std']:8.3f}")
            lines.append(f"    Median (IQR):  {d['diff_quartiles'][1]:8.3f} ({d['diff_iqr']:8.3f})")
            lines.append(f"    Range:         [{d['diff_range'][0]:8.3f}, {d['diff_range'][1]:8.3f}]")
            
            if not np.isnan(d.get('percent_diff_mean', np.nan)):
                lines.append(f"    % Difference:  {d['percent_diff_mean']:8.1f}% ± {d['percent_diff_std']:8.1f}%")
            
            # Data comparison insights
            if not np.isnan(d.get('data_spread_ratio', np.nan)):
                spread_ratio = d['data_spread_ratio']
                if spread_ratio > 1.2:
                    lines.append(f"  💡 Test data has {spread_ratio:.1f}x more variability than reference")
                elif spread_ratio < 0.8:
                    lines.append(f"  💡 Test data has {1/spread_ratio:.1f}x less variability than reference")
                else:
                    lines.append(f"  💡 Similar variability between datasets (ratio: {spread_ratio:.2f})")
        
        # Bland-Altman Analysis (enhanced for bland_altman plot type)
        if plot_type == 'bland_altman' or 'bland_altman' in stats:
            lines.append("\nBLAND-ALTMAN ANALYSIS:")
            if 'error' in stats.get('bland_altman', {}):
                lines.append(f"  Error: {stats['bland_altman']['error']}")
            else:
                ba = stats['bland_altman']
                lines.append(f"  Mean Bias:        {ba['mean_bias']:8.3f} ± {ba['std_bias']:8.3f}")
                
                if not np.isnan(ba.get('bias_relative_to_mean', np.nan)):
                    lines.append(f"  Relative Bias:    {ba['bias_relative_to_mean']:8.1f}% of mean")
                
                lines.append(f"  95% Limits of Agreement:")
                lines.append(f"    Upper LoA:      {ba['loa_upper']:8.3f}")
                lines.append(f"    Lower LoA:      {ba['loa_lower']:8.3f}")
                lines.append(f"    LoA Width:      {ba['loa_width']:8.3f}")
                
                if not np.isnan(ba.get('relative_loa_width', np.nan)):
                    lines.append(f"    Relative Width: {ba['relative_loa_width']:8.1f}% of mean")
                
                lines.append(f"  Within LoA:       {ba['percent_within_loa']:5.1f}% of points")
                
                # Proportional bias assessment
                if ba.get('has_proportional_bias', False):
                    lines.append(f"  ⚠️  Proportional bias detected (r={ba['proportional_bias_r']:.3f}, p={ba['proportional_bias_p']:.3f})")
                    lines.append(f"      Bias increases/decreases with measurement magnitude")
                else:
                    lines.append(f"  ✓ No significant proportional bias detected")
                
                # Clinical interpretation
                if ba['percent_within_loa'] >= 95:
                    lines.append(f"  💡 Excellent agreement - {ba['percent_within_loa']:.1f}% within expected range")
                elif ba['percent_within_loa'] >= 90:
                    lines.append(f"  💡 Good agreement - most points within limits")
                else:
                    lines.append(f"  ⚠️  Poor agreement - many outliers beyond limits")
        
        # Enhanced Error Metrics
        lines.append("\nERROR METRICS:")
        if 'error' in stats.get('error_metrics', {}):
            lines.append(f"  Error: {stats['error_metrics']['error']}")
        else:
            em = stats['error_metrics']
            
            # Basic error metrics
            lines.append(f"  Absolute Errors:")
            lines.append(f"    MAE (Mean):           {em['mae']:8.3f}")
            lines.append(f"    Median AE:            {em['median_abs_error']:8.3f}")
            lines.append(f"    Max AE:               {em['max_abs_error']:8.3f}")
            lines.append(f"    95th percentile AE:   {em['q95_abs_error']:8.3f}")
            lines.append(f"    99th percentile AE:   {em['q99_abs_error']:8.3f}")
            
            # Squared error metrics
            lines.append(f"  Squared Errors:")
            lines.append(f"    MSE (Mean Squared):   {em['mse']:8.3f}")
            lines.append(f"    RMSE (Root MSE):      {em['rmse']:8.3f}")
            
            # Normalized metrics
            if not np.isnan(em.get('nrmse_range', np.nan)):
                lines.append(f"  Normalized RMSE:")
                lines.append(f"    By range:             {em['nrmse_range']:8.3f} ({em['nrmse_range']*100:5.1f}%)")
            if not np.isnan(em.get('nrmse_mean', np.nan)):
                lines.append(f"    By mean:              {em['nrmse_mean']:8.3f} ({em['nrmse_mean']*100:5.1f}%)")
            
            # Percentage-based metrics
            if not np.isnan(em.get('mape', np.nan)):
                lines.append(f"  Percentage Errors:")
                lines.append(f"    MAPE (Mean Abs %):    {em['mape']:8.1f}%")
                lines.append(f"    SMAPE (Symmetric):    {em['smape']:8.1f}%")
            
            # Concordance correlation coefficient
            if not np.isnan(em.get('ccc', np.nan)):
                ccc = em['ccc']
                lines.append(f"  Concordance Corr Coef: {ccc:8.4f}")
                if ccc > 0.99:
                    lines.append(f"    💡 Almost perfect agreement")
                elif ccc > 0.95:
                    lines.append(f"    💡 Substantial agreement") 
                elif ccc > 0.90:
                    lines.append(f"    💡 Moderate agreement")
                else:
                    lines.append(f"    ⚠️  Poor agreement")
        
        # Residual Analysis (for residual plot type)
        if plot_type == 'residual' and 'residual_analysis' in stats:
            lines.append("\nRESIDUAL ANALYSIS:")
            if 'error' in stats['residual_analysis']:
                lines.append(f"  Error: {stats['residual_analysis']['error']}")
            else:
                ra = stats['residual_analysis']
                
                # Heteroscedasticity test
                if ra.get('has_heteroscedasticity', False):
                    lines.append(f"  ⚠️  Heteroscedasticity detected (r={ra['heteroscedasticity_r']:.3f})")
                    lines.append(f"      Error variance changes with measurement level")
                else:
                    lines.append(f"  ✓ Homoscedasticity - consistent error variance")
                
                # Durbin-Watson for autocorrelation
                if not np.isnan(ra.get('durbin_watson', np.nan)):
                    dw = ra['durbin_watson']
                    lines.append(f"  Durbin-Watson: {dw:.3f}")
                    if dw < 1.5:
                        lines.append(f"    ⚠️  Positive autocorrelation in residuals")
                    elif dw > 2.5:
                        lines.append(f"    ⚠️  Negative autocorrelation in residuals")
                    else:
                        lines.append(f"    ✓ No significant autocorrelation")
                
                # Normality test
                if ra.get('residuals_normal') is not None:
                    if ra['residuals_normal']:
                        lines.append(f"  ✓ Residuals are approximately normal (Shapiro p={ra['shapiro_p']:.3f})")
                    else:
                        lines.append(f"  ⚠️  Residuals deviate from normality (Shapiro p={ra['shapiro_p']:.3f})")
        
        # Individual Pair Summary with enhanced details
        lines.append(f"\nINDIVIDUAL PAIRS SUMMARY:")
        for pair_stat in stats.get('pair_stats', []):
            name = pair_stat['name']
            n = pair_stat['n']
            r = pair_stat.get('individual_r', np.nan)
            diff_mean = pair_stat.get('diff_mean', np.nan)
            diff_std = pair_stat.get('diff_std', np.nan)
            
            if not np.isnan(r):
                lines.append(f"  {name}: {n:,} pts, r={r:.3f}, bias={diff_mean:.3f}±{diff_std:.3f}")
            else:
                lines.append(f"  {name}: {n:,} pts, bias={diff_mean:.3f}±{diff_std:.3f}")
        
        # Add plot-specific recommendations
        lines.append(f"\nRECOMMendations for {plot_name}:")
        if plot_type == 'scatter':
            lines.append("  • Points near diagonal line indicate good agreement")
            lines.append("  • Scatter pattern shows precision; bias shown by offset from line")
            lines.append("  • Look for outliers and non-linear patterns")
        elif plot_type == 'bland_altman':
            lines.append("  • Points within LoA indicate acceptable agreement")
            lines.append("  • Horizontal pattern indicates consistent bias")
            lines.append("  • Fan-shaped pattern suggests proportional bias")
        elif plot_type == 'residual':
            lines.append("  • Random scatter around zero indicates good model fit")
            lines.append("  • Patterns indicate systematic errors or non-linearity")
            lines.append("  • Increasing spread suggests heteroscedasticity")
        elif plot_type == 'pearson':
            lines.append("  • r > 0.7 indicates strong linear relationship")
            lines.append("  • R² shows proportion of variance explained")
            lines.append("  • Compare Pearson vs Spearman for linearity assessment")
        
        return "\n".join(lines)

    def _generate_cumulative_preview(self, checked_pairs):
        """Generate preview plot with multiple pairs in different colors"""
        try:
            if not hasattr(self.window, 'canvas') or not self.window.canvas:
                return
                
            fig = self.window.canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            
            if not checked_pairs:
                ax.text(0.5, 0.5, 'No pairs selected for preview', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Data Alignment Preview')
                fig.tight_layout()
                self.window.canvas.draw()
                return
            
            # Color cycle for different pairs
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
            
            all_ref_data = []
            all_test_data = []
            pair_count = 0
            
            for i, pair in enumerate(checked_pairs):
                pair_name = pair['name']
                if pair_name not in self.pair_aligned_data:
                    continue
                    
                aligned_data = self.pair_aligned_data[pair_name]
                ref_data = aligned_data['ref_data']
                test_data = aligned_data['test_data']
                
                if ref_data is None or test_data is None or len(ref_data) == 0:
                    continue
                
                # Filter valid data for plotting
                valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                ref_plot = ref_data[valid_mask]
                test_plot = test_data[valid_mask]
                
                if len(ref_plot) == 0:
                    continue
                
                # Downsample for performance if too many points
                if len(ref_plot) > 2000:
                    indices = np.random.choice(len(ref_plot), 2000, replace=False)
                    ref_plot = ref_plot[indices]
                    test_plot = test_plot[indices]
                
                # Plot this pair
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                ax.scatter(ref_plot, test_plot, 
                          alpha=0.6, s=20, 
                          color=color, marker=marker,
                          label=f"{pair_name} (n={len(ref_plot)})")
                
                # Collect for overall statistics
                all_ref_data.extend(ref_plot)
                all_test_data.extend(test_plot)
                pair_count += 1
            
            if not all_ref_data:
                ax.text(0.5, 0.5, 'No valid data for preview', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Data Alignment Preview - No Valid Data')
                fig.tight_layout()
                self.window.canvas.draw()
                return
            
            # Convert to arrays for statistics
            all_ref_data = np.array(all_ref_data)
            all_test_data = np.array(all_test_data)
            
            # Add 1:1 line
            min_val = min(np.min(all_ref_data), np.min(all_test_data))
            max_val = max(np.max(all_ref_data), np.max(all_test_data))
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 line')
            
            # Set labels (use first pair's labels as representative)
            first_pair_name = checked_pairs[0]['name']
            if first_pair_name in self.pair_aligned_data:
                first_aligned = self.pair_aligned_data[first_pair_name]
                ax.set_xlabel(first_aligned.get('ref_label', 'Reference'))
                ax.set_ylabel(first_aligned.get('test_label', 'Test'))
            
            # Add title with summary
            try:
                r_value, _ = scipy_stats.pearsonr(all_ref_data, all_test_data)
                r_str = f"r = {r_value:.3f}" if not np.isnan(r_value) else "r = N/A"
            except:
                r_str = "r = N/A"
            
            title = f'Cumulative Preview: {pair_count} pairs, {len(all_ref_data)} pts, {r_str}'
            ax.set_title(title)
            
            # Legend and grid
            if pair_count <= 10:  # Only show legend if not too many pairs
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            try:
                fig.tight_layout()
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"[ComparisonWizard] tight_layout failed: {e}, using subplots_adjust fallback")
                try:
                    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
                except Exception:
                    pass  # If both fail, continue without layout adjustment
            self.window.canvas.draw()
            
        except Exception as e:
            print(f"[ComparisonWizard] Cumulative preview error: {str(e)}")
            try:
                # Try to show error message on plot
                if hasattr(self.window, 'canvas') and self.window.canvas:
                    fig = self.window.canvas.figure
                    fig.clear()
                    ax = fig.add_subplot(111)
                    ax.text(0.5, 0.5, f'Preview error:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Data Alignment Preview - Error')
                    try:
                        fig.tight_layout()
                    except (np.linalg.LinAlgError, ValueError) as layout_error:
                        print(f"[ComparisonWizard] Error plot tight_layout failed: {layout_error}")
                        try:
                            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9)
                        except Exception:
                            pass  # If both fail, continue without layout adjustment
                    self.window.canvas.draw()
            except:
                pass



    def _calculate_cumulative_statistics(self, checked_pairs):
        """Calculate cumulative statistics for checked pairs"""
        if not checked_pairs:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': 0}
        
        # Collect all data from checked pairs
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                ref_data = aligned_data['ref_data']
                test_data = aligned_data['test_data']
                
                if ref_data is not None and test_data is not None:
                    # Filter valid data
                    valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
                    all_ref_data.extend(ref_data[valid_mask])
                    all_test_data.extend(test_data[valid_mask])
        
        if not all_ref_data or not all_test_data:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': len(checked_pairs)}
        
        # Convert to numpy arrays
        all_ref_data = np.array(all_ref_data)
        all_test_data = np.array(all_test_data)
        
        # Calculate cumulative statistics
        n_points = len(all_ref_data)
        
        # Correlation
        try:
            if np.var(all_ref_data) > 0 and np.var(all_test_data) > 0:
                r_value, p_value = scipy_stats.pearsonr(all_ref_data, all_test_data)
            else:
                r_value = np.nan
        except:
            r_value = np.nan
        
        # RMS
        try:
            differences = all_test_data - all_ref_data
            rms_value = np.sqrt(np.mean(differences ** 2))
        except:
            rms_value = np.nan
        
        return {
            'r': r_value,
            'rms': rms_value,
            'n': n_points,
            'pairs': len(checked_pairs)
        }
        
    def _format_cumulative_stats(self, stats, n_pairs):
        """Format cumulative statistics for display"""
        r_str = f"{stats['r']:.3f}" if not np.isnan(stats['r']) else "N/A"
        rms_str = f"{stats['rms']:.3f}" if not np.isnan(stats['rms']) else "N/A"
        
        return f"Cumulative Stats: r = {r_str}, RMS = {rms_str}, N = {stats['n']:,} points ({n_pairs} pairs shown)"