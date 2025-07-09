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
    from comparison.comparison_registry import ComparisonRegistry
    from comparison import load_all_comparisons
    from comparison.base_comparison import BaseComparison
    COMPARISON_AVAILABLE = True
    print("[ComparisonWizardManager] Comparison registry imported successfully")
except ImportError as e:
    print(f"[ComparisonWizardManager] Warning: Could not import comparison registry: {e}")
    COMPARISON_AVAILABLE = False
    
    # Create dummy classes if comparison module is not available
    class ComparisonRegistry:
        @staticmethod
        def get_all_methods():
            return ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis"]
        
        @staticmethod
        def get_all_categories():
            return ["Statistical", "Agreement", "Error Analysis"]
        
        @staticmethod
        def get_methods_by_category(category):
            if category == "Statistical":
                return ["Correlation Analysis"]
            elif category == "Agreement":
                return ["Bland-Altman Analysis"]
            elif category == "Error Analysis":
                return ["Residual Analysis"]
            return []
        
        @staticmethod
        def get_method_info(method_name):
            return {
                'name': method_name,
                'description': f'Description for {method_name}',
                'parameters': {},
                'category': 'Statistical'
            }
        
        @staticmethod
        def create_method(method_name, **kwargs):
            return None
    
    def load_all_comparisons(directory=None):
        print(f"[ComparisonWizardManager] Warning: Comparison module not available")
        return False
    
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
        
        # Connect signals first
        self._connect_signals()
        
        # Refresh window with comparison registry data after everything is set up
        try:
            self.window._refresh_comparison_data()
        except Exception as e:
            print(f"[ComparisonWizardManager] Warning: Could not refresh comparison data: {e}")
            # This is not critical, the window will use static controls
        
        # Log initialization
        self._log_state_change("Manager initialized successfully")
    
    def _initialize_comparison_methods(self):
        """Initialize comparison methods from the comparison folder"""
        try:
            if COMPARISON_AVAILABLE:
                # Set the comparison registry reference
                self.comparison_registry = ComparisonRegistry
                
                # Load all comparison methods from the comparison folder
                load_all_comparisons()
                
                # Log loaded methods
                methods = self.comparison_registry.all_comparisons()
                print(f"[ComparisonWizardManager] Using {len(methods)} comparison methods")
                print(f"  Methods: {', '.join(methods)}")
                
                self._log_state_change("Comparison methods loaded successfully")
            else:
                self._log_state_change("Comparison methods not available - using basic calculations only")
        except Exception as e:
            print(f"[ComparisonWizardManager] Warning: Could not load comparison methods: {e}")
            import traceback
            traceback.print_exc()
            self._log_state_change("Failed to load comparison methods - using fallback")
    

    
    def get_available_comparison_methods(self):
        """Get list of available comparison methods"""
        if COMPARISON_AVAILABLE:
            try:
                # Get registry names and convert to display names using the comparison classes
                registry_names = self.comparison_registry.all_comparisons()
                display_names = []
                for name in registry_names:
                    try:
                        comparison_cls = self.comparison_registry.get(name)
                        if comparison_cls:
                            # Generate clean display name
                            display_name = self._generate_clean_display_name(comparison_cls, name)
                            display_names.append(display_name)
                        else:
                            display_names.append(name.replace('_', ' ').title() + ' Analysis')
                    except Exception as e:
                        print(f"[ComparisonWizardManager] Error getting display name for {name}: {e}")
                        display_names.append(name.replace('_', ' ').title() + ' Analysis')
                return display_names
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting comparison methods: {e}")
        return ["Correlation Analysis", "Bland-Altman Analysis", "Residual Analysis"]
    
    def get_comparison_categories(self):
        """Get list of comparison method categories"""
        if COMPARISON_AVAILABLE:
            try:
                # Get categories from all registered comparison methods
                categories = set()
                for method_name in self.comparison_registry.all_comparisons():
                    try:
                        comparison_cls = self.comparison_registry.get(method_name)
                        if comparison_cls:
                            categories.add(comparison_cls.category)
                    except:
                        pass
                return list(categories)
            except:
                pass
        return ["Statistical", "Agreement", "Error Analysis"]
    
    def get_methods_by_category(self, category):
        """Get comparison methods in a specific category"""
        if COMPARISON_AVAILABLE:
            try:
                # Get methods by category from all registered comparison methods
                display_names = []
                for method_name in self.comparison_registry.all_comparisons():
                    try:
                        comparison_cls = self.comparison_registry.get(method_name)
                        if comparison_cls and comparison_cls.category == category:
                            # Generate clean display name
                            display_name = self._generate_clean_display_name(comparison_cls, method_name)
                            display_names.append(display_name)
                    except Exception as e:
                        print(f"[ComparisonWizardManager] Error getting method for category {category}: {e}")
                        pass
                return display_names
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting methods by category: {e}")
                pass
        # Fallback
        if category == "Statistical":
            return ["Correlation Analysis"]
        elif category == "Agreement":
            return ["Bland-Altman Analysis"]
        elif category == "Error Analysis":
            return ["Residual Analysis"]
        return []
    
    def get_method_info(self, method_name_or_display):
        """Get detailed information about a comparison method"""
        if COMPARISON_AVAILABLE:
            try:
                # Convert display name to registry name if needed
                registry_name = self.get_registry_name_from_display(method_name_or_display)
                
                # Try to get method information from the comparison registry
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    return comparison_cls().get_info()
                else:
                    print(f"[ComparisonWizardManager] Method {registry_name} not found in registry")
            except Exception as e:
                print(f"[ComparisonWizardManager] Error getting method info for {method_name_or_display}: {e}")
        
        # Fallback
        return {
            'name': method_name_or_display,
            'description': f'Description for {method_name_or_display}',
            'parameters': {},
            'category': 'Statistical'
        }
    
    def get_registry_name_from_display(self, display_name):
        """Convert display name to registry name"""
        if COMPARISON_AVAILABLE:
            try:
                # Check each registered method to find matching display name
                for registry_name in self.comparison_registry.all_comparisons():
                    comparison_cls = self.comparison_registry.get(registry_name)
                    if comparison_cls:
                        # Generate clean display name and check if it matches
                        method_display_name = self._generate_clean_display_name(comparison_cls, registry_name)
                        
                        if method_display_name == display_name:
                            return registry_name
                            
                # Fallback: convert display name to registry name
                return display_name.lower().replace(' analysis', '').replace(' ', '_')
            except Exception as e:
                print(f"[ComparisonWizardManager] Error converting display name {display_name}: {e}")
                return display_name.lower().replace(' analysis', '').replace(' ', '_')
        
        # Fallback conversion
        return display_name.lower().replace(' analysis', '').replace(' ', '_')
        
    def _generate_clean_display_name(self, comparison_cls, registry_name):
        """Generate a clean, user-friendly display name from comparison class"""
        if hasattr(comparison_cls, 'description') and comparison_cls.description:
            description = comparison_cls.description
            
            # If description contains ' - ', take the first part
            if ' - ' in description:
                return description.split(' - ')[0]
            
            # If description contains 'analysis' or 'comparison', extract the key part
            desc_lower = description.lower()
            
            # Common patterns to extract clean names
            if desc_lower.startswith('histogram analysis'):
                return 'Error Distribution Histogram'
            elif desc_lower.startswith('time series analysis'):
                return 'Relative Error Time Series'
            elif desc_lower.startswith('cross-correlation analysis'):
                return 'Time Lag Cross-Correlation'
            elif 'correlation' in desc_lower and 'coefficients' in desc_lower:
                return 'Correlation Analysis'
            elif 'bland-altman' in desc_lower or 'bland altman' in desc_lower:
                return 'Bland-Altman Analysis'
            elif 'residual' in desc_lower and 'analysis' in desc_lower:
                return 'Residual Analysis'
            else:
                # Try to extract first few meaningful words
                words = description.split()
                if len(words) >= 2:
                    # Take first 2-3 significant words and add 'Analysis'
                    significant_words = [w for w in words[:3] if len(w) > 2 and w.lower() not in ['of', 'the', 'and', 'with', 'for']]
                    if significant_words:
                        return ' '.join(significant_words[:2]).title() + ' Analysis'
        
        # Fallback: convert registry name to title case
        return registry_name.replace('_', ' ').title() + ' Analysis'
        
    def get_display_name_from_registry(self, registry_name):
        """Convert registry name to display name"""
        if COMPARISON_AVAILABLE:
            try:
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    return self._generate_clean_display_name(comparison_cls, registry_name)
            except Exception as e:
                print(f"[ComparisonWizardManager] Error converting registry name {registry_name}: {e}")
        
        # Fallback conversion
        return registry_name.replace('_', ' ').title() + ' Analysis'
    
    
    # REMOVED: _fallback_comparison - Not used in current implementation

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
            
            for pair in active_pairs:
                pair_name = pair['name']
                
                # Check if statistics are missing
                if pair_name not in self.pair_statistics:
                    print(f"[ComparisonWizard] Recalculating missing statistics for '{pair_name}'...")
                    self._recalculate_pair_statistics(pair)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error verifying statistics: {str(e)}")
    
    def _recalculate_pair_statistics(self, pair_config):
        """Recalculate statistics for a specific pair"""
        try:
            pair_name = pair_config['name']
            
            # Get or recalculate aligned data
            if pair_name not in self.pair_aligned_data:
                ref_channel = self._get_channel(pair_config['ref_file'], pair_config['ref_channel'])
                test_channel = self._get_channel(pair_config['test_file'], pair_config['test_channel'])
                
                if not ref_channel or not test_channel:
                    print(f"[ComparisonWizard] Cannot find channels for '{pair_name}'")
                    return
                
                aligned_data = self._align_channels(ref_channel, test_channel, pair_config)
                self.pair_aligned_data[pair_name] = aligned_data
            else:
                aligned_data = self.pair_aligned_data[pair_name]
            
            # Recalculate and store statistics
            stats = self._calculate_statistics(aligned_data)
            self.pair_statistics[pair_name] = stats
            self._update_pair_statistics(pair_name, stats)
            
            print(f"[ComparisonWizard] Recalculated statistics for '{pair_name}'")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error recalculating statistics for '{pair_config['name']}': {str(e)}")
            error_stats = {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': f'Recalculation failed: {str(e)}'}
            self.pair_statistics[pair_config['name']] = error_stats
            self._update_pair_statistics(pair_config['name'], error_stats)
        
    def _force_table_refresh(self):
        """Force the active pairs table to refresh and show updates immediately"""
        try:
            self.window.active_pair_table.repaint()
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
            
            print(f"[ComparisonWizard] Pair '{pair_name}': r={r_str}, RMS={rms_str}, N={n_val}")
            
            if 'error' in stats:
                print(f"  Error: {stats['error']}")
                
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
        
        # Generate plots on all tabs
        self._generate_all_visualizations(checked_pairs, plot_config)
    
    def _determine_plot_type_from_pairs(self, checked_pairs):
        """Determine appropriate plot type based on comparison methods"""
        if not checked_pairs:
            return 'scatter'
        
        # Get the comparison method from the first pair
        first_pair = checked_pairs[0]
        comparison_method = first_pair.get('comparison_method', 'Correlation Analysis')
        
        # Convert to registry name and get plot type from comparison class
        if COMPARISON_AVAILABLE:
            try:
                registry_name = self.get_registry_name_from_display(comparison_method)
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls and hasattr(comparison_cls, 'plot_type'):
                    return comparison_cls.plot_type
            except Exception as e:
                print(f"[ComparisonWizardManager] Error determining plot type: {e}")
        
        # Fallback to scatter for all methods
        return 'scatter'
        
    def _generate_all_visualizations(self, checked_pairs, plot_config):
        """Generate plots for all tabs (scatter, histogram, heatmap)"""
        # Generate scatter plot (main comparison plot)
        self._generate_scatter_plot(checked_pairs, plot_config)
        
        # Generate histogram plot
        self._generate_histogram_plot(checked_pairs, plot_config)
        
        # Generate heatmap plot
        self._generate_heatmap_plot(checked_pairs, plot_config)
        
    def _generate_scatter_plot(self, checked_pairs, plot_config):
        """Generate scatter plot (renamed from _generate_multi_pair_plot)"""
        self._generate_multi_pair_plot(checked_pairs, plot_config)
        
    def _generate_histogram_plot(self, checked_pairs, plot_config):
        """Generate histogram plot for distribution comparison"""
        if not hasattr(self.window, 'histogram_canvas') or not self.window.histogram_canvas:
            return
            
        fig = self.window.histogram_canvas.figure
        self._clear_figure_completely(fig)
        
        # Collect all data for histogram
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            if pair_name not in self.pair_aligned_data:
                continue
                
            aligned_data = self.pair_aligned_data[pair_name]
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None or len(ref_data) == 0:
                continue
                
            # Filter valid data
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_clean = ref_data[valid_mask]
            test_clean = test_data[valid_mask]
            
            if len(ref_clean) > 0:
                all_ref_data.extend(ref_clean)
                all_test_data.extend(test_clean)
        
        if not all_ref_data:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No valid data for histogram', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('No Data Available')
            self.window.histogram_canvas.draw()
            return
        
        # Create histogram plot
        ax = fig.add_subplot(111)
        
        # Create side-by-side histograms
        ax.hist(all_ref_data, bins=50, alpha=0.7, label='Reference', color='blue', density=True)
        ax.hist(all_test_data, bins=50, alpha=0.7, label='Test', color='red', density=True)
        
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.set_title('Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        try:
            fig.tight_layout()
        except:
            pass
        self.window.histogram_canvas.draw()
        
    def _generate_heatmap_plot(self, checked_pairs, plot_config):
        """Generate heatmap plot for density visualization"""
        if not hasattr(self.window, 'heatmap_canvas') or not self.window.heatmap_canvas:
            return
            
        fig = self.window.heatmap_canvas.figure
        self._clear_figure_completely(fig)
        
        # Collect all data for heatmap
        all_ref_data = []
        all_test_data = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            if pair_name not in self.pair_aligned_data:
                continue
                
            aligned_data = self.pair_aligned_data[pair_name]
            ref_data = aligned_data['ref_data']
            test_data = aligned_data['test_data']
            
            if ref_data is None or test_data is None or len(ref_data) == 0:
                continue
                
            # Filter valid data
            valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
            ref_clean = ref_data[valid_mask]
            test_clean = test_data[valid_mask]
            
            if len(ref_clean) > 0:
                all_ref_data.extend(ref_clean)
                all_test_data.extend(test_clean)
        
        if not all_ref_data or len(all_ref_data) < 10:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'Insufficient data for heatmap', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Insufficient Data')
            self.window.heatmap_canvas.draw()
            return
        
        # Create heatmap/density plot
        ax = fig.add_subplot(111)
        
        try:
            # Create 2D histogram/heatmap
            all_ref_data = np.array(all_ref_data)
            all_test_data = np.array(all_test_data)
            
            # Create heatmap based on plot type
            plot_type = plot_config.get('plot_type', 'scatter')
            
            if plot_type == 'bland_altman':
                # For Bland-Altman, show heatmap of differences vs means
                means = (all_ref_data + all_test_data) / 2
                diffs = all_test_data - all_ref_data
                
                h, xedges, yedges = np.histogram2d(means, diffs, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
                ax.set_xlabel('Mean of Methods')
                ax.set_ylabel('Difference (Test - Reference)')
                ax.set_title('Bland-Altman Density Heatmap')
                
            else:
                # For correlation/scatter, show heatmap of ref vs test
                h, xedges, yedges = np.histogram2d(all_ref_data, all_test_data, bins=50)
                extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
                
                im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
                ax.set_xlabel('Reference Data')
                ax.set_ylabel('Test Data')
                ax.set_title('Correlation Density Heatmap')
                
                # Add identity line for reference
                min_val = min(np.min(all_ref_data), np.min(all_test_data))
                max_val = max(np.max(all_ref_data), np.max(all_test_data))
                ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.8, linewidth=2, label='y=x')
                ax.legend()
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Point Density')
            
        except Exception as e:
            print(f"[ComparisonWizard] Error creating heatmap: {e}")
            ax.text(0.5, 0.5, f'Error creating heatmap: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Heatmap Error')
        
        ax.grid(True, alpha=0.3)
        
        try:
            fig.tight_layout()
        except:
            pass
        self.window.heatmap_canvas.draw()
        
    def _generate_multi_pair_plot(self, checked_pairs, plot_config):
        """Generate plot with multiple pairs using different markers"""
        if not hasattr(self.window, 'canvas') or not self.window.canvas:
            return
            
        fig = self.window.canvas.figure
        
        # Comprehensive figure clearing to prevent overlapping plots
        self._clear_figure_completely(fig)
        
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
        
        # Try to generate plot using proper comparison methods first
        try:
            dynamic_success = self._generate_dynamic_plot_content(ax, all_ref_data, all_test_data, plot_config, checked_pairs)
            if dynamic_success:
                # If dynamic plot generation succeeded, we're done
                self._apply_common_plot_config(ax, fig, plot_config, checked_pairs)
                print(f"[ComparisonWizard] Plot generated successfully using comparison methods with {len(checked_pairs)} pairs")
                return
        except Exception as e:
            print(f"[ComparisonWizard] Dynamic plot generation failed: {e}, falling back to hardcoded logic")
        
        # Fallback to hardcoded logic only if dynamic generation fails
        print(f"[ComparisonWizard] Using fallback hardcoded plot generation")
        
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
                        
                elif plot_type == 'ccc' or plot_type == 'rmse' or plot_type == 'icc':
                    # CCC, RMSE, and ICC all use hexbin plot of test vs reference
                    # Check for zero range
                    if np.ptp(all_ref_data) == 0 or np.ptp(all_test_data) == 0:
                        # Fallback to scatter plot
                        ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
                    else:
                        hb = ax.hexbin(all_ref_data, all_test_data, gridsize=bin_size, cmap='viridis', mincnt=1)
                
                # Add colorbar only if hexbin was successful
                if 'hb' in locals():
                    try:
                        # Remove any existing colorbars to prevent duplication
                        if hasattr(fig, '_colorbar_list'):
                            for cb in fig._colorbar_list:
                                try:
                                    cb.remove()
                                except:
                                    pass
                            fig._colorbar_list = []
                        
                        cbar = plt.colorbar(hb, ax=ax)
                        cbar.set_label('Point Density', rotation=270, labelpad=15)
                        
                        # Keep track of colorbars for future cleanup
                        if not hasattr(fig, '_colorbar_list'):
                            fig._colorbar_list = []
                        fig._colorbar_list.append(cbar)
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
                elif plot_type == 'ccc' or plot_type == 'rmse' or plot_type == 'icc':
                    # CCC, RMSE, and ICC all use scatter plot of test vs reference
                    ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            
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
                    
                elif plot_type == 'ccc' or plot_type == 'rmse' or plot_type == 'icc':
                    # CCC, RMSE, and ICC all use scatter plot of test vs reference
                    self._create_kde_plot(ax, all_ref_data, all_test_data, kde_bandwidth)
                    
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
                elif plot_type == 'ccc' or plot_type == 'rmse' or plot_type == 'icc':
                    # CCC, RMSE, and ICC all use scatter plot of test vs reference
                    ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            
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
                
                elif plot_type == 'ccc':
                    # CCC uses scatter plot of test vs reference
                    ax.scatter(ref_plot, test_plot, alpha=0.6, s=30,
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
                
                elif plot_type == 'rmse':
                    # RMSE uses scatter plot of test vs reference
                    ax.scatter(ref_plot, test_plot, alpha=0.6, s=30,
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
                
                elif plot_type == 'icc':
                    # ICC uses scatter plot of test vs reference
                    ax.scatter(ref_plot, test_plot, alpha=0.6, s=30,
                              color=color, marker=marker,
                              label=f"{pair_name} (n={len(ref_plot)})")
        
        # Apply common plot configuration (for fallback plots)
        self._apply_common_plot_config(ax, fig, plot_config, checked_pairs)
        
        print(f"[ComparisonWizard] Fallback plot generated successfully with {len(checked_pairs)} pairs")
    
    def _clear_figure_completely(self, fig):
        """
        Comprehensively clear the figure to prevent overlapping plots when switching methods.
        
        Args:
            fig: Matplotlib figure object
        """
        try:
            # Remove any existing colorbars to prevent duplication
            if hasattr(fig, '_colorbar_list'):
                for cb in fig._colorbar_list:
                    try:
                        cb.remove()
                    except:
                        pass
                fig._colorbar_list = []
            
            # Clear all axes and their artists
            for ax in fig.get_axes():
                try:
                    ax.clear()
                    # Remove all artists (lines, patches, text, etc.)
                    for artist in ax.get_children():
                        try:
                            artist.remove()
                        except:
                            pass
                except:
                    pass
            
            # Clear the figure itself
            fig.clear()
            
            # Force a redraw to ensure clean state
            try:
                fig.canvas.draw_idle()
            except:
                pass
            
            print("[ComparisonWizard] Figure cleared completely to prevent overlapping plots")
            
        except Exception as e:
            print(f"[ComparisonWizard] Error during figure clearing: {e}")
            # Fallback to basic clear
            fig.clear()
    
    def _generate_dynamic_plot_content(self, ax, all_ref_data, all_test_data, plot_config, checked_pairs):
        """Generate plot content using plot generators from comparison folder"""
        try:
            plot_type = plot_config.get('plot_type', 'scatter')
            
            # Use comparison methods for plot generation
            
            # Fallback to comparison method dynamic generation
            method_name = None
            if checked_pairs:
                method_name = checked_pairs[0].get('comparison_method')
            
            if not method_name:
                method_name = plot_config.get('comparison_method', 'Correlation Analysis')
            
            # Try to create the comparison method instance
            method_instance = None
            if COMPARISON_AVAILABLE and method_name:
                # Get method parameters from plot config
                method_params = plot_config.get('method_parameters', {})
                # Also include overlay parameters
                overlay_params = {k: v for k, v in plot_config.items() if k.startswith('show_') or k in ['confidence_interval', 'custom_line']}
                method_params.update(overlay_params)
                
                # Use dynamic display name to registry name conversion
                registry_name = self.get_registry_name_from_display(method_name)
                if registry_name:
                    comparison_cls = self.comparison_registry.get(registry_name)
                    if comparison_cls:
                        method_instance = comparison_cls(**method_params)
                        print(f"[ComparisonWizard] Successfully created {registry_name} method instance")
                    else:
                        print(f"[ComparisonWizard] Comparison class not found for registry name: {registry_name}")
                        method_instance = None
                else:
                    print(f"[ComparisonWizard] Could not convert display name to registry name: {method_name}")
                    method_instance = None
            
            if method_instance and hasattr(method_instance, 'generate_plot'):
                print(f"[ComparisonWizard] Using comparison method plot generation for {method_name}")
                # Calculate statistics first
                stats_results = method_instance.calculate_stats(all_ref_data, all_test_data)
                method_instance.generate_plot(ax, all_ref_data, all_test_data, plot_config, stats_results)
                return True
            else:
                # Ultimate fallback to hardcoded methods for compatibility
                print(f"[ComparisonWizard] No plot generator or method found for {plot_type}, using fallback")
                self._generate_fallback_plot_content(ax, all_ref_data, all_test_data, plot_config, plot_type, checked_pairs)
                return False
                
        except Exception as e:
            print(f"[ComparisonWizard] Error in dynamic plot generation: {e}")
            import traceback
            traceback.print_exc()
            # Ultimate fallback
            plot_type = plot_config.get('plot_type', 'scatter')
            ax.text(0.5, 0.5, f'Error generating {plot_type} plot: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            return False
    
    def _enhance_plot_config_with_overlay_options(self, plot_config, checked_pairs):
        """Enhance plot config with overlay options from comparison methods"""
        enhanced_config = plot_config.copy()
        
        if not checked_pairs:
            return enhanced_config
        
        # Get overlay options from the first checked pair's comparison method
        first_pair = checked_pairs[0]
        method_name = first_pair.get('comparison_method')
        
        if method_name and COMPARISON_AVAILABLE:
            try:
                # Use dynamic display name to registry name conversion
                registry_name = self.get_registry_name_from_display(method_name)
                comparison_cls = self.comparison_registry.get(registry_name)
                if comparison_cls:
                    method_info = comparison_cls().get_info()
                    if method_info and 'overlay' in method_info:
                        overlay_options = method_info['overlay']
                        
                        # Merge overlay options with existing plot config
                        for key, value in overlay_options.items():
                            if key not in enhanced_config:
                                enhanced_config[key] = value
                        
                        print(f"[ComparisonWizard] Enhanced plot config with {len(overlay_options)} overlay options from {method_name}")
                    
            except Exception as e:
                print(f"[ComparisonWizard] Error enhancing plot config with overlay options: {e}")
        
        return enhanced_config
    
    def _generate_fallback_plot_content(self, ax, all_ref_data, all_test_data, plot_config, plot_type, checked_pairs):
        """Fallback plot generation for backward compatibility"""
        import numpy as np
        from scipy import stats
        
        print(f"[ComparisonWizard] Fallback plot generated successfully with {len(checked_pairs)} pairs")
        
        # Simple fallback plot based on plot type
        if plot_type == 'scatter' or plot_type == 'correlation':
            # Create scatter plot
            ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            
            # Add identity line if requested
            if plot_config.get('show_identity_line', True):
                min_val = min(np.min(all_ref_data), np.min(all_test_data))
                max_val = max(np.max(all_ref_data), np.max(all_test_data))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y = x')
            
            # Add regression line if requested
            if plot_config.get('show_regression_line', True):
                try:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(all_ref_data, all_test_data)
                    line_x = np.array([np.min(all_ref_data), np.max(all_ref_data)])
                    line_y = slope * line_x + intercept
                    ax.plot(line_x, line_y, 'g-', alpha=0.8, label=f'Regression (R²={r_value**2:.3f})')
                except:
                    pass
            
            # Add statistical results if requested
            if plot_config.get('show_statistical_results', True):
                try:
                    r_val = np.corrcoef(all_ref_data, all_test_data)[0, 1]
                    rmse = np.sqrt(np.mean((all_test_data - all_ref_data)**2))
                    stats_text = f'R = {r_val:.3f}\nRMSE = {rmse:.3f}'
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                except:
                    pass
            
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Test Data')
            ax.set_title('Correlation Analysis')
            
        elif plot_type == 'bland_altman':
            differences = all_test_data - all_ref_data
            averages = (all_ref_data + all_test_data) / 2
            ax.scatter(averages, differences, alpha=0.6, s=20, c='blue')
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Mean difference')
            
            # Add limits of agreement
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            ax.axhline(y=mean_diff + 1.96*std_diff, color='r', linestyle=':', alpha=0.7, label='+1.96 SD')
            ax.axhline(y=mean_diff - 1.96*std_diff, color='r', linestyle=':', alpha=0.7, label='-1.96 SD')
            
            ax.set_xlabel('Average')
            ax.set_ylabel('Difference')
            ax.set_title('Bland-Altman Plot')
            
        else:
            # Default scatter plot
            ax.scatter(all_ref_data, all_test_data, alpha=0.6, s=20, c='blue')
            ax.set_xlabel('Reference Data')
            ax.set_ylabel('Test Data')
            ax.set_title(f'{plot_type.title()} Plot')
        
        # Add legend if there are labeled elements
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        
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
                # Remove any existing colorbars to prevent duplication
                fig = ax.get_figure()
                if hasattr(fig, '_colorbar_list'):
                    for cb in fig._colorbar_list:
                        try:
                            cb.remove()
                        except:
                            pass
                    fig._colorbar_list = []
                
                cbar = plt.colorbar(cs, ax=ax)
                cbar.set_label('Density', rotation=270, labelpad=15)
                
                # Keep track of colorbars for future cleanup
                if not hasattr(fig, '_colorbar_list'):
                    fig._colorbar_list = []
                fig._colorbar_list.append(cbar)
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
        
    # REMOVED: _generate_pearson_plot_content - Now handled by PearsonPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_bland_altman_plot_content - Now handled by BlandAltmanPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_scatter_plot_content - Now handled by ScatterPlotGenerator in comparison/plot_generators.py
        
    # REMOVED: _generate_residual_plot_content - Now handled by ResidualPlotGenerator in comparison/plot_generators.py
        
    def _apply_common_plot_config(self, ax, fig, plot_config, checked_pairs):
        """Apply common plot configuration options"""
        # Apply plot configuration
        if plot_config.get('show_grid', True):
            ax.grid(True, alpha=0.3)
            
        # Only show legend for scatter plots and if requested
        density_type = plot_config.get('density_display', 'scatter')
        if plot_config.get('show_legend', False) and density_type == 'scatter' and len(checked_pairs) <= 10:
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
        
        # Use comprehensive clearing to prevent overlapping plots
        self._clear_figure_completely(fig)
        
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
        """Calculate comparison statistics using statistics calculators"""
        try:
            # Use simple statistics calculation
            return self._calculate_simple_statistics(aligned_data)
            
        except Exception as e:
            print(f"[ComparisonWizard] Statistics calculation error: {str(e)}")
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': str(e)}
    
    def _calculate_simple_statistics(self, aligned_data):
        """Simple statistics calculation as fallback"""
        try:
            # Validate input data
            validation_result = self._validate_aligned_data(aligned_data)
            if not validation_result['valid']:
                return validation_result['error_stats']
            
            ref_clean, test_clean, n_valid, n_total = validation_result['clean_data']
            
            # Calculate basic statistics
            stats_dict = {'n': n_valid, 'valid_ratio': n_valid/n_total}
            
            # Calculate correlation
            stats_dict.update(self._calculate_correlation(ref_clean, test_clean))
            
            # Calculate RMS and difference statistics
            stats_dict.update(self._calculate_difference_stats(ref_clean, test_clean))
            
            return stats_dict
            
        except Exception as e:
            print(f"[ComparisonWizard] Simple statistics calculation error: {str(e)}")
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': str(e)}
    
    def _validate_aligned_data(self, aligned_data):
        """Validate aligned data for statistics calculation"""
        ref_data = aligned_data['ref_data']
        test_data = aligned_data['test_data']
        
        # Check for missing data
        if ref_data is None or test_data is None:
            return {
                'valid': False,
                'error_stats': {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'Missing data'}
            }
        
        # Check for length mismatch
        if len(ref_data) != len(test_data):
            return {
                'valid': False,
                'error_stats': {'r': np.nan, 'rms': np.nan, 'n': 0, 'error': 'Data length mismatch'}
            }
        
        # Clean data
        valid_mask = np.isfinite(ref_data) & np.isfinite(test_data)
        ref_clean = ref_data[valid_mask]
        test_clean = test_data[valid_mask]
        
        n_valid = len(ref_clean)
        n_total = len(ref_data)
        
        # Check for insufficient data
        if n_valid == 0:
            return {
                'valid': False,
                'error_stats': {'r': np.nan, 'rms': np.nan, 'n': 0, 'valid_ratio': 0.0, 'error': 'No valid data points'}
            }
        
        if n_valid < 3:
            return {
                'valid': False,
                'error_stats': {'r': np.nan, 'rms': np.nan, 'n': n_valid, 'valid_ratio': n_valid/n_total, 'error': 'Insufficient data for statistics'}
            }
        
        return {
            'valid': True,
            'clean_data': (ref_clean, test_clean, n_valid, n_total)
        }
    
    def _calculate_correlation(self, ref_clean, test_clean):
        """Calculate correlation coefficient"""
        try:
            if np.var(ref_clean) == 0 or np.var(test_clean) == 0:
                return {'r': np.nan, 'r_error': 'Constant values'}
            
            correlation, p_value = scipy_stats.pearsonr(ref_clean, test_clean)
            return {'r': correlation, 'r_pvalue': p_value}
        except Exception as e:
            return {'r': np.nan, 'r_error': str(e)}
    
    def _calculate_difference_stats(self, ref_clean, test_clean):
        """Calculate RMS and difference statistics"""
        try:
            differences = test_clean - ref_clean
            rms = np.sqrt(np.mean(differences ** 2))
            return {
                'rms': rms,
                'mean_diff': np.mean(differences),
                'std_diff': np.std(differences)
            }
        except Exception as e:
            return {'rms': np.nan, 'rms_error': str(e)}
        
    def _update_pair_statistics(self, pair_name, statistics):
        """Update statistics in the active pairs table"""
        print(f"[ComparisonWizard] Updating table statistics for pair '{pair_name}'...")
        
        table = self.window.active_pair_table
        row = self._find_pair_row(pair_name)
        
        if row is None:
            print(f"[ComparisonWizard] WARNING: Could not find table row for pair '{pair_name}'")
            return
        
        print(f"[ComparisonWizard] Found pair '{pair_name}' at table row {row}")
        
        # Set tooltip for pair name
        self._set_pair_name_tooltip(row, pair_name)
        
        # Create and update table items
        items = self._create_statistics_table_items(statistics)
        self._set_table_items(row, items)
        
        # Set comprehensive tooltips
        self._set_detailed_tooltips(row, statistics)
        
        print(f"[ComparisonWizard] Updated row {row}: r={items['r_text']}, RMS={items['rms_text']}, N={items['n_text']}")
        
        # Refresh the updated cells
        self._refresh_table_cells(row, [2, 3, 4])
    
    def _find_pair_row(self, pair_name):
        """Find the table row for a given pair name"""
        table = self.window.active_pair_table
        for row in range(table.rowCount()):
            name_item = table.item(row, 1)
            if name_item and name_item.text() == pair_name:
                return row
        return None
    
    def _create_statistics_table_items(self, statistics):
        """Create formatted table items for statistics display"""
        from PySide6.QtWidgets import QTableWidgetItem
        from PySide6.QtCore import Qt
        
        # Format correlation
        r_val = statistics.get('r', np.nan)
        if np.isnan(r_val):
            r_text, r_color = "N/A", None
        else:
            r_text = f"{r_val:.3f}"
            r_color = self._get_correlation_color(r_val)
        
        # Format RMS
        rms_val = statistics.get('rms', np.nan)
        if np.isnan(rms_val):
            rms_text, rms_color = "N/A", None
        else:
            rms_text = f"{rms_val:.3f}"
            rms_color = QColor(70, 130, 180)  # Steel blue
        
        # Format sample size
        n_val = statistics.get('n', 0)
        n_text = f"{n_val:,}"
        n_color = QColor(105, 105, 105) if n_val > 0 else QColor(220, 20, 60)
        
        # Create table items
        items = {}
        for key, (text, color) in [('r', (r_text, r_color)), ('rms', (rms_text, rms_color)), ('n', (n_text, n_color))]:
            item = QTableWidgetItem(text)
            if color:
                item.setForeground(color)
                item.setData(Qt.FontRole, self._get_bold_font())
            items[f'{key}_item'] = item
            items[f'{key}_text'] = text
        
        return items
    
    def _get_correlation_color(self, r_val):
        """Get color for correlation value"""
        if abs(r_val) >= 0.7:
            return QColor(34, 139, 34)  # Forest green
        elif abs(r_val) >= 0.3:
            return QColor(255, 140, 0)  # Dark orange
        else:
            return QColor(220, 20, 60)  # Crimson
    
    def _set_table_items(self, row, items):
        """Set table items for statistics columns"""
        table = self.window.active_pair_table
        table.setItem(row, 2, items['r_item'])
        table.setItem(row, 3, items['rms_item'])
        table.setItem(row, 4, items['n_item'])
    
    def _refresh_table_cells(self, row, columns):
        """Refresh specific table cells"""
        table = self.window.active_pair_table
        for col in columns:
            table.update(table.model().index(row, col))
                    
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
            
            # Build tooltip parts using helper methods
            tooltip_parts = []
            tooltip_parts.extend(self._get_basic_statistics_tooltip(statistics))
            tooltip_parts.extend(self._get_data_quality_tooltip(statistics))
            tooltip_parts.extend(self._get_additional_statistics_tooltip(statistics))
            tooltip_parts.extend(self._get_reference_test_tooltip(statistics))
            tooltip_parts.extend(self._get_error_tooltip(statistics))
            
            # Create tooltip text
            tooltip_text = "\n".join(tooltip_parts)
            
            # Apply tooltip to all statistics columns
            for col in [2, 3, 4]:  # r, RMS, N columns
                item = table.item(row, col)
                if item:
                    item.setToolTip(tooltip_text)
                    
        except Exception as e:
            print(f"[ComparisonWizard] Error setting tooltips: {str(e)}")
    
    def _get_basic_statistics_tooltip(self, statistics):
        """Get basic statistics for tooltip"""
        parts = []
        
        r_val = statistics.get('r', np.nan)
        rms_val = statistics.get('rms', np.nan)
        n_val = statistics.get('n', 0)
        
        if not np.isnan(r_val):
            parts.append(f"Correlation: {r_val:.4f}")
            if 'r_pvalue' in statistics:
                parts.append(f"p-value: {statistics['r_pvalue']:.2e}")
        
        if not np.isnan(rms_val):
            parts.append(f"RMS difference: {rms_val:.4f}")
        
        parts.append(f"Sample size: {n_val:,}")
        
        return parts
    
    def _get_data_quality_tooltip(self, statistics):
        """Get data quality information for tooltip"""
        parts = []
        
        if 'valid_ratio' in statistics:
            parts.append(f"Valid data: {statistics['valid_ratio']*100:.1f}%")
        
        return parts
    
    def _get_additional_statistics_tooltip(self, statistics):
        """Get additional statistics for tooltip"""
        parts = []
        
        additional_stats = ['mean_diff', 'std_diff', 'median_diff', 'max_abs_diff']
        stat_labels = ['Mean difference', 'Std difference', 'Median difference', 'Max |difference|']
        
        for stat, label in zip(additional_stats, stat_labels):
            if stat in statistics:
                parts.append(f"{label}: {statistics[stat]:.4f}")
        
        return parts
    
    def _get_reference_test_tooltip(self, statistics):
        """Get reference and test data info for tooltip"""
        parts = []
        
        if 'ref_mean' in statistics and 'ref_std' in statistics:
            parts.append(f"Reference: μ={statistics['ref_mean']:.3f}, σ={statistics['ref_std']:.3f}")
        if 'test_mean' in statistics and 'test_std' in statistics:
            parts.append(f"Test: μ={statistics['test_mean']:.3f}, σ={statistics['test_std']:.3f}")
        
        return parts
    
    def _get_error_tooltip(self, statistics):
        """Get error information for tooltip"""
        parts = []
        
        if 'error' in statistics:
            parts.append(f"⚠️ Error: {statistics['error']}")
        
        return parts
        
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
        
        # Build comprehensive plot configuration including method parameters
        plot_config = {
            'plot_type': plot_type,
            'show_grid': True,
            'show_legend': False,  # Remove legend by default
            'checked_pairs': checked_pairs
        }
        
        # Add method-specific parameters from the first pair for plot configuration
        if checked_pairs:
            first_pair = checked_pairs[0]
            method_params = first_pair.get('method_parameters', {})
            
            # Map method parameters to plot config parameters
            if plot_type == 'bland_altman':
                plot_config['confidence_interval'] = method_params.get('show_ci', True)
                plot_config['agreement_limits'] = method_params.get('agreement_limits', 1.96)
                plot_config['proportional_bias'] = method_params.get('proportional_bias', False)
            elif plot_type == 'scatter' or plot_type == 'pearson':
                plot_config['confidence_level'] = method_params.get('confidence_level', 0.95)
                plot_config['correlation_type'] = method_params.get('correlation_type', 'pearson')
            elif plot_type == 'residual':
                plot_config['normality_test'] = method_params.get('normality_test', 'shapiro')
                plot_config['outlier_detection'] = method_params.get('outlier_detection', 'iqr')
        
        # Use the same plot generation as the Generate Plot button
        self._generate_multi_pair_plot(checked_pairs, plot_config)

    # REMOVED: _update_comprehensive_display - Not used in current implementation
        
    def _calculate_comprehensive_statistics(self, checked_pairs, plot_type='scatter'):
        """Calculate comprehensive statistics for step 2 display using statistics calculators"""
        if not checked_pairs:
            return {'error': 'No pairs selected'}
        
        # Use simple calculation
        return {'error': 'Comprehensive statistics not available'}
        
    def _prepare_pairs_for_statistics_calculator(self, checked_pairs):
        """Prepare pairs data for statistics calculator"""
        enhanced_pairs = []
        
        for pair in checked_pairs:
            pair_name = pair['name']
            enhanced_pair = pair.copy()
            
            # Add aligned data and statistics if available
            if pair_name in self.pair_aligned_data:
                aligned_data = self.pair_aligned_data[pair_name]
                enhanced_pair['statistics'] = {
                    'aligned_data': aligned_data
                }
                
                # Add individual pair statistics if available
                if pair_name in self.pair_statistics:
                    enhanced_pair['statistics']['calculated_stats'] = self.pair_statistics[pair_name]
            
            enhanced_pairs.append(enhanced_pair)
        
        return enhanced_pairs
    
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

    # REMOVED: _generate_cumulative_preview - Not used in current implementation, replaced by _update_cumulative_display

    def _calculate_cumulative_statistics(self, checked_pairs):
        """Calculate cumulative statistics for checked pairs using statistics calculators"""
        if not checked_pairs:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': 0}
        
        # Use simple calculation
        return self._calculate_simple_cumulative_stats(checked_pairs)
    
    def _calculate_simple_cumulative_stats(self, checked_pairs):
        """Simple cumulative statistics calculation as fallback"""
        if not checked_pairs:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': 0}
        
        # Collect all data from checked pairs
        all_ref_data, all_test_data = self._collect_cumulative_data(checked_pairs)
        
        if not all_ref_data or not all_test_data:
            return {'r': np.nan, 'rms': np.nan, 'n': 0, 'pairs': len(checked_pairs)}
        
        # Calculate statistics using the same helper methods
        ref_clean = np.array(all_ref_data)
        test_clean = np.array(all_test_data)
        
        correlation_stats = self._calculate_correlation(ref_clean, test_clean)
        difference_stats = self._calculate_difference_stats(ref_clean, test_clean)
        
        return {
            'r': correlation_stats.get('r', np.nan),
            'rms': difference_stats.get('rms', np.nan),
            'n': len(ref_clean),
            'pairs': len(checked_pairs)
        }
    
    def _collect_cumulative_data(self, checked_pairs):
        """Collect all data from checked pairs"""
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
        
        return all_ref_data, all_test_data
        
    def _format_cumulative_stats(self, stats, n_pairs):
        """Format cumulative statistics for display"""
        r_str = f"{stats['r']:.3f}" if not np.isnan(stats['r']) else "N/A"
        rms_str = f"{stats['rms']:.3f}" if not np.isnan(stats['rms']) else "N/A"
        
        return f"Cumulative Stats: r = {r_str}, RMS = {rms_str}, N = {stats['n']:,} points ({n_pairs} pairs shown)"

