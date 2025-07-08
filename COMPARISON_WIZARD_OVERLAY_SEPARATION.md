# Comparison Wizard: Overlay Options vs Parameter Options Separation

## Summary of Changes

This document summarizes the refactoring done to cleanly separate **overlay options** from **parameter options** in the comparison wizard, eliminating overlapping elements and creating a clear distinction between computational parameters and display toggles.

## Key Principles

### Parameter Options (Computational)
- **Purpose**: Control how comparison metrics are computed
- **Examples**: correlation_type, agreement_multiplier, polynomial_degree, fit_method
- **UI**: Form controls (dropdowns, spinboxes, checkboxes for computational flags)
- **Impact**: Changes require recomputation of statistics

### Overlay Options (Display)
- **Purpose**: Control what computed metrics are displayed on plots
- **Examples**: show_bias_line, highlight_outliers, show_confidence_intervals
- **UI**: Simple checkboxes that toggle display on/off
- **Impact**: Changes only affect plot appearance, no recomputation needed

## Changes Made

### 1. Enhanced Overlay Options (`_create_overlay_options`)

**Added comprehensive overlay options:**
- `show_confidence_intervals` - Display confidence intervals around statistics
- `show_confidence_bands` - For cross-correlation and time series
- `highlight_outliers` - Mark statistical outliers on plots
- `show_bias_line` - Mean bias line (Bland-Altman)
- `show_limits_of_agreement` - ±1.96×SD lines (Bland-Altman)
- `show_regression_line` - Linear regression line
- `show_trend_line` - Trend lines for time series/residuals
- `show_error_bands` - ±RMSE error bands
- `show_residual_statistics` - Display residual stats on plot
- `show_density_overlay` - Kernel density estimation overlay
- `show_histogram_overlay` - Histogram overlay for distributions
- `show_statistical_results` - Display test results on plot
- `custom_line` - Custom reference line

### 2. Method-Specific Overlay Visibility (`_update_overlay_options`)

**Each comparison method now shows only relevant overlays:**

- **Bland-Altman Analysis**: confidence intervals, bias line, limits of agreement, outliers, statistical results
- **Correlation Analysis**: identity line, confidence intervals, regression line, outliers, density overlay, statistical results  
- **Residual Analysis**: outliers, confidence intervals, trend line, residual statistics, histogram overlay, statistical results
- **Statistical Tests**: confidence intervals, outliers, identity line, statistical results
- **Cross-Correlation**: confidence bands, trend line, statistical results

### 3. Cleaned Parameter Controls

**Removed display-related parameters from method controls:**

#### Correlation Analysis
- **Removed**: `confidence_level`, `remove_outliers` (now overlay-controlled)
- **Kept**: `correlation_type`, `include_rmse`, `outlier_method`, `detrend_method`

#### Bland-Altman Analysis  
- **Removed**: Display options (bias/LoA lines now overlay-controlled)
- **Kept**: `agreement_multiplier`, `percentage_difference`, `test_proportional_bias`, `log_transform`

#### Residual Analysis
- **Removed**: `show_residual_stats`, `detect_outliers` (now overlay-controlled)
- **Kept**: `fit_method`, `polynomial_degree`, `normality_test`, `trend_analysis`

### 4. Updated Parameter Filtering (`_create_controls_for_method`)

**Expanded filtering of display-related parameters:**
```python
overlay_params = {
    'show_ci', 'confidence_intervals', 'compute_confidence_intervals', 'confidence_interval',
    'outlier_detection', 'detect_outliers', 'confidence_bands', 'confidence_level',
    'bootstrap_ci', 'show_residual_stats', 'highlight_outliers', 'show_bias_line',
    'show_limits_of_agreement', 'show_regression_line', 'show_error_bands',
    'show_confidence_bands', 'show_identity_line', 'remove_outliers'
}
```

### 5. Enhanced Overlay Parameters (`_get_overlay_parameters`)

**Added support for all new overlay options:**
- `show_identity_line`, `confidence_interval`, `show_confidence_bands`
- `highlight_outliers`, `show_bias_line`, `show_limits_of_agreement`
- `show_regression_line`, `show_trend_line`, `show_error_bands`
- `show_residual_statistics`, `show_density_overlay`, `show_histogram_overlay`
- `show_statistical_results`, `custom_line`

### 6. Updated Signal Connections (`_connect_overlay_signals`)

**Connected all new overlay checkboxes to immediate plot updates:**
```python
overlay_widgets = [
    'y_equals_x_checkbox', 'ci_checkbox', 'confidence_bands_checkbox', 'outlier_checkbox',
    'bias_line_checkbox', 'loa_checkbox', 'regression_line_checkbox', 'trend_line_checkbox',
    'error_bands_checkbox', 'residual_stats_checkbox', 'density_overlay_checkbox',
    'histogram_overlay_checkbox', 'stats_results_checkbox', 'custom_line_checkbox'
]
```

## Benefits Achieved

### 1. Clear Separation of Concerns
- **Parameters**: Purely computational, affect how statistics are calculated
- **Overlays**: Purely visual, control what computed results are displayed

### 2. Better User Experience  
- **Immediate feedback**: Overlay changes update plots instantly
- **Method-specific options**: Only relevant overlays shown for each method
- **No redundancy**: Eliminated overlapping parameter/overlay controls

### 3. Computational Efficiency
- **Always compute everything**: All metrics calculated regardless of overlay state
- **Display toggles**: Overlays only control visualization, no recomputation needed
- **Faster interaction**: Overlay changes don't trigger expensive recalculations

### 4. Maintainable Architecture
- **Single responsibility**: Each control has one clear purpose
- **Extensible**: Easy to add new overlays without affecting parameters
- **Consistent**: Same pattern across all comparison methods

## Usage Example

1. **Select comparison method** (e.g., "Bland-Altman Analysis")
2. **Configure parameters** (e.g., agreement_multiplier = 1.96, percentage_difference = true)
3. **Add comparison pair** with selected parameters
4. **Generate plot** (computes all metrics: bias, LoA, outliers, confidence intervals, etc.)
5. **Toggle overlays** to show/hide computed elements:
   - ✓ Show Bias Line
   - ✓ Show Limits of Agreement  
   - ✗ Highlight Outliers
   - ✓ Show Statistical Results

Plot updates immediately when overlay checkboxes are toggled, showing/hiding the relevant computed elements without recalculation.

## Future Enhancements

- **Overlay configuration persistence**: Save overlay preferences per method
- **Advanced overlay options**: Color, style, transparency controls for overlays
- **Custom overlay plugins**: Allow users to define custom overlay elements
- **Overlay export**: Include overlay configuration in exported plots/data 