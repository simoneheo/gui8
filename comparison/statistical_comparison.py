"""
Statistical Comparison Method

This module implements various statistical tests for comparing two datasets,
including parametric and non-parametric tests for differences in central tendency,
variance, and distribution shape.
"""

import numpy as np
from scipy import stats
from typing import Dict, Any, Optional
from .base_comparison import BaseComparison

class StatisticalComparison(BaseComparison):
    """
    Statistical tests comparison method.
    
    Performs comprehensive statistical testing to compare two datasets,
    including tests for differences in means, variances, and distributions.
    """
    
    name = "Statistical Tests"
    description = "Comprehensive statistical testing for comparing two datasets"
    category = "Statistical"
    version = "1.0.0"
    
    parameters = {
        'significance_level': {
            'type': float,
            'default': 0.05,
            'min': 0.001,
            'max': 0.5,
            'description': 'Significance level for hypothesis testing'
        },
        'test_suite': {
            'type': str,
            'default': 'comprehensive',
            'choices': ['basic', 'comprehensive', 'nonparametric'],
            'description': 'Suite of tests to perform'
        },
        'equal_variance_assumption': {
            'type': str,
            'default': 'test',
            'choices': ['assume_equal', 'assume_unequal', 'test'],
            'description': 'How to handle equal variance assumption'
        },
        'normality_assumption': {
            'type': str,
            'default': 'test',
            'choices': ['assume_normal', 'assume_nonnormal', 'test'],
            'description': 'How to handle normality assumption'
        },
        'paired_data': {
            'type': bool,
            'default': False,
            'description': 'Whether the data represents paired observations'
        }
    }
    
    output_types = ["statistical_tests", "assumption_tests", "effect_sizes", "descriptive_statistics"]
    
    def compare(self, ref_data: np.ndarray, test_data: np.ndarray, 
                ref_time: Optional[np.ndarray] = None, 
                test_time: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform statistical comparison between reference and test data.
        
        Args:
            ref_data: Reference data array
            test_data: Test data array
            ref_time: Optional time array (unused)
            test_time: Optional time array (unused)
            
        Returns:
            Dictionary containing statistical test results
        """
        # Validate and clean input data
        ref_data, test_data = self._validate_input_data(ref_data, test_data)
        ref_clean, test_clean, valid_ratio = self._remove_invalid_data(ref_data, test_data)
        
        # Initialize results
        results = {
            'method': self.name,
            'n_ref': len(ref_clean),
            'n_test': len(test_clean),
            'valid_ratio': valid_ratio,
            'significance_level': self.params['significance_level'],
            'paired_data': self.params['paired_data'],
            'descriptive_statistics': self._calculate_descriptive_stats(ref_clean, test_clean),
            'assumption_tests': {},
            'statistical_tests': {},
            'effect_sizes': {}
        }
        
        # Test assumptions
        results['assumption_tests'] = self._test_assumptions(ref_clean, test_clean)
        
        # Perform statistical tests based on suite
        test_suite = self.params['test_suite']
        
        if test_suite in ['basic', 'comprehensive']:
            results['statistical_tests'].update(self._perform_parametric_tests(ref_clean, test_clean, results['assumption_tests']))
        
        if test_suite in ['nonparametric', 'comprehensive']:
            results['statistical_tests'].update(self._perform_nonparametric_tests(ref_clean, test_clean))
        
        if test_suite == 'comprehensive':
            results['statistical_tests'].update(self._perform_distribution_tests(ref_clean, test_clean))
        
        # Calculate effect sizes
        results['effect_sizes'] = self._calculate_effect_sizes(ref_clean, test_clean)
        
        # Provide interpretation
        results['interpretation'] = self._interpret_results(results)
        
        # Store results
        self.results = results
        return results
    
    def _calculate_descriptive_stats(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate descriptive statistics for both datasets."""
        def calc_stats(data, label):
            return {
                f'{label}_mean': np.mean(data),
                f'{label}_median': np.median(data),
                f'{label}_std': np.std(data, ddof=1),
                f'{label}_var': np.var(data, ddof=1),
                f'{label}_min': np.min(data),
                f'{label}_max': np.max(data),
                f'{label}_q25': np.percentile(data, 25),
                f'{label}_q75': np.percentile(data, 75),
                f'{label}_skewness': stats.skew(data),
                f'{label}_kurtosis': stats.kurtosis(data),
                f'{label}_sem': stats.sem(data)
            }
        
        ref_stats = calc_stats(ref_data, 'ref')
        test_stats = calc_stats(test_data, 'test')
        
        return {**ref_stats, **test_stats}
    
    def _test_assumptions(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Test statistical assumptions for parametric tests."""
        assumptions = {}
        
        # Test normality
        if self.params['normality_assumption'] == 'test':
            assumptions['normality'] = self._test_normality_assumption(ref_data, test_data)
        else:
            assumptions['normality'] = {
                'ref_normal': self.params['normality_assumption'] == 'assume_normal',
                'test_normal': self.params['normality_assumption'] == 'assume_normal',
                'assumed': True
            }
        
        # Test equal variances
        if self.params['equal_variance_assumption'] == 'test':
            assumptions['equal_variance'] = self._test_equal_variance_assumption(ref_data, test_data)
        else:
            assumptions['equal_variance'] = {
                'equal_variances': self.params['equal_variance_assumption'] == 'assume_equal',
                'assumed': True
            }
        
        return assumptions
    
    def _test_normality_assumption(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Test normality assumption for both datasets."""
        alpha = self.params['significance_level']
        
        # Test reference data
        if len(ref_data) < 5000:
            ref_stat, ref_p = stats.shapiro(ref_data)
            ref_test = 'shapiro'
        else:
            ref_stat, ref_p = stats.kstest(ref_data, 'norm', args=(np.mean(ref_data), np.std(ref_data)))
            ref_test = 'kstest'
        
        # Test test data
        if len(test_data) < 5000:
            test_stat, test_p = stats.shapiro(test_data)
            test_test = 'shapiro'
        else:
            test_stat, test_p = stats.kstest(test_data, 'norm', args=(np.mean(test_data), np.std(test_data)))
            test_test = 'kstest'
        
        return {
            'ref_normal': ref_p > alpha,
            'test_normal': test_p > alpha,
            'ref_test': ref_test,
            'test_test': test_test,
            'ref_statistic': ref_stat,
            'test_statistic': test_stat,
            'ref_p_value': ref_p,
            'test_p_value': test_p,
            'both_normal': (ref_p > alpha) and (test_p > alpha)
        }
    
    def _test_equal_variance_assumption(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Test equal variance assumption."""
        alpha = self.params['significance_level']
        
        # Levene's test (more robust)
        levene_stat, levene_p = stats.levene(ref_data, test_data)
        
        # Bartlett's test (assumes normality)
        bartlett_stat, bartlett_p = stats.bartlett(ref_data, test_data)
        
        # F-test for equal variances
        f_stat = np.var(ref_data, ddof=1) / np.var(test_data, ddof=1)
        f_p = 2 * min(stats.f.cdf(f_stat, len(ref_data)-1, len(test_data)-1),
                      1 - stats.f.cdf(f_stat, len(ref_data)-1, len(test_data)-1))
        
        return {
            'equal_variances': levene_p > alpha,  # Use Levene's test as primary
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'bartlett_statistic': bartlett_stat,
            'bartlett_p_value': bartlett_p,
            'f_statistic': f_stat,
            'f_p_value': f_p
        }
    
    def _perform_parametric_tests(self, ref_data: np.ndarray, test_data: np.ndarray, 
                                 assumptions: Dict[str, Any]) -> Dict[str, Any]:
        """Perform parametric statistical tests."""
        tests = {}
        alpha = self.params['significance_level']
        
        # Choose appropriate t-test based on assumptions and data pairing
        if self.params['paired_data']:
            # Paired t-test
            if len(ref_data) == len(test_data):
                stat, p_value = stats.ttest_rel(ref_data, test_data)
                tests['paired_ttest'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_type': 'paired_ttest'
                }
        else:
            # Independent samples t-test
            if assumptions['equal_variance']['equal_variances']:
                stat, p_value = stats.ttest_ind(ref_data, test_data, equal_var=True)
                test_type = 'independent_ttest_equal_var'
            else:
                stat, p_value = stats.ttest_ind(ref_data, test_data, equal_var=False)
                test_type = 'welch_ttest'
            
            tests['independent_ttest'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': test_type
            }
        
        return tests
    
    def _perform_nonparametric_tests(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Perform non-parametric statistical tests."""
        tests = {}
        alpha = self.params['significance_level']
        
        if self.params['paired_data'] and len(ref_data) == len(test_data):
            # Wilcoxon signed-rank test
            try:
                stat, p_value = stats.wilcoxon(ref_data, test_data)
                tests['wilcoxon'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < alpha,
                    'test_type': 'wilcoxon_signed_rank'
                }
            except ValueError as e:
                tests['wilcoxon'] = {'error': str(e)}
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(ref_data, test_data, alternative='two-sided')
            tests['mann_whitney'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < alpha,
                'test_type': 'mann_whitney_u'
            }
        
        # Kruskal-Wallis test (equivalent to Mann-Whitney for 2 groups)
        stat, p_value = stats.kruskal(ref_data, test_data)
        tests['kruskal_wallis'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'test_type': 'kruskal_wallis'
        }
        
        return tests
    
    def _perform_distribution_tests(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, Any]:
        """Perform tests comparing distributions."""
        tests = {}
        alpha = self.params['significance_level']
        
        # Kolmogorov-Smirnov test
        stat, p_value = stats.ks_2samp(ref_data, test_data)
        tests['kolmogorov_smirnov'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'test_type': 'ks_2sample'
        }
        
        # Anderson-Darling test
        try:
            stat, critical_values, significance_level = stats.anderson_ksamp([ref_data, test_data])
            tests['anderson_darling'] = {
                'statistic': stat,
                'critical_values': critical_values.tolist(),
                'significance_level': significance_level,
                'significant': stat > critical_values[2],  # 5% level
                'test_type': 'anderson_darling_2sample'
            }
        except Exception as e:
            tests['anderson_darling'] = {'error': str(e)}
        
        return tests
    
    def _calculate_effect_sizes(self, ref_data: np.ndarray, test_data: np.ndarray) -> Dict[str, float]:
        """Calculate various effect size measures."""
        effect_sizes = {}
        
        # Cohen's d
        pooled_std = np.sqrt(((len(ref_data) - 1) * np.var(ref_data, ddof=1) + 
                             (len(test_data) - 1) * np.var(test_data, ddof=1)) / 
                            (len(ref_data) + len(test_data) - 2))
        
        if pooled_std > 0:
            cohens_d = (np.mean(test_data) - np.mean(ref_data)) / pooled_std
            effect_sizes['cohens_d'] = cohens_d
            effect_sizes['cohens_d_interpretation'] = self._interpret_cohens_d(abs(cohens_d))
        
        # Glass's delta (using reference group std)
        ref_std = np.std(ref_data, ddof=1)
        if ref_std > 0:
            glass_delta = (np.mean(test_data) - np.mean(ref_data)) / ref_std
            effect_sizes['glass_delta'] = glass_delta
        
        # Hedges' g (bias-corrected Cohen's d)
        if pooled_std > 0:
            correction_factor = 1 - (3 / (4 * (len(ref_data) + len(test_data)) - 9))
            hedges_g = cohens_d * correction_factor
            effect_sizes['hedges_g'] = hedges_g
        
        # Common Language Effect Size (probability of superiority)
        if len(ref_data) > 0 and len(test_data) > 0:
            comparisons = []
            for ref_val in ref_data:
                for test_val in test_data:
                    if test_val > ref_val:
                        comparisons.append(1)
                    elif test_val == ref_val:
                        comparisons.append(0.5)
                    else:
                        comparisons.append(0)
            
            cles = np.mean(comparisons)
            effect_sizes['common_language_effect_size'] = cles
        
        return effect_sizes
    
    def _interpret_cohens_d(self, abs_d: float) -> str:
        """Interpret Cohen's d effect size."""
        if abs_d >= 0.8:
            return 'large'
        elif abs_d >= 0.5:
            return 'medium'
        elif abs_d >= 0.2:
            return 'small'
        else:
            return 'negligible'
    
    def _interpret_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Provide interpretation of statistical test results."""
        interpretations = {}
        
        # Descriptive comparison
        desc_stats = results['descriptive_statistics']
        mean_diff = desc_stats['test_mean'] - desc_stats['ref_mean']
        interpretations['descriptive'] = f"Test mean is {mean_diff:.3f} units {'higher' if mean_diff > 0 else 'lower'} than reference"
        
        # Assumption tests
        assumptions = results['assumption_tests']
        if 'normality' in assumptions and not assumptions['normality'].get('assumed', False):
            if assumptions['normality']['both_normal']:
                interpretations['normality'] = "Both datasets appear normally distributed"
            else:
                interpretations['normality'] = "One or both datasets deviate from normality"
        
        if 'equal_variance' in assumptions and not assumptions['equal_variance'].get('assumed', False):
            if assumptions['equal_variance']['equal_variances']:
                interpretations['variance'] = "Equal variances assumption is satisfied"
            else:
                interpretations['variance'] = "Unequal variances detected"
        
        # Statistical tests
        stat_tests = results['statistical_tests']
        significant_tests = [name for name, test in stat_tests.items() 
                           if isinstance(test, dict) and test.get('significant', False)]
        
        if significant_tests:
            interpretations['significance'] = f"Significant difference detected by: {', '.join(significant_tests)}"
        else:
            interpretations['significance'] = "No significant difference detected"
        
        # Effect sizes
        effect_sizes = results['effect_sizes']
        if 'cohens_d' in effect_sizes:
            d_value = effect_sizes['cohens_d']
            d_interp = effect_sizes.get('cohens_d_interpretation', 'unknown')
            interpretations['effect_size'] = f"Effect size is {d_interp} (Cohen's d = {d_value:.3f})"
        
        return interpretations 