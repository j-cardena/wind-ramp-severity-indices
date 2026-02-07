"""
Validation Framework for Ramp Severity Indices

Implements a six-test validation framework:
1. Unique Information Content
2. Discriminative Power
3. Statistical Robustness
4. Construct Validity
5. Temporal Consistency
6. Sensitivity Analysis

Reference:
    Cardenas-Barrera, J. (2026). "Beyond Magnitude and Rate: Shape-Based 
    Severity Indices for Wind Power Ramp Events with Validated Unique 
    Information Content."

Author: Julian Cardenas-Barrera
License: MIT
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.cluster import KMeans
from scipy import stats
import warnings


@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_name: str
    metric_name: str
    value: float
    threshold: float
    passed: bool
    details: Optional[Dict] = None


@dataclass  
class ValidationReport:
    """Complete validation report."""
    results: List[ValidationResult]
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)
    
    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def n_total(self) -> int:
        return len(self.results)
    
    def summary(self) -> str:
        lines = ["Validation Summary", "=" * 50]
        for r in self.results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            lines.append(f"{r.test_name}: {r.metric_name} = {r.value:.3f} "
                        f"(threshold: {r.threshold}) {status}")
        lines.append("=" * 50)
        lines.append(f"Overall: {self.n_passed}/{self.n_total} tests passed")
        return "\n".join(lines)


class ValidationFramework:
    """
    Six-test validation framework for ramp severity indices.
    
    Tests:
    1. Unique Information Content: R² analysis
    2. Discriminative Power: Clustering + ANOVA
    3. Statistical Robustness: Bootstrap + outliers
    4. Construct Validity: Correlation analysis
    5. Temporal Consistency: Diurnal/seasonal patterns
    6. Sensitivity Analysis: Weight perturbation
    """
    
    # Default thresholds
    THRESHOLDS = {
        'unique_variance': 0.15,
        'silhouette': 0.50,
        'eta_squared': 0.14,
        'cohens_d': 0.20,
        'outlier_sensitivity': 0.10,
        'weight_stability': 0.90
    }
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = self.THRESHOLDS.copy()
        if thresholds:
            self.thresholds.update(thresholds)
    
    def run_all_tests(self,
                      index_values: np.ndarray,
                      magnitude: np.ndarray,
                      rate: np.ndarray,
                      duration: np.ndarray,
                      start_times: np.ndarray,
                      component_values: Optional[Dict[str, np.ndarray]] = None,
                      weights: Optional[Dict[str, float]] = None) -> ValidationReport:
        """
        Run all validation tests.
        
        Args:
            index_values: Array of ECSI or other index values
            magnitude: Array of ramp magnitudes
            rate: Array of ramp rates
            duration: Array of ramp durations
            start_times: Array of start hours (0-23)
            component_values: Dict with RAI, RSCI, OSI, GIP arrays
            weights: Dict with component weights
            
        Returns:
            ValidationReport with all test results
        """
        results = []
        
        # Test 1: Unique Information Content
        results.append(self.test_unique_variance(
            index_values, magnitude, rate, duration
        ))
        
        # Test 2: Discriminative Power
        results.extend(self.test_discriminative_power(index_values))
        
        # Test 3: Statistical Robustness
        results.extend(self.test_robustness(index_values))
        
        # Test 4: Construct Validity (if component values provided)
        if component_values:
            results.append(self.test_construct_validity(
                index_values, component_values
            ))
        
        # Test 5: Temporal Consistency
        results.append(self.test_temporal_consistency(
            index_values, start_times
        ))
        
        # Test 6: Sensitivity Analysis (if weights provided)
        if weights and component_values:
            results.append(self.test_sensitivity(
                component_values, weights
            ))
        
        return ValidationReport(results=results)
    
    def test_unique_variance(self,
                             index_values: np.ndarray,
                             magnitude: np.ndarray,
                             rate: np.ndarray,
                             duration: np.ndarray) -> ValidationResult:
        """
        Test 1: Unique Information Content
        
        Calculates proportion of variance NOT explained by basic parameters.
        """
        # Build feature matrix
        X = np.column_stack([magnitude, rate, duration])
        y = index_values
        
        # Fit regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        unique_var = 1 - r2
        
        return ValidationResult(
            test_name="Information Content",
            metric_name="Unique Variance",
            value=unique_var,
            threshold=self.thresholds['unique_variance'],
            passed=unique_var > self.thresholds['unique_variance'],
            details={'r_squared': r2, 'coefficients': model.coef_.tolist()}
        )
    
    def test_discriminative_power(self,
                                   index_values: np.ndarray) -> List[ValidationResult]:
        """
        Test 2: Discriminative Power
        
        Uses K-means clustering and ANOVA to assess class separation.
        """
        results = []
        
        # K-means clustering (2 clusters: high/low severity)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(index_values.reshape(-1, 1))
        
        # Silhouette score
        sil_score = silhouette_score(index_values.reshape(-1, 1), labels)
        results.append(ValidationResult(
            test_name="Discriminative Power",
            metric_name="Silhouette",
            value=sil_score,
            threshold=self.thresholds['silhouette'],
            passed=sil_score > self.thresholds['silhouette']
        ))
        
        # ANOVA and eta-squared
        groups = [index_values[labels == 0], index_values[labels == 1]]
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Calculate eta-squared
        ss_between = sum(len(g) * (np.mean(g) - np.mean(index_values))**2 
                        for g in groups)
        ss_total = np.sum((index_values - np.mean(index_values))**2)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
        
        results.append(ValidationResult(
            test_name="Discriminative Power",
            metric_name="η²",
            value=eta_sq,
            threshold=self.thresholds['eta_squared'],
            passed=eta_sq > self.thresholds['eta_squared'],
            details={'f_statistic': f_stat, 'p_value': p_value}
        ))
        
        return results
    
    def test_robustness(self,
                        index_values: np.ndarray,
                        n_bootstrap: int = 1000) -> List[ValidationResult]:
        """
        Test 3: Statistical Robustness
        
        Assesses bootstrap confidence interval and outlier sensitivity.
        """
        results = []
        
        # Bootstrap confidence interval
        bootstrap_means = []
        n = len(index_values)
        for _ in range(n_bootstrap):
            sample = np.random.choice(index_values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        ci_width = ci_upper - ci_lower
        
        # Outlier sensitivity
        q1, q3 = np.percentile(index_values, [25, 75])
        iqr = q3 - q1
        outlier_mask = (index_values < q1 - 1.5 * iqr) | (index_values > q3 + 1.5 * iqr)
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers > 0:
            mean_with = np.mean(index_values)
            mean_without = np.mean(index_values[~outlier_mask])
            sensitivity = abs(mean_with - mean_without) / mean_with
        else:
            sensitivity = 0.0
        
        results.append(ValidationResult(
            test_name="Robustness",
            metric_name="Outlier Sensitivity",
            value=sensitivity,
            threshold=self.thresholds['outlier_sensitivity'],
            passed=sensitivity < self.thresholds['outlier_sensitivity'],
            details={
                'n_outliers': int(n_outliers),
                'ci_95': (ci_lower, ci_upper),
                'ci_width': ci_width
            }
        ))
        
        return results
    
    def test_construct_validity(self,
                                index_values: np.ndarray,
                                component_values: Dict[str, np.ndarray]) -> ValidationResult:
        """
        Test 4: Construct Validity
        
        Verifies theoretical relationships between ECSI and components.
        """
        correlations = {}
        for name, values in component_values.items():
            corr, _ = stats.pearsonr(index_values, values)
            correlations[name] = corr
        
        # All components should positively correlate with composite
        min_corr = min(correlations.values())
        all_positive = all(c > 0.3 for c in correlations.values())
        
        return ValidationResult(
            test_name="Construct Validity",
            metric_name="Min Component Correlation",
            value=min_corr,
            threshold=0.30,
            passed=all_positive,
            details={'correlations': correlations}
        )
    
    def test_temporal_consistency(self,
                                   index_values: np.ndarray,
                                   start_times: np.ndarray) -> ValidationResult:
        """
        Test 5: Temporal Consistency
        
        Tests for expected diurnal patterns (peak vs off-peak).
        """
        # Define peak hours (7-9, 17-21)
        is_peak = ((start_times >= 7) & (start_times <= 9)) | \
                  ((start_times >= 17) & (start_times <= 21))
        
        peak_values = index_values[is_peak]
        offpeak_values = index_values[~is_peak]
        
        # Cohen's d effect size
        pooled_std = np.sqrt(
            ((len(peak_values) - 1) * np.std(peak_values)**2 +
             (len(offpeak_values) - 1) * np.std(offpeak_values)**2) /
            (len(peak_values) + len(offpeak_values) - 2)
        )
        
        cohens_d = (np.mean(peak_values) - np.mean(offpeak_values)) / pooled_std \
                   if pooled_std > 0 else 0
        
        # T-test
        t_stat, p_value = stats.ttest_ind(peak_values, offpeak_values)
        
        return ValidationResult(
            test_name="Temporal Validity",
            metric_name="Cohen's d (peak/off-peak)",
            value=abs(cohens_d),
            threshold=self.thresholds['cohens_d'],
            passed=abs(cohens_d) > self.thresholds['cohens_d'],
            details={
                'peak_mean': np.mean(peak_values),
                'offpeak_mean': np.mean(offpeak_values),
                't_statistic': t_stat,
                'p_value': p_value
            }
        )
    
    def test_sensitivity(self,
                         component_values: Dict[str, np.ndarray],
                         weights: Dict[str, float],
                         n_trials: int = 1000,
                         perturbation: float = 0.5) -> ValidationResult:
        """
        Test 6: Sensitivity Analysis
        
        Tests stability under weight perturbation.
        """
        # Calculate baseline ECSI
        baseline = sum(weights[k] * self._normalize(component_values[k])
                      for k in weights)
        
        correlations = []
        for _ in range(n_trials):
            # Perturb weights
            perturbed_weights = {}
            for k, w in weights.items():
                perturb = 1 + np.random.uniform(-perturbation, perturbation)
                perturbed_weights[k] = w * perturb
            
            # Renormalize
            total = sum(perturbed_weights.values())
            perturbed_weights = {k: v/total for k, v in perturbed_weights.items()}
            
            # Calculate perturbed ECSI
            perturbed = sum(perturbed_weights[k] * self._normalize(component_values[k])
                           for k in perturbed_weights)
            
            # Correlation with baseline
            corr, _ = stats.pearsonr(baseline, perturbed)
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        return ValidationResult(
            test_name="Sensitivity",
            metric_name="Weight Stability",
            value=mean_corr,
            threshold=self.thresholds['weight_stability'],
            passed=mean_corr > self.thresholds['weight_stability'],
            details={
                'correlation_mean': mean_corr,
                'correlation_std': np.std(correlations),
                'correlation_95ci': (np.percentile(correlations, 2.5),
                                    np.percentile(correlations, 97.5))
            }
        )
    
    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Min-max normalize values to [0, 1]."""
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            return np.full_like(values, 0.5)
        return (values - min_val) / (max_val - min_val)


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.std(group1, ddof=1)**2 + 
         (n2 - 1) * np.std(group2, ddof=1)**2) / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"
