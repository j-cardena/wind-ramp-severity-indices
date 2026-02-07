"""
Wind Power Ramp Severity Indices

This module implements novel severity indices for wind power ramp events:
- RAI: Ramp Acceleration Index (captures onset suddenness)
- RSCI: Ramp Shape Complexity Index (captures trajectory complexity)
- OSI: Operational Stress Index (captures asymmetric operational risk)
- GIP: Grid Impact Potential (captures context-dependent severity)
- ECSI: Enhanced Composite Severity Index (weighted combination)

Reference:
    Cardenas-Barrera, J. (2026). "Beyond Magnitude and Rate: Shape-Based 
    Severity Indices for Wind Power Ramp Events with Validated Unique 
    Information Content."

Author: Julian Cardenas-Barrera
License: MIT
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import warnings


@dataclass
class RampEvent:
    """
    Represents a single wind power ramp event.
    
    Attributes:
        power: Array of power values during the ramp (normalized 0-1)
        timestamps: Array of timestamps (hours from start)
        start_time: Hour of day when ramp started (0-23)
        direction: 'up' or 'down'
    """
    power: np.ndarray
    timestamps: np.ndarray
    start_time: int
    direction: str
    
    @property
    def magnitude(self) -> float:
        """Absolute change in power (ΔP)."""
        return abs(self.power[-1] - self.power[0])
    
    @property
    def duration(self) -> float:
        """Duration in hours (Δt)."""
        return self.timestamps[-1] - self.timestamps[0]
    
    @property
    def rate(self) -> float:
        """Average rate of change (ΔP/Δt)."""
        if self.duration == 0:
            return 0.0
        return self.magnitude / self.duration
    
    @property
    def start_power(self) -> float:
        """Power at ramp start."""
        return self.power[0]
    
    @property
    def end_power(self) -> float:
        """Power at ramp end."""
        return self.power[-1]
    
    @property
    def n_points(self) -> int:
        """Number of data points in the ramp."""
        return len(self.power)


class RampSeverityCalculator:
    """
    Calculate severity indices for wind power ramp events.
    
    This class implements four novel indices plus a weighted composite:
    - RAI: Ramp Acceleration Index
    - RSCI: Ramp Shape Complexity Index  
    - OSI: Operational Stress Index
    - GIP: Grid Impact Potential
    - ECSI: Enhanced Composite Severity Index
    
    Example:
        >>> calculator = RampSeverityCalculator()
        >>> ramp = RampEvent(power=np.array([0.8, 0.6, 0.4, 0.2]),
        ...                  timestamps=np.array([0, 1, 2, 3]),
        ...                  start_time=18, direction='down')
        >>> results = calculator.calculate_all(ramp)
        >>> print(f"ECSI: {results['ECSI']:.3f}")
    """
    
    # Default ECSI weights (determined by unique variance)
    DEFAULT_WEIGHTS = {
        'RAI': 0.452,
        'RSCI': 0.271,
        'OSI': 0.107,
        'GIP': 0.170
    }
    
    # Peak hour definitions
    PEAK_HOURS = {
        'evening': (17, 21),  # 5 PM - 9 PM
        'morning': (7, 9),    # 7 AM - 9 AM
        'midday': (12, 14)    # 12 PM - 2 PM
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the calculator.
        
        Args:
            weights: Optional custom weights for ECSI. If None, uses default
                    weights derived from unique variance analysis.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        
        # Normalization bounds (will be updated during batch processing)
        self._norm_bounds = {
            'RAI': (0, 10),
            'RSCI': (1, 3),
            'OSI': (0, 2),
            'GIP': (0, 1)
        }
    
    def calculate_rai(self, ramp: RampEvent) -> float:
        """
        Calculate Ramp Acceleration Index (RAI).
        
        RAI = max|r_{i+1} - r_i| / σ(r)
        
        Captures the "surprise factor" - how suddenly a ramp begins or
        changes character. Based on the second derivative of power.
        
        Args:
            ramp: RampEvent object
            
        Returns:
            RAI value (dimensionless, higher = more sudden onset)
        """
        if ramp.n_points < 3:
            return 0.0
        
        # Calculate instantaneous rates
        rates = np.diff(ramp.power)
        
        if len(rates) < 2:
            return 0.0
        
        # Calculate rate changes (acceleration)
        accelerations = np.diff(rates)
        
        # Maximum absolute acceleration
        max_accel = np.max(np.abs(accelerations))
        
        # Standard deviation of rates
        rate_std = np.std(rates)
        
        if rate_std == 0:
            return 0.0
        
        return max_accel / rate_std
    
    def calculate_rsci(self, ramp: RampEvent) -> float:
        """
        Calculate Ramp Shape Complexity Index (RSCI).
        
        RSCI = (L_actual / L_straight) × (1 + N_inflection / n)
        
        Quantifies deviation from an ideal linear trajectory. Complex,
        non-monotonic ramps are harder to forecast and manage.
        
        Args:
            ramp: RampEvent object
            
        Returns:
            RSCI value (dimensionless, ≥1, higher = more complex)
        """
        if ramp.n_points < 2:
            return 1.0
        
        n = ramp.n_points
        
        # Calculate actual path length
        # L = Σ sqrt(1 + r_i^2) for normalized time steps
        rates = np.diff(ramp.power)
        path_segments = np.sqrt(1 + rates**2)
        L_actual = np.sum(path_segments)
        
        # Calculate straight-line distance
        L_straight = np.sqrt((n - 1)**2 + ramp.magnitude**2)
        
        if L_straight == 0:
            return 1.0
        
        # Count inflection points (direction changes)
        if len(rates) < 2:
            n_inflections = 0
        else:
            signs = np.sign(rates)
            # Remove zeros for sign comparison
            signs = signs[signs != 0]
            if len(signs) < 2:
                n_inflections = 0
            else:
                n_inflections = np.sum(np.diff(signs) != 0)
        
        # RSCI formula
        path_ratio = L_actual / L_straight
        inflection_factor = 1 + n_inflections / n
        
        return path_ratio * inflection_factor
    
    def calculate_osi(self, ramp: RampEvent) -> float:
        """
        Calculate Operational Stress Index (OSI).
        
        OSI = ΔP(1+r̄) × (1+R_reserve) × (1+r̄^1.5) × D(t₀)
        
        Incorporates physics-informed operational constraints with
        ASYMMETRIC treatment of up-ramps vs down-ramps.
        
        Args:
            ramp: RampEvent object
            
        Returns:
            OSI value (dimensionless, higher = more operational stress)
        """
        # Base stress
        base_stress = ramp.magnitude * (1 + ramp.rate)
        
        # Asymmetric reserve risk
        if ramp.direction == 'down':
            # Down-ramp: risk of reserve depletion at low power
            R_reserve = np.exp(-3 * ramp.end_power)
        else:
            # Up-ramp: risk of curtailment at high power
            R_reserve = np.exp(-3 * (1 - ramp.end_power))
        
        reserve_factor = 1 + R_reserve
        
        # Ramping reserve requirement (non-linear with rate)
        ramp_reserve = 1 + ramp.rate ** 1.5
        
        # Demand factor based on time of day
        demand_factor = self._get_demand_factor(ramp.start_time)
        
        return base_stress * reserve_factor * ramp_reserve * demand_factor
    
    def calculate_gip(self, ramp: RampEvent) -> float:
        """
        Calculate Grid Impact Potential (GIP).
        
        GIP = f₁(physical) × f₂(context) × f₃(asymmetry)
        
        Uses multiplicative interactions for context-dependent severity.
        A ramp is only severe when multiple factors align.
        
        Args:
            ramp: RampEvent object
            
        Returns:
            GIP value (dimensionless, higher = greater grid impact)
        """
        # f1: Physical characteristics
        # Square root transforms compress extremes
        mag_term = np.sqrt(ramp.magnitude)
        rate_term = np.sqrt(ramp.rate) if ramp.rate > 0 else 0
        duration_mod = self._duration_modifier(ramp.duration)
        f1 = mag_term * rate_term * duration_mod
        
        # f2: Contextual factors
        avg_power = (ramp.start_power + ramp.end_power) / 2
        # Penalize ramps at extreme operating points
        extreme_penalty = 1 + 4 * (avg_power - 0.5) ** 2
        time_factor = self._get_time_factor(ramp.start_time)
        f2 = extreme_penalty * time_factor
        
        # f3: Asymmetry factor
        f3 = self._asymmetry_factor(ramp)
        
        return f1 * f2 * f3
    
    def calculate_ecsi(self, ramp: RampEvent, 
                       normalize: bool = True) -> float:
        """
        Calculate Enhanced Composite Severity Index (ECSI).
        
        ECSI = w₁×RAI + w₂×RSCI + w₃×OSI + w₄×GIP
        
        Weights are determined by unique information content (variance
        not explained by basic parameters).
        
        Args:
            ramp: RampEvent object
            normalize: Whether to normalize components to [0,1]
            
        Returns:
            ECSI value (0-1 if normalized, higher = more severe)
        """
        # Calculate individual indices
        rai = self.calculate_rai(ramp)
        rsci = self.calculate_rsci(ramp)
        osi = self.calculate_osi(ramp)
        gip = self.calculate_gip(ramp)
        
        if normalize:
            rai = self._normalize(rai, 'RAI')
            rsci = self._normalize(rsci, 'RSCI')
            osi = self._normalize(osi, 'OSI')
            gip = self._normalize(gip, 'GIP')
        
        # Weighted sum
        ecsi = (self.weights['RAI'] * rai +
                self.weights['RSCI'] * rsci +
                self.weights['OSI'] * osi +
                self.weights['GIP'] * gip)
        
        return ecsi
    
    def calculate_all(self, ramp: RampEvent) -> Dict[str, float]:
        """
        Calculate all severity indices for a ramp event.
        
        Args:
            ramp: RampEvent object
            
        Returns:
            Dictionary with all index values and basic parameters
        """
        return {
            # Basic parameters
            'magnitude': ramp.magnitude,
            'duration': ramp.duration,
            'rate': ramp.rate,
            'direction': ramp.direction,
            'start_time': ramp.start_time,
            
            # Novel indices
            'RAI': self.calculate_rai(ramp),
            'RSCI': self.calculate_rsci(ramp),
            'OSI': self.calculate_osi(ramp),
            'GIP': self.calculate_gip(ramp),
            
            # Composite
            'ECSI': self.calculate_ecsi(ramp)
        }
    
    def calculate_batch(self, ramps: List[RampEvent]) -> List[Dict[str, float]]:
        """
        Calculate indices for multiple ramp events.
        
        This method first computes normalization bounds from the batch,
        then calculates all indices.
        
        Args:
            ramps: List of RampEvent objects
            
        Returns:
            List of dictionaries with index values
        """
        # First pass: compute raw values for normalization
        raw_values = {
            'RAI': [],
            'RSCI': [],
            'OSI': [],
            'GIP': []
        }
        
        for ramp in ramps:
            raw_values['RAI'].append(self.calculate_rai(ramp))
            raw_values['RSCI'].append(self.calculate_rsci(ramp))
            raw_values['OSI'].append(self.calculate_osi(ramp))
            raw_values['GIP'].append(self.calculate_gip(ramp))
        
        # Update normalization bounds
        for key in raw_values:
            values = np.array(raw_values[key])
            self._norm_bounds[key] = (np.min(values), np.max(values))
        
        # Second pass: calculate all indices with proper normalization
        return [self.calculate_all(ramp) for ramp in ramps]
    
    def _get_demand_factor(self, hour: int) -> float:
        """Get demand factor based on hour of day."""
        if self.PEAK_HOURS['evening'][0] <= hour <= self.PEAK_HOURS['evening'][1]:
            return 1.5  # Evening peak
        elif self.PEAK_HOURS['morning'][0] <= hour <= self.PEAK_HOURS['morning'][1]:
            return 1.3  # Morning peak
        elif self.PEAK_HOURS['midday'][0] <= hour <= self.PEAK_HOURS['midday'][1]:
            return 1.1  # Midday
        else:
            return 0.8  # Off-peak
    
    def _get_time_factor(self, hour: int) -> float:
        """Get time factor for GIP calculation."""
        # Normalize to 0-1 range with peak hours higher
        if 7 <= hour <= 21:
            return 0.7 + 0.3 * np.sin(np.pi * (hour - 7) / 14)
        return 0.5
    
    def _duration_modifier(self, duration: float) -> float:
        """
        Penalize very short or very long ramps.
        Optimal duration around 5-10 hours.
        """
        if duration < 3:
            return 0.7 + 0.1 * duration  # Short ramps
        elif duration > 15:
            return 1.0 - 0.02 * (duration - 15)  # Long ramps
        else:
            return 1.0  # Optimal range
    
    def _asymmetry_factor(self, ramp: RampEvent) -> float:
        """Calculate asymmetry factor for GIP."""
        if ramp.direction == 'down':
            # Higher factor if ending at low power during peak
            if ramp.end_power < 0.3 and self._is_peak_hour(ramp.start_time):
                return 1.5
            return 1.2
        else:
            # Higher factor if ending at high power during peak
            if ramp.end_power > 0.7 and self._is_peak_hour(ramp.start_time):
                return 1.3
            return 1.0
    
    def _is_peak_hour(self, hour: int) -> bool:
        """Check if hour is during peak demand."""
        return (self.PEAK_HOURS['evening'][0] <= hour <= self.PEAK_HOURS['evening'][1] or
                self.PEAK_HOURS['morning'][0] <= hour <= self.PEAK_HOURS['morning'][1])
    
    def _normalize(self, value: float, index_name: str) -> float:
        """Min-max normalize a value to [0, 1]."""
        min_val, max_val = self._norm_bounds[index_name]
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)


def classify_severity(ecsi: float) -> str:
    """
    Classify ramp severity based on ECSI value.
    
    Args:
        ecsi: ECSI value (0-1)
        
    Returns:
        Severity class: 'low', 'moderate', 'high', or 'critical'
    """
    if ecsi < 0.25:
        return 'low'
    elif ecsi < 0.5:
        return 'moderate'
    elif ecsi < 0.75:
        return 'high'
    else:
        return 'critical'


def get_unique_variance(index_values: np.ndarray,
                        magnitude: np.ndarray,
                        rate: np.ndarray,
                        duration: np.ndarray) -> float:
    """
    Calculate unique variance of an index beyond basic parameters.
    
    Unique Variance = 1 - R²(Index | magnitude, rate, duration)
    
    Args:
        index_values: Array of index values
        magnitude: Array of ramp magnitudes
        rate: Array of ramp rates
        duration: Array of ramp durations
        
    Returns:
        Unique variance (0-1, higher = more unique information)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # Build feature matrix
    X = np.column_stack([magnitude, rate, duration])
    y = index_values
    
    # Fit regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R²
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Unique variance is what's NOT explained
    return 1 - r2


# Convenience functions
def calculate_severity(power: np.ndarray,
                       timestamps: np.ndarray,
                       start_time: int,
                       direction: str) -> Dict[str, float]:
    """
    Convenience function to calculate all severity indices.
    
    Args:
        power: Array of power values (normalized 0-1)
        timestamps: Array of timestamps (hours)
        start_time: Hour of day (0-23)
        direction: 'up' or 'down'
        
    Returns:
        Dictionary with all index values
    """
    ramp = RampEvent(
        power=np.array(power),
        timestamps=np.array(timestamps),
        start_time=start_time,
        direction=direction
    )
    calculator = RampSeverityCalculator()
    return calculator.calculate_all(ramp)
