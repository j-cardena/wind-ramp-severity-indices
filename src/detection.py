"""
Wind Power Ramp Detection

This module implements ramp detection algorithms for wind power time series.

Methods:
- Slope threshold detection
- Swinging door algorithm (simplified)

Reference:
    Cardenas-Barrera, J. (2026). "Beyond Magnitude and Rate: Shape-Based 
    Severity Indices for Wind Power Ramp Events with Validated Unique 
    Information Content."

Author: Julian Cardenas-Barrera
License: MIT
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .ramp_indices import RampEvent


@dataclass
class DetectionConfig:
    """Configuration for ramp detection."""
    threshold: float = 0.02  # Minimum rate to qualify as ramping (per hour)
    min_duration: int = 3    # Minimum ramp duration (hours)
    min_magnitude: float = 0.1  # Minimum total magnitude
    merge_gap: int = 1       # Maximum gap to merge adjacent ramps (hours)


def detect_ramps(power: np.ndarray,
                 timestamps: Optional[np.ndarray] = None,
                 config: Optional[DetectionConfig] = None) -> List[RampEvent]:
    """
    Detect ramp events in a wind power time series.
    
    Uses a slope-threshold approach:
    1. Compute rolling first differences
    2. Identify periods where |δP| > threshold
    3. Merge consecutive same-sign periods
    4. Filter by minimum duration and magnitude
    
    Args:
        power: Array of power values (normalized 0-1)
        timestamps: Optional array of timestamps (hours). If None, assumes
                   hourly data starting at hour 0.
        config: Detection configuration. If None, uses defaults.
        
    Returns:
        List of RampEvent objects
    """
    if config is None:
        config = DetectionConfig()
    
    if timestamps is None:
        timestamps = np.arange(len(power), dtype=float)
    
    # Calculate hourly differences
    diffs = np.diff(power)
    
    # Find ramping periods (above threshold)
    is_ramping = np.abs(diffs) > config.threshold
    
    # Identify contiguous ramping segments
    segments = _find_segments(is_ramping, diffs)
    
    # Merge close segments with same direction
    segments = _merge_segments(segments, config.merge_gap)
    
    # Convert to RampEvent objects
    ramps = []
    for start_idx, end_idx, direction in segments:
        # Extract ramp data
        ramp_power = power[start_idx:end_idx + 1]
        ramp_times = timestamps[start_idx:end_idx + 1] - timestamps[start_idx]
        
        # Check minimum duration
        duration = ramp_times[-1] - ramp_times[0]
        if duration < config.min_duration:
            continue
        
        # Check minimum magnitude
        magnitude = abs(ramp_power[-1] - ramp_power[0])
        if magnitude < config.min_magnitude:
            continue
        
        # Get start hour (for diurnal analysis)
        start_hour = int(timestamps[start_idx]) % 24
        
        ramp = RampEvent(
            power=ramp_power,
            timestamps=ramp_times,
            start_time=start_hour,
            direction=direction
        )
        ramps.append(ramp)
    
    return ramps


def detect_ramps_swinging_door(power: np.ndarray,
                                tolerance: float = 0.05,
                                min_duration: int = 3) -> List[RampEvent]:
    """
    Detect ramps using simplified Swinging Door Algorithm.
    
    Based on: Cui et al. (2016) "An optimized swinging door algorithm 
    for identifying wind ramping events"
    
    Args:
        power: Array of power values (normalized 0-1)
        tolerance: Deviation tolerance for linear approximation
        min_duration: Minimum ramp duration (hours)
        
    Returns:
        List of RampEvent objects
    """
    n = len(power)
    if n < 3:
        return []
    
    segments = []
    i = 0
    
    while i < n - 1:
        # Find end of current segment
        j = i + 1
        while j < n:
            # Check if point j fits within tolerance of line from i
            if j > i + 1:
                # Linear interpolation from i to j
                expected = power[i] + (power[j] - power[i]) * \
                          np.arange(1, j - i) / (j - i)
                actual = power[i + 1:j]
                
                if np.max(np.abs(actual - expected)) > tolerance:
                    break
            j += 1
        
        # Record segment
        if j - i >= min_duration:
            direction = 'up' if power[j - 1] > power[i] else 'down'
            segments.append((i, j - 1, direction))
        
        i = j if j > i + 1 else i + 1
    
    # Convert to RampEvent objects
    timestamps = np.arange(n, dtype=float)
    ramps = []
    
    for start_idx, end_idx, direction in segments:
        ramp_power = power[start_idx:end_idx + 1]
        ramp_times = timestamps[start_idx:end_idx + 1] - timestamps[start_idx]
        start_hour = start_idx % 24
        
        ramp = RampEvent(
            power=ramp_power,
            timestamps=ramp_times,
            start_time=start_hour,
            direction=direction
        )
        ramps.append(ramp)
    
    return ramps


def _find_segments(is_ramping: np.ndarray, 
                   diffs: np.ndarray) -> List[Tuple[int, int, str]]:
    """Find contiguous ramping segments."""
    segments = []
    in_segment = False
    start_idx = 0
    current_sign = 0
    
    for i, (ramping, diff) in enumerate(zip(is_ramping, diffs)):
        if ramping:
            sign = 1 if diff > 0 else -1
            
            if not in_segment:
                # Start new segment
                in_segment = True
                start_idx = i
                current_sign = sign
            elif sign != current_sign:
                # Direction change - end current, start new
                direction = 'up' if current_sign > 0 else 'down'
                segments.append((start_idx, i, direction))
                start_idx = i
                current_sign = sign
        else:
            if in_segment:
                # End segment
                direction = 'up' if current_sign > 0 else 'down'
                segments.append((start_idx, i, direction))
                in_segment = False
    
    # Handle final segment
    if in_segment:
        direction = 'up' if current_sign > 0 else 'down'
        segments.append((start_idx, len(diffs), direction))
    
    return segments


def _merge_segments(segments: List[Tuple[int, int, str]], 
                    max_gap: int) -> List[Tuple[int, int, str]]:
    """Merge segments that are close together with same direction."""
    if len(segments) < 2:
        return segments
    
    merged = [segments[0]]
    
    for start, end, direction in segments[1:]:
        prev_start, prev_end, prev_dir = merged[-1]
        
        # Check if can merge
        if direction == prev_dir and start - prev_end <= max_gap:
            # Merge with previous
            merged[-1] = (prev_start, end, direction)
        else:
            merged.append((start, end, direction))
    
    return merged


def calculate_ramp_statistics(ramps: List[RampEvent]) -> dict:
    """
    Calculate summary statistics for a set of ramps.
    
    Args:
        ramps: List of RampEvent objects
        
    Returns:
        Dictionary with statistics
    """
    if not ramps:
        return {}
    
    magnitudes = [r.magnitude for r in ramps]
    durations = [r.duration for r in ramps]
    rates = [r.rate for r in ramps]
    
    n_up = sum(1 for r in ramps if r.direction == 'up')
    n_down = len(ramps) - n_up
    
    return {
        'n_ramps': len(ramps),
        'n_up': n_up,
        'n_down': n_down,
        'magnitude_mean': np.mean(magnitudes),
        'magnitude_std': np.std(magnitudes),
        'magnitude_max': np.max(magnitudes),
        'duration_mean': np.mean(durations),
        'duration_std': np.std(durations),
        'rate_mean': np.mean(rates),
        'rate_std': np.std(rates),
        'rate_max': np.max(rates)
    }
