"""
Wind Power Ramp Severity Indices (ECSI)

Novel shape-based severity indices for wind power ramp events.

Indices:
- RAI: Ramp Acceleration Index
- RSCI: Ramp Shape Complexity Index
- OSI: Operational Stress Index
- GIP: Grid Impact Potential
- ECSI: Enhanced Composite Severity Index

Reference:
    Cardenas-Barrera, J. (2026). "Beyond Magnitude and Rate: Shape-Based 
    Severity Indices for Wind Power Ramp Events with Validated Unique 
    Information Content."
"""

from .ramp_indices import (
    RampEvent,
    RampSeverityCalculator,
    classify_severity,
    get_unique_variance,
    calculate_severity
)

from .detection import (
    DetectionConfig,
    detect_ramps,
    detect_ramps_swinging_door,
    calculate_ramp_statistics
)

from .validation import (
    ValidationResult,
    ValidationReport,
    ValidationFramework,
    calculate_cohens_d,
    interpret_cohens_d
)

__version__ = "1.0.0"
__author__ = "Julian Cardenas-Barrera"
__email__ = "julian.cardenas@unb.ca"

__all__ = [
    # Core classes
    'RampEvent',
    'RampSeverityCalculator',
    'DetectionConfig',
    'ValidationFramework',
    'ValidationResult',
    'ValidationReport',
    
    # Functions
    'detect_ramps',
    'detect_ramps_swinging_door',
    'calculate_severity',
    'classify_severity',
    'get_unique_variance',
    'calculate_ramp_statistics',
    'calculate_cohens_d',
    'interpret_cohens_d'
]
