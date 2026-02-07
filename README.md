# Wind Power Ramp Severity Indices (ECSI)

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/10.xxxx/xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Novel shape-based severity indices for wind power ramp events with validated unique information content.**

This repository contains the implementation and validation code for the paper:

> Cardenas-Barrera, J. (2026). "Beyond Magnitude and Rate: Shape-Based Severity Indices for Wind Power Ramp Events with Validated Unique Information Content." *[Journal Name]*.

## 🎯 Key Contributions

We propose four novel indices that capture **50.7% unique variance** beyond conventional ramp parameters—a **15× improvement** over existing approaches:

| Index | What It Captures | Unique Variance |
|-------|------------------|-----------------|
| **RAI** (Ramp Acceleration Index) | Onset suddenness (2nd derivative) | 90.1% |
| **RSCI** (Ramp Shape Complexity Index) | Trajectory complexity | 54.0% |
| **OSI** (Operational Stress Index) | Asymmetric reserve/curtailment risk | 21.3% |
| **GIP** (Grid Impact Potential) | Context-dependent severity | 33.8% |
| **ECSI** (Enhanced Composite) | Weighted combination | 50.7% |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/[username]/wind-ramp-severity-indices.git
cd wind-ramp-severity-indices

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Usage

```python
from ramp_indices import RampSeverityCalculator, detect_ramps
import pandas as pd

# Load your wind power data (normalized 0-1)
data = pd.read_csv('your_wind_data.csv')
power = data['power'].values

# Detect ramp events
ramps = detect_ramps(power, threshold=0.02, min_duration=3)

# Calculate severity indices
calculator = RampSeverityCalculator()
results = calculator.calculate_all(ramps)

# Access individual indices
print(f"ECSI: {results['ECSI']:.3f}")
print(f"RAI:  {results['RAI']:.3f}")
print(f"RSCI: {results['RSCI']:.3f}")
print(f"OSI:  {results['OSI']:.3f}")
print(f"GIP:  {results['GIP']:.3f}")
```

## 📊 Why These Indices Matter

### The Problem with Existing Metrics

Conventional ramp metrics (magnitude, rate, duration) fail to distinguish operationally different scenarios:

```
Ramp A (Gradual onset):     Ramp B (Sudden onset):
    ____                        ________
   /                                    \
  /                                      \____
 /                                       
Magnitude: 50 MW ✓              Magnitude: 50 MW ✓
Rate: 25 MW/h ✓                 Rate: 25 MW/h ✓
Duration: 2h ✓                  Duration: 2h ✓

RAI: LOW                        RAI: HIGH ← Captures the difference!
```

### Asymmetric Risk (OSI)

```
Down-ramp to 5% capacity  →  Reserve depletion risk (HIGH OSI)
Up-ramp to 95% capacity   →  Curtailment stress (HIGH OSI)
Ramp at 50% capacity      →  Manageable (LOW OSI)
```

## 📁 Repository Structure

```
wind-ramp-severity-indices/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── CITATION.cff              # Citation information
│
├── src/
│   ├── __init__.py
│   ├── ramp_indices.py       # Core index calculations
│   ├── detection.py          # Ramp detection algorithms
│   ├── validation.py         # Six-test validation framework
│   └── visualization.py      # Plotting utilities
│
├── notebooks/
│   ├── 01_demo.ipynb         # Quick start demonstration
│   ├── 02_validation.ipynb   # Full validation analysis
│   └── 03_case_studies.ipynb # Real-world examples
│
├── data/
│   ├── README.md             # Data description
│   └── sample_ramps.csv      # Sample ramp events for testing
│
├── results/
│   ├── validation_results.json
│   └── figures/
│
└── docs/
    ├── mathematical_formulation.md
    └── api_reference.md
```

## 📐 Mathematical Formulations

### Ramp Acceleration Index (RAI)

```
RAI = max|rᵢ₊₁ - rᵢ| / σ(r)
```

Where `rᵢ = P(tᵢ₊₁) - P(tᵢ)` is the instantaneous rate.

### Ramp Shape Complexity Index (RSCI)

```
RSCI = (L_actual / L_straight) × (1 + N_inflection / n)
```

Where:
- `L_actual` = path length of the ramp trajectory
- `L_straight` = direct Euclidean distance
- `N_inflection` = number of direction changes

### Operational Stress Index (OSI)

```
OSI = ΔP(1 + r̄) × (1 + R_reserve) × (1 + r̄^1.5) × D(t₀)
```

With **asymmetric** reserve risk:
```
R_reserve = exp(-3 × P_end)           for down-ramps
          = exp(-3 × (1 - P_end))     for up-ramps
```

### Grid Impact Potential (GIP)

```
GIP = f₁(physical) × f₂(context) × f₃(asymmetry)
```

### Enhanced Composite Severity Index (ECSI)

```
ECSI = 0.452×RAI + 0.271×RSCI + 0.107×OSI + 0.170×GIP
```

Weights determined by unique variance contribution.

## ✅ Validation Results

All indices pass a rigorous six-test validation framework:

| Test | Metric | Value | Threshold | Result |
|------|--------|-------|-----------|--------|
| Information Content | Unique Variance | 0.507 | > 0.15 | ✅ Pass |
| Discriminative Power | Silhouette | 0.558 | > 0.50 | ✅ Pass |
| Discriminative Power | η² | 0.629 | > 0.14 | ✅ Pass |
| Temporal Validity | Cohen's d | 0.618 | > 0.20 | ✅ Pass |
| Robustness | Outlier Sensitivity | 1.5% | < 10% | ✅ Pass |
| Sensitivity | Weight Stability | 0.994 | > 0.90 | ✅ Pass |

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{cardenasbarrera2026ramp,
  title={Beyond Magnitude and Rate: Shape-Based Severity Indices for 
         Wind Power Ramp Events with Validated Unique Information Content},
  author={Cardenas-Barrera, Julian},
  journal={[Journal Name]},
  year={2026},
  volume={},
  pages={},
  doi={10.xxxx/xxxxx}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Julian Cardenas-Barrera**  
Department of Electrical and Computer Engineering  
University of New Brunswick  
NB Power Research Chair in Smart Grid Technologies

- Email: [your.email@unb.ca]
- Website: [https://your-website.com]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NB Power Research Chair in Smart Grid Technologies
- Atlantic Digital Grid Consortium
- NB Power for providing wind power data
