# Mathematical Formulation

This document provides the complete mathematical formulation of the wind power ramp severity indices.

## Notation

| Symbol | Description |
|--------|-------------|
| $P(t_i)$ | Power at time $t_i$ (normalized 0-1) |
| $\Delta P$ | Ramp magnitude |
| $\Delta t$ | Ramp duration (hours) |
| $r_i$ | Instantaneous rate at step $i$ |
| $\bar{r}$ | Average rate |
| $n$ | Number of data points |

## Index Definitions

### 1. Ramp Acceleration Index (RAI)

**Formula:**
$$\text{RAI} = \frac{\max_i |r_{i+1} - r_i|}{\sigma(\{r_i\})}$$

**Where:**
- $r_i = P(t_{i+1}) - P(t_i)$ (instantaneous rate)
- $\sigma(\{r_i\})$ = standard deviation of instantaneous rates

**Interpretation:**
- Measures the maximum "jerk" (rate of change of rate)
- High RAI → sudden onset (step-like)
- Low RAI → gradual transition

**Properties:**
- Range: $[0, \infty)$, dimensionless
- Unique variance: 90.1%

---

### 2. Ramp Shape Complexity Index (RSCI)

**Formula:**
$$\text{RSCI} = \frac{L_{\text{actual}}}{L_{\text{straight}}} \times \left(1 + \frac{N_{\text{inflection}}}{n}\right)$$

**Where:**
- $L_{\text{actual}} = \sum_{i=0}^{n-1} \sqrt{1 + r_i^2}$ (path length)
- $L_{\text{straight}} = \sqrt{n^2 + (\Delta P)^2}$ (direct distance)
- $N_{\text{inflection}} = |\{i : \text{sign}(r_{i+1}) \neq \text{sign}(r_i)\}|$ (direction changes)

**Interpretation:**
- Measures deviation from ideal linear trajectory
- High RSCI → erratic, non-monotonic path
- RSCI = 1 for perfectly linear ramps

**Properties:**
- Range: $[1, \infty)$, dimensionless
- Unique variance: 54.0%

---

### 3. Operational Stress Index (OSI)

**Formula:**
$$\text{OSI} = \Delta P(1 + \bar{r}) \times (1 + R_{\text{reserve}}) \times (1 + \bar{r}^{1.5}) \times D(t_0)$$

**Reserve Risk (ASYMMETRIC):**
$$R_{\text{reserve}} = \begin{cases}
\exp(-3 \cdot P_{\text{end}}) & \text{if ramp-down} \\
\exp(-3 \cdot (1 - P_{\text{end}})) & \text{if ramp-up}
\end{cases}$$

**Demand Factor:**
$$D(t_0) = \begin{cases}
1.5 & t_0 \in [17, 21] \text{ (evening peak)} \\
1.3 & t_0 \in [7, 9] \text{ (morning peak)} \\
1.1 & t_0 \in [12, 14] \text{ (midday)} \\
0.8 & \text{otherwise}
\end{cases}$$

**Interpretation:**
- Down-ramps ending at low power → reserve depletion risk
- Up-ramps ending at high power → curtailment stress
- Peak hours amplify both effects

**Properties:**
- Asymmetric by design
- Unique variance: 21.3%

---

### 4. Grid Impact Potential (GIP)

**Formula:**
$$\text{GIP} = f_1 \times f_2 \times f_3$$

**Where:**
$$f_1 = \sqrt{\Delta P} \cdot \sqrt{\bar{r}} \cdot D_{\text{mod}}(\Delta t)$$
$$f_2 = (1 + 4(\bar{P} - 0.5)^2) \cdot T(t_0)$$
$$f_3 = A(\text{type}, P_{\text{end}}, t_0)$$

- $\bar{P} = (P_{\text{start}} + P_{\text{end}})/2$
- $D_{\text{mod}}$ penalizes very short (<3h) or very long (>15h) ramps
- $T(t_0)$ is a time-of-day factor

**Interpretation:**
- Multiplicative structure: severity requires multiple factors to align
- Square root transforms compress extremes

**Properties:**
- Context-dependent
- Unique variance: 33.8%

---

### 5. Enhanced Composite Severity Index (ECSI)

**Formula:**
$$\text{ECSI} = w_1 \cdot \widetilde{\text{RAI}} + w_2 \cdot \widetilde{\text{RSCI}} + w_3 \cdot \widetilde{\text{OSI}} + w_4 \cdot \widetilde{\text{GIP}}$$

**Where:**
- $\widetilde{X} = (X - X_{\min})/(X_{\max} - X_{\min})$ (min-max normalization)

**Weights (by unique variance):**
$$w_i = \frac{1 - R^2_i}{\sum_j (1 - R^2_j)}$$

| Component | Unique Variance | Weight |
|-----------|-----------------|--------|
| RAI | 0.901 | 0.452 |
| RSCI | 0.540 | 0.271 |
| GIP | 0.338 | 0.170 |
| OSI | 0.213 | 0.107 |

**Properties:**
- Range: [0, 1]
- Unique variance: 50.7% (15× improvement over conventional composites)

---

## Unique Variance Calculation

$$\text{Unique Variance} = 1 - R^2(\text{Index} \mid \Delta P, \bar{r}, \Delta t)$$

Where $R^2$ is the coefficient of determination from regressing the index on basic parameters.

- High unique variance → index captures novel information
- Low unique variance → index is redundant
- Threshold: 15% minimum for validation
