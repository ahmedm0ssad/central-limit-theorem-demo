# Central Limit Theorem Demonstration 📊

A comprehensive demonstration of the Central Limit Theorem using Monte Carlo simulations with exponential distributions. This project shows how sample means converge to normality regardless of the underlying population distribution.

## 🎯 Project Overview

This simulation demonstrates one of statistics' most fundamental theorems - the **Central Limit Theorem (CLT)**. Through Monte Carlo simulations, we show how the sampling distribution of means approaches normality as sample size increases, even when the original population follows a highly skewed exponential distribution.

## 🚀 Quick Start

### Python Version
```bash
# Install dependencies
pip install -r requirements.txt

# Run demonstration
python clt_demonstration.py
```

### R Version
```bash
# Install R packages (run once)
Rscript install_packages.R

# Run demonstration
Rscript clt_demonstration.R
```

## 📊 What This Simulation Does

- **Simulates 1,500 samples** from an exponential distribution (λ = 0.5, μ = 2, σ = 2)
- **Tests three sample sizes**: n = 10, 50, and 100
- **Creates comprehensive visualizations** showing convergence to normality
- **Generates statistical summaries** with theoretical comparisons
- **Saves all results** in the `outputs/` folder for further analysis

## 📁 Project Structure

```
central-limit-theorem-demo/
├── clt_demonstration.py         # Python implementation
├── clt_demonstration.R          # R implementation
├── install_packages.R           # R package installer
├── requirements.txt             # Python dependencies
├── outputs/                     # Generated results
│   ├── clt_demonstration_plots.png
│   ├── clt_simulation_results.npz
│   └── clt_simulation_results.RData
├── README.md                    # This file
├── LICENSE                      # MIT license
└── .gitignore                   # Git ignore rules
```

## 🔬 Key Findings

The simulation demonstrates three critical aspects of the Central Limit Theorem:

### 1. 🎯 Normality Convergence
Sample mean distributions become increasingly normal as sample size increases, regardless of the original exponential distribution's skewness.

### 2. 📈 Mean Convergence  
All sample means converge to the true population mean (μ = 2), demonstrating unbiased estimation.

### 3. 📉 Variance Reduction
Standard deviation of sample means decreases proportionally to 1/√n, showing increased precision with larger samples.

## 📈 Expected Results

| Sample Size (n) | Theoretical SE | Distribution Shape | Convergence Quality |
|-----------------|----------------|-------------------|-------------------|
| 10              | 0.632          | Somewhat normal   | ✅ Good           |
| 50              | 0.283          | Very normal       | ✅ Better         |
| 100             | 0.200          | Highly normal     | ✅ Best           |

*Standard Error (SE) = σ/√n = 2/√n*

## 🛠️ Technical Requirements

### Python Dependencies
- **numpy**: Numerical computations and random sampling
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization
- **scipy**: Statistical functions and tests

### R Dependencies  
- **ggplot2**: Advanced data visualization
- **gridExtra**: Multi-panel plot arrangements
- **grid**: Low-level graphics functions

## 📚 Educational Applications

This project is perfect for:
- 🎓 **Statistics Students**: Visual understanding of CLT fundamentals
- 📊 **Data Science Learning**: Practical application of theoretical concepts  
- 👨‍🏫 **Educators**: Teaching statistical inference with concrete examples
- 🔬 **Researchers**: Reference implementation for simulation studies

## 🎨 Visualization Features

- **Histogram overlays** with theoretical normal curves
- **Density estimation** using kernel methods
- **Multi-panel layouts** for sample size comparison
- **Statistical annotations** with means and standard deviations
- **Professional styling** ready for presentations

## 🔧 Methodology

- **Population Distribution**: Exponential(λ = 0.5)
- **Population Parameters**: μ = 2, σ = 2
- **Sample Sizes**: n ∈ {10, 50, 100}
- **Monte Carlo Runs**: 1,500 simulations per sample size
- **Reproducibility**: Fixed random seeds for consistent results

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Ahmed Mosad**  
*Statistical Inference DSAI 307*  
*June 2025*

---

⭐ *"The Central Limit Theorem is not just a mathematical curiosity—it's the foundation that makes statistical inference possible."*

🎯 **Understanding statistics through simulation and visualization**
