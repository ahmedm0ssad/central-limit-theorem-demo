#!/usr/bin/env python3
"""
Central Limit Theorem Demonstration - Python Implementation
===========================================================

This script demonstrates the Central Limit Theorem using Monte Carlo 
simulations with exponential distributions.

Author: Ahmed Mosad
Course: Statistical Inference DSAI 307
Date: June 2025

Description: Shows how sample means converge to normality as sample size 
increases, regardless of the underlying population distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation parameters
N_SIMULATIONS = 1500
SAMPLE_SIZES = [10, 50, 100]
EXPONENTIAL_SCALE = 2  # Mean of exponential distribution
POPULATION_MEAN = EXPONENTIAL_SCALE
POPULATION_STD = EXPONENTIAL_SCALE
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

print("=" * 65)
print("CENTRAL LIMIT THEOREM DEMONSTRATION - PYTHON")
print("=" * 65)
print(f"Simulation Parameters:")
print(f"- Number of simulations: {N_SIMULATIONS}")
print(f"- Sample sizes: {SAMPLE_SIZES}")
print(f"- Population mean (μ): {POPULATION_MEAN}")
print(f"- Population std (σ): {POPULATION_STD}")
print(f"- Distribution: Exponential with scale = {EXPONENTIAL_SCALE}")
print("=" * 65)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_sample_means(n_samples: int, sample_size: int, scale: float = 2) -> np.ndarray:
    """
    Generate sample means from exponential distribution.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    sample_size : int
        Size of each sample
    scale : float
        Scale parameter (mean) of exponential distribution
    
    Returns:
    --------
    np.ndarray
        Array of sample means
    """
    samples = np.random.exponential(scale, size=(n_samples, sample_size))
    return np.mean(samples, axis=1)


def calculate_theoretical_std(population_std: float, sample_size: int) -> float:
    """
    Calculate theoretical standard deviation of sample means.
    
    Parameters:
    -----------
    population_std : float
        Population standard deviation
    sample_size : int
        Sample size
    
    Returns:
    --------
    float
        Theoretical standard error
    """
    return population_std / np.sqrt(sample_size)


def perform_normality_test(sample_means: np.ndarray, test_size: int = 1000) -> Tuple[float, float]:
    """
    Perform Shapiro-Wilk normality test.
    
    Parameters:
    -----------
    sample_means : np.ndarray
        Array of sample means
    test_size : int
        Maximum size for test
    
    Returns:
    --------
    Tuple[float, float]
        Test statistic and p-value
    """
    test_sample = sample_means[:test_size] if len(sample_means) > test_size else sample_means
    return stats.shapiro(test_sample)

# ============================================================================
# MAIN SIMULATION
# ============================================================================

print("\nRunning simulations...\n")

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 16))
fig.suptitle('Central Limit Theorem Demonstration\\nSampling Distribution of Means from Exponential Distribution', 
             fontsize=16, fontweight='bold', y=0.98)

# Store results for analysis
results = {}

# Generate and plot for each sample size
for i, n in enumerate(SAMPLE_SIZES):
    print(f"Processing sample size n = {n}...")
    
    # Generate sample means
    sample_means = generate_sample_means(N_SIMULATIONS, n, EXPONENTIAL_SCALE)
    
    # Calculate statistics
    observed_mean = np.mean(sample_means)
    observed_std = np.std(sample_means, ddof=1)
    theoretical_std = calculate_theoretical_std(POPULATION_STD, n)
    
    # Store results
    results[n] = {
        'sample_means': sample_means,
        'observed_mean': observed_mean,
        'observed_std': observed_std,
        'theoretical_std': theoretical_std
    }
    
    # Create histogram
    axes[i].hist(sample_means, bins=40, density=True, alpha=0.7, 
                color='skyblue', edgecolor='darkblue', linewidth=0.5)
    
    # Add observed density curve
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    kde = stats.gaussian_kde(sample_means)
    axes[i].plot(x, kde(x), 'b-', linewidth=2, label='Observed Distribution')
    
    # Add theoretical normal curve
    theoretical_curve = stats.norm.pdf(x, POPULATION_MEAN, theoretical_std)
    axes[i].plot(x, theoretical_curve, 'r--', linewidth=2, label='Theoretical Normal')
    
    # Add reference lines
    axes[i].axvline(POPULATION_MEAN, color='red', linestyle=':', linewidth=2, 
                   label=f'Population Mean (μ = {POPULATION_MEAN})')
    axes[i].axvline(observed_mean, color='blue', linestyle=':', linewidth=1, 
                   label=f'Sample Mean ({observed_mean:.3f})')
    
    # Formatting
    axes[i].set_title(f'Sample Size n = {n}\\n'
                     f'Mean = {observed_mean:.3f}, Std = {observed_std:.3f} '
                     f'(Theoretical: {theoretical_std:.3f})', fontsize=12)
    axes[i].set_xlabel('Sample Mean')
    axes[i].set_ylabel('Density')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

print("\\n" + "=" * 65)
print("STATISTICAL SUMMARY")
print("=" * 65)

# Create summary table
summary_data = []
for n in SAMPLE_SIZES:
    result = results[n]
    # Perform normality test
    stat, p_value = perform_normality_test(result['sample_means'])
    
    summary_data.append({
        'Sample Size (n)': n,
        'Observed Mean': f"{result['observed_mean']:.4f}",
        'Observed Std': f"{result['observed_std']:.4f}",
        'Theoretical Std': f"{result['theoretical_std']:.4f}",
        'Error': f"{abs(result['observed_std'] - result['theoretical_std']):.4f}",
        'Shapiro W': f"{stat:.4f}",
        'p-value': f"{p_value:.6f}",
        'Normal?': 'Yes' if p_value > 0.05 else 'No'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print(f"\\nKey Observations:")
print(f"• Population mean (μ) = {POPULATION_MEAN}")
print(f"• All sample means converge to population mean")
print(f"• Standard deviation decreases as sample size increases (∝ 1/√n)")
print(f"• Observed standard deviations closely match theoretical values")

# ============================================================================
# DETAILED NORMALITY ANALYSIS
# ============================================================================

print("\\n" + "=" * 65)
print("NORMALITY TESTS (Shapiro-Wilk)")
print("=" * 65)

for n in SAMPLE_SIZES:
    result = results[n]
    sample_means = result['sample_means']
    
    # Perform normality test
    stat, p_value = perform_normality_test(sample_means)
    
    print(f"Sample Size n = {n:3d}: W = {stat:.4f}, p-value = {p_value:.6f}", end="")
    
    if p_value > 0.05:
        print(" → Distribution appears normal ✓")
    else:
        print(" → Distribution may not be perfectly normal")

print("\\nNote: As sample size increases, the distribution of sample means")
print("becomes more normal, demonstrating the Central Limit Theorem.")

# ============================================================================
# ADVANCED VISUALIZATIONS
# ============================================================================

print("\\n" + "=" * 65)
print("CREATING ADVANCED VISUALIZATIONS")
print("=" * 65)

# Create comparative visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Central Limit Theorem: Comprehensive Analysis', fontsize=16, fontweight='bold')

# Plot 1: Comparison of distributions
ax1 = axes[0, 0]
for n in SAMPLE_SIZES:
    sample_means = results[n]['sample_means']
    ax1.hist(sample_means, bins=30, alpha=0.5, density=True, label=f'n = {n}')
ax1.set_title('Distribution Comparison')
ax1.set_xlabel('Sample Mean')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Convergence of means
ax2 = axes[0, 1]
sample_sizes_extended = range(5, 101, 5)
means_convergence = []
for n in sample_sizes_extended:
    temp_means = generate_sample_means(500, n, EXPONENTIAL_SCALE)
    means_convergence.append(np.mean(temp_means))

ax2.plot(sample_sizes_extended, means_convergence, 'bo-', markersize=4)
ax2.axhline(y=POPULATION_MEAN, color='red', linestyle='--', label=f'Population Mean = {POPULATION_MEAN}')
ax2.set_title('Convergence of Sample Means')
ax2.set_xlabel('Sample Size')
ax2.set_ylabel('Mean of Sample Means')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Standard error comparison
ax3 = axes[1, 0]
sample_sizes_plot = SAMPLE_SIZES
observed_stds = [results[n]['observed_std'] for n in sample_sizes_plot]
theoretical_stds = [results[n]['theoretical_std'] for n in sample_sizes_plot]

x_pos = np.arange(len(sample_sizes_plot))
width = 0.35

ax3.bar(x_pos - width/2, observed_stds, width, label='Observed', alpha=0.7)
ax3.bar(x_pos + width/2, theoretical_stds, width, label='Theoretical', alpha=0.7)
ax3.set_title('Standard Deviation Comparison')
ax3.set_xlabel('Sample Size')
ax3.set_ylabel('Standard Deviation')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(sample_sizes_plot)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Q-Q plots for normality assessment
ax4 = axes[1, 1]
colors = ['red', 'green', 'blue']
for i, n in enumerate(SAMPLE_SIZES):
    sample_means = results[n]['sample_means'][:200]  # Use subset for clarity
    stats.probplot(sample_means, dist="norm", plot=ax4)
    ax4.get_lines()[-2].set_color(colors[i])
    ax4.get_lines()[-1].set_color(colors[i])
    ax4.get_lines()[-2].set_label(f'n = {n}')

ax4.set_title('Q-Q Plots for Normality')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STATISTICAL INSIGHTS
# ============================================================================

print("\\n" + "=" * 65)
print("STATISTICAL INSIGHTS")
print("=" * 65)

# Calculate effect sizes
print("Effect of Sample Size on Variability:")
for i, n in enumerate(SAMPLE_SIZES[:-1]):
    current_std = results[n]['observed_std']
    next_std = results[SAMPLE_SIZES[i+1]]['observed_std']
    reduction = ((current_std - next_std) / current_std) * 100
    print(f"• n = {n} to n = {SAMPLE_SIZES[i+1]}: {reduction:.1f}% reduction in std deviation")

# Theoretical vs observed comparison
print("\\nAccuracy of Theoretical Predictions:")
for n in SAMPLE_SIZES:
    result = results[n]
    accuracy = (1 - abs(result['observed_std'] - result['theoretical_std']) / result['theoretical_std']) * 100
    print(f"• n = {n}: {accuracy:.1f}% accuracy")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\\n" + "=" * 65)
print("CONCLUSIONS")
print("=" * 65)
print("This Python implementation demonstrates the Central Limit Theorem:")
print()
print("1. CONVERGENCE TO NORMALITY:")
print("   As sample size increases (n: 10 → 50 → 100), the distribution")
print("   of sample means becomes increasingly normal, despite the")
print("   underlying exponential distribution being highly skewed.")
print()
print("2. MEAN CONVERGENCE:")
print("   All sample means converge to the true population mean (μ = 2),")
print("   regardless of sample size.")
print()
print("3. VARIANCE REDUCTION:")
print("   The standard deviation of sample means decreases as sample")
print("   size increases, following the σ/√n relationship.")
print()
print("4. THEORETICAL VALIDATION:")
print("   Our observed standard deviations closely match theoretical")
print("   predictions, confirming the mathematical foundation of CLT.")
print()
print("PRACTICAL APPLICATIONS:")
print("• Quality control in manufacturing")
print("• Survey research and polling")
print("• Clinical trials and medical research")
print("• Financial risk assessment")
print("• A/B testing in data science")
print()
print("=" * 65)
print("SIMULATION COMPLETE!")
print("=" * 65)

# Save results
print("\\nSaving results...")
# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Convert integer keys to strings for np.savez
results_for_save = {f'n_{k}': v for k, v in results.items()}
np.savez('outputs/clt_simulation_results.npz', **results_for_save)
print("Results saved to 'outputs/clt_simulation_results.npz'")

if __name__ == "__main__":
    print("\\nCentral Limit Theorem demonstration completed successfully!")
    print("Check the generated plots and statistical summaries above.")
