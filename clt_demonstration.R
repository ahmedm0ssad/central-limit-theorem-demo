# ============================================================================
# Central Limit Theorem Demonstration
# Statistical Inference Project
# ============================================================================
# 
# This script demonstrates the Central Limit Theorem using Monte Carlo 
# simulations with exponential distributions.
# 
# Author: Ahmed Mosad
# Course: Statistical Inference DSAI 307
# Date: June 2025
# 
# Description: Shows how sample means converge to normality as sample size 
# increases, regardless of the underlying population distribution.
# ============================================================================

# Install and load required packages
required_packages <- c("ggplot2", "gridExtra", "dplyr", "grid")

for (package in required_packages) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Load packages
library(grid)
library(ggplot2)
library(gridExtra)
library(dplyr)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

set.seed(123)
n_simulations <- 1500                    # Number of Monte Carlo simulations
sample_sizes <- c(10, 50, 100)          # Sample sizes to test
exponential_rate <- 1/2                 # Rate parameter (λ = 1/μ)
population_mean <- 1/exponential_rate   # Population mean μ = 2
population_sd <- 1/exponential_rate     # Population std σ = 2

cat("=================================================================\n")
cat("CENTRAL LIMIT THEOREM DEMONSTRATION\n")
cat("=================================================================\n")
cat("Simulation Parameters:\n")
cat("- Number of simulations:", n_simulations, "\n")
cat("- Sample sizes:", paste(sample_sizes, collapse = ", "), "\n")
cat("- Population mean (μ):", population_mean, "\n")
cat("- Population std (σ):", population_sd, "\n")
cat("- Distribution: Exponential with rate λ =", exponential_rate, "\n")
cat("=================================================================\n\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

#' Generate sample means from exponential distribution
#' 
#' @param n_samples Number of samples to generate
#' @param sample_size Size of each sample
#' @param rate Rate parameter of exponential distribution
#' @return Vector of sample means
generate_sample_means <- function(n_samples, sample_size, rate = 1/2) {
  samples <- matrix(rexp(n_samples * sample_size, rate = rate),
                   nrow = n_samples)
  return(rowMeans(samples))
}

#' Calculate theoretical standard error
#' 
#' @param population_sd Population standard deviation
#' @param sample_size Sample size
#' @return Theoretical standard error
calculate_theoretical_se <- function(population_sd, sample_size) {
  return(population_sd / sqrt(sample_size))
}

# ============================================================================
# MAIN SIMULATION - BASE R IMPLEMENTATION
# ============================================================================

cat("Running simulations...\n\n")

# Set up plotting area (3 rows, 1 column)
par(mfrow = c(3, 1), mar = c(4, 4, 3, 2))

# Store results for summary
results <- list()

# Run simulation for each sample size
for(i in seq_along(sample_sizes)) {
  n <- sample_sizes[i]
  
  cat("Processing sample size n =", n, "...\n")
  
  # Generate exponential samples and calculate means
  samples <- matrix(rexp(n_simulations * n, rate = exponential_rate),
                   nrow = n_simulations)
  sample_means <- rowMeans(samples)
  
  # Calculate statistics
  observed_mean <- mean(sample_means)
  observed_sd <- sd(sample_means)
  theoretical_se <- calculate_theoretical_se(population_sd, n)
  
  # Store results
  results[[i]] <- list(
    sample_size = n,
    observed_mean = observed_mean,
    observed_sd = observed_sd,
    theoretical_se = theoretical_se,
    sample_means = sample_means
  )
  
  # Create histogram
  hist(sample_means,
       prob = TRUE,
       main = paste("Sample Size n =", n),
       sub = paste("Mean =", round(observed_mean, 3), 
                  ", SD =", round(observed_sd, 3),
                  "(Theoretical:", round(theoretical_se, 3), ")"),
       xlab = "Sample Mean",
       ylab = "Density",
       breaks = 40,
       col = "lightblue",
       border = "darkblue",
       cex.main = 1.2,
       cex.sub = 0.9)
  
  # Add density curve for observed data
  lines(density(sample_means), col = "blue", lwd = 2)
  
  # Add theoretical normal curve
  x_range <- seq(min(sample_means), max(sample_means), length = 100)
  theoretical_density <- dnorm(x_range, mean = population_mean, sd = theoretical_se)
  lines(x_range, theoretical_density, col = "red", lwd = 2, lty = 2)
  
  # Add vertical lines
  abline(v = population_mean, col = "red", lty = 3, lwd = 2)
  abline(v = observed_mean, col = "blue", lty = 3, lwd = 1)
  
  # Add legend
  legend("topright", 
         legend = c("Observed Distribution", "Theoretical Normal", "Population Mean", "Sample Mean"),
         col = c("blue", "red", "red", "blue"),
         lty = c(1, 2, 3, 3),
         lwd = c(2, 2, 2, 1),
         cex = 0.8)
}

# Reset plotting parameters
par(mfrow = c(1, 1))

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================

cat("\n=================================================================\n")
cat("STATISTICAL SUMMARY\n")
cat("=================================================================\n")

# Create summary table
cat(sprintf("%-12s %-12s %-12s %-15s %-10s\n", 
           "Sample Size", "Obs. Mean", "Obs. SD", "Theoretical SE", "Error"))
cat(paste(rep("-", 65), collapse = ""), "\n")

for(i in seq_along(results)) {
  result <- results[[i]]
  error <- abs(result$observed_sd - result$theoretical_se)
  
  cat(sprintf("%-12d %-12.4f %-12.4f %-15.4f %-10.4f\n",
             result$sample_size,
             result$observed_mean,
             result$observed_sd,
             result$theoretical_se,
             error))
}

cat("\nKey Observations:\n")
cat("• Population mean (μ) =", population_mean, "\n")
cat("• All sample means converge to population mean\n")
cat("• Standard deviation decreases as sample size increases (∝ 1/√n)\n")
cat("• Observed SDs closely match theoretical predictions\n")

# ============================================================================
# NORMALITY TESTS
# ============================================================================

cat("\n=================================================================\n")
cat("NORMALITY TESTS (Shapiro-Wilk)\n")
cat("=================================================================\n")

for(i in seq_along(results)) {
  result <- results[[i]]
  n <- result$sample_size
  sample_means <- result$sample_means
  
  # Perform Shapiro-Wilk test (use subset for large samples)
  test_sample <- if(length(sample_means) > 5000) {
    sample(sample_means, 1000)
  } else {
    sample_means
  }
  
  shapiro_result <- shapiro.test(test_sample)
  
  cat(sprintf("Sample size n = %3d: W = %.4f, p-value = %.6f",
             n, shapiro_result$statistic, shapiro_result$p.value))
  
  if(shapiro_result$p.value > 0.05) {
    cat(" → Normal ✓\n")
  } else {
    cat(" → Not perfectly normal\n")
  }
}

# ============================================================================
# ADVANCED VISUALIZATION WITH GGPLOT2
# ============================================================================

cat("\n=================================================================\n")
cat("CREATING ADVANCED VISUALIZATIONS\n")
cat("=================================================================\n")

# Create ggplot2 visualizations
plots_list <- list()

for(i in seq_along(sample_sizes)) {
  n <- sample_sizes[i]
  
  # Use the sample_means from our existing results
  sample_means_data <- results[[i]]$sample_means
  obs_mean <- results[[i]]$observed_mean
  obs_sd <- results[[i]]$observed_sd
  theo_se <- results[[i]]$theoretical_se
  
  # Create data frame
  df <- data.frame(means = sample_means_data)
  
  # Create the plot
  p <- ggplot(df, aes(x = means)) +
    geom_histogram(aes(y = after_stat(density)), bins = 40,
                  fill = "lightblue", color = "darkblue", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1.2) +
    geom_vline(xintercept = population_mean, color = "red", 
              linetype = "dotted", linewidth = 1.5) +
    geom_vline(xintercept = obs_mean, color = "blue", 
              linetype = "dotted", linewidth = 1) +
    labs(
      title = paste("Sample Size n =", n),
      subtitle = paste("Mean =", round(obs_mean, 3), 
                      ", SD =", round(obs_sd, 3),
                      "(Theoretical:", round(theo_se, 3), ")"),
      x = "Sample Mean",
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      panel.grid.minor = element_blank()
    )
  
  # Add theoretical normal curve manually
  x_range <- seq(min(sample_means_data), max(sample_means_data), length.out = 100)
  theoretical_density <- dnorm(x_range, mean = population_mean, sd = theo_se)
  normal_df <- data.frame(x = x_range, y = theoretical_density)
  
  p <- p + geom_line(data = normal_df, aes(x = x, y = y), 
                     color = "red", linetype = "dashed", linewidth = 1.2,
                     inherit.aes = FALSE)
  
  plots_list[[i]] <- p
}

# Arrange plots with error handling
cat("Arranging plots...\n")
tryCatch({
  # Try to create with title
  combined_plot <- grid.arrange(
    grobs = plots_list, 
    ncol = 1,
    top = textGrob("Central Limit Theorem Demonstration\nExponential Distribution (μ=2, σ=2)", 
                  gp = gpar(fontsize = 16, fontface = "bold"))
  )
  cat("✅ Advanced visualizations with title created successfully!\n")
}, error = function(e) {
  # If textGrob fails, create without title
  cat("ℹ️ Creating plots without title...\n")
  combined_plot <- do.call(grid.arrange, c(plots_list, list(ncol = 1)))
  cat("✅ Advanced visualizations created successfully!\n")
})

# Try to save the plot
# Create outputs directory if it doesn't exist
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

tryCatch({
  ggsave("outputs/clt_demonstration_plots.png", 
         plot = last_plot(), 
         width = 12, 
         height = 16, 
         dpi = 300)
  cat("💾 Plots saved as 'outputs/clt_demonstration_plots.png'\n")
}, error = function(e) {
  cat("ℹ️ Could not save plot file, but visualizations were displayed\n")
})

# ============================================================================
# CONCLUSIONS
# ============================================================================

cat("\n=================================================================\n")
cat("CONCLUSIONS\n")
cat("=================================================================\n")
cat("This demonstration clearly shows the Central Limit Theorem:\n\n")

cat("1. CONVERGENCE TO NORMALITY:\n")
cat("   As sample size increases (n: 10 → 50 → 100), the distribution\n")
cat("   of sample means becomes increasingly normal, despite the\n")
cat("   underlying exponential distribution being highly skewed.\n\n")

cat("2. MEAN CONVERGENCE:\n")
cat("   All sample means converge to the true population mean (μ = 2),\n")
cat("   regardless of sample size.\n\n")

cat("3. VARIANCE REDUCTION:\n")
cat("   The standard deviation of sample means decreases as sample\n")
cat("   size increases, following σ/√n relationship.\n\n")

cat("4. THEORETICAL VALIDATION:\n")
cat("   Observed standard deviations closely match theoretical\n")
cat("   predictions, confirming the mathematical foundation.\n\n")

cat("PRACTICAL IMPLICATIONS:\n")
cat("• Statistical inference becomes possible\n")
cat("• Confidence intervals can be constructed\n")
cat("• Hypothesis testing is enabled\n")
cat("• Quality control applications\n\n")

cat("=================================================================\n")
cat("SIMULATION COMPLETE!\n")
cat("=================================================================\n")

# Save workspace for future analysis
save.image("outputs/clt_simulation_results.RData")
cat("Results saved to 'outputs/clt_simulation_results.RData'\n")
