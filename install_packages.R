# R Dependencies
# Install required R packages with this script

# Check if packages are installed, install if not
required_packages <- c("ggplot2", "gridExtra", "dplyr", "knitr", "rmarkdown")

install_if_missing <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}

# Install and load packages
sapply(required_packages, install_if_missing)

cat("All required R packages installed successfully!\n")
cat("Packages installed:\n")
cat(paste("-", required_packages, collapse = "\n"))
cat("\n")
