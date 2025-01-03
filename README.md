[<img src="img/krmm.png"/>]()
# KRMM: Kernel Ridge Mixed Model

##### Licence, status and metrics
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Lifecycle:Experimental](https://img.shields.io/badge/Lifecycle-Experimental-339999)]()
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![GitHub repo size](https://img.shields.io/github/repo-size/ljacquin/KRMM)
![GitHub language count](https://img.shields.io/github/languages/count/ljacquin/KRMM)
![GitHub top language](https://img.shields.io/github/languages/top/ljacquin/KRMM)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/ljacquin/KRMM)  
![GitHub all releases](https://img.shields.io/github/downloads/ljacquin/KRMM/total)
![GitHub stars](https://img.shields.io/github/stars/ljacquin/KRMM)  

##### Languages and technologies
[![R Badge](https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white)](https://www.r-project.org/)

## Overview

The KRMM package provides advanced tools for solving kernel ridge regression within the following mixed model framework:

$$
Y = X\beta + Zu + \varepsilon
$$

where $X$ and $Z$ are design matrices of predictors with fixed and random effects, respectively. The random effect $u$ follows a multivariate normal distribution $N_n(0, K_{\sigma^2_{K}})$, where $K$ is the genomic covariance matrix (also known as the Gram matrix) built using different kernels.

The package offers flexibility in kernel choice, including linear, polynomial, Gaussian, Laplacian, and ANOVA kernels. The RR-BLUP (Random Regression BLUP) or GBLUP (Genomic BLUP) ```method``` is associated with the linear kernel, while the RKHS (Reproducing Kernel Hilbert Space) ```method``` is associated with the other kernels, with the Gaussian kernel set as the default.

The package utilizes the expectation-maximization (EM) algorithm for estimating model components (fixed and random effects) and variance parameters. Additionally, it provides the ability to obtain estimates such as the BLUP of random effects for the covariates of the linear kernel, also known as RR-BLUP of random effects, and BLUP of dual variables within the kernel ridge regression context.


## Installation

You can install the latest version of the KRMM package with:

```R
install.packages("devtools")
library(devtools)
install_github("ljacquin/KRMM")
```

## Key Features

    Kernel Regression: Solving ridge regression with a variety of kernels.
    Mixed Models: Integrating fixed and random effects into the model.
    EM Algorithm: Utilizing the EM algorithm for parameter estimation.
    BLUP Estimates: Ability to obtain BLUP of dual variables and random predictor effects for the linear kernel (RR-BLUP).

## Main Functions

    krmm: Main function for fitting the KRMM model.
    predict_krmm: Predicting values for new data.
    tune_krmm: Tuning the kernel decay rate parameter through K-folds cross-validation.
    em_reml_mm: EM algorithm for restricted maximum likelihood estimation (REML) within the mixed model framework.

## Examples

### Fitting the KRMM Model

Here's a simple example illustrating the use of the krmm function:

```R
# load libraries
library(KRMM)

# simulate data
set.seed(123)
p <- 500                                                          # e.g. number of SNP markers
n <- 300                                                          # e.g. number of phenotypes for a trait
gamma <- rnorm(p, mean = 0, sd = 0.5)                             # e.g. random effects of SNP markers
M <- matrix(runif(p * n, min = 0, max = 1), ncol = p, byrow = T)  # matrix of covariates, e.g. genotype matrix
f <- tcrossprod(gamma, M)                                         # data generating process (DGP), e.g. true genetic model
eps <- rnorm(n, mean = 0, sd = 0.1)                               # add residuals
Y <- f + eps                                                      # e.g. measured phenotypes (i.e. genetic values 
                                                                  # + residuals)
# split data into training and test set
n_train <- floor(n * 0.67)
idx_train <- sample(1:n, size = n_train, replace = F)

# get train data
M_train <- M[idx_train, ]
y_train <- Y[idx_train]

# train krmm with linear kernel (i.e. dot product)
linear_krmm_model <- krmm(Y = y_train, Matrix_covariates = M_train, method = "RR-BLUP")

summary(linear_krmm_model)
print(linear_krmm_model$beta_hat)
hist(linear_krmm_model$gamma_hat)
hist(linear_krmm_model$vect_alpha)

# train krmm with non linear gaussian kernel (gaussian is the default kernel for RKHS method)
non_linear_krmm_model <- krmm(Y = y_train, Matrix_covariates = M_train, method = "RKHS")

summary(non_linear_krmm_model)
print(non_linear_krmm_model$beta_hat)
hist(non_linear_krmm_model$vect_alpha)
```

### Making Predictions with predict_krmm

Here's an example of how to use the predict_krmm function to make predictions on new data:

```R
# get test data from matrix of covariates for prediction
M_test <- M[-idx_train, ]

# get unknown true value generated by DGP we want to predict using predict_krmm
f_test <- f[-idx_train]

# -- prediction with linear kernel

# without fixed effects
f_hat_test <- predict_krmm(linear_krmm_model, Matrix_covariates = M_test)
dev.new()
plot(f_hat_test, f_test, main = "Linear RKHS regression without fixed effects")
cor(f_hat_test, f_test)

# with added fixed effects
f_hat_test <- predict_krmm(linear_krmm_model, Matrix_covariates = M_test, add_fixed_effects = T)
dev.new()
plot(f_hat_test, f_test, main = "Linear RKHS regression with fixed effects added")
cor(f_hat_test, f_test)

# -- prediction with non linear gaussian kernel

# without fixed effects
f_hat_test <- predict_krmm(non_linear_krmm_model, Matrix_covariates = M_test)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression without fixed effects,
     and default rate of decay (not optimized)")
cor(f_hat_test, f_test)

# with added fixed effects
f_hat_test <- predict_krmm(non_linear_krmm_model, Matrix_covariates = M_test, add_fixed_effects = T)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with fixed effects added,
     and default rate of decay (not optimized)")
cor(f_hat_test, f_test)
```

### Tuning the rate of decay of specific kernels with K-folds cross-validation 

Here's an example of how to use the tune_krmm function for tuning the rate of decay of Gaussian, Laplacian or ANOVA kernels (see documentation for these kernels):

```R
# -- tune krmm model with a gaussian kernel and make predictions for test data
non_linear_opt_krmm_obj <- tune_krmm(
  Y = y_train, Matrix_covariates = M_train,
  rate_decay_grid = seq(0.01, 0.1, length.out = 5), nb_folds = 3,
  method = "RKHS", kernel = "Gaussian",
)
print(non_linear_opt_krmm_obj$optimal_h)
plot(non_linear_opt_krmm_obj$rate_decay_grid,
     non_linear_opt_krmm_obj$mean_loss_grid, type = "l")

# get the optimized krmm model
non_linear_opt_krmm_model <- non_linear_opt_krmm_obj$optimized_model

# without fixed effects
f_hat_test <- predict_krmm(non_linear_opt_krmm_model, Matrix_covariates = M_test, add_fixed_effects = F)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with optimized rate of decay")
cor(f_hat_test, f_test)

# with added fixed effects
f_hat_test <- predict_krmm(non_linear_opt_krmm_model, Matrix_covariates = M_test, add_fixed_effects = T)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with optimized rate of decay")
cor(f_hat_test, f_test)
```

## Authors and References

* Author: Laval Jacquin
* Maintainer: Laval Jacquin jacquin.julien@gmail.com

## References

* Jacquin et al. (2016). A Unified and Comprehensible View of Parametric and Kernel Methods for Genomic Prediction with Application to Rice. Front. Genet. 7:145
* Robinson, G. K. (1991). That blup is a good thing: the estimation of random effects. Statistical Science, 534 15-32
* Foulley, J.-L. (2002). Algorithme em: théorie et application au modèle mixte. Journal de la Société française de Statistique 143, 57-109

