\name{tune_krmm}
\alias{tune_krmm}
\encoding{utf8}

\title{
  Tune Kernel Ridge Regression in the Mixed Model Framework
}

\description{
  The \code{tune_krmm} function tunes the rate of decay parameter of kernels for kernel ridge regression using K-folds cross-validation.
}

\usage{
  tune_krmm(Y, X = rep(1, length(Y)), Z = diag(1, length(Y)),
             Matrix_covariates, method = "RKHS", kernel = "Gaussian", rate_decay_kernel = 0.1,
             degree_poly = 2, scale_poly = 1, offset_poly = 1, degree_anova = 3,
             init_sigma2K = 2, init_sigma2E = 3, convergence_precision = 1e-8,
             nb_iter = 1000, display = FALSE,
             rate_decay_grid = seq(0.1, 1.0, length.out = 10), nb_folds = 5, loss = "mse")
}

\arguments{
  \item{Y}{numeric vector; response vector}
  \item{X}{numeric matrix; design matrix of predictors with fixed effects (default is a vector of ones)}
  \item{Z}{numeric matrix; design matrix of predictors with random effects (default is identity matrix)}
  \item{Matrix_covariates}{numeric matrix; entries used to build the kernel matrix}
  \item{method}{character string; RKHS, GBLUP, or RR-BLUP}
  \item{kernel}{character string; Gaussian, Laplacian, or ANOVA}
  \item{rate_decay_grid}{grid over which the rate of decay is tuned by K-folds cross-validation}
  \item{nb_folds}{number of folds for cross-validation (default is 5)}
  \item{loss}{loss function to optimize; "mse" for mean square error or "cor" for correlation (default is "mse")}
  \item{rate_decay_kernel}{numeric scalar; hyperparameter of the kernel (default is 0.1)}
  \item{degree_poly, scale_poly, offset_poly}{numeric scalars; parameters for polynomial kernel (defaults are 2, 1, and 1)}
  \item{degree_anova}{numeric scalar; parameter for ANOVA kernel (default is 3)}
  \item{init_sigma2K, init_sigma2E}{numeric scalars; initial guess values for variance parameters in the EM-REML algorithm}
  \item{convergence_precision, nb_iter}{numeric scalars; convergence precision and maximum iterations for the EM-REML algorithm}
  \item{display}{boolean; display estimated components at each iteration}
}

\value{
  \item{optimized_model}{the tuned model (a \code{krmm} object)}
  \item{mean_loss_grid}{average loss for each rate of decay tested over the grid}
  \item{optimal_h}{rate of decay minimizing the average loss}
}

\author{
  Laval Jacquin

  Maintainer: Laval Jacquin <jacquin.julien@gmail.com>
}

\examples{
# load libraries
library(KRMM)

# simulate data
set.seed(123)
p <- 500
n <- 300
gamma <- rnorm(p, mean = 0, sd = 0.5)
M <- matrix(runif(p * n, min = 0, max = 1), ncol = p, byrow = T)  # matrix of covariates
f <- tcrossprod(gamma, M)                                         # data generating process
eps <- rnorm(n, mean = 0, sd = 0.1)                               # add residuals
Y <- f + eps                                                      # data generating process (DGP)

# split data into training and test set
n_train <- floor(n * 0.67)
idx_train <- sample(1:n, size = n_train, replace = F)

# train
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
f_hat_test <- predict_krmm(linear_krmm_model, Matrix_covariates = M_test, add_flxed_effects = T)
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
f_hat_test <- predict_krmm(non_linear_krmm_model, Matrix_covariates = M_test, add_flxed_effects = T)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with fixed effects added,
     and default rate of decay (not optimized)")
cor(f_hat_test, f_test)

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
f_hat_test <- predict_krmm(non_linear_opt_krmm_model, Matrix_covariates = M_test, add_flxed_effects = F)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with optimized rate of decay")
cor(f_hat_test, f_test)

# with added fixed effects
f_hat_test <- predict_krmm(non_linear_opt_krmm_model, Matrix_covariates = M_test, add_flxed_effects = T)
dev.new()
plot(f_hat_test, f_test, main = "Gaussian RKHS regression with optimized rate of decay")
cor(f_hat_test, f_test)
}
