# load libraries
library(KRMM)
library(MASS)
library(kernlab)
library(cvTools)

# simulate data
set.seed(123)
p <- 100
n <- 200
beta <- rnorm(p, mean = 0, sd = 1.0) # random effects
X <- matrix(runif(p * n, min = 0, max = 1), ncol = p, byrow = T) # matrix of covariates
f <- tcrossprod(beta, X) # data generating process
eps <- rnorm(n, mean = 0, sd = 0.9) # add residuals
Y <- f + eps

# split data into training and test set
n_train <- floor(n * 0.67)
idx_train <- sample(1:n, size = n_train, replace = F)

# train
x_train <- X[idx_train, ]
y_train <- Y[idx_train]
length(y_train)

# test
x_test <- X[-idx_train, ]
y_test <- Y[-idx_train]
f_test <- f[-idx_train] # true value generated by DGP we want to predict
length(y_test)

# train and predict with krmm linear kernel
linear_krmm_model <- krmm(
  Y = y_train, Matrix_covariates = x_train,
  method = "RR-BLUP"
)

# without fixed effects
f_hat_test <- predict_krmm(linear_krmm_model,
  Matrix_covariates = x_test
)

dev.new()
plot(f_hat_test, f_test,
  main = "Linear RKHS regression with default rate of decay (not optimized)"
)
cor(f_hat_test, f_test)

# add fixed effects
f_hat_test <- predict_krmm(linear_krmm_model,
  Matrix_covariates = x_test, add_flxed_effects = T
)

dev.new()
plot(f_hat_test, f_test,
  main = "Linear RKHS regression with default rate of decay (not optimized)"
)
cor(f_hat_test, f_test)

# train and predict with krmm gaussian kernel (default kernel for RKHS method)
non_linear_krmm_model <- krmm(
  Y = y_train, Matrix_covariates = x_train,
  method = "RKHS"
)

# without fixed effects
f_hat_test <- predict_krmm(non_linear_krmm_model,
  Matrix_covariates = x_test
)

dev.new()
plot(f_hat_test, f_test,
  main = "Gaussian RKHS regression with default rate of decay (not optimized)"
)
cor(f_hat_test, f_test)

# add fixed effects
f_hat_test <- predict_krmm(non_linear_krmm_model,
  Matrix_covariates = x_test, add_flxed_effects = T
)

dev.new()
plot(f_hat_test, f_test,
  main = "Gaussian RKHS regression with default rate of decay (not optimized)"
)
cor(f_hat_test, f_test)

\dontrun{

# tune krmm model with a gaussian kernel
non_linear_opt_krmm_obj <- tune_krmm(
  Y = y_train, Matrix_covariates = x_train,
  rate_decay_grid = seq(0.01, 0.1, length.out = 5), nb_folds = 3,
  method = "RKHS"
)
non_linear_opt_krmm_obj$optimal_h

plot(non_linear_opt_krmm_obj$rate_decay_grid,
  non_linear_opt_krmm_obj$mean_loss_grid,
  type = "l"
)

non_linear_opt_krmm_model <- non_linear_opt_krmm_obj$optimized_model

# add fixed effects
f_hat_test <- predict_krmm(non_linear_opt_krmm_model,
  Matrix_covariates = x_test, add_flxed_effects = T
)
dev.new()
plot(f_hat_test, f_test,
  main = "Gaussian RKHS regression with optimized rate of decay"
)
cor(f_hat_test, f_test)

}
