tune_krmm <- function(Y, X = rep(1, length(Y)), Z = diag(1, length(Y)),
                      Matrix_covariates, method = "RKHS", kernel = "Gaussian",
                      rate_decay_kernel = 0.1,
                      degree_poly = 2, scale_poly = 1, offset_poly = 1, degree_anova = 3,
                      init_sigma2K = 2, init_sigma2E = 3,
                      convergence_precision = 1e-8, nb_iter = 1000, display = F,
                      rate_decay_grid = seq(0.01, 0.1, length.out = 3),
                      nb_folds = 3, loss = "mse") {
  # set exported functions and packages
  exported_funct_ <- c("krmm", "predict_krmm", "em_reml_mm")
  exported_pkgs_ <- c(
    "doParallel", "foreach", "kernlab", "MASS",
    "Matrix", "cvTools"
  )

  # detect number of cores and make cluster
  cl <- makeCluster(detectCores())
  registerDoParallel(cl)

  # center and scale matrix of covariates
  Matrix_covariates_scaled <- scale(Matrix_covariates,
    center = T, scale = T
  )

  # test if arguments are compliant
  n <- length(Y)
  nrow_mat_ <- nrow(Matrix_covariates_scaled)
  if (n > nrow_mat_) {
    warning(
      "
        The length of Y exceeds the number of rows in the covariate matrix.
        Please rebuild the model using 'krmm' with the optimal_h estimated
        from the 'tune_krmm' function.
        The optimal model from 'tune_krmm' will not be suitable.
        "
    )
    Y <- Y[1:nrow_mat_]
    X <- X[1:nrow_mat_]
    Z <- Z[1:nrow_mat_, 1:nrow_mat_]
    n <- nrow_mat_
  }
  idx_pop_ <- 1:n
  folds_ <- cvFolds(n, nb_folds, type = "consecutive")

  # based on the folds, compute a vector of losses for each h
  vect_mean_loss_grid <- foreach(
    h = rate_decay_grid,
    .export = exported_funct_,
    .packages = exported_pkgs_,
    .combine = "c"
  ) %dopar% {
    vect_loss_fold_ <- foreach(
      fold_ = 1:nb_folds,
      .export = exported_funct_,
      .packages = exported_pkgs_,
      .combine = "c"
    ) %dopar% {
      idx_valid_ <- which(folds_$which == fold_)
      y_valid_ <- Y[idx_valid_]
      Matrix_covariates_valid_ <- Matrix_covariates_scaled[idx_valid_, ]

      idx_train_ <- idx_pop_[-idx_valid_]
      y_train_ <- Y[idx_train_]
      x_train_ <- if (is.data.frame(X)) X[idx_train_, ] else X[idx_train_]
      z_train <- Z[idx_train_, idx_train_]
      Matrix_covariates_train_ <- Matrix_covariates_scaled[idx_train_, ]

      model_krmm_ <- krmm(
        Y = y_train_, X = x_train_,
        Z = z_train, Matrix_covariates = Matrix_covariates_train_,
        method, kernel, rate_decay_kernel = h, degree_poly,
        scale_poly, offset_poly, degree_anova, init_sigma2K, init_sigma2E,
        convergence_precision, nb_iter, display
      )
      f_hat_valid_ <- predict_krmm(model_krmm_, Matrix_covariates_valid_)

      if (identical(loss, "mse")) {
        mean((f_hat_valid_ - y_valid_)^2)
      } else {
        0.5 * (1 - cor(f_hat_valid_, y_valid_))
      }
    }
    mean(vect_loss_fold_)
  }

  # get the optimal h
  optimal_h <- rate_decay_grid[which.min(vect_mean_loss_grid)]

  # compute the tuned model for the optimal_h
  optimized_model <- krmm(Y, X, Z, Matrix_covariates, method, kernel,
    rate_decay_kernel = optimal_h, degree_poly, scale_poly, offset_poly,
    degree_anova, init_sigma2K, init_sigma2E,
    convergence_precision, nb_iter, display
  )

  # stop the parallel cluster
  stopCluster(cl)

  return(list(
    "optimized_model" = optimized_model, "optimal_h" = optimal_h, "loss" = loss,
    "mean_loss_grid" = vect_mean_loss_grid, "rate_decay_grid" = rate_decay_grid
  ))
}
