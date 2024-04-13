tune_krmm <-
  function(Y, X = rep(1, length(Y)), Z = diag(1, length(Y)),
           Matrix_covariates, method = "RKHS", kernel = "Gaussian", rate_decay_kernel = 0.1,
           degree_poly = 2, scale_poly = 1, offset_poly = 1, degree_anova = 3,
           init_sigma2K = 2, init_sigma2E = 3, convergence_precision = 1e-8, nb_iter = 1000, display = F,
           rate_decay_grid = seq(0.1, 1.0, length.out = 10), nb_folds = 5, loss = "mse") {

    # center and scale matrix of covariates
    Matrix_covariates_scaled <- scale(Matrix_covariates,
      center = T, scale = T
    )

    # based on the folds, compute a vector of losses for each h
    n <- length(Y)
    idx_pop_ <- 1:n
    folds_ <- cvFolds(n, nb_folds, type = "consecutive")
    vect_mean_loss_grid <- rep(0, length(rate_decay_grid))
    l <- 1
    for (h in rate_decay_grid)
    {
      vect_loss_fold_ <- rep(0, nb_folds)
      for (fold_ in 1:nb_folds)
      {
        # get validation set
        idx_valid_ <- which(folds_$which == fold_)
        y_valid_ <- Y[idx_valid_]
        Matrix_covariates_valid_ <- Matrix_covariates_scaled[idx_valid_, ]

        # get training set
        idx_train_ <- idx_pop_[-idx_valid_]
        y_train_ <- Y[idx_train_]
        if (is.data.frame(X)) {
          x_train_ <- X[idx_train_, ]
        } else {
          x_train_ <- X[idx_train_]
        }
        z_train <- Z[idx_train_, idx_train_]
        Matrix_covariates_train_ <- Matrix_covariates_scaled[idx_train_, ]

        # build krmm model from training data and predict values for validation set
        model_krmm_ <- krmm(
          Y = y_train_, X = x_train_,
          Z = z_train, Matrix_covariates = Matrix_covariates_train_,
          method, kernel, rate_decay_kernel = h, degree_poly,
          scale_poly, offset_poly, degree_anova, init_sigma2K, init_sigma2E,
          convergence_precision, nb_iter, display
        )
        f_hat_valid_ <- predict_krmm(model_krmm_, Matrix_covariates_valid_)

        # compute loss for each fold_
        if (identical(loss, "mse")) {
          vect_loss_fold_[fold_] <- mean((f_hat_valid_ - y_valid_)^2)
        } else {
          vect_loss_fold_[fold_] <- 0.5 * (1 - cor(f_hat_valid_, y_valid_))
        }
      }
      # compute mean loss over the folds for h
      vect_mean_loss_grid[l] <- mean(vect_loss_fold_)
      l <- l + 1
    }
    # get the optimal h
    optimal_h <- rate_decay_grid[which.min(vect_mean_loss_grid)]

    # compute the tuned model for the optimal_h
    optimized_model <- krmm(Y, X, Z, Matrix_covariates, method, kernel,
      rate_decay_kernel = optimal_h, degree_poly, scale_poly, offset_poly,
      degree_anova, init_sigma2K, init_sigma2E,
      convergence_precision, nb_iter, display
    )

    return(list(
      "optimized_model" = optimized_model, "optimal_h" = optimal_h, "loss" = loss,
      "mean_loss_grid" = vect_mean_loss_grid, "rate_decay_grid" = rate_decay_grid
    ))
  }
