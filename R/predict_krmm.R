predict_krmm <-
  function(krmm_model, Matrix_covariates,
           X = NULL,
           Z = NULL,
           add_flxed_effects = F) {
    # center and scale target data
    if (length(krmm_model$covariates_center) > 0 &&
      length(krmm_model$covariates_scale) > 0) {
      Matrix_covariates <- scale(Matrix_covariates,
        center = krmm_model$covariates_center,
        scale = krmm_model$covariates_scale
      )
    } else if (length(krmm_model$covariates_center) > 0) {
      Matrix_covariates <- scale(Matrix_covariates,
        center = krmm_model$covariates_center,
        scale = F
      )
    } else if (length(krmm_model$covariates_scale) > 0) {
      Matrix_covariates <- scale(Matrix_covariates,
        center = F,
        scale = krmm_model$covariates_scale
      )
    }
    # get number of observations and features
    n <- nrow(Matrix_covariates)
    p <- ncol(Matrix_covariates)

    if (identical(krmm_model$method, "RKHS")) {
      # gaussian kernel
      if (identical(krmm_model$kernel, "Gaussian")) {
        kernel_function <- rbfdot(sigma = (1 / p) * krmm_model$rate_decay_kernel)

        # laplacian kernel
      } else if (identical(krmm_model$kernel, "Laplacian")) {
        kernel_function <- laplacedot(sigma = (1 / p) * krmm_model$rate_decay_kernel)

        # polynomial kernel
      } else if (identical(krmm_model$kernel, "Polynomial")) {
        kernel_function <- polydot(
          degree = krmm_model$degree_poly, scale = krmm_model$scale_poly,
          offset = krmm_model$offset_poly
        )

        # anova kernel
      } else if (identical(krmm_model$kernel, "ANOVA")) {
        kernel_function <- anovadot(
          sigma = (1 / p) * krmm_model$rate_decay_kernel,
          degree = krmm_model$degree_anova
        )
      }
      K <- kernelMatrix(
        kernel_function, Matrix_covariates,
        krmm_model$Matrix_covariates
      )
    } else if (identical(krmm_model$method, "RR-BLUP") ||
      identical(krmm_model$method, "GBLUP")) {
      K <- tcrossprod(Matrix_covariates, krmm_model$Matrix_covariates)
    }

    # compute u_hat
    u_hat <- crossprod(t(K), krmm_model$vect_alpha)

    # get fixed effects from model
    beta_hat <- krmm_model$beta_hat
    n_beta <- length(beta_hat)

    #  compute f_hat if add_fixed_effect = TRUE
    if (add_flxed_effects) {
      if (is.null(X)) {
        X <- rep(1, n)
        if (n_beta > 1) {
          warning("
                  Multiple estimated fixed effects detected in the krmm model.
                  Please provide the design matrix X for the new data.
                  Otherwise, X will be interpreted as a matrix with multiple columns
                  of ones for the estimated fixed effects (one column per effect).
                  ")
          X <- matrix(1, nrow = n, ncol = n_beta)
        }
      }
      if (is.null(Z)) {
        Z <- diag(1, n)
      }

      if (n_beta > 1) {
        f_hat <- crossprod(t(X), beta_hat) + crossprod(t(Z), u_hat)
      } else {
        f_hat <- X * beta_hat + crossprod(t(Z), u_hat)
      }

      return(f_hat)
    } else {
      return(u_hat)
    }
  }
