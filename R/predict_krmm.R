predict_krmm <-
  function(krmm_model, Matrix_covariates,
           X = rep(1, nrow(Matrix_covariates)),
           Z = diag(1, nrow(Matrix_covariates)),
           add_flxed_effects = F) {

    # center and scale target data
    Matrix_covariates <- scale(Matrix_covariates,
      center = krmm_model$covariates_center,
      scale = krmm_model$covariates_scale
    )
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
      K <- Matrix_covariates %*% t(krmm_model$Matrix_covariates)
    }

    # compute u_hat
    u_hat <- Z %*% K %*% krmm_model$vect_alpha

    #  compute f_hat if add_fixed_effect = TRUE
    if (add_flxed_effects) {
      beta_hat <- krmm_model$beta_hat
      if (length(beta_hat) > 1) {
        f_hat <- X %*% beta_hat + u_hat
      } else {
        f_hat <- X * beta_hat + u_hat
      }
      return(f_hat)
    } else {
      return(u_hat)
    }
  }
