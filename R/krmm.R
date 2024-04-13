# kernel Ridge regression (aka RKHS regression)
krmm <-
  function(Y, X = rep(1, length(Y)), Z = diag(1, length(Y)),
           Matrix_covariates, method = "RKHS", kernel = "Gaussian",
           rate_decay_kernel = 0.1, degree_poly = 2, scale_poly = 1,
           offset_poly = 1, degree_anova = 3,
           init_sigma2K = 2, init_sigma2E = 3, convergence_precision = 1e-08,
           nb_iter = 1000, display = F) {

    # get number of covariates (i.e. features)
    Matrix_covariates <- scale(Matrix_covariates, center = T, scale = T)
    covariates_center <- attr(Matrix_covariates, "scaled:center")
    covariates_scale <- attr(Matrix_covariates, "scaled:scale")
    p <- ncol(Matrix_covariates)

    # get defined method
    if (identical(method, "RKHS")) {
      # gaussian kernel
      if (identical(kernel, "Gaussian")) {
        kernel_function <- rbfdot(sigma = (1 / p) * rate_decay_kernel)

        # laplacian kernel
      } else if (identical(kernel, "Laplacian")) {
        kernel_function <- laplacedot(sigma = (1 / p) * rate_decay_kernel)

        # polynomial kernel
      } else if (identical(kernel, "Polynomial")) {
        kernel_function <- polydot(
          degree = degree_poly,
          scale = scale_poly, offset = offset_poly
        )

        # anova kernel
      } else if (identical(kernel, "ANOVA")) {
        kernel_function <- anovadot(
          sigma = (1 / p) * rate_decay_kernel,
          degree = degree_anova
        )
      }
      K <- kernelMatrix(kernel_function, Matrix_covariates)
    } else {
      # special case : linear kernel, i.e. rr-blup and gblup
      if (identical(method, "GBLUP") || identical(method, "RR-BLUP")) {
        K <- crossprod(t(Matrix_covariates))
      }
    }
    n <- length(Y)
    K_inv <- ginv(K)

    MM_components_solved <- em_reml_mm(
      K_inv, Y, X, Z, init_sigma2K, init_sigma2E,
      convergence_precision, nb_iter, display
    )
    beta_hat <- as.vector(MM_components_solved$beta_hat)
    sigma2K_hat <- as.vector(MM_components_solved$sigma2K_hat)
    sigma2E_hat <- as.vector(MM_components_solved$sigma2E_hat)
    lambda <- (sigma2E_hat / sigma2K_hat)
    var_y_div_sig2_alpha <- crossprod(Z, crossprod(K, t(Z))) + lambda * diag(1, n)

    vect_alpha <- crossprod(
      crossprod(t(Z), ginv(var_y_div_sig2_alpha)),
      Y - crossprod(t(X), beta_hat)
    )

    if (identical(method, "RKHS") || identical(method, "GBLUP")) {
      return(list(
        "Matrix_covariates" = Matrix_covariates,
        "beta_hat" = beta_hat,
        "sigma2K_hat" = sigma2K_hat, "sigma2E_hat" = sigma2E_hat,
        "vect_alpha" = vect_alpha,
        "method" = method, "kernel" = kernel,
        "rate_decay_kernel" = rate_decay_kernel,
        "degree_anova" = degree_anova,
        "degree_poly" = degree_poly, "scale_poly" = scale_poly,
        "offset_poly" = offset_poly,
        "covariates_center" = covariates_center,
        "covariates_scale" = covariates_scale
      ))
    } else if (identical(method, "RR-BLUP")) {
      gamma_hat <- crossprod(Matrix_covariates, vect_alpha)
      return(list(
        "Matrix_covariates" = Matrix_covariates,
        "beta_hat" = beta_hat, "gamma_hat" = gamma_hat,
        "sigma2K_hat" = sigma2K_hat, "sigma2E_hat" = sigma2E_hat,
        "vect_alpha" = vect_alpha,
        "method" = method,
        "covariates_center" = covariates_center,
        "covariates_scale" = covariates_scale
      ))
    }
  }
