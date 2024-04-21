em_reml_mm <-
  function(Mat_K_inv, Y, X, Z, init_sigma2K, init_sigma2E, convergence_precision,
           nb_iter, display) {
    # build mme components
    n <- length(Y)
    XpX <- crossprod(X)
    XpZ <- crossprod(X, Z)
    ZpX <- crossprod(Z, X)
    ZpZ <- crossprod(Z)
    XpY <- crossprod(X, Y)
    ZpY <- crossprod(Z, Y)
    RHS <- c(XpY, ZpY)

    rankX <- qr(X, LAPACK = TRUE)$rank
    rankK <- qr(Mat_K_inv, LAPACK = TRUE)$rank
    nb_rows_mme_ <- length(RHS)
    l <- nb_rows_mme_ - nrow(ZpZ) + 1

    # initialize parameters
    old_sigma2E <- init_sigma2E
    old_sigma2K <- init_sigma2K
    precision1 <- 1
    precision2 <- 1
    i <- 0

    # iterate over variance components through EM steps
    while (precision1 > convergence_precision && precision2 > convergence_precision) {
      i <- i + 1

      lambda <- as.vector((old_sigma2E / old_sigma2K))
      LHS <- rbind(cbind(XpX, XpZ), cbind(ZpX, ZpZ + Mat_K_inv * lambda))

      ginv_LHS <- ginv(LHS)

      gamma <- crossprod(ginv_LHS, RHS)
      eps <- Y - crossprod(t(cbind(X, Z)), gamma)

      Cuu <- ginv_LHS[l:nb_rows_mme_, l:nb_rows_mme_]
      u <- gamma[l:length(gamma)]

      new_sigma2E <- crossprod(eps, Y) / (n - rankX)
      new_sigma2K <- (crossprod(u, crossprod(Mat_K_inv, u)) +
        sum(Mat_K_inv * Cuu) * old_sigma2E) / rankK

      precision1 <- abs(new_sigma2E - old_sigma2E)
      precision2 <- abs(new_sigma2K - old_sigma2K)

      old_sigma2E <- new_sigma2E
      old_sigma2K <- new_sigma2K

      if (i >= nb_iter) {
        break
      }
    }

    if (display) {
      cat("\n")
      cat("iteration ", i, "\n")
      cat("sigma2E_hat", new_sigma2E, "\n")
      cat("sigma2K_hat", new_sigma2K, "\n")
      cat("\n")
    }

    return(list(
      "beta_hat" = gamma[1:(l - 1)], "u_hat" = u,
      "sigma2E_hat" = new_sigma2E, "sigma2K_hat" = new_sigma2K
    ))
  }
