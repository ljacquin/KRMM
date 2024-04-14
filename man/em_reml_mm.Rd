
\name{em_reml_mm}
\alias{em_reml_mm}
\encoding{utf8}

\title{
  Expectation-Maximization (EM) Algorithm for Restricted Maximum Likelihood (REML) in Mixed Models
}

\description{
  \code{em_reml_mm} estimates the components and variance parameters of the mixed model \eqn{Y = X\beta + Zu + \epsilon} using the EM-REML algorithm.
}

\usage{
  em_reml_mm(Mat_K_inv, Y, X, Z, init_sigma2K, init_sigma2E,
             convergence_precision, nb_iter, display)
}

\arguments{
  \item{Mat_K_inv}{numeric matrix; inverse of the kernel matrix}
  \item{Y}{numeric vector; response vector}
  \item{X}{numeric matrix; design matrix of fixed effects}
  \item{Z}{numeric matrix; design matrix of random effects}
  \item{init_sigma2K, init_sigma2E}{numeric scalars; initial guess values for variance parameters in the EM-REML algorithm}
  \item{convergence_precision, nb_iter}{numeric scalars; convergence precision and maximum iterations for the EM-REML algorithm}
  \item{display}{boolean; display estimated components at each iteration}
}

\value{
  \item{beta_hat}{estimated fixed effect(s)}
  \item{u_hat}{estimated random effect(s)}
  \item{sigma2K_hat, sigma2E_hat}{estimated variance components}
}

\references{
  Foulley, J.-L. (2002). Algorithme em: théorie et application au modèle mixte. Journal de la Société française de Statistique 143, 57-109
}

\author{Laval Jacquin <jacquin.julien@gmail.com>}
