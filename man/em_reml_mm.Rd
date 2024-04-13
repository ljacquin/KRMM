
\name{em_reml_mm}
\alias{em_reml_mm}
\encoding{utf8}

%- Also NEED an '\alias' for EACH other topic documented here.
\title{Expectation-Maximization (EM) algorithm for the restricted maximum likelihood (REML) associated to the mixed model
%%  ~~function to do ... ~~
}
\description{
em_reml_mm estimates the components and variance parameters of the following mixed model; $Y =X\beta + ZU + \varepsilon$, using the EM-REML algorithm.
}
\usage{ em_reml_mm(Mat_K_inv, Y, X, Z, init_sigma2K, init_sigma2E,
                    convergence_precision, nb_iter, display)}

%- maybe also 'usage' for other objects documented here.

\arguments{

	\item{Mat_K_inv}{numeric matrix; the inverse of the kernel matrix}

	\item{Y}{numeric vector; response vector}

	\item{X}{numeric matrix; design matrix of fixed effects}

	\item{Z}{numeric matrix; design matrix of random effects}

	\item{init_sigma2K, init_sigma2E}{
		numeric scalars;
		initial guess values, associated to the mixed model variance parameters,
		for the EM-REML algorithm
		}

		\item{convergence_precision, nb_iter}{
		convergence precision (i.e. tolerance) associated to the mixed model
		variance parameters, for the EM-REML algorithm, and number of maximum
		iterations allowed if convergence is not reached
	}

	\item{display}{
		boolean (TRUE or FALSE character string);
		should estimated components be displayed at each iteration
	}
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
