# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3 # of the License, or (at your option) any later version.
# # This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#' Bayesian generalized spatial-temporal aggregated predictor(STAP) models via Rcpp 
#'
#' Generalized linear modeling with spatial temporal aggregated predictors using
#' prior distributions for the coefficients, intercept, spatial-temporal scales, and auxiliary parameters.
#'
#' @export
#'
#' @param formula Same as for \code{\link[stats]{glm}}. Note that in-formula transformations will not be passed ot the final design matrix. Covariates that have "_scale" or "_shape" in their name are not advised as this text is parsed for in the final model fit.
#' @param family Same as \code{\link[stats]{glm}} for gaussian, binomial, and poisson families.
#' @param subject_data a data.frame that contains data specific to the subject or subjects on whom the outcome is measured. Must contain one column that has the subject_ID  on which to join the distance and time_data
#' @param distance_data a (minimum) three column data.frame that contains (1) an id_key (2) The sap/tap/stap features and (3) the distances between subject with a given id and the built environment feature in column (2), the distance column must be the only column of type "double" and the sap/tap/stap features must be specified in the dataframe exactly as they are in the formula.
#' @param time_data same as distance_data except with time that the subject has been exposed to the built environment feature, instead of distance 
#' @param subject_ID  name of column(s) to join on between subject_data and bef_data
#' @details The \code{stap_glm2} function is similar in syntax to
#' \code{\link[rstanarm]{stan_glm}} except instead of performing full bayesian
#' inference for a generalized linear model stap_glm also incorporates spatial-temporal covariates
#' @seealso The various vignettes for \code{stap_glm} at
#'   \url{https://biostatistics4socialimpact.github.io/rstap/articles} and the \href{http://arxiv.org/abs/1812.10208}{preprint} article.  
#'
#'@export stap_glm
#'
stap_glm2 <- function(formula,
					  family = gaussian(),
                     subject_data = NULL,
                     distance_data = NULL,
                     time_data = NULL,
					 iter_max = 2E3,
					 warm_up = 1E3,
                     adapt_delta = 0.65,
					 seed = NULL){

	if(family$family != "gaussian")
		stop("Currently only gaussian outcomes are supported")
	if(is.null(seed))
		seed <- 32432
	stopifnot(iter_max > warm_up)
    call <- match.call(expand.dots = TRUE)
    mf <-  match.call(expand.dots = FALSE)
    mf$formula <- formula 
    m <- match(c("formula", "weights", "offset"),
               table = names(mf), nomatch=0L)
    mf <- mf[c(1L,m)]
    mf$data <- subject_data
    mf$drop.unused.levels <- TRUE

    mf[[1L]] <- as.name("model.frame")
    mf <- eval(mf, parent.frame())
    mt <- attr(mf, "terms")
    if(is.empty.model(mt))
        stop("No intercept or predictors specified.", call. = FALSE)
    Z <- model.matrix(mt, mf)
	has_intercept <- attr(terms(formula),"intercept")
	if(has_intercept)
		Z <- Z[,2:ncol(Z),drop=F]
	subject_array <-  as(as.matrix(Matrix::fac2sparse(1:nrow(Z))),"sparseMatrix")
	subj_n <- matrix(rep(0,nrow(Z)) ,ncol=1)
	Y <- stats::model.response(mf,type='any')

    stapfit <- stap_diffndiff_stfit(y = Y,
									Z = Z, 
									distances = distance_data$distances,
									u_crs = distance_data$u_crs,
									times = time_data$times,
									u_tcrs = time_data$u_crs,
									subj_array_ = subject_array,
									subj_n = subj_n,
									stap_par_code = c(has_intercept*length(Y),ncol(Z),1,1,1,1),
									adapt_delta = adapt_delta,
									iter_max = iter_max,
									max_treedepth = 10,
									warmup = warm_up,
									seed = seed,
									diagnostics = 0
                            )
    fit <- list(stapfit = stapfit, 
				 call = call,
                 stan_function = "stap_glm2")
    out <- stapreg2(fit,warm_up,iter_max)

    return(out)
}

#' @rdname stap_glm
#' @export
stap_lm2 <- 
  function(formula,
           subject_data = NULL,
           distance_data = NULL,
           time_data = NULL,
           subject_ID = NULL,
           adapt_delta = NULL){

  mc <- call <- match.call(expand.dots = TRUE)
  if (!"formula" %in% names(call))
    names(call)[2L] <- "formula"
  mc[[1L]] <- quote(stap_glm)
  mc$family <- "gaussian"
  out <- eval(mc, parent.frame())
  out$call <- call
  out$stan_function <- "stap_lm2"
  return(out)
}
