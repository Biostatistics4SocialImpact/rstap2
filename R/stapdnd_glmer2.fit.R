# Part of the rstap2 package for estimating model parameters
# Copyright (c) 2018
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
# # This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the # GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#' Bayesian spatial-temporal generalized linear models with group-specific terms via Stan
#' 
#' Bayesian inference for stap-glms with group-specific coefficients that have 
#' unknown covariance matrices with flexible priors.
#' 
#'@export
#'@param y an N-length vector
#'@param z N x p design matrix of subject specific covariates
#'@param dists_crs (q_s+q_st) x M matrix of distances between outcome 
#' observations and built environment features with a hypothesized spatial scale
#'@param u_s N x (q *2) matrix of compressed row storage array indices for dists_crs
#'@param subj_matrix an n x N matrix indexing the subjects observations in the y and Z matrices
#'@param subj_n an n x q matrix where each row holds the reciprocal of the number of observations for the ith subject
#'@param stap_data object of class "stap_data" that contains information on all the spatial-temporal predictors in the model
#'@param max_distance the upper bound on any and all distances included in the model 
#'@param max_time the upper bound on any and all times included in the model
#'@param weights weights to be added to the likelihood observation for a given subject
#'@param offset offset term to be added to the outcome for a given subject
#'@param family distributional family - only binomial gaussian or poisson currently allowed
#' @seealso The Longituinal \href{https://biostatistics4socialimpact.github.io/rstap/articles/longitudinal-I.html}{Vignette} for \code{stap_glmer} 
#' and the \href{http://arxiv.org/abs/1812.10208}{preprint} article available through arXiv.
#'
stapdnd_glmer2.fit <- function(y,z,w,
                             dists_crs,u_s,
                             subj_matrix,subj_n,
                             family = gaussian(),
                             stap_par_code,
                             iter_max, 
                             warmup,
                             adapt_delta,
                             max_treedepth,
                             seed,
                             diagnostics,
                             cores = 1,
                             prior_dnd = normal(),
                             prior_intercept = normal(),
                             prior_bar = normal(),
                             prior_theta = lognormal(),
                             prior_sigma = cauchy()){
    
    out <- stapdnd_glmer(y = y, Z = z, W = w,distances = dists_crs,
                         u_crs = u_s, subj_matrix = subj_matrix, subj_n = subj_n,
                         stap_par_code = stap_par_code, adapt_delta = adapt_delta,
                         iter_max = iter_max, max_treedepth = max_treedepth,
                         warmup = warmup, seed = seed,
                         diagnostics = diagnostics)
    return(out)
}
