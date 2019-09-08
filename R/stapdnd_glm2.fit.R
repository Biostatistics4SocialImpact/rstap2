#' Fitting Difference in Difference STAP glms
#'
#'
#'@param y an N-length vector
#'@param z N x p design matrix of subject specific covariates
#'@param dists_crs (q_s+q_st) x M matrix of distances between outcome 
#' observations and built environment features with a hypothesized spatial scale
#'@param u_s N x (q *2) matrix of compressed row storage array indices for dists_crs
#'@param subj_matrix an n x N matrix indexing the subjects observations in the y and Z matrices
#'@param subj_n
#'@param stap_data object of class "stap_data" that contains information on all the spatial-temporal predictors in the model
#'@param max_distance the upper bound on any and all distances included in the model 
#'@param max_time the upper bound on any and all times included in the model
#'@param weights weights to be added to the likelihood observation for a given subject
#'@param offset offset term to be added to the outcome for a given subject
#'@param family distributional family - only binomial gaussian or poisson currently allowed
#'@param prior_dnd,prior_intercept,prior_bar,prior_theta,prior_sigma 
#'@export stapdnd_glm
stapdnd_glm2.fit <- function(y,z,
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
                            prior_stap = normal(),
                            prior_intercept = normal(),
                            prior_bar = normal(),
                            prior_theta = lognormal(),
                            prior_sigma = cauchy(),
                            include_warmup = FALSE
                           ){


    supported_families <- c("binomial","gaussian","poisson")
    fam <- which(pmatch(supported_families, family$family, nomatch = 0L) == 1L)
    if(!length(fam))
        stop("'family' must be one of ", paste(supported_families, collapse = ', '))

    link <- family$link
    if(link != "identity")
         stop("'link' must be one of", paste( supported_links, collapse = ', '))
    
    fit <- stap_diffndiff(y = y,
                          Z = z,
                          distances = dists_crs,
                          u_crs = u_s,
                          subj_array = subj_matrix,
                          subj_n = subj_n,
                          stap_par_code = stap_par_code,
                          adapt_delta = adapt_delta, 
                          iter_max = iter_max,
                          max_treedepth = max_treedepth,
                          warmup = warmup,
                          seed = seed,
                          diagnostics = diagnostics)

}
