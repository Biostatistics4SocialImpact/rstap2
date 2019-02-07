
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tictoc)
library(coda)


# Data Generation ---------------------------------------------------------

set.seed(24)
num_subj <- 300
num_bef <- 40
theta_s <- .5
beta <- 1.2
sigma <- 1
dists_1 <- matrix(rexp(n = num_subj*num_bef,
                       rate = 1),
                  nrow = num_subj,
                  ncol = num_bef)
dists_2 <- matrix(rexp(n = num_subj*num_bef,
                       rate = 1),
                  nrow = num_subj,
                  ncol = num_bef)
dists_3 <- matrix(rexp(n= num_subj*num_bef,
                       rate = .5),
                  nrow = num_subj, ncol = num_bef)
dists <- rbind(dists_1,dists_2,dists_3)
colnames(dists) <- stringr::str_c("BEF_",1:ncol(dists))
dists <- as_tibble(dists)

dists <- dists %>% mutate(id=expand.grid(subj=1:num_subj,time = 1:3)$subj,
                         time = expand.grid(1:num_subj,time = 1:3)$time) %>%
    gather(contains("BEF"),key = "BEF", value="Distance") %>% 
    filter(Distance<=5)

X <- dists %>% group_by(id,time) %>% summarise(Exposure = sum(exp(-Distance/theta_s)))
X_bar <- X %>% group_by(id) %>% summarise(MN_Exposure = mean(Exposure))

X_diff <- X %>% left_join(X_bar,by='id') %>% mutate(X_diff = Exposure - MN_Exposure)

y <- beta*X_diff$X_diff + rnorm(n = num_subj*3,
                         mean = 0,
                         sd = sigma)



# Ragged Array Distance Structure Set up ----------------------------------------------------------------


dists_crs <- matrix(dists$Distance,nrow=1)
u_crs <- as_tibble(xtabs(~id + time,data=dists,addNA = T)) %>% 
    mutate(start = replace(dplyr::lag(cumsum(n)),is.na(dplyr::lag(cumsum(n))),0)+1,
           stop = cumsum(n)) %>% select(start,stop) %>% mutate(start = start - 1,
                                                               stop = stop - 1)

# Ragged Array Distance Structure Sampling --------------------------------



Rcpp::sourceCpp("src/Rinterface.cpp")
iter_max <- 750
warmup <- 500
beta_init2 <- runif(n = 1,-2,-2)
theta_init2 <- runif(n = 1,-2,-2)
tic()
fit1 <- stap_diffndiff(y = y,
                       beta = beta_init2,
                       theta = theta_init2,
                       distances = dists,
                       d_one = d_one ,
                       d_two = d_two,
                       d_three = d_three,
                       adapt_delta = .65,
                       warmup = warmup, 
                       iter_max = iter_max,
                       max_treedepth = 10,
                       sd_beta = 1,
                       sd_theta = 1,
                       seed = 23 )
toc()



samples <- tibble(chain=1,
                  beta = fit1$beta_samps,
                  theta = fit1$theta_samps,
                  acceptance = fit1$acceptance) %>% mutate(ix = 1:n())