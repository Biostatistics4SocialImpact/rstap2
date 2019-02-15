
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tictoc)
library(coda)


# Data Generation ---------------------------------------------------------

set.seed(24)
num_subj <- 300
num_bef <- 10
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


# R Function Declarations -------------------------------------------------


# create_X_diff2 <- function(dists,theta){
#     X <- dists %>% group_by(id,time) %>% summarise(Exposure = sum(exp(-Distance/theta)))
#     X_bar <- X %>% group_by(id) %>% summarise(MN_Exposure = mean(Exposure))
#     X_diff <- X %>% left_join(X_bar,by='id') %>% mutate(X_diff = Exposure - MN_Exposure)
#     return(X_diff$X_diff)
# }
# 
# get_ll2 <- function(y,dists,theta,beta,sigma = log(1)){
#     sigma <- exp(sigma)
#     theta_tilde <- 10/(1+exp(-theta))
#     X_diff <- create_X_diff2(dists,theta_tilde)
#     sum(dnorm(x = y,mean = beta*X_diff,sd = sigma,log = T))
# }
# 
# get_energy2 <- function(y,dists,theta,beta){
#     ll <-  get_ll2(y,dists,theta,beta)
#     theta_tilde <- 10 / (1+exp(-theta))
#     ll <- ll + dnorm(beta,0,3,log=T)
#     ll <-  ll + dlnorm(theta_tilde,meanlog = 1,sdlog=1,log = T)
#     ll <-  ll +  (log(10) - theta - 2*log(1+exp(-theta)))
#     return(ll)
# }
# 
# 
# btgrd <- expand.grid(thetas = seq(from = -4,to=1,by=0.05),betas = seq(from = 0,to=3,by=0.1))
# Rrslts <- tibble(energy = map2_dbl(btgrd$thetas,btgrd$betas,function(v,w) get_energy2(y,dists,v,w)),
#                  beta = btgrd$betas, theta = btgrd$thetas,
#                  mn_Xd = map2_dbl(btgrd$thetas,btgrd$betas,function(v,w) mean(create_X_diff2(dists,v))))
# Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% filter(beta<1.2) %>% 
#     ggplot(aes(x=theta,y=energy,color=factor(beta))) + geom_line() + theme_bw()

# Ragged Array Distance Structure Set up ----------------------------------------------------------------


dists_crs <- matrix(dists %>% arrange(id,time) %>% select(Distance) %>% pull(),nrow=1)
u_crs <- as_tibble(xtabs(~id + time,data = dists, addNA = T)) %>% 
    mutate(id = as.integer(id),time = as.integer(time)) %>% 
    arrange(id,time) %>% 
    mutate(start = replace(dplyr::lag(cumsum(n)),is.na(dplyr::lag(cumsum(n))),0)+1,
           stop = cumsum(n)) %>% select(start,stop) %>% mutate(start = start - 1,
                                                               stop = stop - 1)

subj_mat1 <- as.matrix(Matrix::fac2sparse(as.factor(X_diff$id)))
subj_n <- rep(1/3,300)

# Sigma Gradient ----------------------------------------------------------

# sigs <- seq(-1,1,0.05)
# sigrslts <- tibble(sigma = sigs,
#                    gradient = map_dbl(sigs,function(x) calculate_sigma_gradient(1.2,.5,x,y,dists_crs,as.matrix(u_crs),subj_mat1,subj_n)),
#                    energy = map_dbl(sigs,function(x) calculate_total_energy(1.2,-2.944,x,y,dists_crs,as.matrix(u_crs),subj_mat1,subj_n)))
# 
# sigrslts %>% ggplot(aes(x=sigma,y=gradient)) + geom_line() + theme_bw()
# sigrslts %>% ggplot(aes(x=sigma,y=energy)) + geom_line() + theme_bw()
# Ragged Array Distance Structure Sampling --------------------------------

Rcpp::sourceCpp("src/Rinterface.cpp")
iter_max <- 500
warmup <- 250
sink("~/Desktop/Routput.txt")
tic()
fit1 <- stap_diffndiff(y = y,
                       u_crs = as.matrix(u_crs),
                       subj_array = subj_mat1,
                       subj_n = subj_n,
                       stap_par_code = c(0,0,1,0,1),
                       distances = dists_crs,
                       adapt_delta = .65,
                       warmup = warmup, 
                       iter_max = iter_max,
                       max_treedepth = 10,
                       seed = 2431,
                       diagnostics = 1)
toc()
sink()



samples <- tibble(chain=1,
                  beta = fit1$beta_samps,
                  theta = fit1$theta_samps,
                  sigma = fit1$sigma_samps,
                  acceptance = fit1$acceptance) %>% mutate(ix = 1:n())

samples %>% filter(acceptance==1,ix>warmup) %>% 
    gather(beta,theta,sigma,key="Parameters",value="Samples") %>% 
    mutate(Truth = (Parameters=="beta")*1.2 + (Parameters =="theta")*.5 + (Parameters=="sigma")*1) %>% 
    ggplot(aes(x=Samples)) + geom_histogram() + theme_bw() + 
    geom_vline(aes(xintercept=Truth),linetype=2) +
    facet_grid(~Parameters) + ggtitle("Posterior Samples") + 
    theme(strip.background = element_blank()) + ggsave("~/Desktop/stapdnd_progresspic.png",width = 7, height = 5)


calculate_sigma_gradient(cur_beta = beta_init,cur_theta = 10/(1+exp(-theta_init)),
                         cur_sigma = sigma_init,y = y,dists = dists_crs,
                         u_crs = as.matrix(u_crs),subj_array = subj_mat1,subj_n = subj_n)
