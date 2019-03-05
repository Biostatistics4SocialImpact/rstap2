
# Libraries ---------------------------------------------------------------

library(tidyverse)
library(tictoc)


# Data Generation ---------------------------------------------------------

theta_transform <- function(theta) 10 / (1 + exp(-theta))

set.seed(24)
num_subj <- 400
num_bef <- 10
Z <- rep(rbinom(n = num_subj, size = 1, prob = .5),3)
delta <- -.5
theta_s <- .5
alpha <- 22
beta <- 1.2
beta_bar <- 1
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


y <- alpha + Z * delta + beta*X_diff$X_diff + X_diff$MN_Exposure*beta_bar +  rnorm(n = num_subj*3,
                         mean = 0,
                         sd = sigma)

Z <- matrix(Z,ncol=1)

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






# Ragged Array Distance Structure Sampling --------------------------------


Rcpp::sourceCpp("src/Rinterface.cpp")
iter_max <- 1500
warmup <- 500
sink("~/Desktop/Routput.txt")
tic()
fit <- stap_diffndiff(y = y, Z = Z,
                       u_crs = as.matrix(u_crs),
                       subj_array = subj_mat1,
                       subj_n = subj_n,
                       stap_par_code = c(length(y),1,1,1,1),
                       distances = dists_crs,
                       adapt_delta = .65,
                       warmup = warmup, 
                       iter_max = iter_max,
                       max_treedepth = 10,
                       seed = 2341,
                       diagnostics = 0)

toc()
sink()


tibble(sim_ix = 1:length(fit$epsilons),
       epsilons = fit$epsilons) %>% ggplot(aes(x=sim_ix,y=log(epsilons))) + geom_line() + theme_bw() + 
        ggtitle("Epsilon Decay - Log Scale") + ggsave("~/Desktop/epdecay1.png")

tibble(sim_ix = 1:length(fit$epsilons),
              epsilons = fit$epsilons) %>% ggplot(aes(x=sim_ix,y=(epsilons))) + geom_line() + theme_bw() + ggtitle("Epsilon Decay - unit scale") + ggsave("~/Desktop/epdecay2.png")


samples <- tibble(chain=1,
                  alpha = fit$alpha_samps,
                  delta = fit$delta_samps[,1],
                  beta = fit$beta_samps[,1],
                  beta_bar = fit$beta_bar_samps[,1],
                  theta = fit$theta_samps,
                  sigma = fit$sigma_samps,
                  acceptance = fit$acceptance) %>% mutate(ix = 1:n())

samples %>% filter(acceptance==1,ix>warmup) %>% 
    gather(delta,beta,beta_bar,theta,sigma,alpha,key="Parameters",value="Samples") %>% 
    mutate(Truth = (Parameters=="beta")*1.2 + (Parameters =="theta")*.5 + (Parameters=="sigma")*1 + 
           (Parameters=="delta") *-.5 +  (Parameters=="beta_bar")*1 + 
           (Parameters=="alpha")*22) %>% 
    ggplot(aes(x=Samples)) + geom_density() + theme_bw() + 
    geom_vline(aes(xintercept=Truth),linetype=2) +
    facet_wrap(~Parameters,scales="free") + ggtitle("Custom NUTS - Posterior Samples") + 
    theme(strip.background = element_blank()) + ggsave("~/Desktop/stapdnd_progresspic.png",width = 7, height = 5)


samples %>% filter(acceptance==1,ix>warmup) %>%
    gather(beta,delta,theta,sigma,alpha,key= "Parameters", value = "Samples") %>%
    group_by(Parameters) %>% summarise(lower = quantile(Samples,0.025), med = median(Samples),
                                           upper = quantile(Samples,0.975))


# CPP Grad Checks ---------------------------------------------------------


Rcpp::sourceCpp("src/Rinterface.cpp")
thetas <- seq(from = -5, to =1.0, by =0.05);
sink("~/Desktop/Routput.txt")
out <- test_grads(y,Z,beta_bar,beta,dists_crs,as.matrix(u_crs),subj_mat1,subj_n,thetas,c(length(y),1,1,1,1),seed = 1241)
sink()

tibble(theta = theta_transform(thetas), energy = out$energy) %>% ggplot(aes(x=theta,y=energy)) + geom_line() + theme_bw()  + geom_vline(aes(xintercept = .5),linetype = 2) 

tibble(theta = theta_transform(thetas), grad = out$grad) %>% ggplot(aes(x=theta,y=grad)) + geom_line() + theme_bw()  + geom_vline(aes(xintercept = .5),linetype = 2) 


