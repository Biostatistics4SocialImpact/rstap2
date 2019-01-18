# 
# library(coda)
# library(tictoc)

# Static HMC --------------------------------------------------------------

# num_subj <- 2.5E3
# set.seed(23)
# 
# X <- cbind(1,
#            rbinom(n = num_subj,
#                     size = 1,
#                     prob = .5),
#            rnorm(n = num_subj))
# beta <- c(-2,1,3)
# sigma <- 1.5
# 
# y <- X %*% beta + rnorm(n = num_subj,mean = 0,sd = sigma)
# hist(y)
# 
# tic()
# fit <- hmc_lm(X,y,beta = rnorm(3),
#                L_naught = 20,
#                epsilon_naught = 0.05,
#                iter_max = 2E3,
#                seed = 23)
# toc()
# 
# beta_samps <- as.mcmc(fit$beta_samps[which(fit$accepted_ix==1),])
# plot(beta_samps)



# NUTS --------------------------------------------------------------------


# Rcpp::sourceCpp("src/rcppeigen_hello_world.cpp")
# num_subj <- 6.5E2
# set.seed(23)
# 
# X <- cbind(1,
#            rbinom(n = num_subj,
#                   size = 1,
#                   prob = .5))
# beta <- c(-2,2)
# sigma <- 1.5
# 
# y <- X %*% beta + rnorm(n = num_subj,mean = 0,sd = sigma)
# hist(y)
# beta_init <- rnorm(3)
# beta_init
# 
# fit <- nuts_lm(X,y,
#                beta = beta_init,
#                adapt_delta = .65,
#                warmup = 5,
#               iter_max = 10,
#               seed = 23)
# 
# 
# beta_samps <- as.mcmc(fit$beta_samps)
# plot(beta_samps)





# NUTS-STAP-DiffnDiff -----------------------------------------------------

set.seed(24)
num_subj <- 350
num_bef <- 40
theta_s <- 1
beta <- 1.2
sigma <- 1
dists_1 <- matrix(rexp(n = num_subj*num_bef,
                       rate = 2),
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
X_theta_one <- rowSums(exp( - dists_1 / theta_s))
X_theta_two <- rowSums(exp( - dists_2 / theta_s))
X_theta_three <- rowSums(exp( - dists_3 / theta_s))
d_one <- rbind(dists_1,dists_1,dists_1)
d_two <- rbind(dists_2,dists_2,dists_2)
d_three <- rbind(dists_3,dists_3,dists_3)
X <- apply(dists,1,function(x) sum(exp(-x/theta_s))  )
X_mean <- rowSums(exp(-d_one / theta_s)) + rowSums(exp(-d_two / theta_s)) + rowSums(exp(-d_three / theta_s))
X_mean <- X_mean / 3.0
X_diff <- X - X_mean
y <- beta*X_diff + rnorm(n = num_subj*3,
                         mean = 0,
                         sd = sigma)

Rcpp::sourceCpp("src/Rinterface.cpp")
beta_init <- rnorm(1)
theta_init <- rnorm(1)

fit <- stap_diffndiff(y = y,
                      beta = beta_init,
                      theta = -3,
                      distances = dists,
                      d_one = d_one , d_two = d_two,d_three = d_three,
                      adapt_delta = .85,
                      warmup = 100, iter_max = 150,
                      seed = 23 )



# Checking Gradient Functions ---------------------------------------------

library(tidyverse)



create_X_diff <- function(dists,d_one,d_two,d_three,theta){
    X <-  apply(dists,1,function(x) sum(exp(-x/theta)))
    X_mean <- rowSums(exp(-d_one / theta)) + rowSums(exp(-d_two / theta)) + rowSums(exp(-d_three / theta))
    X_mean <- X_mean / 3.0
    return(X-X_mean)
}

get_ll <- function(y,dists,d_one,d_two,d_three,theta,beta){
    theta <- 10/(1+exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta)
    sum(dnorm(x = y,mean = beta*X_diff,sd = 1,log = T))
}


get_ll_h <- function(y,dists,d_one,d_two,d_three,theta,beta,h){
    theta_tilde <- 10/(1+exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    ll <- sum(dnorm(x = y,mean = beta*X_diff,sd = 1,log = T))
    theta_tilde <- 10/(1+exp(-(theta+h)))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    (sum(dnorm(x = y,mean = beta*X_diff,sd = 1,log = T)) - ll)/h
}

get_X_prime <- function(dists,theta){
    theta_tilde <- 10 / (1 + exp(-theta))
    X_p <- rowSums(dists / theta_tilde^2 * exp(-dists/theta_tilde))
    X_p <- X_p * 10 * exp(-theta) / (1 + exp(-theta))^2
}

get_X_mean_prime <- function(d_one,d_two,d_three,theta){
    theta_tilde <- 10 / (1 + exp(-theta))
    X_mp <- rowSums(exp(-d_one/ theta_tilde) * d_one / theta_tilde^2)
    X_mp <-  X_mp + rowSums(exp(-d_two/ theta_tilde) * d_two / theta_tilde^2)
    print (head(X_mp))
    X_mp <-  X_mp + rowSums(exp(-d_three/ theta_tilde) * d_three / theta_tilde^2)
    X_mp <-  X_mp / 3.0 * 10 * exp(-theta) / (1 + exp(-theta))^2
}

create_Xp_diff <- function(dists,d_one,d_two,d_three,theta){
    X_mp <- get_X_mean_prime(d_one,d_two,d_three,theta)
    X_p <- get_X_prime(dists,theta)
    print("X_prime")
    print("----------")
    print(head(X_p))
    print("X_prime mean")
    print("----------")
    print(head(X_mp))
    return(X_p - X_mp)
}

get_theta_grad <- function(y,dists,d_one,d_two,d_three,theta,beta){
    theta_tilde <- 10 / (1 + exp(-theta))
    Xp_diff <- create_Xp_diff(dists,d_one,d_two,d_three,theta)
    
    theta_grad <- sum(y*Xp_diff)*beta - beta^2 * sum(Xp_diff)
    theta_grad <- theta_grad - theta_tilde^-1 * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <- theta_grad - (log(theta_tilde)-1)/theta_tilde * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <- theta_grad - length(y)+1 * exp(-theta) / ( 1 + exp(-theta)) * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <-  theta_grad + length(y)+1 * exp(-theta) / (1 - 1 /(1+exp(-theta)) * (1 + exp(-theta))^2) * 10 * exp(-theta) / (1 + exp(-theta))^2
}


Rrslts <- data_frame( ll = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll(y,dists,d_one,d_two,d_three,z,w)),
                      grad_theta = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll_h(y,dists,d_one,d_two,d_three,z,w,1E-4)),
                      theta_grad = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_theta_grad(y,dists,d_one,d_two,d_three,z,w)),
            beta = rslts$beta,
            theta = rslts$theta) 

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1) %>% filter(theta<1) %>% 
    ggplot(aes(x=theta,y=ll,color=factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Log Likelihood - Theta perspective")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1) %>% filter(theta<1) %>% 
    ggplot(aes(x=theta,y=theta_grad,color=factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Analytic Gradient")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>%
    filter(theta<=0.5,theta>0.45) %>%  
    ggplot(aes(x=beta,y=ll,color=factor(theta))) + 
    geom_line() + theme_bw() + ggtitle("Log Likelihood - Beta perspective")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>=.80) %>% filter(theta<1) %>% 
    ggplot(aes(x=theta,y=grad_theta,color= factor(beta) )) +
    geom_line() + theme_bw() + ggtitle("Numerical Gradient")



fit <- test_grads(y = y,
                      beta = beta_init,
                      theta = -2.944,
                      distances = dists,
                      d_one = d_one , d_two = d_two,d_three = d_three,
                      adapt_delta = .85,theta_grid = seq(from = -3, to = 0, by =0.05),
                      beta_grid = seq(from=0,to=3,by=0.05),
                      warmup = 5, iter_max = 10,
                      seed = 23 )



rslts <- data_frame(bg = fit$beta_gradient,tg=fit$theta_gradient,energy=fit$energy,
           beta = fit$beta_grid, theta = fit$theta_grid)

rslts %>% filter(theta<=-2.8) %>%  ggplot(aes(x=beta,y=energy,color=factor(theta))) + geom_line() + theme_bw() + ggtitle("Likelihood")
rslts  %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>%  
    filter(beta>1,beta<=1.3,transformed_theta<=3) %>% 
    ggplot(aes(x=transformed_theta,y=energy,color=factor(beta))) + geom_line() + theme_bw() + ggtitle("Likelihood")

rslts  %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>%  
    filter(beta>1,beta<=1.3,transformed_theta<=3) %>% 
    ggplot(aes(x= theta, y = energy, color = factor(beta))) + geom_line() + theme_bw() + ggtitle("Likelihood")


rslts %>% filter(beta>0) %>% filter(energy == max(.$energy))

rslts %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>% 
    filter(beta>1.11,beta<1.21) %>% 
    ggplot(aes(x=transformed_theta,y=tg,color=factor(beta) )) + 
    geom_line() + theme_bw() + ggtitle("Theta gradient")

rslts %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>% 
    filter(beta>1.11,beta<1.21) %>% ggplot(aes(x=theta,y=tg,color=factor(beta) )) + geom_line() + theme_bw() + ggtitle("Theta gradient")

rslts %>% filter(theta %in% c(-1.9,-2.95,-2)) %>%  ggplot(aes(x=beta,y=bg,color= factor(theta) )) + geom_line() + theme_bw() + ggtitle("beta gradient")
