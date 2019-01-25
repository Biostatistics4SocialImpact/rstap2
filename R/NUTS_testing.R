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
num_subj <- 300
num_bef <- 40
theta_s <- .5
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
X_1 <- rowSums(exp(-dists_1 / theta_s))
y_stapreg <- 1 + beta*X_1 + rnorm(n = length(X_1),mean = 0, sd = 1)
beta_init <- rnorm(1)
theta_init <- rnorm(1)








create_X_diff <- function(dists,d_one,d_two,d_three,theta){
    X <-  apply(dists,1,function(x) sum(exp(-x/theta)))
    X_mean <- rowSums(exp(-d_one / theta)) + rowSums(exp(-d_two / theta)) + rowSums(exp(-d_three / theta))
    X_mean <- X_mean / 3.0
    return(X-X_mean)
}

get_ll <- function(y,dists,d_one,d_two,d_three,theta,beta,sigma = log(1)){
    sigma <- exp(sigma)
    theta_tilde <- 10/(1+exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    sum(dnorm(x = y,mean = beta*X_diff,sd = sigma,log = T))
}

optim(c(0,0,0),function(x) 0-get_ll(y,dists,d_one,d_two,d_three,x[1],x[2],sigma = x[3]))

get_ll <- function(y,dists,d_one,d_two,d_three,theta,beta){
    theta_tilde <- 10/(1+exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    sum(dnorm(x = y,mean = beta*X_diff,sd = sigma,log = T))
}
# Checking Gradient Functions ---------------------------------------------

library(tidyverse)

get_ll_stapreg <- function(y,dists,theta,beta){
    theta_tilde <- 10/(1+exp(-theta))
    X <- apply(dists,1,function(x) sum(exp(-x/theta_tilde)))
    ll <- dnorm(y,1 + beta*X,1,T)
    lp <- dlnorm(theta_tilde,1,1,T)
    bp <- dnorm(beta,0,3)
    sum(ll + lp + bp)
}

get_energy <- function(y,dists,d_one,d_two,d_three,theta,beta){
    ll <-  get_ll(y,dists,d_one,d_two,d_three,theta,beta)
    theta_tilde <- 10 / (1+exp(-theta))
    ll <- ll + dnorm(beta,0,3,log=T)
    ll <-  ll + dlnorm(theta_tilde,meanlog = 1,sdlog=1,log = T)
    ll <-  ll +  (log(10) - theta - 2*log(1+exp(-theta)))
    return(ll)
}


optim(c(0,0),function(x) 0-get_energy(y,dists,d_one,d_two,d_three,x[1],x[2]))

optim(c(5,5),function(x) 0 - get_energy(y,dists,d_one,d_two,d_three,x[1],x[2]))


get_ll_h <- function(y,dists,d_one,d_two,d_three,theta,beta,h){
    theta_tilde <- 10/(1+exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    ll <- sum(dnorm(x = y,mean = beta*X_diff,sd = 1,log = T))
    theta_tilde <- 10/(1+exp(-(theta+h)))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta_tilde)
    (sum(dnorm(x = y,mean = beta*X_diff,sd = 1,log = T)) - ll)/h
}

get_ll_h_stapreg <- function(y,dists,theta,beta,h){
    theta_tilde <- 10/(1+exp(-theta))
    X_h <- apply(dists,1,function(x) sum(exp(-x/theta_tilde+h)))
    X <- apply(dists,1,function(x) sum(exp(-x/theta_tilde)))
    (sum(dnorm(x=y,mean=beta*X_h,sd=1,log=T)) -  sum(dnorm(y,beta*X,1,T)) )/h
}

get_energy_h <- function(y,dists,d_one,d_two,d_three,theta,beta,h){
    e1 <- get_energy(y,dists,d_one,d_two,d_three,theta,beta)
    e2 <- get_energy(y,dists,d_one,d_two,d_three,theta+h,beta)
    return((e2-e1)/h)
}
get_X_prime <- function(dists,theta){
    theta_tilde <- 10 / (1 + exp(-theta))
    X_p <- rowSums(dists / 10 * exp(-dists/theta_tilde - theta))
    X_p <- X_p #10 * exp(-theta) / (1 + exp(-theta))^2
}

get_X_mean_prime <- function(d_one,d_two,d_three,theta){
    theta_tilde <- 10 / (1 + exp(-theta))
    X_mp <- rowSums(exp(-d_one/ theta_tilde - theta) * d_one / 10)
    X_mp <-  X_mp + rowSums(exp(-d_two/ theta_tilde - theta) * d_two / 10 )
    # print (head(X_mp))
    X_mp <-  X_mp + rowSums(exp(-d_three/ theta_tilde - theta) * d_three / 10)
    X_mp <-  X_mp / 3.0
}

create_Xp_diff <- function(dists,d_one,d_two,d_three,theta){
    X_mp <- get_X_mean_prime(d_one,d_two,d_three,theta)
    X_p <- get_X_prime(dists,theta)
    return(X_p - X_mp)
}

get_theta_grad <- function(y,dists,d_one,d_two,d_three,theta,beta){
    theta_tilde <- 10 / (1 + exp(-theta))
    X_diff <- create_X_diff(dists,d_one,d_two,d_three,theta = theta_tilde)
    Xp_diff <- create_Xp_diff(dists,d_one,d_two,d_three,theta)
    theta_grad <- sum((y-beta*X_diff)*Xp_diff) #(y*Xp_diff)*beta - beta^2 * sum(Xp_diff)
    theta_grad <- theta_grad - theta_tilde^-1 * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <- theta_grad - (log(theta_tilde)-1)/theta_tilde * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <- theta_grad -  exp(-theta) / ( 1 + exp(-theta)) * 10 * exp(-theta) / (1 + exp(-theta))^2
    theta_grad <-  theta_grad + exp(-theta) / (1 - 1 /(1+exp(-theta)) * (1 + exp(-theta))^2) * 10 * exp(-theta) / (1 + exp(-theta))^2
}

uniroot(f = function(x) get_theta_grad(y,dists,d_one,d_two,d_three,x,1.2),interval = c(-5,5))

Rrslts <- data_frame(#ll = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll(y,dists,d_one,d_two,d_three,z,w)),
                     # grad_theta = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll_h(y,dists,d_one,d_two,d_three,z,w,1E-4)),
                     # theta_grad = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_theta_grad(y,dists,d_one,d_two,d_three,z,w)),
                     # energy = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_energy(y,dists,d_one,d_two,d_three,z,w)),
                     # energy_h = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_energy_h(y,dists,d_one,d_two,d_three,z,w,1E-4)),
                     ll_stapreg = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll_stapreg(y,dists,z,w)),
                     beta = rslts$beta,
                     theta = rslts$theta)

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1) %>% mutate(beta = factor(beta)) %>% 
    ggplot(aes(x=theta,y=ll_stapreg,color =beta)) + geom_line() + 
    theme_bw() + ggtitle("Log Posterior (Stapreg) - Theta Perspective") + 
    ylab("log posterior") +
    ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_22_19/energy_theta_persp_stapreg.png",
           width = 7, height =5)
    

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1) %>% filter(theta<5) %>% 
    ggplot(aes(x=theta,y=ll,color=factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Log Likelihood - Theta perspective")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1)%>% 
    ggplot(aes(x=theta,y=energy,color=factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Energy - Theta perspective")


Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>1) %>% filter(theta<3) %>% 
    ggplot(aes(x=theta,y=theta_grad,color=factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Analytic Gradient")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>%
    filter(theta <1 ) %>%
    ggplot(aes(x=beta,y=ll,color=factor(theta))) + 
    geom_line() + theme_bw() + ggtitle("Log Likelihood - Beta perspective")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>=.80) %>% filter(theta<3) %>% 
    ggplot(aes(x=theta,y=grad_theta,color= factor(beta) )) +
    geom_line() + theme_bw() + ggtitle("Numerical Gradient")

Rrslts %>% mutate(theta = 10/(1+exp(-theta))) %>% 
    filter(beta<1.3,beta>=.80) %>% filter(theta<3) %>% 
    ggplot(aes(x=theta,y=energy_h,color= factor(beta) )) +
    geom_line() + theme_bw() + ggtitle("Numerical Gradient - Energy")


Rrslts_stapreg <- data_frame(ll = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll_stapreg(y_stapreg,dists,z,w)),
                             grad_theta = map2_dbl(rslts$theta,rslts$beta,function(z,w) get_ll_h_stapreg(y_stapreg,dists,z,w,1E-4)),
                             beta = rslts$beta,
                             theta = rslts$theta)

Rrslts_stapreg %>% filter(beta>1,beta<1.3) %>% 
    mutate(theta = 10 / (1+exp(-theta))) %>% filter(theta<1) %>% 
    ggplot(aes(x=theta,y=grad_theta,color= factor(beta))) + geom_line() + 
    theme_bw()


# Cpp - gradient testing  ---------------------------------------------------------
Rcpp::sourceCpp("src/Rinterface.cpp")


sink("garbage.txt")
fit <- test_grads(y = y,
                      beta = beta_init,
                      theta = -2.944,
                      distances = dists,
                      d_one = d_one , d_two = d_two,d_three = d_three,
                      adapt_delta = .85,
                      theta_grid = seq(from = -4, to = 3, by =0.05),
                      beta_grid = seq(from=0, to=5, by=0.05),
                      warmup = 5, iter_max = 10,
                      seed = 23 )
sink()


rslts <- data_frame(bg = fit$beta_gradient,tg=fit$theta_gradient,energy=fit$energy,
           beta = fit$beta_grid, theta = fit$theta_grid)

rslts %>% mutate(theta = round(10 / (1+exp(-theta)),2)) %>% 
    filter(theta<.6,theta>.45) %>%  mutate(theta = factor(theta)) %>% 
    ggplot(aes(x=beta,y=energy,color=theta)) + 
    geom_line() + theme_bw() + ggtitle("Energy - beta perspective") + 
    xlab("Beta") + ylab("Log Posterior (unnormalized)") + 
    ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_22_19/energy_beta_persp.png",
           width = 7, height = 5)



rslts  %>% mutate(theta = 10/(1.0 + exp(-theta))) %>%  
    filter(beta>1,beta<=1.5,theta<=3) %>% mutate(beta = factor(beta)) %>% 
    ggplot(aes(x=theta,y=energy,color=beta)) + 
    geom_line() + theme_bw() + ggtitle("Log Posterior - theta perspective") + 
    xlab("Theta") + ylab("Log Posterior") + 
    ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_22_19/energy_theta_persp.png",
           width = 7, height = 5)




rslts  %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>%  
    filter(beta>1,beta<=1.5,transformed_theta<=3) %>% 
    ggplot(aes(x= theta, y = energy, color = factor(beta))) + 
    geom_line() + theme_bw() + ggtitle("Log Posterior, theta perspective")


rslts %>% filter(beta>0) %>% filter(energy == max(.$energy))

rslts %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>% 
    filter(beta>1,beta<1.3,transformed_theta<3) %>% 
    mutate(beta = factor(beta)) %>% 
    ggplot(aes(x=transformed_theta,y=tg,color=beta )) + 
    geom_line() + theme_bw() + ggtitle("Theta gradient") + 
    xlab("Theta") + ylab("Gradient") + 
    ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_22_19/grad_theta.png",
           width = 7, height = 5)

rslts %>% mutate(transformed_theta = 10/(1.0 + exp(-theta))) %>% 
    filter(beta>1.11,beta<1.21) %>% 
    ggplot(aes(x=theta,y=tg,color=factor(beta) )) + 
    geom_line() + theme_bw() + ggtitle("Theta gradient")

rslts %>% filter(theta%in% c(-2.95,1.5)) %>%
    mutate(theta = factor(round(10/(1+exp(-theta)),2 ))) %>% 
    ggplot(aes(x=beta,y=bg,color= theta )) + 
    geom_line() + theme_bw() + ggtitle("beta gradient") +
    xlab("Beta") + ylab("Beta Gradient") + 
    ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_22_19/grad_beta.png",
           width = 7, height = 5)




# Stap_diffndiff testing --------------------------------------------------

Rcpp::sourceCpp("src/Rinterface.cpp")
library(coda)
library(tictoc)
iter_max <- 2E3
warmup <- 1E3
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
beta_init2 <- runif(n = 1,-2,-2)
theta_init2 <- runif(n = 1,-2,-2)
tic()
fit2 <- stap_diffndiff(y = y,
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

beta_init2 <- runif(n = 1,-2,-2)
theta_init2 <- runif(n = 1,-2,-2)
tic()
fit3 <- stap_diffndiff(y = y,
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



beta_init2 <- runif(n = 1,-2,-2)
theta_init2 <- runif(n = 1,-2,-2)
tic()
fit4 <- stap_diffndiff(y = y,
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
       acceptance = fit1$acceptance) %>% mutate(ix = 1:n()) %>% 
    rbind(.,
          tibble(chain = 2,
                 beta = fit2$beta_samps,
                 theta = fit2$theta_samps,
                 acceptance = fit2$acceptance) %>% mutate(ix = 1:n())) %>% 
    rbind(.,
          tibble(chain = 3,
                   beta = fit3$beta_samps,
                   theta = fit3$theta_samps,
                   acceptance = fit3$acceptance) %>% mutate(ix = 1:n())) %>% 
    rbind(.,
          tibble(chain = 4, 
                 beta = fit4$beta_samps,
                 theta = fit4$theta_samps,
                 acceptance = fit4$acceptance) %>% mutate(ix = 1:n()))




final_samples <- samples %>% filter(ix>1000,acceptance==1)

final_samples %>% filter(chain!=2,chain!=4) %>% 
    gather(beta,theta,key="Parameters",value="Samples") %>% 
    mutate(true_par = 0.5*(Parameters=="theta") + 1.2*(Parameters=="beta")) %>% 
    ggplot(aes(x=Samples)) + geom_histogram() + theme_bw() + 
    facet_wrap(~Parameters,scales = "free_x") + 
    theme(strip.background = element_blank()) +
    geom_vline(aes(xintercept=true_par),linetype=2) +
    ggtitle("Sample Estimates") + 
        ggsave("~/Google Drive/Academic/UM-Biostatistics/PhD/Brisa Meetings/1_29_19/causaldiffupdate/par_samples.png",
               width =7, height = 5)

theta1 <- mcmc(final_samples %>% filter(chain==1) %>% select(theta) %>% pull())
theta2 <- mcmc(final_samples %>% filter(chain==2) %>% select(theta) %>% pull())
theta3 <- mcmc(final_samples %>% filter(chain==3) %>% select(theta) %>% pull())
theta4 <- mcmc(final_samples %>% filter(chain==4) %>% select(theta) %>% pull())

ml <- mcmc.list(mcmc(theta1[1:771]),
          theta3)
beta1 <- mcmc(final_samples %>% filter(chain==1) %>% select(beta) %>% pull())
mlb <- mcmc.list(mcmc(beta1[1:771]),
             mcmc(final_samples %>% filter(chain==3) %>% select(beta) %>% pull()))

summary(ml)
summary(mlb)
gelman.diag(ml)
gelman.diag(mlb)
effectiveSize(ml)
effectiveSize(mlb)
# Diagnostics -------------------------------------------------------------



beta_trace <- fit$beta_tracel %>% as_data_frame() %>%
    mutate(sample_id = 1:n()) %>% 
    gather(contains("V"),key="step",value="position") %>% 
    mutate(new_step = as.numeric(str_remove(step,"V")))

theta_trace <- fit$theta_tracer %>% as_data_frame() %>% 
    mutate(sample_id = 1:n()) %>% 
    gather(contains("V"),key="step",value="theta_position") %>% 
    mutate(new_step = as.numeric(str_remove(step,"V")))


library(gganimate)
p <- beta_trace %>% left_join(theta_trace,by=c("sample_id","new_step")) %>%
    filter(sample_id %in% samp) %>% mutate(step = as.integer(new_step)) %>% 
    ggplot(aes(x=theta_position,y=position)) + geom_line() + 
    facet_wrap(~sample_id) + ggtitle('NUTS Step') + xlab("Theta") +
    ylab("Beta") + transition_reveal(step) + theme_minimal()

animate(p)
