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
num_subj <- 1E4
num_bef <- 40
theta_s <- .5
beta <- 1.2
sigma <- 1
dists_1 <- matrix(rexp(n = num_subj*num_bef,
                       rate = 2),
                nrow = num_subj,
                ncol = num_bef)
dists_2 <- matrix(rexp(n = num_subj*num_bef,
                       rate = 1.5),
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
X <- c(X_theta_one,X_theta_two,X_theta_three)
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
                theta = theta_init,
                distances = dists,
                d_one = d_one , d_two = d_two,d_three = d_three,
               adapt_delta = .85,
               warmup = 25, iter_max = 35,
                seed = 23 )

plot(10/(1+exp(-fit$theta_grid)),fit$energy,type='l')
