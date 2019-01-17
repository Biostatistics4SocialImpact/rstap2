// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
//#include <cmath>

// via the depends attribute we tell Rcpp to create hooks for
#include "STAP.hpp"
#include "STAP_Tree.hpp"
#include "STAP_MathHelpers.hpp"
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

//
//
// [[Rcpp::export]]
Rcpp::List stap_diffndiff(Eigen::VectorXd &y,
                          double beta, 
                          double theta,
                          Eigen::MatrixXd &distances,
                          Eigen::MatrixXd &d_one,
                          Eigen::MatrixXd &d_two,
                          Eigen::MatrixXd &d_three,
                          const double &adapt_delta,
                          const int iter_max,
                          const int warmup,
                          const int seed) {

        Eigen::VectorXd acceptance(iter_max);
        Eigen::VectorXd beta_out(iter_max); 
        beta_out = Eigen::VectorXd::Zero(iter_max);
        Eigen::VectorXd theta_out(iter_max);
        theta_out = Eigen::VectorXd::Zero(iter_max);
        Eigen::VectorXd max_treedepth(iter_max);
        std::mt19937 rng;
        rng = std::mt19937(seed);
        std::uniform_real_distribution<double> coin_flip(0.0,1.0);
        double beta_init = beta;
        double theta_init = theta;
        double beta_left; 
        double beta_right;
        double theta_left;
        double theta_right;
        double bm;
        double bml;
        double bmr; 
        double tm;
        double tml;
        double tmr;

        int n ,s, j, vj;
        double p;
        double epsilon = 1.0;
        double epsilon_bar = 1.0;
        double H_bar = 0;
        double gamma = 0.05;
        double t_naught = 10;
        double kappa = 0.75;
        double mu = log(10*epsilon);
        double u;
        double UTI_one, UTI_two;
        STAP stap_object(distances,d_one,d_two,d_three,y);
        STAP_Tree tree;



        for(int iter_ix = 1; iter_ix <= iter_max; iter_ix++){
            Rcpp::Rcout << "Beginning of iteration: " << iter_ix << std::endl;
            Rcpp::Rcout << "-------------------------------------" << std::endl;
            Rcpp::Rcout << "epsilon for this iteration is: " << epsilon << std::endl;
            Rcpp::Rcout << "Theta init: " << 10.0 /( 1 + exp(-theta_init)) << std::endl;
            bm = GaussianNoise_scalar(rng);
            tm = GaussianNoise_scalar(rng);
            u = stap_object.sample_u(beta_init,theta_init,bm,tm,rng);
            Rcpp::Rcout << "u is: " << u << std::endl;
            //equate variables
            beta_left = beta_init;
            beta_right = beta_init;
            theta_left = theta_init;
            theta_right = theta_init;
            // equate momenta
            bmr = bm;
            bml = bm;
            tml = tm;
            tmr = tm;
            n = 1;
            s = 1;
            j = 0;
            while(s == 1){
                Rcpp::Rcout << "Growing Tree with j = " << j << std::endl;
                vj = coin_flip(rng) <= .5 ? 1: -1;
                if(vj == -1){
                    tree.BuildTree(stap_object,beta_left,theta_left,beta_init,theta_init,bml,tml,bm,tm,u,vj,j,epsilon,rng);
                    beta_left = tree.get_bl();
                    bml = tree.get_bml();
                    theta_left = tree.get_tl();
                    tml = tree.get_tml();
                }else{
                    tree.BuildTree(stap_object,beta_right,theta_right,beta_init,theta_init,bmr,tmr,bm,tm,u,vj,j,epsilon,rng);
                    beta_right = tree.get_br();
                    bmr = tree.get_bmr();
                    theta_right = tree.get_tr();
                    tmr = tree.get_tmr();
                }
                if(tree.get_s_prime() == 1){
                    p = std::min(1.0, tree.get_n_prime() / n);
                    if(coin_flip(rng) <= p){
                        Rcpp::Rcout << "Sample Accepted" << std::endl;
                        acceptance(iter_ix-1) = 1;
                        beta = tree.get_beta_new();
                        theta = tree.get_theta_new();
                        beta_out(iter_ix-1) = beta;
                        theta_out(iter_ix-1) = theta; 
                    }
                }
                UTI_one = pow((beta_right - beta_left)*bml,2) + pow((theta_right -theta_left)*tml,2);
                UTI_two = pow((beta_right - beta_left)*bmr,2) + pow((theta_right - theta_left)*tmr,2);
                n = n + tree.get_n_prime();
                s = (UTI_one >= 0.0 & UTI_two >= 0.0) ? tree.get_s_prime() : 0;
                j++;
                if(j > 15){
                    Rcpp::Rcout << "Iteration: " << iter_ix << "Exceeded Max Treedepth: " << j << std::endl;
                    break;
                }
            }
            if(iter_ix <= warmup){
                H_bar = (1.0 - 1.0 / (iter_ix + t_naught)) * H_bar + (1.0 /(iter_ix + t_naught)) * (adapt_delta - tree.get_alpha_prime() / tree.get_n_alpha());
                epsilon = exp(mu - sqrt(iter_ix) / gamma * H_bar);
                epsilon_bar = exp(pow(iter_ix,-kappa) * log(epsilon) + (1.0 - pow(iter_ix,-kappa)) * log(epsilon_bar));
            }
            else 
                epsilon = epsilon_bar;
            
            beta_init = beta;
            theta_init = theta;
            max_treedepth(iter_ix-1) = j;
            Rcpp::Rcout << "End of iteration: " << iter_ix << std::endl;
            Rcpp::Rcout << "epsilon at end of this iteration is: " << epsilon << std::endl;
        }

    
    return Rcpp::List::create(Rcpp::Named("beta_samps") =  beta_out,
                              Rcpp::Named("theta_samps") = theta_out,
                              Rcpp::Named("treedepth") = max_treedepth,
                              Rcpp::Named("epsilon") = epsilon );
   
}
