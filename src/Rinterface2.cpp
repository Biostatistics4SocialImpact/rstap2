// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
#include <random>

// via the depends attribute we tell Rcpp to create hooks for
#include "STAP_MathHelpers.hpp"
#include "STAP_Vars.hpp"
#include "STAPdnd.hpp"
#include "STAPdnd_Tree.hpp"
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

//
//
// [[Rcpp::export]]
Rcpp::List stap_diffndiff(Eigen::VectorXd &y,
                          Eigen::MatrixXd &Z,
                          Eigen::VectorXd delta,
                          Eigen::VectorXd beta, 
                          Eigen::VectorXd theta,
                          double sigma,
                          const Eigen::MatrixXd &distances,
                          const Eigen::MatrixXi &u_crs,
                          const Eigen::VectorXi &subj_array,
                          const Eigen::VectorXi &subj_n,
                          const Eigen::VectorXi &diff_coefs,
                          const int family,
                          const int link,
                          const double &adapt_delta, 
                          const int iter_max,
                          const double sd_beta,
                          const double sd_theta,
                          const int max_treedepth,
                          const int warmup,
                          const int seed) {

        Eigen::VectorXi acceptance(iter_max);
        acceptance = Eigen::VectorXi::Zero(iter_max);
        Eigen::VectorXd epsilons(iter_max);
        Eigen::MatrixXd delta_out(iter_max,delta.size()); 
        Eigen::MatrixXd beta_out(iter_max,beta.size()); 
        Eigen::MatrixXd theta_out(iter_max,theta.size()); 
        Eigen::VectorXd sigma_out(iter_max); 
        Eigen::VectorXi treedepth(iter_max);
        delta_out = Eigen::MatrixXd::Zero(iter_max,delta.size());
        beta_out = Eigen::MatrixXd::Zero(iter_max,beta.size());
        theta_out = Eigen::MatrixXd::Zero(iter_max,theta.size());
        sigma_out = Eigen::VectorXd::Zero(iter_max);
        std::mt19937 rng;
        rng = std::mt19937(seed);
        std::uniform_real_distribution<double> coin_flip(0.0,1.0);
        STAP stap_object(distances,u_crs,subj_array,subj_n,diff_coefs,Z,y);
        STAP_Vars sv(delta,beta,theta,sigma);
        STAP_Vars svl(delta,beta,theta,sigma);
        STAP_Vars svr(delta,beta,theta,sigma);

        int n ,s, j, vj;
        double p;
        double epsilon_bar = 1.0;
        double H_bar = 0.0;
        double gamma = 0.05;
        double t_naught = 1;
        double kappa = 0.75;
        double log_z;
        double UTI_one, UTI_two;
        double epsilon = stap_object.FindReasonableEpsilon(sv,rng);
        double mu = log(10*epsilon);


        Rcpp::Rcout << "Beginning Sampling" << std::endl;

        for(int iter_ix = 1; iter_ix <= iter_max; iter_ix++){
            if(iter_ix % 10 == 0){
                Rcpp::Rcout << "Beginning of iteration: " << iter_ix << std::endl;
                Rcpp::Rcout << "-------------------------------------" << std::endl;
            }
            sv.update_momenta(rng);
            svl.copy_momenta(sv);
            svr.copy_momenta(sv);
            svl.position_update(sv);
            svr.position_update(sv);
            log_z = stap_object.sample_log_u(sv,rng);
            n = 1;
            s = 1;
            j = 0;
            STAP_Tree tree(delta.size(),beta.size(),theta.size());
            while(s == 1){
                //Rcpp::Rcout << "Growing Tree with j = " << j << std::endl;
                vj = coin_flip(rng) <= .5 ? 1: -1;
                if(vj == -1){
                //Rcpp::Rcout << "Growing Tree to the left " << j << std::endl;
                    tree.BuildTree(stap_object,svl,sv,log_z,vj,j,epsilon,rng);
                    //Rcpp::Rcout << "left branch at top: " << tree.get_bl() << std::endl;
                    svl = tree.get_svl();
                }else{
                //Rcpp::Rcout << "Growing Tree to the right " << j << std::endl;
                    tree.BuildTree(stap_object,svr,sv,log_z,vj,j,epsilon,rng);
                    //Rcpp::Rcout << "right branch at top: " << tree.get_bl() << std::endl;
                    svr = tree.get_svr();
                }
                if(tree.get_s_prime() == 1){
                    p = std::min(1.0, tree.get_n_prime() / n);
                    if(coin_flip(rng) <= p){
                        //Rcpp::Rcout << "sample accepted" << std::endl;
                        acceptance(iter_ix-1) = 1;
                        sv.position_update(tree.get_sv_new());
                        delta_out.row(iter_ix-1) = sv.delta;
                        beta_out.row(iter_ix-1) = sv.beta;
                        theta_out.row(iter_ix-1) = sv.get_transformed_theta(); 
                        sigma_out(iter_ix-1) = sv.get_transformed_sigma();
                    }
                }
                UTI_one = get_UTI_one(svl,svr);
                UTI_two = get_UTI_two(svl,svr);
                n = n + tree.get_n_prime();
                s = (UTI_one && UTI_two) ? tree.get_s_prime() : 0;
                j++;
                epsilons(iter_ix-1) = epsilon;
                if(iter_ix > warmup && (acceptance(iter_ix -1) == 0))
                    iter_ix = iter_ix - 1;
            }
            if(iter_ix <= warmup){
                H_bar = (1.0 - 1.0 / (iter_ix + t_naught)) * H_bar + (1.0 /(iter_ix + t_naught)) * (adapt_delta - tree.get_alpha_prime() / tree.get_n_alpha());
                epsilon = exp(mu - sqrt(iter_ix) / gamma * H_bar);
                epsilon_bar = exp(pow(iter_ix,-kappa) * log(epsilon) + (1.0 - pow(iter_ix,-kappa)) * log(epsilon_bar));
            }
            else 
                epsilon = epsilon_bar;
            sv.position_update(sv); 
            treedepth(iter_ix-1) = j;
        }

    return Rcpp::List::create(Rcpp::Named("beta_samps") =  beta_out,
                              Rcpp::Named("theta_samps") = theta_out,
                              Rcpp::Named("treedepth") = treedepth,
                              Rcpp::Named("acceptance") = acceptance,
                              Rcpp::Named("epsilons") = epsilons,
                              Rcpp::Named("epsilon") = epsilon );
   
}
