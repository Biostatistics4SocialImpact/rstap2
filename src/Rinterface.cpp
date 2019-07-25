// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us #include <RcppEigen.h>
#include<chrono>
//#include <cmath>

// via the depends attribute we tell Rcpp to create hooks for
#include "STAP_MathHelpers.hpp"
#include "STAP.hpp"
#include "STAP_Tree.hpp"
#include "STAP_glmer.hpp"
#include "STAP_Tree_glmer.hpp"
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

//
//
// [[Rcpp::export]]
Rcpp::List stap_diffndiff(Eigen::VectorXd& y,
                          Eigen::MatrixXd& Z,
                          Eigen::ArrayXXd& distances,
                          Eigen::ArrayXXi& u_crs,
                          Eigen::MatrixXd& subj_array,
                          Eigen::MatrixXd& subj_n,
                          Eigen::ArrayXi& stap_par_code,
                          const double& adapt_delta,
                          const int& iter_max,
                          const int& max_treedepth,
                          const int& warmup,
                          const int& seed,
                          const bool& diagnostics) {

        auto start = std::chrono::high_resolution_clock::now();
        Eigen::VectorXi acceptance(iter_max);
        acceptance = Eigen::VectorXi::Zero(iter_max);
        // declare placeholder items  for list return
        Eigen::VectorXd treedepth(iter_max);
        Eigen::VectorXd epsilons(iter_max);
        Eigen::VectorXd loglik_out(iter_max);
        Eigen::VectorXd alpha_out(iter_max); 
        Eigen::MatrixXd delta_out(iter_max,Z.cols());
        Eigen::MatrixXd beta_out(iter_max,stap_par_code(2)); 
        Eigen::MatrixXd beta_bar_out(iter_max,stap_par_code(3)); 
        Eigen::VectorXd sigma_out(iter_max); 
        Eigen::VectorXd theta_out(iter_max);
        // fill objects with zer0s
        alpha_out = Eigen::VectorXd::Zero(iter_max);
        delta_out = Eigen::VectorXd::Zero(iter_max);
        beta_bar_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(3));
        beta_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(2));
        theta_out = Eigen::VectorXd::Zero(iter_max);
        sigma_out = Eigen::VectorXd::Zero(iter_max);
        loglik_out = Eigen::VectorXd::Zero(iter_max);
        // random number generator
        std::mt19937 rng;
        rng = std::mt19937(seed);
        std::uniform_real_distribution<double> coin_flip(0.0,1.0);
        // declare stap variable classes 
        SV sv(stap_par_code,rng,diagnostics);
        SV svl(stap_par_code,rng,diagnostics);
        SV svr(stap_par_code,rng,diagnostics);
        sv.initialize_momenta(rng);
        svl.copy_SV(sv);
        svr.copy_SV(sv);

        int n ,s, j, vj;
        double p;
        double epsilon_bar = 1.0;
        double H_bar = 0.0;
        double gamma = 0.05;
        double t_naught = 10;
        double kappa = 0.75;
        double log_z;
        double UTI_one, UTI_two;
        STAP stap_object(distances,u_crs,subj_array,subj_n,Z,y,diagnostics);
        double epsilon = stap_object.FindReasonableEpsilon(sv,rng);
        double mu_beta = log(10*epsilon);
        
        Rcpp::Rcout << "Beginning Sampling" << std::endl;

       for(int iter_ix = 1; iter_ix <= iter_max; iter_ix++){
           if(diagnostics){
                Rcpp::Rcout << "Beginning of iteration: " << iter_ix << std::endl;
                Rcpp::Rcout << "-------------------------------------" << std::endl;
           }else if(iter_ix % (int)round(.1 * iter_max) == 0 ){
               std::string str = iter_ix <= warmup ? "\t [Warmup] " : "\t [Sampling]";
                Rcpp::Rcout << "Beginning of iteration: " << iter_ix << " / " << iter_max << str  << std::endl;
           }
           sv.initialize_momenta(rng);
           sv.initialize_momenta(rng);
           log_z = stap_object.sample_u(sv,rng);
            if(diagnostics)
                Rcpp::Rcout << "log z is : " << log_z << std::endl;
            //equate variables
            svl.copy_SV(sv);
            svr.copy_SV(sv);
            n = 1;
            s = 1;
            j = 0;
            STAP_Tree tree(stap_par_code,diagnostics,rng);
            while(s == 1){
                if(diagnostics)
                    Rcpp::Rcout << "\n Growing Tree with j = " << j << std::endl;
                vj = coin_flip(rng) <= .5 ? 1: -1;
                if(vj == -1){
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the left " << j << std::endl;
                    tree.BuildTree(stap_object,svl,sv,log_z,vj,j,epsilon,rng);
                    svl.copy_SV(tree.get_svl());
                }else{
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the right " << j << std::endl;
                    tree.BuildTree(stap_object,svr,sv,log_z,vj,j,epsilon,rng);
                    svr.copy_SV(tree.get_svr());
                }
                if(tree.get_s_prime() == 1){
                    p = std::min(1.0, tree.get_n_prime() / n);
                    if(coin_flip(rng) <= p){
                        if(diagnostics)
                            Rcpp::Rcout << "sample accepted" << std::endl;
                        acceptance(iter_ix-1) = 1;
                    }
                }
                UTI_one = get_UTI_one(svl,svr);
                UTI_two = get_UTI_two(svl,svr);
                n = n + tree.get_n_prime();
                s = (UTI_one && UTI_two) ? tree.get_s_prime() : 0;
                j++;
                if(j == max_treedepth){
                    Rcpp::Rcout << "Iteration: " << iter_ix << "Exceeded Max Treedepth: " << j << std::endl;
                    break;
                }
                epsilons(iter_ix-1) = epsilon;
              }
            if(iter_ix <= warmup){
                H_bar = (1.0 - 1.0 / (iter_ix + t_naught)) * H_bar + (1.0 /(iter_ix + t_naught)) * (adapt_delta - tree.get_alpha_prime() / tree.get_n_alpha());
                epsilon = exp(mu_beta - (sqrt(iter_ix) / gamma) * H_bar);
                epsilon_bar = exp(pow(iter_ix,-kappa) * log(epsilon) + (1.0 - pow(iter_ix,-kappa)) * log(epsilon_bar));
                if(diagnostics){
                    Rcpp::Rcout << "H_bar: " << H_bar << std::endl;
                    Rcpp::Rcout << "tree alpha " << tree.get_alpha_prime() << std::endl;
                    Rcpp::Rcout << "tree n alpha " << tree.get_n_alpha() << std::endl;
                    Rcpp::Rcout << "epsilon for next iteration is " << epsilon << std::endl;
                }
            }
            else{ 
                epsilon = epsilon_bar;
            }
            treedepth(iter_ix-1) = j;
            if(acceptance(iter_ix-1) == 1){
                alpha_out(iter_ix -1 ) = tree.get_alpha_new();
                delta_out.row(iter_ix -1 ) = tree.get_delta_new();
                beta_bar_out.row(iter_ix - 1) = tree.get_beta_bar_new();
                beta_out.row(iter_ix-1) = tree.get_beta_new();
                theta_out.row(iter_ix-1) = tree.get_theta_new_transformed(); 
                sigma_out(iter_ix-1) = tree.get_sigma_new_transformed(); 
                sv.alpha = tree.get_alpha_new();
                sv.delta = tree.get_delta_new();
                sv.beta_bar = tree.get_beta_bar_new();
                sv.beta = tree.get_beta_new();
                sv.theta = tree.get_theta_new();
                sv.sigma = tree.get_sigma_new();
                loglik_out(iter_ix-1) = stap_object.calculate_ll(sv);
            }
            if((acceptance(iter_ix-1) == 0  && iter_ix > warmup) && diagnostics == false)
                iter_ix = iter_ix - 1;
       }
       auto stop = std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
       double sampling_time = duration.count();

    return Rcpp::List::create(Rcpp::Named("alpha_samps") = alpha_out, 
                              Rcpp::Named("delta_samps") = delta_out,
                              Rcpp::Named("beta_samps") =  beta_out,
                              Rcpp::Named("beta_bar_samps") = beta_bar_out,
                              Rcpp::Named("theta_samps") = theta_out,
                              Rcpp::Named("sigma_samps") = sigma_out,
                              Rcpp::Named("treedepth") = treedepth,
                              Rcpp::Named("acceptance") = acceptance,
                              Rcpp::Named("epsilons") = epsilons,
                              Rcpp::Named("epsilon") = epsilon,
                              Rcpp::Named("loglik") = loglik_out,
                              Rcpp::Named("sampling_time") = sampling_time );
}

     
Rcpp::List test_grads(Eigen::VectorXd& y,
                      Eigen::MatrixXd& Z,
                      Eigen::VectorXd& beta_bar,
                      Eigen::VectorXd& beta,
                      Eigen::ArrayXXd &distances,
                      Eigen::ArrayXXi &u_crs,
                      Eigen::MatrixXd &subj_array,
                      Eigen::MatrixXd &subj_n,
                      Eigen::VectorXd &par_grid,
                      Eigen::ArrayXi& stap_par_code,
                      const int seed) {

        std::mt19937 rng;
        rng = std::mt19937(seed);
        STAP stap_object(distances,u_crs,subj_array,subj_n,Z,y,true);
        Eigen::VectorXd grad_grid(par_grid.size());
        Eigen::VectorXd energy_grid(par_grid.size());
        SV sv(stap_par_code,rng,true);
        sv.beta_bar(0) = 1.0;
        sv.beta(0) = 1.2;
        sv.alpha = 22;
        sv.delta(0) = -.5;
        sv.sigma = 0;
        sv.am = 0;
        sv.bm = Eigen::VectorXd::Zero(1);
        sv.bbm =Eigen::VectorXd::Zero(1);
        sv.tm = Eigen::VectorXd::Zero(1);
        sv.dm = Eigen::VectorXd::Zero(1);
        sv.theta(0) = log(1.0 / 19.0);

        for(int i = 0; i < par_grid.size(); i++){
            sv.theta(0) = par_grid(i);
            stap_object.calculate_gradient(sv);
            grad_grid(i) = stap_object.sg.theta_grad(0);
            energy_grid(i) = stap_object.calculate_total_energy(sv);
        }



    return Rcpp::List::create(Rcpp::Named("energy") = energy_grid,
                              Rcpp::Named("grad") = grad_grid);
}
