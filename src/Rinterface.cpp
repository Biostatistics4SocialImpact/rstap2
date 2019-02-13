// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppEigen.h which pulls Rcpp.h in for us
#include <RcppEigen.h>
//#include <cmath>

// via the depends attribute we tell Rcpp to create hooks for
#include "STAP.hpp"
#include "STAP_Tree.hpp"
#include "STAP_MathHelpers.hpp"
#include "STAP_Vars.hpp"
// RcppEigen so that the build process will know what to do
//
// [[Rcpp::depends(RcppEigen)]]

//
//
// [[Rcpp::export]]
Rcpp::List stap_diffndiff(Eigen::VectorXd& y,
                          double beta, 
                          double theta,
                          double sigma,
                          Eigen::ArrayXXd& distances,
                          Eigen::ArrayXXi& u_crs,
                          Eigen::MatrixXd& subj_array,
                          Eigen::ArrayXd& subj_n,
                          Eigen::ArrayXi& stap_par_code,
                          Eigen::ArrayXi& stap_bar_code,
                          const double& adapt_delta,
                          const int& iter_max,
                          const int& max_treedepth,
                          const int& warmup,
                          const int& seed,
                          const bool& diagnostics) {

        Eigen::VectorXi acceptance(iter_max);
        acceptance = Eigen::VectorXi::Zero(iter_max);
        Eigen::VectorXd epsilon_betas(iter_max);
        Eigen::VectorXd epsilon_thetas(iter_max);
        Eigen::VectorXd beta_out(iter_max); 
        Eigen::VectorXd sigma_out(iter_max); 
        Eigen::VectorXd alpha_out(iter_max); 
        Eigen::VectorXd nalpha_out(iter_max);
        beta_out = Eigen::VectorXd::Zero(iter_max);
        Eigen::VectorXd theta_out(iter_max);
        theta_out = Eigen::VectorXd::Zero(iter_max);
        Eigen::VectorXd treedepth(iter_max);
        std::mt19937 rng;
        rng = std::mt19937(seed);
        std::uniform_real_distribution<double> coin_flip(0.0,1.0);
        double beta_init = beta;
        double theta_init = theta;
        double sigma_init = sigma;
        double beta_left; 
        double beta_right;
        double theta_left;
        double theta_right;
        double sigma_left;
        double sigma_right;
        double bm;
        double bml;
        double bmr; 
        double tm;
        double tml;
        double tmr;
        double sm;
        double sml;
        double smr;

        int n ,s, j, vj;
        double p;
        double epsilon_bar_beta = 1.0;
        double epsilon_bar_theta = 1.0;
        double H_bar = 0.0;
        double gamma = 0.05;
        double t_naught = 10;
        double kappa = 0.75;
        double log_z;
        double UTI_one, UTI_two;
        STAP stap_object(distances,u_crs,subj_array,subj_n,y,diagnostics);
        bm = GaussianNoise_scalar(rng); 
        tm = GaussianNoise_scalar(rng);
        sm = GaussianNoise_scalar(rng);
        double epsilon_theta = stap_object.FindReasonableEpsilon(beta,theta,sigma,bm,tm,sm,rng);
        double epsilon_beta = epsilon_theta;
        double mu_theta = log(10*epsilon_theta);
        double mu_beta = log(10*epsilon_theta);


        Rcpp::Rcout << "Beginning Sampling" << std::endl;
        
        
       for(int iter_ix = 1; iter_ix <= iter_max; iter_ix++){
           if(diagnostics){
                Rcpp::Rcout << "Beginning of iteration: " << iter_ix << std::endl;
                Rcpp::Rcout << "-------------------------------------" << std::endl;
           }else if(iter_ix % 10 == 0 ){
                Rcpp::Rcout << "Beginning of iteration: " << iter_ix << std::endl;
                Rcpp::Rcout << "-------------------------------------" << std::endl;
           }
            bm = GaussianNoise_scalar(rng);
            tm = GaussianNoise_scalar(rng);
            sm = GaussianNoise_scalar(rng);
            log_z = stap_object.sample_u(beta_init,theta_init,sigma_init,bm,tm,sm,rng);
            if(diagnostics)
                Rcpp::Rcout << "log z is : " << log_z << std::endl;
            //equate variables
            beta_left = beta_init;
            beta_right = beta_init;
            theta_left = theta_init;
            theta_right = theta_init;
            sigma_left = sigma_init;
            sigma_right = sigma_init;
            // equate momenta
            bmr = bm;
            bml = bm;
            tml = tm;
            tmr = tm;
            smr = sm;
            sml = sm;
            n = 1;
            s = 1;
            j = 0;
            STAP_Tree tree(diagnostics);
            while(s == 1){
                if(diagnostics)
                    Rcpp::Rcout << "\n Growing Tree with j = " << j << std::endl;
                vj = coin_flip(rng) <= .5 ? 1: -1;
                if(vj == -1){
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the left " << j << std::endl;
                    tree.BuildTree(stap_object,beta_left,theta_left,sigma_left,beta_init,theta_init,sigma_init,bml,tml,sml,bm,tm,sm,log_z,vj,j,epsilon_beta,epsilon_theta,rng);
                    beta_left = tree.get_bl();
                    bml = tree.get_bml();
                    theta_left = tree.get_tl();
                    tml = tree.get_tml();
                    sigma_left = tree.get_sl();
                    sml = tree.get_sml();
                }else{
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the right " << j << std::endl;
                    tree.BuildTree(stap_object,beta_right,theta_right,sigma_right,beta_init,theta_init,sigma_init,bmr,tmr,smr,bm,tm,sm,log_z,vj,j,epsilon_beta,epsilon_theta,rng);
                    beta_right = tree.get_br();
                    bmr = tree.get_bmr();
                    theta_right = tree.get_tr();
                    tmr = tree.get_tmr();
                    sigma_right = tree.get_sr();
                    smr = tree.get_smr();
                }
                if(tree.get_s_prime() == 1){
                    p = std::min(1.0, tree.get_n_prime() / n);
                    if(coin_flip(rng) <= p){
                        if(diagnostics)
                            Rcpp::Rcout << "sample accepted" << std::endl;
                        acceptance(iter_ix-1) = 1;
                        beta = tree.get_beta_new();
                        theta = tree.get_theta_new();
                        sigma = tree.get_sigma_new();
                        beta_out(iter_ix-1) = beta;
                        theta_out(iter_ix-1) = 10 / (1 + exp(-theta)) ; 
                        sigma_out(iter_ix-1) = exp(sigma);
                    }
                }
                UTI_one = ( ( (beta_right - beta_left) * bmr + (theta_right - theta_left)*tmr) + ((sigma_right - sigma_left) * smr) >=0 ) ;
                UTI_two = ( ( (beta_right - beta_left) * bml + (theta_right - theta_left)*tml) + ((sigma_right - sigma_left) * sml) >=0 );
                n = n + tree.get_n_prime();
                s = (UTI_one && UTI_two) ? tree.get_s_prime() : 0;
                j++;
                if((j == max_treedepth && iter_ix > warmup) || (j>12) ){
                    Rcpp::Rcout << "Iteration: " << iter_ix << "Exceeded Max Treedepth: " << j << std::endl;
                    break;
                }
                epsilon_betas(iter_ix-1) = epsilon_beta;
                epsilon_thetas(iter_ix-1) = epsilon_theta;
            }
            if(iter_ix <= warmup){
                alpha_out(iter_ix-1) = tree.get_alpha_prime();
                nalpha_out(iter_ix-1) = tree.get_n_alpha();
                H_bar = (1.0 - 1.0 / (iter_ix + t_naught)) * H_bar + (1.0 /(iter_ix + t_naught)) * (adapt_delta - tree.get_alpha_prime() / tree.get_n_alpha());
                epsilon_beta = exp(mu_beta - (sqrt(iter_ix) / gamma) * H_bar);
                epsilon_theta = exp(mu_theta - (sqrt(iter_ix) / gamma) * H_bar);
                epsilon_bar_theta = exp(pow(iter_ix,-kappa) * log(epsilon_theta) + (1.0 - pow(iter_ix,-kappa)) * log(epsilon_bar_theta));
                epsilon_bar_beta = exp(pow(iter_ix,-kappa) * log(epsilon_beta) + (1.0 - pow(iter_ix,-kappa)) * log(epsilon_bar_beta));
                if(diagnostics){
                    Rcpp::Rcout << "tree alpha " << tree.get_alpha_prime() << std::endl;
                    Rcpp::Rcout << "tree n alpha " << tree.get_n_alpha() << std::endl;
                    Rcpp::Rcout << "epsilon for next iteration is " << epsilon_beta << std::endl;
                }
            }
            else 
                epsilon_theta = epsilon_bar_theta;
                epsilon_beta = epsilon_bar_beta;
            
            beta_init = beta;
            theta_init = theta;
            sigma_init = sigma;
            treedepth(iter_ix-1) = j;
            if((acceptance(iter_ix-1) == 0  && iter_ix > warmup) && diagnostics==false) iter_ix = iter_ix - 1;
        }
    
    return Rcpp::List::create(Rcpp::Named("beta_samps") =  beta_out,
                              Rcpp::Named("theta_samps") = theta_out,
                              Rcpp::Named("sigma_samps") = sigma_out,
                              Rcpp::Named("alpha") = alpha_out,
                              Rcpp::Named("n_alpha") = nalpha_out,
                              Rcpp::Named("treedepth") = treedepth,
                              Rcpp::Named("acceptance") = acceptance,
                              Rcpp::Named("epsilon_betas") = epsilon_betas,
                              Rcpp::Named("epsilon_thetas") = epsilon_thetas,
                              Rcpp::Named("epsilon_beta") = epsilon_beta,
                              Rcpp::Named("epsilon_theta") = epsilon_theta);
   
}

/*Rcpp::List test_grads(Eigen::VectorXd &y,
                          double beta, 
                          double theta,
                          Eigen::ArrayXXd &distances,
                          Eigen::ArrayXXi &u_crs,
                          Eigen::MatrixXd &subj_array,
                          Eigen::ArrayXd &subj_n,
                          Eigen::VectorXd &theta_grid,
                          Eigen::VectorXd &beta_grid
                          ) {

        double bm = 0.0;
        double tm = 0.0;

        STAP stap_object(distances,u_crs,subj_array,subj_n,y);
        Eigen::VectorXd beta_one(theta_grid.size() * beta_grid.size());
        Eigen::VectorXd theta_one(theta_grid.size() * beta_grid.size());
        Eigen::VectorXd th_grad_grid(theta_grid.size() * beta_grid.size());
        Eigen::VectorXd bt_grad_grid(theta_grid.size() * beta_grid.size());
        Eigen::VectorXd energy_grid(theta_grid.size() * beta_grid.size());
        Eigen::VectorXd mean_Xd_grid(theta_grid.size() * beta_grid.size());
        int cntr = 0;
        

        for(int i = 0; i < theta_grid.size(); i++){
            for(int j =0; j<beta_grid.size();j++){
                beta_one(cntr) = beta_grid(j);
                theta_one(cntr) = theta_grid(i);
                stap_object.calculate_gradient(beta_grid(j),theta_grid(i));
                energy_grid(cntr) = stap_object.calculate_total_energy(beta_grid(j),theta_grid(i),bm,tm);
                th_grad_grid(cntr) = stap_object.get_theta_grad(); 
                bt_grad_grid(cntr) = stap_object.get_beta_grad();
                mean_Xd_grid(cntr) = (stap_object.get_X_diff()).mean();
                cntr ++ ;
            }
        }



    return Rcpp::List::create(Rcpp::Named("beta_gradient") =  bt_grad_grid,
                              Rcpp::Named("theta_gradient") = th_grad_grid,
                              Rcpp::Named("beta_grid") = beta_one,
                              Rcpp::Named("Xmn_grid") = mean_Xd_grid,
                              Rcpp::Named("theta_grid") = theta_one,
                              Rcpp::Named("energy") = energy_grid );
}*/
