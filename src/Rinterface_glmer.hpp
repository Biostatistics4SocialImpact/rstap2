#include "STAP_glmer.hpp"
#include "STAP_Tree_glmer.hpp"


//[[Rcpp::export]]
Rcpp::List test_sparse_matrix(Eigen::VectorXd& b,
                              const SEXP WW){

  try{
      const Eigen::MappedSparseMatrix<double> W(Rcpp::as<Eigen::MappedSparseMatrix<double> >(WW));
      Rcpp::Rcout << "W' * b" << (W.transpose() * b).head(5) << std::endl;
      Rcpp::Rcout << "W' " << (W.block(0,0,5,5))<< std::endl;
    }
  catch (std::exception& _ex_) {
    forward_exception_to_r(_ex_);
  }


  return (Rcpp::List::create(Rcpp::Named("out") = 0 ) );

}





//' NUTS estimation for STAP model with either subj int or subj int and slope
//' @param y vector of continuous outcomes
//' @param Z matrix of confounders 
//' @param W matrix of group-level effect slope covariates should be num_subj x 600 
//' @param distances array of distances
//' @param u_crs  compressed row storage indices for distances
//' @param subj_matrix_ N X n sparse matrix that has a 1 in element ij if subject i has an outcome in y (i+j) 
//' @param subj_n  n x 1 matrix that has 1/n_i in element i1 for the ith subject
//' @param stap_par_code detailed code for estimation set-up - see details
//' @param adapt_delta tuning parameter for adaptation
//' @param iter_max  maximum number of iterations
//' @param max_treedepth  maximum number of branches to grow for NUTS
//' @param warmup number of iterations for which to tune sampler
//' @param seed seed to initialize random number generator
//' @param diagnostics -for development use only
//[[Rcpp::export]]
Rcpp::List stapdnd_glmer(Eigen::VectorXd& y,
                          Eigen::MatrixXd& Z,
                          const SEXP WW,
                          Eigen::ArrayXXd& distances,
                          Eigen::ArrayXXi& u_crs,
                          SEXP subj_matrix_,
                          Eigen::MatrixXd& subj_n,
                          Eigen::ArrayXi& stap_par_code,
                          const double& adapt_delta,
                          const int& iter_max,
                          const int& max_treedepth,
                          const int& warmup,
                          const int& seed,
                          const bool& diagnostics) {

        auto start = std::chrono::high_resolution_clock::now();

        const Eigen::MappedSparseMatrix<double> W(Rcpp::as<Eigen::MappedSparseMatrix<double> >(WW));
        Eigen::MappedSparseMatrix<double> subj_matrix(Rcpp::as<Eigen::MappedSparseMatrix<double> >(subj_matrix_));
        Eigen::VectorXi acceptance(iter_max);
        acceptance = Eigen::VectorXi::Zero(iter_max);
        // declare placeholder items for list return
        Eigen::VectorXd treedepth(iter_max);
        Eigen::VectorXd epsilons(iter_max);
        Eigen::VectorXd loglik_out(iter_max);
        Eigen::VectorXd alpha_out(iter_max); 
        Eigen::MatrixXd delta_out(iter_max,Z.cols());
        Eigen::MatrixXd beta_out(iter_max,stap_par_code(2)); 
        Eigen::MatrixXd beta_bar_out(iter_max,stap_par_code(3)); 
        Eigen::VectorXd sigma_out(iter_max); 
        Eigen::VectorXd theta_out(iter_max);
        Eigen::MatrixXd b1_out(iter_max,stap_par_code(4));
        Eigen::MatrixXd b2_out(iter_max,stap_par_code(4));
        Eigen::MatrixXd cov_out(iter_max,stap_par_code(5) == 1 ? 1 : 4);

        // fill objects with zer0s
        alpha_out = Eigen::VectorXd::Zero(iter_max);
        b1_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(4));
        b2_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(4));
        delta_out = Eigen::VectorXd::Zero(iter_max);
        beta_bar_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(2));
        beta_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(2));
        theta_out = Eigen::VectorXd::Zero(iter_max);
        sigma_out = Eigen::VectorXd::Zero(iter_max);
        loglik_out = Eigen::VectorXd::Zero(iter_max);
        cov_out = Eigen::MatrixXd::Zero(iter_max,stap_par_code(5) == 1 ? 1 : 4 );

        // random number generator
        std::mt19937 rng;
        rng = std::mt19937(seed);
        std::uniform_real_distribution<double> coin_flip(0.0,1.0);

        // declare stap variable classes 
        SV_glmer sv(stap_par_code,rng,diagnostics);
        SV_glmer svl(stap_par_code,rng,diagnostics);
        SV_glmer svr(stap_par_code,rng,diagnostics);
        sv.initialize_momenta(rng);
        svl.copy_SV_glmer(sv);
        svr.copy_SV_glmer(sv);

        // standardize Z
        
        Eigen::VectorXd z_bar = Z.colwise().mean();
        Eigen::MatrixXd Z_std = Z.rowwise() - z_bar.transpose();

        double y_bar = 0; 
        double y_sd = 1;


        int n ,s, j, vj;
        double p;
        double epsilon_bar = 1.0;
        double H_bar = 0.0;
        double gamma = 0.05;
        double t_naught = 10;
        double kappa = 0.75;
        double log_z;
        double UTI_one, UTI_two;
        STAP_glmer stap_object(distances,u_crs,subj_matrix,subj_n,Z_std,W,y,diagnostics);
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
           log_z = stap_object.sample_u(sv,rng);
            if(diagnostics)
                Rcpp::Rcout << "log z is : " << log_z << std::endl;
            //equate variables
            svl.copy_SV_glmer(sv);
            svr.copy_SV_glmer(sv);
            n = 1;
            s = 1;
            j = 0;
            STAP_Tree_glmer tree(stap_par_code,diagnostics,rng);
            while(s == 1){
                if(diagnostics)
                    Rcpp::Rcout << "\n Growing Tree with j = " << j << std::endl;
                vj = coin_flip(rng) <= .5 ? 1: -1;
                if(vj == -1){
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the left " << j << std::endl;
                    tree.BuildTree(stap_object,svl,sv,log_z,vj,j,epsilon,rng);
                    svl.copy_SV_glmer(tree.get_svl());
                }else{
                    if(diagnostics)
                        Rcpp::Rcout << "Growing Tree to the right " << j << std::endl;
                    tree.BuildTree(stap_object,svr,sv,log_z,vj,j,epsilon,rng);
                    svr.copy_SV_glmer(tree.get_svr());
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
                if(diagnostics){
                    Rcpp::Rcout << "Checking s" << std::endl;
                    Rcpp::Rcout << "UTI_one: " << UTI_one << std::endl;
                    Rcpp::Rcout << "UTI_two: " << UTI_two << std::endl;
                    Rcpp::Rcout << "Tree s: " << tree.get_s_prime() << std::endl;
                }
                j++;
                if(j == max_treedepth ){
                    if(iter_ix > warmup)
                      Rcpp::Rcout << "Iteration: " << iter_ix << "Exceeded Max Treedepth: " << j << std::endl;
                    break;
                }
              }

            if(iter_ix <= warmup){
                epsilons(iter_ix-1) = epsilon;
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
                epsilons(iter_ix-1) = epsilon;
                epsilon = epsilon_bar;
            }
            treedepth(iter_ix-1) = j;
            if(acceptance(iter_ix-1) == 1){
                sv.alpha = tree.get_alpha_new();
                sv.delta = tree.get_delta_new();
                sv.beta_bar = tree.get_beta_bar_new();
                sv.beta = tree.get_beta_new();
                sv.theta = tree.get_theta_new();
                sv.sigma = tree.get_sigma_new();
                sv.Sigma = tree.get_Sigma_new();
                sv.b = tree.get_b_new();
                sv.b_slope = tree.get_b_slope_new();
                loglik_out(iter_ix-1) = stap_object.calculate_ll(sv);
                alpha_out(iter_ix -1 ) = adjust_alpha(stap_object,sv,y_bar,y_sd,z_bar); 
                delta_out.row(iter_ix -1 ) = tree.get_delta_new() * y_sd;
                beta_bar_out.row(iter_ix - 1) = tree.get_beta_bar_new() * y_sd;
                beta_out.row(iter_ix-1) = tree.get_beta_new() * y_sd;
                sigma_out(iter_ix-1) = exp(tree.get_sigma_new()) * y_sd ; 
                theta_out.row(iter_ix-1) = tree.get_theta_new_transformed();
                b1_out.row(iter_ix-1) = sv.adjust_b() * y_sd;
                b2_out.row(iter_ix-1) = sv.adjust_b_slope() * y_sd;
                if(stap_par_code(5) == 1)
                    cov_out(iter_ix-1,0) = exp(tree.get_Sigma_new_transformed()(0,0));
                else 
                    cov_out.row(iter_ix-1) = tree.get_Sigma_new_transformed();
            }
            if((acceptance(iter_ix-1) == 0  && iter_ix > warmup) && diagnostics == false)
                iter_ix = iter_ix - 1;
       }

       auto stop = std::chrono::high_resolution_clock::now();
       auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop-start);
       double sampling_time = duration.count();

    return(Rcpp::List::create(Rcpp::Named("alpha_samps") = alpha_out, 
                              Rcpp::Named("delta_samps") = delta_out,
                              Rcpp::Named("beta_samps") =  beta_out,
                              Rcpp::Named("beta_bar_samps") = beta_bar_out,
                              Rcpp::Named("b_int_samps") = b1_out,
                              Rcpp::Named("b_slope_samps") = b2_out,
                              Rcpp::Named("theta_samps") = theta_out,
                              Rcpp::Named("sigma_samps") = sigma_out,
                              Rcpp::Named("Sigma_samps") = cov_out,
                              Rcpp::Named("treedepth") = treedepth,
                              Rcpp::Named("acceptance") = acceptance,
                              Rcpp::Named("sampling_time") = sampling_time,
                              Rcpp::Named("epsilons") = epsilons,
                              Rcpp::Named("epsilon") = epsilon,
                              Rcpp::Named("loglik") = loglik_out));
   
}

//[[Rcpp::export]]
Rcpp::List test_grads_glmer(Eigen::VectorXd& y,
                      Eigen::MatrixXd& Z,
                      const SEXP WW,
                      Eigen::VectorXd& true_b_int,
                      Eigen::VectorXd& true_b_slope,
                      Eigen::VectorXd& beta_bar,
                      Eigen::VectorXd& beta,
                      Eigen::ArrayXXd &distances,
                      Eigen::ArrayXXi &u_crs,
                      SEXP subj_array_,
                      Eigen::MatrixXd &subj_n,
                      Eigen::VectorXd &par_grid,
                      Eigen::ArrayXi& stap_par_code,
                      const int seed) {

        const Eigen::MappedSparseMatrix<double> W(Rcpp::as<Eigen::MappedSparseMatrix<double> >(WW));
        Eigen::MappedSparseMatrix<double> subj_array(Rcpp::as<Eigen::MappedSparseMatrix<double> >(subj_array_));
        std::mt19937 rng;
        rng = std::mt19937(seed);
        Eigen::VectorXd grad_grid(par_grid.size());
        grad_grid = Eigen::VectorXd::Zero(par_grid.size());
        Eigen::VectorXd energy_grid(par_grid.size());
        STAP_glmer stap_object(distances,u_crs,subj_array,subj_n,Z,W,y,true);
        SV_glmer sv(stap_par_code,rng,true);
        sv.alpha = R::runif(-2,2);
        sv.beta(0) = R::runif(-2,2);
        sv.beta_bar(0) = R::runif(-2,2);
        sv.delta(0) = R::runif(-2,2);
        sv.sigma = R::runif(-2,2);
        sv.Sigma(0) =  R::runif(-2,2);
        sv.Sigma(1) = R::runif(-2,2); 
        sv.Sigma(2) = R::runif(-2,2);
        sv.b = true_b_int;
        sv.b_slope = true_b_slope;
        sv.am = 0;
        sv.bm = Eigen::VectorXd::Zero(1);
        sv.bbm = Eigen::VectorXd::Zero(1);
        sv.tm = Eigen::VectorXd::Zero(1);
        sv.dm = Eigen::VectorXd::Zero(1);
        sv.b_m = Eigen::VectorXd::Zero(true_b_int.rows());
        sv.bs_m = Eigen::VectorXd::Zero(true_b_int.rows());
        sv.S_m = Eigen::VectorXd::Zero(3);
        sv.sm = 0.0;
        sv.theta(0) = -2.94;

        for(int i = 0; i < par_grid.size(); i++){
            sv.Sigma(2) = par_grid(i);
            energy_grid(i) = stap_object.calculate_glmer_energy(sv);
            stap_object.calculate_gradient(sv);
            grad_grid(i) = stap_object.sgg.subj_sig_grad(2);
        }



    return (Rcpp::List::create(Rcpp::Named("energy") = energy_grid,
                              Rcpp::Named("grad") = grad_grid));

}
