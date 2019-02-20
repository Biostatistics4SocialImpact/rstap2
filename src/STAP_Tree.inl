void STAP_Tree::BuildTree(STAP& stap_object,
        SV& sv_proposed, SV& sv_init,
        double& u, int v, int j, 
        double& epsilon, std::mt19937& rng){

    if( j == 0 ){
        if(diagnostics)
            Rcpp::Rcout << "Base Case Reached:" << std::endl;
        double total_energy_init = stap_object.calculate_total_energy(sv_init);
        this->Leapfrog(stap_object,sv_proposed,v*epsilon);
        double total_energy = stap_object.calculate_total_energy(sv_proposed);
        n_prime = u <= total_energy ? 1: 0;
        s_prime = u < (1000 + total_energy) ? 1:0;
        svl.copy_SV(svn);
        svr.copy_SV(svn);
        Rcpp::Rcout << svn.alpha << std::endl;
        Rcpp::Rcout << svr.alpha << std::endl;
        alpha_prime = std::min(1.0,exp(total_energy - total_energy_init));
        n_alpha = 1.0;
        if(diagnostics){
            Rcpp::Rcout << "-------------------" << std::endl;
            Rcpp::Rcout << "Energy Check: u = " << u <<std::endl;
            Rcpp::Rcout << "Energy Check: init energy: " << total_energy_init << std::endl;
            Rcpp::Rcout << "Energy Check: proposed energy = " << total_energy << std::endl;
            Rcpp::Rcout << "Energy Check: n = " << n_prime << std::endl;
            Rcpp::Rcout << "Energy Check: s = " << s_prime << std::endl;
            Rcpp::Rcout << "alpha_prime: " << alpha_prime << std::endl;
            Rcpp::Rcout << "Base Case Terminated" << std::endl;
            Rcpp::Rcout << "-------------------" << std::endl;
        }
    }else{
        if(diagnostics){
            Rcpp::Rcout << "j is: " << j << std::endl;
            Rcpp::Rcout << "Building subtree" << std::endl;
            Rcpp::Rcout << "beta proposed " << sv_proposed.beta << " beta init " << sv_init.beta << std::endl;
            Rcpp::Rcout << "sigma proposed" << exp(sv_proposed.sigma) << " sigma  init " << exp(sv_init.sigma) << std::endl;
        }
        STAP_Tree subtree(spc,diagnostics,rng);
        subtree.BuildTree(stap_object,sv_proposed, sv_init,u,v,j-1,epsilon,rng);
        s_prime = subtree.get_s_prime();
        n_prime = subtree.get_n_prime();
        n_alpha = subtree.get_n_alpha();
        svn.copy_SV(subtree.get_svn());
        svl.copy_SV(subtree.get_svl());
        svr.copy_SV(subtree.get_svr());
        alpha_prime = subtree.get_alpha_prime();
        if(subtree.get_s_prime() == 1){
            STAP_Tree subsubtree(spc,diagnostics,rng);
            if( v == -1 ){
                subsubtree.BuildTree(stap_object,svl,sv_init,u,v,j-1,epsilon,rng);
                svl.copy_SV(subsubtree.get_svl());
            }else{
                subsubtree.BuildTree(stap_object,svr,sv_init,u,v,j-1,epsilon,rng);
                svr.copy_SV(subsubtree.get_svr());
            }
            double p = (subsubtree.get_n_prime() == 0.0 && subtree.get_n_prime() ==0.0) ? 0.0 : subsubtree.get_n_prime() / (subtree.get_n_prime() + subsubtree.get_n_prime());
            std::uniform_real_distribution<double> die(0.0,1.0);
            if(die(rng) <= p)
                svn.copy_SV(subsubtree.get_svn());
            alpha_prime = subsubtree.get_alpha_prime() + subtree.get_alpha_prime();
            n_alpha = subtree.get_n_alpha() +  subsubtree.get_n_alpha();
            double UTI_one = get_UTI_one(svl,svr);
            double UTI_two = get_UTI_two(svl,svr);
            s_prime = (UTI_one && UTI_two ) ? subsubtree.get_s_prime() : 0 ;
            n_prime = subtree.get_n_prime() + subsubtree.get_n_prime();
            if(diagnostics)
                Rcpp::Rcout << " SubSubtree portion completed" << std::endl;
        }
    }
}

void STAP_Tree::Leapfrog(STAP& stap_object,SV& sv, double epsilon){

    if(diagnostics){
        Rcpp::Rcout << "Leapfrogging" << std::endl;
        Rcpp::Rcout << "alpha_init: " << sv.alpha << std::endl;
        Rcpp::Rcout << "beta_init: " << sv.beta << std::endl;
        Rcpp::Rcout << "betab_init: " << sv.beta_bar << std::endl;
        Rcpp::Rcout << "theta_init: " << sv.theta_transformed() << std::endl;
        Rcpp::Rcout << "sigma_init: " << exp(sv.sigma) << std::endl;
    } 
    stap_object.calculate_gradient(sv);
    double alpha_grad = stap_object.get_alpha_grad();
    Eigen::VectorXd beta_bar_grad = stap_object.get_beta_bar_grad();
    Eigen::VectorXd beta_grad = stap_object.get_beta_grad();
    Eigen::VectorXd theta_grad = stap_object.get_theta_grad();
    double sigma_grad = stap_object.get_sigma_grad();
    if(diagnostics){ 
        Rcpp::Rcout << "alpha_grad: " << alpha_grad << std::endl;
        Rcpp::Rcout << "beta_grad: " << beta_grad << std::endl;
        Rcpp::Rcout << "beta_bar_grad: " << beta_bar_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
    }
    svn.am = sv.am + epsilon * alpha_grad / 2.0;
    svn.bm = sv.bm + epsilon * beta_grad / 2.0;
    svn.bbm = sv.bbm + epsilon * beta_bar_grad / 2.0;
    svn.tm = sv.tm +  epsilon * theta_grad / 2.0;
    svn.sm = sv.sm + epsilon * sigma_grad / 2.0;
    svn.alpha = sv.alpha + epsilon * svn.am;
    svn.beta = sv.beta + epsilon * svn.bm; // full step
    svn.beta_bar = sv.beta_bar + epsilon * svn.bbm; 
    svn.theta = sv.theta +  epsilon * svn.tm;
    svn.sigma = sv.sigma + epsilon * svn.sm;
    if(diagnostics){ 
        Rcpp::Rcout << "beta_momentum_new: " << svn.bm << std::endl;
        Rcpp::Rcout << "alpha_new: " << svn.alpha << std::endl;
        Rcpp::Rcout << "beta_new: " << svn.beta << std::endl;
        Rcpp::Rcout << "beta_bar_new: " << svn.beta_bar << std::endl;
        Rcpp::Rcout << "theta_new: " << svn.theta_transformed() << std::endl;
        Rcpp::Rcout << "sigma_new: " << exp(svn.sigma) << std::endl;
        Rcpp::Rcout << "\n"  << std::endl;
    }
    stap_object.calculate_gradient(sv);
    alpha_grad = stap_object.get_alpha_grad();
    beta_grad = stap_object.get_beta_grad();
    beta_bar_grad = stap_object.get_beta_bar_grad();
    theta_grad = stap_object.get_theta_grad();
    sigma_grad = stap_object.get_sigma_grad();
    svn.am = svn.am + epsilon * alpha_grad / 2.0;
    svn.bm = svn.bm + epsilon * beta_grad / 2.0 ;
    svn.bbm = svn.bbm + epsilon * beta_bar_grad / 2.0 ;
    svn.tm = svn.tm +  epsilon * theta_grad / 2.0;
    svn.sm = svn.sm + epsilon * sigma_grad / 2.0;
}
