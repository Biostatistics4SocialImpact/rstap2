void STAP_Tree_glmer::BuildTree(STAP_glmer& stap_object,
        SV_glmer& sv_proposed, SV_glmer& sv_init,
        double& u, int v, int j, 
        double& epsilon, std::mt19937& rng){

    if( j == 0 ){
        if(diagnostics)
            Rcpp::Rcout << "Base Case Reached:" << std::endl;
        double total_energy_init = stap_object.calculate_total_energy(sv_init);
        this->Leapfrog(stap_object,sv_proposed,v*epsilon);
        double total_energy = stap_object.calculate_total_energy(svn);
        n_prime = u <= total_energy ? 1: 0;
        s_prime = u < (1000 + total_energy) ? 1:0;
        svl.copy_SV(svn);
        svr.copy_SV(svn);
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
        STAP_Tree_glmer subtree(spc,diagnostics,rng);
        subtree.BuildTree(stap_object,sv_proposed, sv_init,u,v,j-1,epsilon,rng);
        s_prime = subtree.get_s_prime();
        n_prime = subtree.get_n_prime();
        n_alpha = subtree.get_n_alpha();
        svn.copy_SV(subtree.get_svn());
        svl.copy_SV(subtree.get_svl());
        svr.copy_SV(subtree.get_svr());
        alpha_prime = subtree.get_alpha_prime();
        if(subtree.get_s_prime() == 1){
            STAP_Tree_glmer subsubtree(spc,diagnostics,rng);
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

void STAP_Tree_glmer::Leapfrog(STAP_glmer& stap_object,SV_glmer& sv, double epsilon){

    stap_object.calculate_gradient(sv);

    svn.momenta_leapfrog_other(sv,epsilon,stap_object.sgg);

    svn.momenta_leapfrog_position(sv,epsilon);

    stap_object.calculate_gradient(svn);

    svn.momenta_leapfrog_self(epsilon,stap_object.sgg);
}
