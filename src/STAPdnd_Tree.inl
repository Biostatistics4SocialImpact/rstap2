void STAP_Tree::BuildTree(STAP &stap_object, STAP_Vars sv,
                          STAP_Vars svi,
                          double u, int v,int j,
                          double &epsilon, std::mt19937 &rng){

    if( j == 0 ){
        //Rcpp::Rcout << "Base Case Reached" << std::endl;
        double total_energy_init = stap_object.calculate_total_energy(svi);
        this->Leapfrog(stap_object,sv,v*epsilon);
        double total_energy = stap_object.calculate_total_energy(sv);
        //Rcpp::Rcout << "Energy Check: u = " << u <<std::endl;
        //Rcpp::Rcout << "Energy Check: total energy = " << total_energy << std::endl;
        n_prime = u <= total_energy ? 1: 0;
        //Rcpp::Rcout << "Energy Check: n = " << n_prime << std::endl;
        s_prime = u < (1000 + total_energy) ? 1:0;
        svl = sv_new;
        svr = sv_new;
        alpha_prime = std::min(1.0,exp(total_energy - total_energy_init));
        n_alpha = 1.0;
        //Rcpp::Rcout << "Base Case Terminated" << std::endl;
    }else{
        //Rcpp::Rcout << "j is: " << j << std::endl;
        //Rcpp::Rcout << "Building subtree" << std::endl;
        STAP_Tree subtree(sv.delta.size(),sv.beta.size(),sv.theta.size());
        subtree.BuildTree(stap_object,sv,svi,u,v,j-1,epsilon,rng);
        if(subtree.get_s_prime() == 1){
            STAP_Tree subsubtree(sv.delta.size(),sv.beta.size(),sv.theta.size());
            if( v == -1 ){
                //Rcpp::Rcout << "j is: " << j << std::endl;
                //Rcpp::Rcout << "Building left subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_svl(),svi,u,v,j-1,epsilon,rng);
                svl = subsubtree.get_svl();
                svr = subtree.get_svr();
                //Rcpp::Rcout << "left subsubtree completed" << std::endl;
            }else{
                //Rcpp::Rcout << "j is: " << j << std::endl;
                //Rcpp::Rcout << "Building right subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_svr(),svi,u,v,j-1,epsilon,rng);
                svl = subtree.get_svl();
                svr = subsubtree.get_svr();
                //Rcpp::Rcout << "right subsubtree completed" << std::endl;
            }
            double p = (subsubtree.get_n_prime() == 0.0 && subtree.get_n_prime() == 0.0) ? 0.0 : subsubtree.get_n_prime() / (subtree.get_n_prime() + subsubtree.get_n_prime());
            std::uniform_real_distribution<double> die(0.0,1.0);
            if(die(rng) <= p){
                sv_new = subsubtree.get_sv_new();
            }else{
                sv_new = subtree.get_sv_new();
            }
            alpha_prime = subsubtree.get_alpha_prime() + subtree.get_alpha_prime();
            n_alpha = subtree.get_n_alpha() +  subsubtree.get_n_alpha();
            double UTI_one = get_UTI_one(svl,svr); 
            double UTI_two = get_UTI_two(svl,svr);
            s_prime = (UTI_one && UTI_two ) ? subsubtree.get_s_prime() :0 ;
            n_prime = subtree.get_n_prime() + subsubtree.get_n_prime();
            //Rcpp::Rcout << " SubSubtree portion completed" << std::endl;
        }else{
            s_prime = subtree.get_s_prime();
            n_prime = subtree.get_n_prime();
            n_alpha = subtree.get_n_alpha();
            sv_new  = subtree.get_sv_new();
            svl = subtree.get_svl();
            svr = subtree.get_svr();
            alpha_prime = subtree.get_alpha_prime();
            //Rcpp::Rcout << " Subtree portion completed" << std::endl;
            }
    }
}


void STAP_Tree::Leapfrog(STAP &stap_object, STAP_Vars &sv, double epsilon){

    //Rcpp::Rcout << "Leapfrogging" << std::endl;

    //Rcpp::Rcout << "Beta_naught: " << cur_beta << std::endl;
    //Rcpp::Rcout << "Theta_naught: " << 10.0 /(1 + exp(- cur_theta)) << std::endl;
    stap_object.calculate_gradient(sv);
    sv_new.dm = sv.dm + epsilon * stap_object.get_delta_grad() / 2.0;
    sv_new.bm = sv.bm + epsilon * stap_object.get_beta_grad() / 2.0 ;
    sv_new.tm = sv.tm + epsilon * stap_object.get_theta_grad() / 2.0;
    sv_new.sm = sv.sm + epsilon * stap_object.get_sigma_grad() / 2.0;
    //Rcpp::Rcout << "beta grad" << beta_grad << std::endl;
    //Rcpp::Rcout << "theta grad" << theta_grad << std::endl;
    sv_new.delta = sv.delta + epsilon * sv_new.dm;
    sv_new.beta = sv.beta + epsilon * sv_new.bm;
    sv_new.theta = sv.theta + epsilon * sv_new.tm;
    sv_new.sm = sv.sigma + epsilon * sv_new.sm;
    //Rcpp::Rcout << "beta_new: " << beta_new << std::endl;
    //Rcpp::Rcout << "theta_new: " << 10.0 / (1 + exp(-theta_new)) << std::endl;
    stap_object.calculate_gradient(sv_new);
    sv_new.dm = sv_new.dm + epsilon * stap_object.get_delta_grad() / 2.0;
    sv_new.bm = sv_new.bm + epsilon * stap_object.get_beta_grad() / 2.0 ;
    sv_new.tm = sv_new.tm + epsilon * stap_object.get_theta_grad() / 2.0;
    sv_new.sm = sv_new.sm + epsilon * stap_object.get_sigma_grad() / 2.0;
}

const int STAP_Tree::get_s_prime() const{
    return(s_prime);
}

const double STAP_Tree::get_n_prime() const{
    return(n_prime);
}

const double STAP_Tree::get_alpha_prime() const{
    return(alpha_prime);
}

const double STAP_Tree::get_n_alpha() const{
    return(n_alpha);
}

const STAP_Vars STAP_Tree::get_sv_new() const{
    return(sv_new);
}

const STAP_Vars STAP_Tree::get_svl() const{
    return(svl);
}

const STAP_Vars STAP_Tree::get_svr() const{
    return(svr);
}
