void STAP_Tree::BuildTree(STAP &stap_object,
        double beta_proposed, double theta_proposed, 
        double beta_init, double theta_init, 
        double bmp, double tmp,
        double bmi, double tmi,
        double u, int v, int j, double &epsilon, 
        std::mt19937 &rng){

    if( j == 0 ){
        Rcpp::Rcout << "Base Case Reached" << std::endl;
        double total_energy_init = stap_object.calculate_total_energy(beta_init,theta_init,bmi,tmi);
        this->Leapfrog(stap_object,beta_proposed,theta_proposed,bmp,tmp,v*epsilon);
        double total_energy = stap_object.calculate_total_energy(beta_new,theta_new,bmn,tmn);
        Rcpp::Rcout << "Energy Check: u = " << u <<std::endl;
        Rcpp::Rcout << "Energy Check: total energy = " << exp(total_energy) << std::endl;
        n_prime = u <= exp(total_energy) ? 1: 0;
        Rcpp::Rcout << "Energy Check: n = " << n_prime << std::endl;
        s_prime = u < exp(1000 + total_energy) ? 1:0;
        bl = beta_new;
        tl = theta_new;
        br = beta_new;
        tr = theta_new;
        bml = bmn;
        tml = tmn;
        bmr = bmn;
        tmr = tmn;
        alpha_prime = std::min(1.0,exp(total_energy - total_energy_init));
        n_alpha = 1.0;
        Rcpp::Rcout << "Base Case Terminated" << std::endl;
    }else{
        Rcpp::Rcout << "j is: " << j << std::endl;
        Rcpp::Rcout << "Building subtree" << std::endl;
        STAP_Tree subtree;
        subtree.BuildTree(stap_object,beta_proposed,theta_proposed,beta_init,theta_init,bmp,tmp,bmi,tmi,u,v,j-1,epsilon,rng);
        if(subtree.get_s_prime() == 1){
            STAP_Tree subsubtree;
            if( v == -1 ){
                Rcpp::Rcout << "j is: " << j << std::endl;
                Rcpp::Rcout << "Building left subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_bl(),subtree.get_tl(),beta_init,theta_init,subtree.get_bml(),subtree.get_tml(),bmi,tmi,u,v,j-1,epsilon,rng);
                bl = subsubtree.get_bl();
                tl = subsubtree.get_tl();
                br = subtree.get_br();
                tr = subtree.get_tr();
                bml = subsubtree.get_bml();
                tml = subsubtree.get_tml();
                bmr = subtree.get_bmr();
                tmr = subtree.get_tmr();
                Rcpp::Rcout << "left subsubtree completed" << std::endl;
            }else{
                Rcpp::Rcout << "j is: " << j << std::endl;
                Rcpp::Rcout << "Building right subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_br(),subtree.get_tr(),beta_init,theta_init,subtree.get_bmr(),subtree.get_tmr(),bmi,tmi,u,v,j-1,epsilon,rng);
                bl = subtree.get_bl();
                tl = subtree.get_tl();
                br = subsubtree.get_br();
                tr = subsubtree.get_tr();
                bml = subtree.get_bml();
                tml = subtree.get_tml();
                bmr = subsubtree.get_bmr();
                tmr = subsubtree.get_tmr();
                Rcpp::Rcout << "right subsubtree completed" << std::endl;
            }
            double p = (subsubtree.get_n_prime() == 0.0 & subtree.get_n_prime() ==0.0) ? 0.0 : subsubtree.get_n_prime() / (subtree.get_n_prime() + subsubtree.get_n_prime());
            std::uniform_real_distribution<double> die(0.0,1.0);
            if(die(rng) <= p){
                beta_new = subsubtree.get_beta_new();
                theta_new = subsubtree.get_theta_new();
            }else{
                beta_new = subtree.get_beta_new();
                theta_new = subtree.get_theta_new();
            }
            alpha_prime = subsubtree.get_alpha_prime() + subtree.get_alpha_prime();
            n_alpha = subtree.get_n_alpha() +  subsubtree.get_n_alpha();
            double UTI_one = pow((br -bl)*bml,2) + pow((tr - tl)*tml ,2); 
            double UTI_two = pow((br -bl)*bmr,2) + pow((tr - tl)*tmr ,2); 
            s_prime = (UTI_one >= 0.0 & UTI_two >= 0.0 ) ? subsubtree.get_s_prime() :0 ;
            n_prime = subtree.get_n_prime() + subsubtree.get_n_prime();
            Rcpp::Rcout << " SubSubtree portion completed" << std::endl;
        }else{
            s_prime = subtree.get_s_prime();
            n_prime = subtree.get_n_prime();
            n_alpha = subtree.get_n_alpha();
            beta_new = subtree.get_beta_new();
            theta_new = subtree.get_theta_new();
            bl = subtree.get_bl();
            br = subtree.get_br();
            tl = subtree.get_tl();
            tr = subtree.get_tr();
            tml = subtree.get_tml();
            tmr = subtree.get_tmr();
            bml = subtree.get_bml();
            bmr = subtree.get_bmr();
            alpha_prime = subtree.get_alpha_prime();
            Rcpp::Rcout << " Subtree portion completed" << std::endl;
            }
    }
}


void STAP_Tree::Leapfrog(STAP &stap_object,double &cur_beta, double &cur_theta, double bm, double tm, double epsilon){

    Rcpp::Rcout << "Leapfrogging" << std::endl;

    Rcpp::Rcout << "Beta_naught: " << cur_beta << std::endl;
    Rcpp::Rcout << "Theta_naught: " << 10.0 /(1 + exp(- cur_theta)) << std::endl;
    stap_object.calculate_gradient(cur_beta,cur_theta);
    double beta_grad;
    beta_grad = stap_object.get_beta_grad();
    double theta_grad;
    theta_grad = stap_object.get_theta_grad();
    bmn = bm + epsilon * beta_grad / 2.0 ;
    tmn = tm + epsilon * theta_grad / 2.0;
    Rcpp::Rcout << "beta grad" << beta_grad << std::endl;
    Rcpp::Rcout << "theta grad" << theta_grad << std::endl;
    beta_new = cur_beta + epsilon * bmn; // full step
    theta_new = cur_theta + epsilon * tmn;
    Rcpp::Rcout << "beta_new: " << beta_new << std::endl;
    Rcpp::Rcout << "theta_new: " << 10.0 / (1 + exp(-theta_new)) << std::endl;
    stap_object.calculate_gradient(beta_new,theta_new);
    theta_grad = stap_object.get_theta_grad();
    beta_grad = stap_object.get_beta_grad();
    bmn = bmn + epsilon * beta_grad / 2.0 ;
    tmn = tmn + epsilon * theta_grad / 2.0;

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

const double STAP_Tree::get_beta_new() const{
    return(beta_new);
}

const double STAP_Tree::get_bl() const{
    return(bl);
}

const double STAP_Tree::get_br() const{
    return(br);
}

const double STAP_Tree::get_bml() const{
    return(bml);
}

const double STAP_Tree::get_bmr() const{
    return(bmr);
}

const double STAP_Tree::get_theta_new() const{
    return(theta_new);
}

const double STAP_Tree::get_tr() const{
    return(tr);
}

const double STAP_Tree::get_tl() const{
    return(tl);
}

const double STAP_Tree::get_tml() const{
    return(tml);
}

const double STAP_Tree::get_tmr() const{
    return(tmr);
}
