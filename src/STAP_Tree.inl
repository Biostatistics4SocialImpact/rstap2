void STAP_Tree::BuildTree(STAP& stap_object,
        double beta_proposed, 
        double theta_proposed, 
        double sigma_proposed,
        double beta_init, 
        double theta_init, 
        double sigma_init,
        double bmp, double tmp, double smp,
        double bmi, double tmi, double smi,
        double u, int v, int j,
        double& epsilon_beta, 
        double& epsilon_theta,
        std::mt19937& rng){

    if( j == 0 ){
        if(diagnostics)
            Rcpp::Rcout << "Base Case Reached:" << std::endl;
        double total_energy_init = stap_object.calculate_total_energy(beta_init,theta_init,sigma_init,bmi,tmi,smi);
        this->Leapfrog(stap_object,beta_proposed,theta_proposed,sigma_proposed,bmp,tmp,smp,v*epsilon_theta,v*epsilon_beta);
        double total_energy = stap_object.calculate_total_energy(beta_new,theta_new,sigma_new,bmn,tmn,smn);
        n_prime = u <= total_energy ? 1: 0;
        s_prime = u < (1000 + total_energy) ? 1:0;
        bl = beta_new;
        tl = theta_new;
        sl = sigma_new;
        br = beta_new;
        tr = theta_new;
        sr = sigma_new;
        bml = bmn;
        tml = tmn;
        sml = smn;
        bmr = bmn;
        tmr = tmn;
        smr = smn;
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
            Rcpp::Rcout << "beta proposed " << beta_proposed << " beta init " << beta_init << std::endl;
            Rcpp::Rcout << "sigma proposed" << exp(sigma_proposed) << " sigma  init " << exp(sigma_init) << std::endl;
        }
        STAP_Tree subtree(diagnostics);
        subtree.BuildTree(stap_object,beta_proposed,theta_proposed,sigma_proposed,beta_init,theta_init,sigma_init,bmp,tmp,smp,bmi,tmi,smi,u,v,j-1,epsilon_beta,epsilon_theta,rng);
        if(subtree.get_s_prime() == 1){
            STAP_Tree subsubtree(diagnostics);
            if( v == -1 ){
                //Rcpp::Rcout << "j is: " << j << std::endl;
                //Rcpp::Rcout << "Building left subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_bl(),subtree.get_tl(),subtree.get_sl(),beta_init,theta_init,sigma_init,subtree.get_bml(),subtree.get_tml(),subtree.get_sml(),bmi,tmi,smi,u,v,j-1,epsilon_beta,epsilon_theta,rng);
                bl = subsubtree.get_bl();
                tl = subsubtree.get_tl();
                sl = subsubtree.get_sl();
                br = subtree.get_br();
                tr = subtree.get_tr();
                sr = subtree.get_sr();
                bml = subsubtree.get_bml();
                tml = subsubtree.get_tml();
                sml = subsubtree.get_sml();
                bmr = subtree.get_bmr();
                tmr = subtree.get_tmr();
                smr = subtree.get_smr();
                //Rcpp::Rcout << "left subsubtree completed" << std::endl;
            }else{
                //Rcpp::Rcout << "j is: " << j << std::endl;
                //Rcpp::Rcout << "Building right subsubtree" << std::endl;
                subsubtree.BuildTree(stap_object,subtree.get_br(),subtree.get_tr(),subtree.get_sr(),beta_init,theta_init,sigma_init,subtree.get_bmr(),subtree.get_tmr(),subtree.get_smr(),bmi,tmi,smi,u,v,j-1,epsilon_theta,epsilon_beta,rng);
                bl = subtree.get_bl();
                tl = subtree.get_tl();
                sl = subtree.get_sl();
                br = subsubtree.get_br();
                tr = subsubtree.get_tr();
                sr = subsubtree.get_sr();
                bml = subtree.get_bml();
                tml = subtree.get_tml();
                sml = subtree.get_sml();
                bmr = subsubtree.get_bmr();
                tmr = subsubtree.get_tmr();
                smr = subsubtree.get_smr();
                //Rcpp::Rcout << "right subsubtree completed" << std::endl;
            }
            double p = (subsubtree.get_n_prime() == 0.0 && subtree.get_n_prime() ==0.0) ? 0.0 : subsubtree.get_n_prime() / (subtree.get_n_prime() + subsubtree.get_n_prime());
            std::uniform_real_distribution<double> die(0.0,1.0);
            if(die(rng) <= p){
                beta_new = subsubtree.get_beta_new();
                theta_new = subsubtree.get_theta_new();
                sigma_new = subsubtree.get_sigma_new();
            }else{
                beta_new = subtree.get_beta_new();
                theta_new = subtree.get_theta_new();
                sigma_new = subtree.get_sigma_new();
            }
            alpha_prime = subsubtree.get_alpha_prime() + subtree.get_alpha_prime();
            n_alpha = subtree.get_n_alpha() +  subsubtree.get_n_alpha();
            double UTI_one = (  (br-bl)*bml + ((tr-tl)*tml) + ((sr-sl)*sml)  >= 0 );
            double UTI_two = ( ( (br-bl)*bmr + (tr-tl)*tmr) + ((sr-sl)*smr) >= 0 );
            s_prime = (UTI_one && UTI_two ) ? subsubtree.get_s_prime() : 0 ;
            n_prime = subtree.get_n_prime() + subsubtree.get_n_prime();
            if(diagnostics)
                Rcpp::Rcout << " SubSubtree portion completed" << std::endl;
        }else{
            s_prime = subtree.get_s_prime();
            n_prime = subtree.get_n_prime();
            n_alpha = subtree.get_n_alpha();
            beta_new = subtree.get_beta_new();
            theta_new = subtree.get_theta_new();
            sigma_new = subtree.get_sigma_new();
            bl = subtree.get_bl();
            br = subtree.get_br();
            tl = subtree.get_tl();
            tr = subtree.get_tr();
            sl = subtree.get_sl();
            sr = subtree.get_sr();
            tml = subtree.get_tml();
            tmr = subtree.get_tmr();
            bml = subtree.get_bml();
            bmr = subtree.get_bmr();
            sml = subtree.get_sml();
            smr = subtree.get_smr();
            alpha_prime = subtree.get_alpha_prime();
            if(diagnostics)
                Rcpp::Rcout << " Subtree portion completed" << std::endl;
            }
    }
}


void STAP_Tree::Leapfrog(STAP& stap_object,double& cur_beta, double& cur_theta,double& cur_sigma, double bm, double tm,double sm, double epsilon_beta, double epsilon_theta){

    if(diagnostics){
        Rcpp::Rcout << "Leapfrogging" << std::endl;
        Rcpp::Rcout << "beta_init: " << cur_beta << std::endl;
        Rcpp::Rcout << "theta_init: " << 10 / (1 + exp(-cur_theta)) << std::endl;
        Rcpp::Rcout << "sigma_init: " << exp(cur_sigma) << std::endl;
    } 
    stap_object.calculate_gradient(cur_beta,cur_theta,cur_sigma);
    double beta_grad = stap_object.get_beta_grad();
    double theta_grad = stap_object.get_theta_grad();
    double sigma_grad = stap_object.get_sigma_grad();
    if(diagnostics){ 
        Rcpp::Rcout << "beta_grad: " << beta_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
    }
    bmn = bm + epsilon_beta * beta_grad / 2.0 ;
    tmn = tm + epsilon_theta * theta_grad / 2.0;
    smn = sm + epsilon_beta * sigma_grad / 2.0;
    beta_new = cur_beta + epsilon_beta * bmn; // full step
    theta_new = cur_theta + epsilon_theta * tmn;
    sigma_new = cur_sigma + epsilon_beta * smn;
    if(diagnostics){ 
        Rcpp::Rcout << "beta_momentum_new: " << bmn << std::endl;
        Rcpp::Rcout << "beta_momentum * epsilon: " << bmn * epsilon_beta << std::endl;
        Rcpp::Rcout << "beta_new: " << beta_new << std::endl;
        Rcpp::Rcout << "theta_new: " << 10.0 / (1 + exp(-theta_new)) << std::endl;
        Rcpp::Rcout << "sigma_new: " << exp(sigma_new) << std::endl;
        Rcpp::Rcout << "\n"  << std::endl;
        }
    stap_object.calculate_gradient(beta_new,theta_new,sigma_new);
    beta_grad = stap_object.get_beta_grad();
    theta_grad = stap_object.get_theta_grad();
    sigma_grad = stap_object.get_sigma_grad();
    bmn = bmn + epsilon_beta * beta_grad / 2.0 ;
    tmn = tmn + epsilon_theta * theta_grad / 2.0;
    smn = smn + epsilon_beta * sigma_grad / 2.0;

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

const double STAP_Tree::get_sigma_new() const{
    return(sigma_new);
}

const double STAP_Tree::get_sl() const{
    return(sl);
}

const double STAP_Tree::get_sr() const{
    return(sr);
}

const double STAP_Tree::get_sml() const{
    return(sml);
}

const double STAP_Tree::get_smr() const{
    return(smr);
}

