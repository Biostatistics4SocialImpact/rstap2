STAP::STAP(Eigen::ArrayXXd& input_dists,
           Eigen::ArrayXXi& input_ucrs,
           Eigen::MatrixXd& input_subj_array,
           Eigen::ArrayXd& input_subj_n,
           Eigen::VectorXd& input_y,
           const bool& input_diagnostics){

    dists = input_dists;
    u_crs = input_ucrs;
    subj_array = input_subj_array;
    subj_n = input_subj_n;
    y = input_y;
    X = Eigen::MatrixXd::Zero(y.size(),dists.rows());
    X_prime = Eigen::MatrixXd::Zero(y.size(),dists.rows());
    diagnostics = input_diagnostics;

}

double STAP::calculate_total_energy(SV& sv){
            
     if(diagnostics){
        Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
        Rcpp::Rcout << " Parameter positions:" << std::endl;
        Rcpp::Rcout << "alpha: " << sv.alpha << std::endl;
        Rcpp::Rcout << "beta: " << sv.beta << std::endl;
        Rcpp::Rcout << "beta_bar: " << sv.beta_bar << std::endl;
        Rcpp::Rcout << "theta: " << sv.theta_transformed()  << std::endl;
        Rcpp::Rcout << "sigma: " <<  sv.sigma_transformed() << std::endl;
        Rcpp::Rcout <<  "------------------" << std::endl;
    }
    
    double out = 0;
    this->calculate_X_diff(sv.theta(0));

    out -= y.size() / 2.0 * log(M_PI * 2 * sv.sigma_sq_transformed() ); 

    // likelihood kernel
    out += - sv.precision_transformed() * (pow((y - Eigen::VectorXd::Ones(y.size()) * sv.alpha - X_diff * sv.beta - X_mean * sv.beta_bar).array(),2)).sum();

    if(diagnostics)
        Rcpp::Rcout << "likelihood" << out << std::endl;
    
    // alpha ~N(25,5)  prior
    out += R::dnorm(sv.alpha,25,5,TRUE);

    // beta ~ N(0,3) prior
    out += R::dnorm(sv.beta(0),0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // beta_bar ~ N(0,3) prior
    out += R::dnorm(sv.beta_bar(0),0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // log(theta) ~ N(1,1) prior 
    out +=  R::dlnorm(sv.theta_transformed()(0),1,1,TRUE);//-log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);

    // sigma ~ C(0,5)
    out +=  R::dcauchy(sv.sigma_transformed(),0,5,TRUE);

    // theta constraints jacobian adjustment 
    out +=  ( log(10.0) - sv.theta(0)  - 2 * log(1+exp(-sv.theta(0))));   

    if(diagnostics)
        Rcpp::Rcout << "jacobian I" << out << std::endl;
    
    
    // sigma jacobian
    out += sv.sigma;

    if(diagnostics){
        Rcpp::Rcout << "jacobian II" << out << std::endl;
        Rcpp::Rcout << "------------------ \n " << std::endl;
    }

    // Incorporate Kinetic Energy
    out -= sv.kinetic_energy();
    if(diagnostics)
        Rcpp::Rcout << "Kinetic Energy " << out << std::endl;

    out = (isinf(-out) || isnan(out)) ? (-1 * DBL_MAX) : out;

    if(diagnostics)
        Rcpp::Rcout << "Energy out " << out << std::endl;
   
    return(out);

}

double STAP::sample_u(SV& sv, std::mt19937& rng){

    if(diagnostics)
        Rcpp::Rcout << "Sample U Energy Calculation" << std::endl;
    double energy = this->calculate_total_energy(sv);
    std::uniform_real_distribution<double> runif(0.0,1.0);
    double log_z = log(runif(rng));
    return(energy + log_z);
}

void STAP::calculate_X( double& theta){
    
    int start_col;
    int range_len;
    int col_lim = X.cols();
    for(int bef_ix = 0; bef_ix < col_lim ; bef_ix ++){
        for(int subj_ix = 0; subj_ix < u_crs.rows(); subj_ix ++){
            start_col = u_crs(subj_ix,bef_ix);
            range_len = u_crs(subj_ix,bef_ix+1) - start_col + 1;
            X(subj_ix,bef_ix) = (exp(- dists.block(bef_ix,start_col,1,range_len) / theta  )).sum();
        }
    }
}

void STAP::calculate_X_diff(double& theta){
    
    double transformed_theta = 10 / (1 + exp(-theta));
    this->calculate_X(transformed_theta);
    this->calculate_X_mean();
    X_diff = X - X_mean;
}

void STAP::calculate_X_mean(){

    X_mean =  subj_array.transpose() * ((subj_array * X).array() * subj_n).matrix();
}

void STAP::calculate_X_prime(double& theta,double& cur_theta){ 

    int start_col;
    int range_len;
    int col_lim = X.cols();
    for(int bef_ix = 0; bef_ix < col_lim ; bef_ix ++){
        for(int subj_ix = 0; subj_ix < u_crs.rows(); subj_ix ++){
            start_col = u_crs(subj_ix,bef_ix);
            range_len = u_crs(subj_ix,bef_ix+1) - start_col + 1;
            X(subj_ix,bef_ix) = (exp(- dists.block(bef_ix,start_col,1,range_len) / theta  )).sum();
            X_prime(subj_ix,bef_ix) = (dists.block(bef_ix,start_col,1,range_len) /  10.0  * exp(- dists.block(bef_ix,start_col,1,range_len) / theta - cur_theta  )).sum() / pow(theta,2);
        }
    }
}

void STAP::calculate_X_mean_prime(){ 

    X_mean_prime = subj_array.transpose() * ((subj_array * X_prime).array() * subj_n).matrix();

}

void STAP::calculate_X_prime_diff(double& theta, double& cur_theta){

    this->calculate_X_prime(theta,cur_theta);
    this->calculate_X_mean_prime();
    this->calculate_X_mean();
    X_prime_diff = (X_prime - X_mean_prime);
    X_diff = (X - X_mean);
}

void STAP::calculate_gradient(SV& sv){

    double theta_exponentiated = sv.theta_transformed()(0) / (10.0 - sv.theta_transformed()(0) );
    double theta_transformed = sv.theta_transformed()(0);
    double theta = sv.theta(0);
    double lp_prior_I = pow(sv.theta_transformed()(0),-1) * (10 * exp(-sv.theta(0) )) / pow(1 + exp(-sv.theta(0)),2);
    double lp_prior_II = 2*(log(sv.theta_transformed()(0) ) -1)/ (1 + exp(-sv.theta(0) ));
    double precision = sv.precision_transformed();
    this->calculate_X_prime_diff(theta_transformed,theta); // also calculates X
    // likelihood
    alpha_grad = sv.spc(0) == 0 ? 0 : precision * ( y.sum() - (X_diff * sv.beta).sum()  - (X_mean * sv.beta_bar).sum() - y.size() * sv.alpha);

    beta_grad = precision * ((y.transpose() - sv.alpha_vec.transpose()) * X_diff    -  X_diff.transpose() * X_diff * sv.beta - X_diff.transpose() * X_mean * sv.beta_bar );

    beta_bar_grad = precision * ((y.transpose() - sv.alpha_vec.transpose()) * X_mean - X_mean.transpose() * X_mean * sv.beta_bar - X_mean.transpose() * X_diff * sv.beta);

    sigma_grad = precision * (pow((y - sv.alpha_vec - X_diff * sv.beta - X_mean * sv.beta_bar).array(),2) ).sum() - y.size();

    theta_grad =  (y.transpose() - sv.alpha_vec.transpose()) * X_prime_diff * sv.beta;

    theta_grad = theta_grad + (y.transpose()- sv.alpha_vec.transpose()) * X_mean_prime * sv.beta_bar;

    theta_grad = theta_grad  - sv.beta.transpose() *(X_prime_diff.transpose() * X_diff + X_diff.transpose() * X_prime_diff )  * sv.beta ; 

    theta_grad = theta_grad +   (sv.beta.transpose() * ( X_prime_diff.transpose() * X_mean + X_diff.transpose() * X_mean_prime ) * sv.beta_bar);
    theta_grad = theta_grad * precision;

    // prior components
    alpha_grad += -1.0 / 25 * sv.alpha;
    beta_grad = beta_grad -1.0 / 9.0 * sv.beta;
    beta_bar_grad = beta_bar_grad -1.0 / 9.0 * sv.beta_bar;
    theta_grad(0)  = theta_grad(0)   - lp_prior_I - lp_prior_II;
    theta_grad(0) = theta_grad(0) + (1 - theta_exponentiated) / (theta_exponentiated +1); // jacobian factor
    sigma_grad +=  -(2*sv.sigma_sq_transformed()) / (5+ sv.sigma_sq_transformed()) + 1; 

}

double STAP::FindReasonableEpsilon(SV& sv, std::mt19937& rng){

    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon Start \n " << std::endl;
    double epsilon = 1.0;
    int a;
    SV sv_prop(sv.spc,rng,diagnostics);
    double ratio,initial_energy,propose_energy;
    this->calculate_gradient(sv);
    if(diagnostics){
        Rcpp::Rcout << "Initial Parameters:  " << std::endl;
        Rcpp::Rcout << "alpha: " << sv.alpha << std::endl;
        Rcpp::Rcout << "beta: " << sv.beta << std::endl;
        Rcpp::Rcout << "beta bar: " << sv.beta_bar << std::endl;
        Rcpp::Rcout << "theta: " << sv.theta_transformed() << std::endl;
        Rcpp::Rcout << "sigma: " << sv.sigma_transformed() << std::endl;
        Rcpp::Rcout << "Initial Gradients: \n " << std::endl;
        Rcpp::Rcout << "alpha_grad: "  << alpha_grad << std::endl;
        Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
        Rcpp::Rcout << "betab_grad: "  << beta_bar_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
        Rcpp::Rcout << "Initial Momenta: \n " << std::endl;
        Rcpp::Rcout << "beta_m: "  << sv.bm << std::endl;
        Rcpp::Rcout << "theta_m: " << sv.tm << std::endl;
        Rcpp::Rcout << "sigma_m: " << sv.sm << std::endl;
        Rcpp::Rcout << "-------------------- \n " << std::endl;
    }
    sv_prop.bm = sv.bm + epsilon * beta_grad / 2.0;
    sv_prop.tm = sv.tm + epsilon * theta_grad / 2.0;
    sv_prop.sm = sv.sm + epsilon * sigma_grad / 2.0;
    sv_prop.beta = sv.beta + epsilon * sv_prop.bm;
    sv_prop.theta = sv.theta + epsilon * sv_prop.tm;
    sv_prop.sigma = sv.sigma + epsilon * sv_prop.sm;
    this->calculate_gradient(sv_prop);
    sv_prop.bm = sv_prop.bm + epsilon * beta_grad / 2.0;
    sv_prop.tm = sv_prop.tm + epsilon * theta_grad / 2.0;
    sv_prop.sm = sv_prop.sm + epsilon * sigma_grad / 2.0;
    if(diagnostics){
        Rcpp::Rcout << "Proposed Parameters:" << std::endl;
        Rcpp::Rcout << "beta: " << sv_prop.beta << std::endl;
        Rcpp::Rcout << "theta: " << sv_prop.theta_transformed() << std::endl;
        Rcpp::Rcout << "sigma: " << sv_prop.sigma_transformed() << std::endl;
        Rcpp::Rcout << "Proposed Gradients: " << std::endl;
        Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << "\n" << std::endl;
        Rcpp::Rcout << "Proposed Momenta: " << std::endl;
        Rcpp::Rcout << "beta_m: "  << sv_prop.bm << std::endl;
        Rcpp::Rcout << "theta_m: " << sv_prop.tm << std::endl;
        Rcpp::Rcout << "sigma_m: " << sv_prop.sm << std::endl;
        Rcpp::Rcout << "-------------------- \n " << std::endl;
    }
    initial_energy = this->calculate_total_energy(sv);
    propose_energy = this->calculate_total_energy(sv_prop);
    ratio =  propose_energy - initial_energy;
    if(diagnostics)
        Rcpp::Rcout << "ratio calc" << propose_energy << " " << initial_energy << " " << propose_energy - initial_energy << std::endl;
    ratio = isinf(-ratio) ? -DBL_MAX: ratio ;
    a = ratio > log(.5) ? 1 : -1;
    if(diagnostics){
        Rcpp::Rcout << "a: " << a << std::endl;
        Rcpp::Rcout << "ratio: " << ratio << std::endl;
        Rcpp::Rcout << "a * ratio: " << a * ratio << std::endl;
        Rcpp::Rcout << "-a * log(2): " << -a * log(2)  << std::endl;
    }
    while ( a * ratio > -a * log(2)){
        epsilon = pow(2,a) * epsilon;
        if(diagnostics)
            Rcpp::Rcout << "epsilon for loop in Find Reasonable Epsilon" << epsilon << std::endl;
        this->calculate_gradient(sv);
        sv_prop.bm = sv.bm + epsilon * beta_grad / 2.0;
        sv_prop.tm = sv.tm + epsilon * theta_grad / 2.0;
        sv_prop.sm = sv.sm + epsilon * sigma_grad / 2.0;
        sv_prop.beta = sv.beta + epsilon * sv_prop.bm;
        sv_prop.theta = sv.theta + epsilon * sv_prop.tm;
        sv_prop.sigma = sv.sigma + epsilon * sv_prop.sm;
        this->calculate_gradient(sv_prop);
        sv_prop.bm = sv_prop.bm + epsilon * beta_grad / 2.0;
        sv_prop.tm = sv_prop.tm + epsilon * theta_grad / 2.0;
        sv_prop.sm = sv_prop.sm + epsilon * sigma_grad / 2.0;
        if(diagnostics){
            Rcpp::Rcout << "Proposed Parameters:" << std::endl;
            Rcpp::Rcout << "beta: " << sv_prop.beta << std::endl;
            Rcpp::Rcout << "theta: " << sv_prop.theta_transformed() << std::endl;
            Rcpp::Rcout << "sigma: " << sv_prop.sigma_transformed() << std::endl;
            Rcpp::Rcout << "Proposed Gradients: " << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << "\n" << std::endl;
            Rcpp::Rcout << "Proposed Momenta: " << std::endl;
            Rcpp::Rcout << "beta_m: "  << sv_prop.bm << std::endl;
            Rcpp::Rcout << "theta_m: " << sv_prop.tm << std::endl;
            Rcpp::Rcout << "sigma_m: " << sv_prop.sm << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;
        }
        propose_energy = this->calculate_total_energy(sv_prop);
        ratio =  propose_energy - initial_energy;
    }
    
    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon End with epsilon =  " <<  epsilon << "\n \n \n " << std::endl;
    return(epsilon);
}
