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
    X = Eigen::VectorXd::Zero(input_ucrs.rows(),dists.rows());
    X_prime = Eigen::VectorXd::Zero(input_ucrs.rows(),dists.rows());
    diagnostics = input_diagnostics;

}

double STAP::calculate_total_energy(double& cur_alpha, double& cur_beta,double& cur_beta_bar, double& cur_theta, double& cur_sigma, double& cur_am, double& cur_bm,double& cur_bbm, double& cur_tm, double& cur_sm){
            
    // likelihood normalizing constant 
    double transformed_theta = 10 / (1.0 + exp(-cur_theta));
    double transformed_sigma_sq = pow(exp(cur_sigma),2);
    if(diagnostics){
        Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
        Rcpp::Rcout << " Parameter positions:" << std::endl;
        Rcpp::Rcout << "alpha: " << cur_alpha << std::endl;
        Rcpp::Rcout << "beta: " << cur_beta << std::endl;
        Rcpp::Rcout << "beta_bar: " << cur_beta_bar << std::endl;
        Rcpp::Rcout << "theta: " << 10 /  (1 + exp(-cur_theta) )  << std::endl;
        Rcpp::Rcout << "sigma: " <<  exp(cur_sigma)  << std::endl;
        Rcpp::Rcout <<  "------------------" << std::endl;
    }
    double out = 0;
    this->calculate_X_diff(cur_theta);

    out -= y.size() / 2.0 * log(M_PI * 2 * transformed_sigma_sq); 

    // likelihood kernel
    out += - (pow((y - Eigen::VectorXd::Constant(y.size(),cur_alpha) - X_diff * cur_beta - X_mean * cur_beta_bar).array(),2) ).sum() * .5 / transformed_sigma_sq; 

    if(diagnostics)
        Rcpp::Rcout << "likelihood" << out << std::endl;
    
    // alpha ~N(25,5)  prior
    out += R::dnorm(cur_alpha,25,5,TRUE);

    // beta ~ N(0,3) prior
    out += R::dnorm(cur_beta,0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // beta_bar ~ N(0,3) prior
    out += R::dnorm(cur_beta_bar,0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // log(theta) ~ N(1,1) prior 
    out +=  R::dlnorm(transformed_theta,1,1,TRUE);//-log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);

    // sigma ~ C(0,5)
    out +=  R::dcauchy(exp(cur_sigma),0,5,TRUE);

    // theta constraints jacobian adjustment 
    out +=  ( log(10.0) - cur_theta  - 2 * log(1+exp(-cur_theta)));   

    if(diagnostics)
        Rcpp::Rcout << "jacobian I" << out << std::endl;
    
    
    // sigma jacobian
    out += cur_sigma;

    if(diagnostics){
        Rcpp::Rcout << "jacobian II" << out << std::endl;
        Rcpp::Rcout << "------------------ \n " << std::endl;
    }

    // Incorporate Kinetic Energy
    out -= (pow(cur_tm,2)  + pow(cur_bm,2) + pow(cur_sm,2))/2.0;
    out -= (pow(cur_am,2) + pow(cur_bbm,2))/2.0;
    if(diagnostics)
        Rcpp::Rcout << "Kinetic Energy " << out << std::endl;

    out = (isinf(-out) || isnan(out)) ? (-1 * DBL_MAX) : out;

    if(diagnostics)
        Rcpp::Rcout << "Energy out " << out << std::endl;
   
    return(out);

}

double STAP::calculate_total_energy(double& cur_beta,  double& cur_theta, double& cur_sigma, double& cur_bm,  double& cur_tm, double& cur_sm){
    

    // likelihood normalizing constant 
    double transformed_theta = 10 / (1.0 + exp(-cur_theta));
    double transformed_sigma_sq = pow(exp(cur_sigma),2);
    if(diagnostics){
        Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
        Rcpp::Rcout << " Parameter positions:" << std::endl;
        Rcpp::Rcout << "beta: " << cur_beta << std::endl;
        Rcpp::Rcout << "theta: " << 10 /  (1 + exp(-cur_theta) )  << std::endl;
        Rcpp::Rcout << "sigma: " <<  exp(cur_sigma)  << std::endl;
        Rcpp::Rcout <<  "------------------" << std::endl;
    }
    double out = 0;
    this->calculate_X_diff(cur_theta);

    out -= y.size() / 2.0 * log(M_PI * 2 * transformed_sigma_sq); 

    // likelihood kernel
    out += - (pow((y - X_diff * cur_beta).array(),2) ).sum() * .5 / transformed_sigma_sq; 

    if(diagnostics)
        Rcpp::Rcout << "likelihood" << out << std::endl;
    

    // beta ~ N(0,3) prior
    out += R::dnorm(cur_beta,0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // log(theta) ~ N(1,1) prior 
    out +=  R::dlnorm(transformed_theta,1,1,TRUE);//-log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);

    // sigma ~ C(0,5)
    out +=  R::dcauchy(exp(cur_sigma),0,5,TRUE);

    // theta constraints jacobian adjustment 
    out +=  ( log(10.0) - cur_theta  - 2 * log(1+exp(-cur_theta)));   

    if(diagnostics)
        Rcpp::Rcout << "jacobian I" << out << std::endl;
    
    
    // sigma jacobian
    out += cur_sigma;

    if(diagnostics){
        Rcpp::Rcout << "jacobian II" << out << std::endl;
        Rcpp::Rcout << "------------------ \n " << std::endl;
    }

    // Incorporate Kinetic Energy
    out -= (pow(cur_tm,2)  + pow(cur_bm,2) + pow(cur_sm,2))/2.0;
    if(diagnostics)
        Rcpp::Rcout << "Kinetic Energy " << out << std::endl;

    out = (isinf(-out) || isnan(out)) ? (-1 * DBL_MAX) : out;

    if(diagnostics)
        Rcpp::Rcout << "Energy out " << out << std::endl;
   
    return(out);
}

double STAP::sample_u(double& cur_beta, double& cur_theta, double& cur_sigma, double& cur_bm,  double& cur_tm, double& cur_sm, std::mt19937& rng){

    if(diagnostics)
        Rcpp::Rcout << "Sample U Energy Calculation" << std::endl;
    double energy = this->calculate_total_energy(cur_beta,cur_theta,cur_sigma,cur_bm,cur_tm,cur_sm);
    std::uniform_real_distribution<double> runif(0.0,1.0);
    double log_z = log(runif(rng));
    return(energy + log_z);
}

void STAP::calculate_X( double& theta){
    
    int start_col;
    int range_len;
    for(int bef_ix = 0; bef_ix <= (dists.rows()-1); bef_ix ++){
        for(int subj_ix = 0; subj_ix < u_crs.rows(); subj_ix ++){
            start_col = u_crs(subj_ix,bef_ix);
            range_len = u_crs(subj_ix,bef_ix+1) - start_col + 1;
            X(subj_ix) = (exp(- dists.block(bef_ix,start_col,1,range_len) / theta  )).sum();
        }
    }
}

void STAP::calculate_X_diff( double& theta){
    
    double transformed_theta = 10 / (1 + exp(-theta));
    this->calculate_X(transformed_theta);
    this->calculate_X_mean();
    X_diff = X - X_mean;
}

void STAP::calculate_X_mean(){

    X_mean =  subj_array.transpose() * ((subj_array * X).array() * subj_n).matrix();
}

void STAP::calculate_X_prime(double& theta, double& cur_theta){ 

    int start_col;
    int range_len;
    // Can probably combine below with calculate_X to remove N*q separate operations...
    for(int bef_ix = 0; bef_ix <= (dists.rows()-1); bef_ix ++){
        for(int subj_ix = 0; subj_ix < u_crs.rows(); subj_ix ++){
            start_col = u_crs(subj_ix,bef_ix);
            range_len = u_crs(subj_ix,bef_ix+1) - start_col + 1;
            X_prime(subj_ix) = (dists.block(bef_ix,start_col,1,range_len) /  10.0  * exp(- dists.block(bef_ix,start_col,1,range_len) / theta - cur_theta  )).sum() / pow(theta,2);
        }
    }
}

void STAP::calculate_X_mean_prime(){ 

    X_mean_prime = subj_array.transpose() * ((subj_array * X_prime).array() * subj_n).matrix();

}

void STAP::calculate_X_prime_diff(double& theta,double& cur_theta){

    this->calculate_X_prime(theta,cur_theta);
    this->calculate_X_mean_prime();
    X_prime_diff = (X_prime - X_mean_prime);
}

void STAP::calculate_gradient(double& cur_beta, double& cur_theta, double& cur_sigma){


    double theta_transformed =  10.0  / (1.0 + exp( - cur_theta));
    double sigma_transformed = exp(cur_sigma);
    double theta_exponentiated = theta_transformed / (10.0 - theta_transformed);
    double lp_prior_I = pow(theta_transformed,-1) * (10 * exp(-cur_theta)) / pow(1 + exp(-cur_theta),2);
    double lp_prior_II = 2*(log(theta_transformed) -1)/ (1 + exp(-cur_theta));
    this->calculate_X_diff(cur_theta);
    this->calculate_X_prime_diff(theta_transformed,cur_theta);
    theta_grad = pow(sigma_transformed,-2) * (y - cur_beta * X_diff).dot(X_prime_diff); //likelihood theta grad  
    theta_grad = theta_grad - lp_prior_I ; // log theta prior 
    theta_grad = theta_grad - lp_prior_II; // log theta prior
    theta_grad = theta_grad +  (1 - theta_exponentiated) / (theta_exponentiated + 1);  //Jacobian factor
    beta_grad = pow(sigma_transformed,-2) * (y-cur_beta * X_diff).dot(X_diff) - 1.0/9.0 * cur_beta;
    sigma_grad = pow(sigma_transformed,-2) * (y-cur_beta*X_diff).dot(y-cur_beta*X_diff) - y.size()  - (2*sigma_transformed) / (5+sigma_transformed)+ 1; 

}

void STAP::calculate_gradient(double& cur_alpha, double& cur_beta,double& cur_beta_bar, double& cur_theta, double& cur_sigma){

    double theta_transformed = 10.0 / (1.0 + exp(- cur_theta));
    double precision = pow(exp(cur_sigma),-2);
    double theta_exponentiated = theta_transformed / (10.0 - theta_transformed);
    double lp_prior_I = pow(theta_transformed,-1) * (10 * exp(-cur_theta)) / pow(1 + exp(-cur_theta),2);
    double lp_prior_II = 2*(log(theta_transformed) -1)/ (1 + exp(-cur_theta));
    this->calculate_X_diff(cur_theta);
    this->calculate_X_prime_diff(theta_transformed,cur_theta);
    // likelihood
    alpha_grad = precision * ( y.sum() - cur_beta * X_diff.sum() - cur_beta_bar * X_mean.sum() - y.size() * cur_alpha);
    beta_grad = precision * (y.dot(X_diff) - cur_beta *X_diff.dot(X_diff) - X_diff.dot(X_mean) * cur_beta_bar - X_diff.sum() * cur_alpha);
    beta_bar_grad = precision * (y.dot(X_mean) - cur_beta * X_diff.dot(X_mean) - cur_beta_bar * X_diff.dot(X_mean) - X_mean.sum() * cur_alpha);
    sigma_grad = precision * (pow((y - Eigen::VectorXd::Constant(y.size(),cur_alpha) - X_diff * cur_beta - X_mean * cur_beta_bar).array(),2) ).sum() - y.size();
    theta_grad = precision * ( y.dot(X_prime_diff)*cur_beta - y.dot(X_mean_prime) * cur_beta_bar - pow(cur_beta,2) * X_prime_diff.dot(X_diff)); 
    theta_grad += precision * (cur_beta * X_prime_diff.dot(X_mean) + X_diff.dot(X_mean_prime) * cur_beta_bar - cur_beta * X_prime_diff.sum() * cur_alpha);
    theta_grad +=  precision * (cur_beta_bar * X_prime_diff.dot(X_mean) * cur_beta - cur_beta_bar * X_mean_prime.sum() * cur_alpha);
    // prior components
    alpha_grad += -1.0 / 25 * cur_alpha;
    beta_grad += -1.0 / 9.0 * cur_beta;
    beta_bar_grad += -1.0 / 9.0 * cur_beta_bar;
    theta_grad +=  -lp_prior_I - lp_prior_II;
    theta_grad += theta_grad + (1 - theta_exponentiated) / (theta_exponentiated +1); // jacobian factor
    sigma_grad +=  -(2*pow(precision,-2)) / (5+pow(precision,-2)) + 1; 

}

double STAP::FindReasonableEpsilon(double& cur_beta, double& cur_theta,double& cur_sigma, double& bm, double& tm,double& sm,std::mt19937& rng){

    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon Start \n " << std::endl;
    double epsilon = 1.0;
    int a;
    double beta_prop,theta_prop,sigma_prop,bm_prop,tm_prop,sm_prop,ratio,initial_energy,propose_energy;
    this->calculate_gradient(cur_beta, cur_theta,cur_sigma);
    if(diagnostics){
        Rcpp::Rcout << "Initial Parameters:  " << std::endl;
        Rcpp::Rcout << "beta: " << cur_beta << std::endl;
        Rcpp::Rcout << "theta: " << 10 / (1+exp(-cur_theta)) << std::endl;
        Rcpp::Rcout << "sigma: " << exp(cur_sigma) << std::endl;
        Rcpp::Rcout << "Initial Gradients: \n " << std::endl;
        Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
        Rcpp::Rcout << "Initial Momenta: \n " << std::endl;
        Rcpp::Rcout << "beta_m: "  << bm << std::endl;
        Rcpp::Rcout << "theta_m: " << tm << std::endl;
        Rcpp::Rcout << "sigma_m: " << sm << std::endl;
        Rcpp::Rcout << "-------------------- \n " << std::endl;
    }
    bm_prop = bm + epsilon * beta_grad / 2.0;
    tm_prop = tm + epsilon * theta_grad / 2.0;
    sm_prop = sm + epsilon * sigma_grad / 2.0;
    beta_prop = cur_beta + epsilon * bm_prop;
    theta_prop = cur_theta + epsilon * tm_prop;
    sigma_prop = cur_sigma + epsilon * sm_prop;
    sigma_prop = isinf(exp(sigma_prop)) ? log(DBL_MAX) : sigma_prop;
    this->calculate_gradient(beta_prop,theta_prop,sigma_prop);
    bm_prop = bm_prop + epsilon * beta_grad / 2.0;
    tm_prop = tm_prop + epsilon * theta_grad / 2.0;
    sm_prop = sm_prop + epsilon * sigma_grad / 2.0;
    if(diagnostics){
        Rcpp::Rcout << "Proposed Parameters:" << std::endl;
        Rcpp::Rcout << "beta: " << beta_prop << std::endl;
        Rcpp::Rcout << "theta: " << 10 / (1+exp(-theta_prop)) << std::endl;
        Rcpp::Rcout << "sigma: " << exp(sigma_prop) << std::endl;
        Rcpp::Rcout << "Proposed Gradients: " << std::endl;
        Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
        Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
        Rcpp::Rcout << "sigma_grad: " << sigma_grad << "\n" << std::endl;
        Rcpp::Rcout << "Proposed Momenta: " << std::endl;
        Rcpp::Rcout << "beta_m: "  << bm_prop << std::endl;
        Rcpp::Rcout << "theta_m: " << tm_prop << std::endl;
        Rcpp::Rcout << "sigma_m: " << sm_prop << std::endl;
        Rcpp::Rcout << "-------------------- \n " << std::endl;
    }
    initial_energy = this->calculate_total_energy(cur_beta,cur_theta,cur_sigma,bm,tm,sm);
    propose_energy = this->calculate_total_energy(beta_prop,theta_prop,sigma_prop,bm_prop,tm_prop,sm_prop);
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
        this->calculate_gradient(cur_beta,cur_theta,cur_sigma);
        if(diagnostics){
            Rcpp::Rcout << "Initial Parameters:  " << std::endl;
            Rcpp::Rcout << "beta: " << cur_beta << std::endl;
            Rcpp::Rcout << "theta: " << 10 / (1+exp(-cur_theta)) << std::endl;
            Rcpp::Rcout << "sigma: " << exp(cur_sigma) << std::endl;
            Rcpp::Rcout << "Initial Gradients: \n " << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
            Rcpp::Rcout << "Initial Momenta: \n " << std::endl;
            Rcpp::Rcout << "beta_m: "  << bm << std::endl;
            Rcpp::Rcout << "theta_m: " << tm << std::endl;
            Rcpp::Rcout << "sigma_m: " << sm << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;
        }
        bm_prop = bm + epsilon * beta_grad / 2.0;
        tm_prop = tm + epsilon * theta_grad / 2.0;
        sm_prop = sm + epsilon * sigma_grad / 2.0;
        beta_prop = cur_beta + epsilon * bm_prop;
        theta_prop = cur_theta + epsilon * tm_prop;
        sigma_prop = cur_sigma + epsilon * sm_prop;
        sigma_prop = isinf(exp(sigma_prop)) ? log(DBL_MAX) : sigma_prop;
        this->calculate_gradient(beta_prop,theta_prop,sigma_prop);
        bm_prop = bm_prop + epsilon * beta_grad / 2.0;
        tm_prop = tm_prop + epsilon * theta_grad / 2.0;
        sm_prop = sm_prop + epsilon * sigma_grad / 2.0;
        if(diagnostics){
            Rcpp::Rcout << "Proposed Parameters:" << std::endl;
            Rcpp::Rcout << "beta: " << beta_prop << std::endl;
            Rcpp::Rcout << "theta: " << 10 / (1+exp(-theta_prop)) << std::endl;
            Rcpp::Rcout << "sigma: " << exp(sigma_prop) << std::endl;
            Rcpp::Rcout << "Proposed Gradients: " << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << "\n" << std::endl;
            Rcpp::Rcout << "Proposed Momenta: " << std::endl;
            Rcpp::Rcout << "beta_m: "  << bm_prop << std::endl;
            Rcpp::Rcout << "theta_m: " << tm_prop << std::endl;
            Rcpp::Rcout << "sigma_m: " << sm_prop << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;
        }
        propose_energy = this->calculate_total_energy(beta_prop,theta_prop,sigma_prop,bm_prop,tm_prop,sm_prop);
        propose_energy = isinf(-propose_energy) ? -DBL_MAX : propose_energy;
        ratio =  propose_energy - initial_energy;
    }
    
    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon End with epsilon =  " <<  epsilon << "\n \n \n " << std::endl;
    return(epsilon);
}
