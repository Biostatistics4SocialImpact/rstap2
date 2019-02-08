STAP::STAP(Eigen::ArrayXXd& input_dists,
           Eigen::ArrayXXi& input_ucrs,
           Eigen::MatrixXd& input_subj_array,
           Eigen::ArrayXd& input_subj_n,
           Eigen::VectorXd& input_y){

    dists = input_dists;
    u_crs = input_ucrs;
    sigma =  1.0;
    subj_array = input_subj_array;
    subj_n = input_subj_n;
    y = input_y;
    X = Eigen::VectorXd::Zero(input_ucrs.rows(),dists.rows());
    X_prime = Eigen::VectorXd::Zero(input_ucrs.rows(),dists.rows());


}


double STAP::calculate_total_energy(double cur_beta,  double cur_theta, double& cur_bm,  double& cur_tm){
    

    //Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
    // likelihood normalizing constant 
    double transformed_theta = 10 / (1.0 + exp(-cur_theta));
    transformed_theta = transformed_theta == 0 ? DBL_MIN : transformed_theta;
    double out = 0;
    for(int i =0;i<y.rows();i++)
        out += R::dnorm(y[i],X_diff(i)*cur_beta,1,TRUE);

//    out -= y.size() / 2.0 * log(M_PI * 2); 

    // likelihood kernel
 //   out += - (pow((y - X_diff * cur_beta).array(),2) ).sum() * .5; 

    // beta ~ N(0,3) prior
    out += R::dnorm(cur_beta,0,3,TRUE);//- 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // log(theta) ~ N(1,1) prior 
    out +=  R::dlnorm(transformed_theta,1,1,TRUE);//-log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);

    // theta constraints jacobian adjustment 
    out +=  ( log(10.0) - cur_theta  - 2 * log(1+exp(-cur_theta)));   

    // Incorporate Kinetic Energy
    out -= (pow(cur_tm,2)  + pow(cur_bm,2))/2.0;
    
    return(out);
}

double STAP::sample_u( double& cur_beta, double& cur_theta ,  double& cur_bm,  double& cur_tm, std::mt19937& rng){


    this->calculate_X_diff(cur_theta);
    double energy = this->calculate_total_energy(cur_beta,cur_theta,cur_bm,cur_tm);
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
    this->calculate_X_mean(transformed_theta);
    X_diff = X - X_mean;
}

void STAP::calculate_X_mean( double& theta){

    X_mean =  subj_array.transpose() * ((subj_array * X).array() * subj_n).matrix();
}

void STAP::calculate_X_prime(double& theta, double& cur_theta){ 

    int start_col;
    int range_len;
    for(int bef_ix = 0; bef_ix <= (dists.rows()-1); bef_ix ++){
        for(int subj_ix = 0; subj_ix < u_crs.rows(); subj_ix ++){
            start_col = u_crs(subj_ix,bef_ix);
            range_len = u_crs(subj_ix,bef_ix+1) - start_col + 1;
            X_prime(subj_ix) = (dists.block(bef_ix,start_col,1,range_len) /  10.0  * exp(- dists.block(bef_ix,start_col,1,range_len) / theta - cur_theta  )).sum() / pow(theta,2);
        }
    }
    

}

void STAP::calculate_X_mean_prime(double& theta,double& cur_theta){ 

    X_mean_prime = subj_array.transpose() * ((subj_array * X_prime).array() * subj_n).matrix();

}

void STAP::calculate_X_prime_diff(double& theta,double& cur_theta){

    this->calculate_X_prime(theta,cur_theta);
    this->calculate_X_mean_prime(theta,cur_theta);
    X_prime_diff = (X_prime - X_mean_prime);
}

void STAP::calculate_gradient(double& cur_beta, double& cur_theta){


    double theta_transformed;
    theta_transformed =  10.0  / (1.0 + exp( - cur_theta));
    theta_transformed = theta_transformed == 0 ? DBL_MIN : theta_transformed;
    double theta_exponentiated = theta_transformed / (10.0 - theta_transformed);
    double lp_prior_I = pow(theta_transformed,-1) * (10 * exp(-cur_theta)) / pow(1 + exp(-cur_theta),2);
    lp_prior_I = isnan(lp_prior_I) ? 0 : lp_prior_I;
    double lp_prior_II = 2*(log(theta_transformed) -1)/ (1 + exp(-cur_theta));
    this->calculate_X_diff(cur_theta);
    this->calculate_X_prime_diff(theta_transformed,cur_theta);
    theta_grad = (y - cur_beta * X_diff).dot(X_prime_diff); //likelihood theta grad  
    theta_grad = theta_grad - lp_prior_I ; // log theta prior 
    theta_grad = theta_grad - lp_prior_II; // log theta prior
    theta_grad = theta_grad +  (1 - theta_exponentiated) / (theta_exponentiated + 1);  //Jacobian factor
    theta_grad = (theta_grad == 0 && isinf(exp(-cur_theta)))  ?  100 :  theta_grad;
    beta_grad = (y-cur_beta * X_diff).dot(X_diff) - 1.0/9.0 * cur_beta;

};

double STAP::FindReasonableEpsilon(double& cur_beta, double& cur_theta,double& bm, double& tm, std::mt19937& rng){

    double epsilon = 1.0;
    int a;
    double beta_prop,theta_prop,bm_prop,tm_prop,ratio,initial_energy,propose_energy;
    this->calculate_gradient(cur_beta, cur_theta);
    bm_prop = bm + epsilon * beta_grad / 2.0;
    tm_prop = tm + epsilon * theta_grad / 2.0;
    beta_prop = cur_beta + epsilon * bm_prop;
    theta_prop = cur_theta + epsilon * tm_prop;
    this->calculate_gradient(beta_prop,theta_prop);
    bm_prop = bm_prop + epsilon * beta_grad / 2.0;
    tm_prop = tm_prop + epsilon * theta_grad / 2.0;
    this->calculate_X_diff(cur_theta);
    initial_energy = this->calculate_total_energy(cur_beta,cur_theta,bm,tm);
    this->calculate_X_diff(theta_prop);
    propose_energy = this->calculate_total_energy(beta_prop,theta_prop,bm_prop,tm_prop);
    ratio =  propose_energy - initial_energy;
    a = ratio > log(.5) ? 1 : -1;
    while ( a * ratio > -a * log(2)){
        epsilon = pow(2,a) * epsilon;
        this->calculate_gradient(cur_beta,cur_theta);
        bm_prop = bm + epsilon * beta_grad / 2.0;
        tm_prop = tm + epsilon * theta_grad / 2.0;
        beta_prop = cur_beta + epsilon * bm_prop;
        theta_prop = cur_theta + epsilon * tm_prop;
        this->calculate_gradient(beta_prop,theta_prop);
        bm_prop = bm_prop + epsilon * beta_grad / 2.0;
        tm_prop = tm_prop + epsilon * theta_grad / 2.0;
        this->calculate_X_diff(theta_prop);
        propose_energy = this->calculate_total_energy(beta_prop,theta_prop,bm_prop,tm_prop);
        propose_energy = isinf(-propose_energy) ? -DBL_MAX : propose_energy;
        ratio =  propose_energy - initial_energy;
    }
    
    return(epsilon);
}
