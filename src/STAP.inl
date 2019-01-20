STAP::STAP(Eigen::MatrixXd input_dists, Eigen::MatrixXd input_d_one, Eigen::MatrixXd input_d_two, Eigen::MatrixXd input_d_three, Eigen::VectorXd input_y){

    dists = input_dists;
    d_one = input_d_one;
    d_two = input_d_two;
    d_three = input_d_three;
    sigma =  1.0;
    y = input_y;

}


double STAP::calculate_total_energy(double cur_beta,  double cur_theta, double &cur_bm,  double &cur_tm){
    
    // likelihood normalizing constant 
    double transformed_theta = 10 / (1.0 + exp(-cur_theta));
    double out = 0;
    out -= y.size() / 2.0 * log(M_PI * 2); 

    // likelihood kernel
    out += - (pow((y - X_diff * cur_beta).array(),2) ).sum() * .5; 

    // beta ~ N(0,3) prior
    out += - 0.5 * log(M_PI * 18.0) - 1.0 / 18.0 * pow(cur_beta,2);

    // log(theta) ~ N(1,1) prior 
    out +=  -log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);

    // theta constraints jacobian adjustment 
    out +=  ( log(10.0) - cur_theta  - 2 * log(1+exp(-cur_theta)));   

    // Incorporate Kinetic Energy

    out -= (pow(cur_tm,2)  + pow(cur_bm,2))/2.0;
   
    
    return(out);
}

double STAP::sample_u( double &cur_beta, double &cur_theta ,  double &cur_bm,  double &cur_tm, std::mt19937 &rng){

    double transformed_theta;
    transformed_theta =  (10.0 / (1.0 + exp(-cur_theta)));

    this->calculate_X_mean(transformed_theta);
    this->calculate_X_diff(transformed_theta);
    double energy = this->calculate_total_energy(cur_beta,cur_theta,cur_bm,cur_tm);
    std::uniform_real_distribution<double> runif(0.0,exp(energy));
    double u = runif(rng);
    return(u);
}

void STAP::calculate_X_diff( double &theta){
    
    X = ( exp(- dists.array() / theta ) ).rowwise().sum();
    X_diff = X - X_mean;
}

void STAP::calculate_X_mean( double &theta){

    Eigen::VectorXd tmp(d_one.rows());
    tmp = (((exp(-d_one.array() / theta ))).matrix()).rowwise().sum();
    X_mean = tmp;
    tmp = (((exp(-d_two.array() / theta ))).matrix()).rowwise().sum();
    X_mean = X_mean + tmp;
    tmp = (((exp(-d_three.array() / theta ))).matrix()).rowwise().sum();
    X_mean = (X_mean + tmp) * 1.0 / 3.0;

}

void STAP::calculate_X_prime(double &theta, double &cur_theta){ 

    X_prime =  (dists.array() / 10.0 * exp( - dists.array() / theta - cur_theta)).rowwise().sum(); 
}

void STAP::calculate_X_mean_prime(double &theta,double &cur_theta){ 

    Eigen::VectorXd tmp(d_one.rows());
    tmp = (d_one.array() / 10.0 * exp( - d_one.array() / theta - cur_theta)).rowwise().sum();
    X_mean_prime = tmp;
    tmp  = (d_two.array() / 10.0 * exp( - d_two.array() / theta - cur_theta)).rowwise().sum();
    X_mean_prime = tmp + X_mean_prime;
    tmp = (d_three.array() / 10.0 * exp( - d_three.array() / theta - cur_theta)).rowwise().sum();
    X_mean_prime = (tmp + X_mean_prime) / 3.0;

}

void STAP::calculate_X_prime_diff(double &theta,double &cur_theta){

    this->calculate_X_prime(theta,cur_theta);
    this->calculate_X_mean_prime(theta,cur_theta);
    X_prime_diff = (X_prime - X_mean_prime);
}


void STAP::calculate_gradient(double &cur_beta, double &cur_theta){


    double theta_transformed;
    theta_transformed =  10.0  / (1.0 + exp( - cur_theta));
    this->calculate_X_mean(theta_transformed);
    this->calculate_X_diff(theta_transformed);
    this->calculate_X_prime_diff(theta_transformed,cur_theta);
    theta_grad = (y - cur_beta * X_diff).dot(X_prime_diff); //likelihood theta grad  
    theta_grad = theta_grad - pow(theta_transformed,-1) *  (10 * exp(-cur_theta)) / pow((1 + exp(-cur_theta)),2); // log theta prior 
    theta_grad = theta_grad - (log(theta_transformed) -1)/theta_transformed *(10 * exp(-cur_theta)) / pow((1 + exp(-cur_theta)),2); // log theta prior
    theta_grad = theta_grad -  (exp(cur_theta) + 3) / (exp(cur_theta) + 1);  //Jacobian factor
    beta_grad = (y-cur_beta * X_diff).dot(X_diff) + 1.0/9.0 * cur_beta;

};
