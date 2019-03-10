void STAP_glmer::calculate_glmer_eta(SV_glmer& svg){

    eta = svg.get_alpha_vector() + X_diff * svg.beta + X_mean * svg.beta_bar + Z * svg.delta + svg.b * W;
}


double STAP_glmer::calculate_glmer_ll(SV_glmer& svg){
    
    double out = 0;
    this->calculate_X_diff(svg.theta(0));
    this->calculate_glmer_eta(svg);

    out -= y.size() / 2.0 * log(M_PI * 2 * svg.sigma_sq_transformed() ); 

    // likelihood kernel
    out += - .5 * svg.precision_transformed() * (pow((y - eta ).array(),2)).sum() +  .5 * (pow(svg.b.array(),2)).sum() / svg.Sigma ;


    return(out);
}

double STAP_glmer::calculate_glmer_energy(SV_glmer& svg){

     if(diagnostics){
        Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
        svg.print_pars();
        svg.print_mom();
    }
    
    double out = 0;
    this->calculate_X_diff(svg.theta(0));
    this->calculate_glmer_eta(svg);

    out -= y.size() / 2.0 * log(M_PI * 2 * svg.sigma_sq_transformed() ); 

    // likelihood kernel
    out += - .5 * svg.precision_transformed() * (pow((y - eta ).array(),2)).sum();
    out +=  .5 * (pow(svg.b.array(),2)).sum() * svg.mer_precision_transformed() ;

    if(diagnostics)
        Rcpp::Rcout << "likelihood " << out << std::endl;
    
    // alpha ~ N(25,5)  prior
    out += R::dnorm(svg.alpha,25,5,TRUE);

    // delta ~ N(0,3)
    if(svg.spc(1) != 0)
        out += - svg.delta.size() / 2.0 * log(M_PI * 2 * 9) - .5 * 1.0 / 9.0  * svg.delta.dot(svg.delta); 

    // beta ~ N(0,3) prior
    if(svg.spc(1) != 0)
        out += - svg.beta.size() / 2.0 * log(M_PI * 2 * 9) - .5 * 1.0 / 9.0  * svg.beta.dot(svg.beta); 

    // beta_bar ~ N(0,3) prior
    if(svg.spc(3) != 0)
        out += - svg.beta_bar.size() / 2.0 * log(M_PI * 2 * 9) - .5 * 1.0 / 9.0  * svg.beta_bar.dot(svg.beta_bar); 

    if(diagnostics)
        Rcpp::Rcout << "bb prior " << out << std::endl;

    // log(theta) ~ N(1,1) prior 
    out +=  R::dlnorm(svg.theta_transformed()(0),0,1,TRUE);//-log(transformed_theta) - .5 * log( 2.0 * M_PI) - .5 * pow(log(transformed_theta) - 1,2);
    if(diagnostics)
        Rcpp::Rcout << "theta prior " << out << std::endl;

    // sigma ~ C(0,5)
    out +=  R::dcauchy(svg.sigma_transformed(),0,5,TRUE);
    if(diagnostics)
        Rcpp::Rcout << "sigma prior " << out << std::endl;

    // theta constraints jacobian adjustment 
    out += 10 /(1+exp(-svg.theta(0))) * (1- 1/(1+exp(-svg.theta(0))));

    //glmer normal density
    out += - svg.b.size() / 2.0 * log(M_PI * 2 * svg.mer_var_transformed()) - .5 * svg.mer_precision_transformed() * (svg.b.transpose() * svg.b).sum();

    // exponential prior on sigma_b  and jacobian
    out += - R::dexp(svg.mer_sd_transformed(),1,TRUE) + svg.Sigma;

    if(diagnostics)
        Rcpp::Rcout << "jacobian I" << out << std::endl;
    
    
    // sigma jacobian
    out += svg.sigma;

    if(diagnostics){
        Rcpp::Rcout << "jacobian II" << out << std::endl;
        Rcpp::Rcout << "------------------ \n " << std::endl;
    }

    // Incorporate Kinetic Energy
    out -= svg.kinetic_energy_glmer();
    if(diagnostics)
        Rcpp::Rcout << "Kinetic Energy " << out << std::endl;

    out = (isinf(-out) || isnan(out)) ? (-1 * DBL_MAX) : out;

    if(diagnostics)
        Rcpp::Rcout << "Energy out " << out << std::endl;
   
    return(out);

}

double STAP_glmer::sample_u(SV_glmer& svg,std::mt19937& rng){

    if(diagnostics)
        Rcpp::Rcout << "Sample U Energy Calculation" << std::endl;
    double energy = this->calculate_glmer_energy(svg);
    std::uniform_real_distribution<double> runif(0.0,1.0);
    double log_z = log(runif(rng));
    return(energy + log_z);

}


void STAP_glmer::calculate_gradient(SV_glmer& svg){

    double theta = svg.theta(0);
    double theta_transformed = 10 / (1 + exp(- theta));
    double theta_exponentiated = theta_transformed / (10.0 - theta_transformed);
    double lp_prior_I = pow(theta_transformed,-1) * (10 * exp(-theta)) / pow(1 + exp(-theta),2);
    double lp_prior_II = 2 * log(theta_transformed) / ( 1 + exp(-theta));
    double precision = svg.precision_transformed();
    Eigen::VectorXd alpha_v  = svg.get_alpha_vector() ;
    this->calculate_X_prime_diff(theta_transformed,theta); // also calculates X
    this->calculate_glmer_eta(svg);

    sgg.delta_grad =  precision * ((y - eta ).transpose() * Z).transpose();

    sgg.alpha_grad = svg.spc(0) == 0 ? 0 : precision * (y - eta).sum();

    sgg.beta_grad = (precision * ( y - eta).transpose() * X_diff).transpose();

    sgg.beta_bar_grad = precision * ((y - eta).transpose() * X_mean ).transpose();

    sgg.sigma_grad = precision * (pow((y - eta ).array(),2) ).sum() - y.size();

    sgg.theta_grad = precision * ((y - eta).transpose() * (X_prime_diff * svg.beta + X_mean_prime * svg.beta_bar) ).transpose();


    sgg.b_grad = precision * ((y-eta)) + svg.b * svg.mer_precision_transformed() ;
    sgg.subj_sig_grad =  -y.size() - svg.b.sum()  * svg.mer_precision_transformed()  ;

    // prior components
    sgg.alpha_grad += -1.0 / 25 * (svg.alpha - 25); 
    sgg.delta_grad = sgg. delta_grad - 1.0 / 9.0 * svg.delta;
    sgg.beta_grad = sgg.beta_grad - 1.0 / 9.0 * svg.beta;
    sgg.beta_bar_grad = sgg.beta_bar_grad - 1.0 / 9.0 * svg.beta_bar;
    sgg.theta_grad  = sgg.theta_grad - Eigen::VectorXd::Constant(sgg.theta_grad.size(),lp_prior_I) - Eigen::VectorXd::Constant(sgg.theta_grad.size(),lp_prior_II) ;
    sgg.theta_grad = sgg.theta_grad + Eigen::VectorXd::Constant(sgg.theta_grad.size(),( 1- theta_exponentiated) / (theta_exponentiated + 1));
    sgg.sigma_grad += - (2 * svg.sigma_transformed()) / (25 + svg.sigma_sq_transformed()) + 1;
    sgg.subj_sig_grad += svg.mer_sd_transformed() + 1; 

    if(svg.spc(1) == 0 )
        sgg.delta_grad = Eigen::VectorXd::Zero(1);
    if(svg.spc(2) == 0)
        sgg.beta_grad = Eigen::VectorXd::Zero(1);
    if(svg.spc(3) == 0)
        sgg.beta_bar_grad = Eigen::VectorXd::Zero(1);

}

double STAP_glmer::FindReasonableEpsilon(SV_glmer& sv, std::mt19937& rng){


    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon Start \n " << std::endl;
    double epsilon = 1.0;
    int a;
    SV_glmer sv_prop(sv.spc,rng,diagnostics);
    double ratio,initial_energy,propose_energy;
    initial_energy = this->calculate_glmer_energy(sv);
    this->calculate_gradient(sv);
    sv_prop.momenta_leapfrog_other(sv,epsilon,sgg);
    sv_prop.momenta_leapfrog_position(sv,epsilon);
    this->calculate_gradient(sv_prop);
    sv_prop.momenta_leapfrog_self(epsilon,sgg);
    propose_energy = this->calculate_glmer_energy(sv_prop);
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
    int cntr = 0;
    while ( a * ratio > -a * log(2)){
        epsilon = pow(2,a) * epsilon;
        if(diagnostics)
            Rcpp::Rcout << "epsilon for loop in Find Reasonable Epsilon" << epsilon << std::endl;
        this->calculate_gradient(sv);
        sv_prop.momenta_leapfrog_other(sv,epsilon,sgg);
        sv_prop.momenta_leapfrog_position(sv,epsilon);
        this->calculate_gradient(sv_prop);
        sv_prop.momenta_leapfrog_self(epsilon,sgg);
        propose_energy = this->calculate_glmer_energy(sv_prop);
        ratio =  propose_energy - initial_energy;
        cntr ++;
        if(cntr > 50)
            break;
    }
    
    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon End with epsilon =  " <<  epsilon << "\n \n \n " << std::endl;
    return(epsilon);
}
