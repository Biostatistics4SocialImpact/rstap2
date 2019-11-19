void STAP_glmer::calculate_glmer_eta(SV_glmer& svg){

    eta = svg.get_alpha_vector() + X_diff * svg.beta + X_mean * svg.beta_bar + Z * svg.delta;
    eta = eta + subj_array.transpose() * svg.adjust_b();
    eta = eta +  W.transpose() * svg.adjust_b_slope();
}

Eigen::VectorXd STAP_glmer::bdel(SV_glmer& svg){

    Eigen::VectorXd out(y.size());
    out =  svg.mer_sd_1() * Eigen::VectorXd::Ones(W.cols()) +  (W.transpose() * svg.mer_sd_2() * svg.get_rho() )  * Eigen::VectorXd::Ones(W.rows());
    return(svg.precision_transformed() * subj_array * ((y-eta).array() * out.array()).matrix() - svg.b ); 
}

Eigen::VectorXd STAP_glmer::bslope_del(SV_glmer& svg){

  return ( svg.mer_L_11() * svg.precision_transformed() * W * (y-eta) - svg.b_slope);
}

double STAP_glmer::rho_del(SV_glmer& svg){

  Eigen::VectorXd temp(y.size());
  double out = 0.0;
  temp = W.transpose() * (svg.b    -  svg.b_slope * svg.get_rho() / sqrt(svg.get_rho_sq_c())   );
  out =  svg.precision_transformed() * svg.mer_sd_2()  * svg.get_rho_derivative() * (y-eta).dot(temp);
  out +=  (1.0 - exp(svg.Sigma(2))) / (exp(svg.Sigma(2)) + 1.0); // jacobian

  return(out);
}


double STAP_glmer::calculate_glmer_ll(SV_glmer& svg){
    
    double out = 0;
    this->calculate_X_diff(svg.theta(0),svg.theta_t(0));
    this->calculate_glmer_eta(svg);

    out += - y.size() / 2.0 * log( M_PI * 2 * svg.sigma_sq_transformed() ); 
    out += - .5 * svg.precision_transformed() * (y-eta).dot(y-eta) ; 

    out += - svg.b.rows() * log(M_PI * 2);

    out += -.5 * svg.b.dot(svg.b);
    if(svg.spc(6) == 2)
      out += -.5 * svg.b_slope.dot(svg.b_slope) ;

    return(out);
}

double STAP_glmer::calculate_glmer_energy(SV_glmer& svg){

     if(diagnostics){
        Rcpp::Rcout << " Energy Calculation \n " << "------------------" << std::endl;
        svg.print_pars();
        svg.print_mom();
     }
    
    double out = 0;
    out = calculate_glmer_ll(svg);
    
    if(diagnostics)
        Rcpp::Rcout << "likelihood " << out << std::endl;
    
    // alpha ~ N(25,5)  prior
    out += R::dnorm(svg.alpha,26,5,TRUE);

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

    // theta ~ N(1,1) prior 
    out +=  R::dlnorm(svg.theta_transformed()(0),0,1,TRUE);
    if(diagnostics)
        Rcpp::Rcout << "theta prior " << out << std::endl;

    out += R::dlnorm(svg.theta_t_transformed()(0),0,1,TRUE) + svg.theta_t(0) ;

    // sigma ~ C(0,5)
    out +=  R::dcauchy(svg.sigma_transformed(),0,5,TRUE);
    if(diagnostics)
        Rcpp::Rcout << "sigma prior " << out << std::endl;

    // theta constraints jacobian adjustment 
    out +=  log_sigmoid_transform_derivative(svg.theta(0),0,10);
    if(diagnostics)
        Rcpp::Rcout << "theta jacobian adjustment " << out << std::endl;

    // exponential prior on sigma_b's  + jacobian
    if(svg.spc(6) == 1){
        out += R::dnorm(svg.mer_sd_1(),0,100,TRUE) + svg.Sigma(0);
        if(diagnostics)
            Rcpp::Rcout << "Sigma priors " << out << std::endl;
    }else{
        // sigma_1b
        out += R::dlnorm(svg.mer_sd_1(),1,1,TRUE) + svg.Sigma(0);
        // sigma_2b
        out += R::dlnorm(svg.mer_sd_2(),1,1,TRUE) + svg.Sigma(1);
        if(diagnostics)
            Rcpp::Rcout << "Sigma priors " << out << std::endl;
        // rho uniform prior and jacobian constraint
        out += log_sigmoid_transform_derivative(svg.Sigma(2),-1,1);
        if(diagnostics)
            Rcpp::Rcout << "corr constraint " << out << std::endl;
    }

    // sigma jacobians
    out += svg.sigma;

    // Incorporate Kinetic Energy
    out -= svg.kinetic_energy_glmer();
    if(diagnostics)
        Rcpp::Rcout << " after Kinetic Energy " << out << std::endl;

    if(diagnostics)
        Rcpp::Rcout << " Kinetic Energy" << svg.kinetic_energy_glmer() << std::endl;

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
    double precision = svg.precision_transformed();
    this->calculate_X_prime_diff(theta_transformed,theta,svg.theta_t_transformed()(0)); // also calculates X
    this->calculate_glmer_eta(svg);
    Eigen::VectorXd e = y - eta;

    sgg.delta_grad =  precision * (e.transpose() * Z).transpose();

    sgg.alpha_grad = svg.spc(0) == 0 ? 0 : precision * e.sum();

    sgg.beta_grad = (precision * e.transpose() * X_diff).transpose();

    sgg.beta_bar_grad = precision * ( e.transpose() * X_mean ).transpose();

    sgg.sigma_grad =  precision * (pow(e.array(),2) ).sum() - y.size();

    sgg.theta_grad = precision * (e.transpose() * (X_prime_diff * svg.beta + X_mean_prime * svg.beta_bar) ).transpose();

    sg.theta_t_grad = precision * ((e).transpose() * (X_tprime_diff * svg.beta + X_mean_tprime * svg.beta_bar) ).transpose();

    sgg.b_grad = bdel(svg);

    sgg.subj_sig_grad = Eigen::VectorXd(svg.Sigma.rows());
    sgg.subj_sig_grad(0) = precision * svg.mer_sd_1() * e.dot( (subj_array.transpose() * svg.b));
    sgg.subj_sig_grad(0) +=  -(log(svg.mer_sd_1()) -1.0) ; // log(sigma_1) ~ normal(1,1) prior and jacobian
    sgg.b_slope_grad = Eigen::VectorXd::Zero(svg.b_slope.rows());

    if(svg.Sigma.rows() == 3){
        sgg.b_slope_grad = bslope_del(svg);
        sgg.subj_sig_grad(1) =  precision * svg.mer_sd_2()  *   e.dot((W.transpose() * (svg.b * svg.get_rho() + svg.b_slope * sqrt(svg.get_rho_sq_c()) ) ));
        sgg.subj_sig_grad(1) +=  - (log(svg.mer_sd_2())-1.0); // switch back to log normal (1,1) 
        sgg.subj_sig_grad(2) = rho_del(svg);
    }

    // prior components
    sgg.alpha_grad += -1.0 / 25.0 * (svg.alpha - 26.0 ); 
    sgg.delta_grad = sgg.delta_grad - 1.0 / 9.0 * svg.delta;
    sgg.beta_grad = sgg.beta_grad - 1.0 / 9.0 * svg.beta;
    sgg.beta_bar_grad = sgg.beta_bar_grad - 1.0 / 9.0 * svg.beta_bar;
    sgg.theta_grad(0) += - pow(theta_transformed,-1) * sigmoid_transform_derivative(svg.theta(0),0,10);
    sgg.theta_grad(0) += - (log(theta_transformed)) * pow(theta_transformed,-1) * sigmoid_transform_derivative(svg.theta(0),0,10);
    sgg.theta_grad(0)  += log_sigmoid_transform_derivative(svg.theta(0),0,10); 
    sg.theta_t_grad(0) += - (svg.theta_t(0));
    sgg.sigma_grad += - (2 * svg.sigma_transformed()) / (25 + svg.sigma_sq_transformed()) + 1;

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
        calculate_gradient(sv);
        sv_prop.momenta_leapfrog_other(sv,epsilon,sgg);
        sv_prop.momenta_leapfrog_position(sv,epsilon);
        calculate_gradient(sv_prop);
        sv_prop.momenta_leapfrog_self(epsilon,sgg);
        propose_energy = calculate_glmer_energy(sv_prop);
        ratio =  propose_energy - initial_energy;
        cntr ++;
        if(cntr > 50){
            break;
            Rcpp::Rcout << " Difficulty finding reasonable stepsize, check model parameterization and possibly report bug " << std::endl;
        }
    }
    
    if(diagnostics)
        Rcpp::Rcout << "Find Reasonable Epsilon End with epsilon =  " <<  epsilon << "\n \n \n " << std::endl;

    return(epsilon);
}
