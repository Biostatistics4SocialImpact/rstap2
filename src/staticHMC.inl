// Public Methods
//

template<typename T>
staticHMC<T>::staticHMC(
        const double &adapt_delta,
        const int seed
        ){

    if (adapt_delta<=0 || adapt_delta>1)
        throw(std::logic_error("Acceptance ratio must be between (0,1]"));

    _rng = std::mt19937(seed);
};



template<typename T>
template< typename S>
void staticHMC<T>::calculate_initial_gradient(S init_beta, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma){

    _i_mt = GaussianNoise(init_beta.size(),_rng);
    _mt = _i_mt; 
    _init_beta = init_beta;
    _init_grad_energy =  pow(sigma,-2) * (-X.transpose() * y + X.transpose() * X * init_beta);
    _grad_energy = _init_grad_energy;

}


template<typename T>
template<typename S>
void staticHMC<T>::Leapfrog(S &beta, int L, double epsilon, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma){

    _mt = _mt - epsilon * _grad_energy / 2.0;
    for(int i = 1; i < L; i++){
        beta = beta + epsilon  *  _mt; // full step for parameters
        _grad_energy  =  pow(sigma,-2) * (-X.transpose() * y + X.transpose() * X * beta); // calculate gradient
        _mt = _mt - epsilon * _grad_energy; // full step for momentum
    }
    beta = beta + epsilon  *  _mt; // final step
    _mt = _mt - epsilon * _grad_energy / 2.0;
}

template<typename T>
template<typename S>
T staticHMC<T>::MH_step(S beta, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma){

    double energy;
    double energy_init;
    double kinetic;
    double kinetic_init;
    double u;
    double out;
    kinetic = _mt.dot(_mt)  / 2.0 ;
    kinetic_init = (_i_mt.dot(_i_mt)) / 2.0;
    energy = -y.size()/2.0 * log(M_PI * 2 * pow(sigma,2)) -  (y - X*beta).dot(y-X*beta) *( 1.0 / (2*pow(sigma,2)));
    energy_init = -y.size()/2.0 * log(M_PI * 2 * pow(sigma,2)) -  (y - X*_init_beta).dot(y-X*_init_beta) *( 1.0 / (2*pow(sigma,2)));
    energy = -energy;
    energy_init = - energy_init;
    out = exp(energy_init - energy  + kinetic_init - kinetic);
    std::uniform_real_distribution<double> die(0,1);
    u = die(_rng);
    if(u<out){
        _accepted = 1;
        return (beta);
    }
    else{
        _accepted = 0;
        return(_init_beta);
    }
}


// Getter Methods
// -----------------------------------------------------------------
//

template <typename T >
const T& staticHMC<T>::momentum() const {
    return(_mt);
}

template < typename T >
const T& staticHMC<T>::grad_energy() const{
    return(_grad_energy);
}

template< typename T >
const int staticHMC<T>::accepted() const{
    return(_accepted);
}

