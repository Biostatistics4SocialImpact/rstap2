#include <cmath>
#include <cstdlib>
#include <Eigen/Core>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <random>
#include <vector>
#ifndef M_PI
#define M_PI REAL(3.1415926535897932384626433832795029)
#endif

double GaussianNoise_scalar(std::mt19937& rng){
    static std::normal_distribution<double> _z(0,1);
    return(_z(rng));
}

Eigen::VectorXd GaussianNoise(const int& q, std::mt19937& rng){
    Eigen::VectorXd out(q);
    for(int i=0;i<q;i++){
        out(i) = GaussianNoise_scalar(rng);
    }
    return(out);
}

double rbeta(const double& shape_alpha, const double& shape_beta, std::mt19937 &rng){

    std::gamma_distribution<double> rgamma_one(shape_alpha,1.0);
    std::gamma_distribution<double> rgamma_two(shape_beta,1.0);

    double out;
    out = rgamma_one(rng);
    out = out / (out + rgamma_two(rng));
    return(out);
}

//' generates random correlation matrices
//' @param d dimension of matrix
//' @param eta concentration parameter
//' @param rng random number generator
Eigen::MatrixXd rlkj(const int& d, const double& eta,std::mt19937& rng){     

    Eigen::MatrixXd out(d,d);
    double beta;
    beta = eta + (d - 2.0) / 2.0;
    out(0,1) = 2 * rbeta(beta,beta,rng) - 1;
    out(1,0) = out(0,1);
    out(0,0) = 1;
    out(1,1) = 1;
    if(d == 2)
        return(out);
    else{
        Rcpp::Rcout << " not implemented yet " << std::endl;
        return(out);
    }

}
