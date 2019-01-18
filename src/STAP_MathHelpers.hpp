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

double GaussianNoise_scalar(std::mt19937 &rng){
    static std::normal_distribution<double> _z(0,1);
    return(_z(rng));
}

Eigen::VectorXd GaussianNoise(const int q, std::mt19937 &rng){
    Eigen::VectorXd out(q);
    for(int i=0;i<q;i++){
        out(i) = GaussianNoise_scalar(rng);
    }
    return(out);
}

