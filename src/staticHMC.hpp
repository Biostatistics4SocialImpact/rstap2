#include<cmath>
#include<random>
#include<vector>
#include <RcppEigen.h>
#include "MathHelpers.h"

template< typename T = double>
class staticHMC 
{
    private :
        T _i_mt;
        T _mt;
        T _init_grad_energy;
        T _grad_energy;
        T _init_beta;
        int _accepted;
        std::uniform_real_distribution<double> _die;
        std::mt19937 _rng;
    public:
        staticHMC(const double &adapt_delta,const int seed);

    template<typename S>
    void calculate_initial_gradient(S init_beta, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma);


    template<typename S>
    void Leapfrog(S &beta, int L, double epsilon, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma);

    template<typename S>
    T MH_step(S beta, const Eigen::MatrixXd &X, const Eigen::VectorXd &y, double sigma);

    //Getter Methods
    const T& momentum() const;
    const T& grad_energy() const;
    const int accepted() const;

};

#include "staticHMC.inl"
