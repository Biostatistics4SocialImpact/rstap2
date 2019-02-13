#include<random>
#include "SV_helpers.hpp"


class SV
{
    public:
        double alpha;
        Eigen::VectorXd delta;
        Eigen::VectorXd beta;
        Eigen::VectorXd beta_bar;
        Eigen::VectorXd theta;
        Eigen::VectorXd theta_transformed;
        double sigma;
        double sigma_transformed;
        Eigen::VectorXd dm;
        Eigen::VectorXd bm;
        Eigen::VectorXd bbm;
        Eigen::VectorXd tm;
        double am;
        double sm;

        SV(Eigen::ArrayXi& stap_par_code_input,std::mt19937& rng){
            sigma = initialize_scalar(rng);
            alpha = stap_par_code_input(0) == 0 ? 0 : initialize_scalar(rng);
            delta = stap_par_code_input(1) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(1),rng);
            beta = stap_par_code_input(2) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(2),rng); 
            beta_bar = stap_par_code_input(3) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(3),rng);
            theta = initialize_vec(stap_par_code_input(4),rng);
        }


        void update_momenta(std::mt19937& rng){
            dm = GaussianNoise(delta.size(),rng); 
            bm = GaussianNoise(beta.size(),rng);
            bbm = GaussianNoise(beta_bar.size(),rng);
            tm = GaussianNoise(theta.size(),rng);
            sm = GaussianNoise_scalar(rng);
        }

        void copy_momenta(SV other){

            dm = other.dm;
            bm = other.bm;
            bbm = other.bbm;
            tm = other.tm;
            sm = other.sm;
        }

        void position_update(SV updated_sv){
            delta = updated_sv.delta;
            beta = updated_sv.beta;
            beta_bar = updated_sv.beta_bar;
            theta = updated_sv.theta;
            sigma = updated_sv.sigma;
            theta_transformed = 10 / (1 + exp(-theta.array() ) );
            sigma_transformed = exp(sigma);
        }

        SV& operator=(SV other){
            std::swap(delta,other.delta);
            std::swap(beta,other.beta);
            std::swap(theta,other.theta);
            std::swap(sigma,other.sigma);
            std::swap(dm,other.dm);
            std::swap(bm,other.bm);
            std::swap(tm,other.tm);
            std::swap(sm,other.sm);
            return *this;
        }

}; 
bool get_UTI_one(SV svl, SV svr){

    double out;
    out = (svr.delta - svl.delta).dot(svl.dm) + (svr.beta - svl.beta).dot(svl.bm) + (svr.beta_bar - svl.beta_bar).dot(svl.bbm) + (svr.theta - svl.theta).dot(svl.tm) + (svr.sigma - svl.sigma) * (svl.sm);
    out = out / 2.0;

    return((out >=0));
}

bool get_UTI_two(SV svl, SV svr){

    double out;
    out = (svr.delta - svl.delta).dot(svr.dm) + (svr.beta - svl.beta).dot(svr.bm) + (svr.beta_bar - svl.beta_bar).dot(svl.bbm) +  (svr.theta - svl.theta).dot(svr.tm) + (svr.sigma - svl.sigma) * (svr.sm);
    out = out / 2.0;

    return((out >=0));
}
