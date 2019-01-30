#include<random>

class STAP_Vars
{
    public:
        Eigen::VectorXd delta;
        Eigen::VectorXd beta;
        Eigen::VectorXd theta;
        Eigen::VectorXd theta_transformed;
        double sigma;
        double sigma_transformed;
        Eigen::VectorXd dm;
        Eigen::VectorXd bm;
        Eigen::VectorXd tm;
        double sm;

        STAP_Vars(Eigen::VectorXd delta_init,
                  Eigen::VectorXd beta_init,
                  Eigen::VectorXd theta_init,
                  double sigma_init){

            delta = delta_init;
            beta = beta_init;
            theta = theta_init;
            sigma = sigma_init;
        }

        STAP_Vars(int p, int q, int thq){
            delta = Eigen::VectorXd::Zero(p);
            beta = Eigen::VectorXd::Zero(q);
            theta = Eigen::VectorXd::Zero(thq);
        }

        void update_momenta(std::mt19937 &rng){
            dm = GaussianNoise(delta.size(),rng); 
            bm = GaussianNoise(beta.size(),rng);
            tm = GaussianNoise(theta.size(),rng);
            sm = GaussianNoise_scalar(rng);
        }

        void copy_momenta(STAP_Vars sv){

            dm = sv.dm;
            bm = sv.bm;
            tm = sv.tm;
            sm = sv.sm;
        }

        void position_update(STAP_Vars updated_sv){
            delta = updated_sv.delta;
            beta = updated_sv.beta;
            theta = updated_sv.theta;
            sigma = updated_sv.sigma;
        }

        Eigen::VectorXd get_delta(){
            return(delta);
        }

        Eigen::VectorXd get_beta(){
            return(beta);
        }

        Eigen::VectorXd get_transformed_theta(){
            return(theta.exp());
        }

        double get_transformed_sigma(){
            return(exp(sigma));
        }

        STAP_Vars& operator=(STAP_Vars other){
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

bool get_UTI_one(STAP_Vars svl, STAP_Vars svr){

    double out;
    out = (svr.delta - svl.delta).dot(svl.dm) + (svr.beta - svl.beta).dot(svl.bm) + (svr.theta - svl.theta).dot(svl.tm) + (svr.sigma - svl.sigma) * (svl.sm);
    out = out / 2.0;

    return((out >=0));
}

bool get_UTI_two(STAP_Vars svl, STAP_Vars svr){

    double out;
    out = (svr.delta - svl.delta).dot(svr.dm) + (svr.beta - svl.beta).dot(svr.bm) + (svr.theta - svl.theta).dot(svr.tm) + (svr.sigma - svl.sigma) * (svr.sm);
    out = out / 2.0;

    return((out >=0));
}
