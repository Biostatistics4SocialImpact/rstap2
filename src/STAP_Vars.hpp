
#include "SV_helpers.hpp"
#include<random>

class SV
{
    public:
        double alpha;
        Eigen::VectorXd delta;
        Eigen::VectorXd beta;
        Eigen::VectorXd beta_bar;
        Eigen::VectorXd theta;
        double sigma;
        Eigen::VectorXd dm;
        Eigen::VectorXd bm;
        Eigen::VectorXd bbm;
        Eigen::VectorXd tm;
        Eigen::ArrayXi spc;
        double am;
        double sm;
        const bool diagnostics;

        SV(Eigen::ArrayXi& stap_par_code_input,std::mt19937& rng,const bool input_diagnostics){
            diagnostics = input_diagnostics
            spc = stap_par_code_input;
            sigma = initialize_scalar(rng);
            alpha = spc(0) == 0 ? 0 : initialize_scalar(rng);
            delta = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(1),rng);
            beta = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(2),rng); 
            beta_bar = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(3),rng);
            theta = initialize_vec(stap_par_code_input(4),rng);
            if(diagnostics){
                Rcpp::Rcout << "Initializing Parameters... " << std::endl;
                Rcpp::Rcout << " initial alpha: " << alpha << std::endl;
                Rcpp::Rcout << " initial delta: " << delta << std::endl;
                Rcpp::Rcout << " initial beta: " << beta << std::endl;
                Rcpp::Rcout << " initial beta_bar: " << beta_bar << std::endl;
                Rcpp::Rcout << " initial theta: " << theta << std::endl;
                Rcpp::Rcout << " initial sigma: " << sigma << std::endl;
            }
        }


        void update_momenta(std::mt19937& rng){
            sm = GaussianNoise_scalar(rng);
            am = spc(0) == 0 ? 0 :  GaussianNoise_scalar(rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1): GaussianNoise(beta_bar.size(),rng);
            tm = GaussianNoise(theta.size(),rng);
        }

        void copy_momenta(SV other){

            dm = other.dm;
            bm = other.bm;
            bbm = other.bbm;
            tm = other.tm;
            sm = other.sm;
        }

        void position_update(SV& updated_sv){
            delta = updated_sv.delta;
            beta = updated_sv.beta;
            beta_bar = updated_sv.beta_bar;
            theta = updated_sv.theta;
            sigma = updated_sv.sigma;
        }

        void operator=(SV other){
            delta = other.delta;
            beta = other.beta;
            theta = other.theta;
            sigma = other.sigma;
            dm = other.dm;
            bm = other.bm;
            tm = other.tm;
            sm = other.sm;
        }

        double precision_transformed(){
            return(pow(exp(sigma),-2));
        }

        double sigma_transformed(){
            return(exp(sigma));
        }

        double sigma_sq_transformed(){
            return(pow(exp(sigma),2));
        }
        
        Eigen::VectorXd theta_transformed(){
            return( 0.1 *  (Eigen::VectorXd::Ones(theta.size()) + (-theta).exp()));
        }

        double kinetic_energy(){
            double out = 0;
            out = dm.dot(dm) + bm.dot(bm) + bbm.dot(bbm) + tm.dot(tm)  + sm * sm + am * am;
            out = out / 2.0;
            return(out);
        }

}; 

bool get_UTI_one(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svl.dm) + (svr.beta - svl.beta).dot(svl.bm) + (svr.beta_bar - svl.beta_bar).dot(svl.bbm) + (svr.theta - svl.theta).dot(svl.tm) + (svr.sigma - svl.sigma) * (svl.sm);
    out = out / 2.0;

    return((out >=0));
}

bool get_UTI_two(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svr.dm) + (svr.beta - svl.beta).dot(svr.bm) + (svr.beta_bar - svl.beta_bar).dot(svl.bbm) +  (svr.theta - svl.theta).dot(svr.tm) + (svr.sigma - svl.sigma) * (svr.sm);
    out = out / 2.0;

    return((out >=0));
}
