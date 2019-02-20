#include<random>
#include<RcppEigen.h>
#include<Eigen/Core>
#include "SV_helpers.hpp"

class SV
{
    public:
        double alpha;
        Eigen::VectorXd alpha_vec;
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
        bool diagnostics;

        SV(Eigen::ArrayXi& stap_par_code_input,std::mt19937& rng,const bool input_diagnostics){
            diagnostics = input_diagnostics;
            spc = stap_par_code_input;
            sigma = initialize_scalar(rng);
            alpha = spc(0) == 0 ? 0 : initialize_scalar(rng);
            if(spc(0) == 0)
                alpha_vec = Eigen::VectorXd::Zero(1);
            else
                alpha_vec = Eigen::VectorXd::Ones(spc(0)) * alpha;
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

        void copy_momenta(SV& other){

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

        void copy_SV(SV other){
            dm = other.dm;
            bm = other.bm;
            bbm = other.bbm;
            tm = other.tm;
            sm = other.sm;
            delta = other.delta;
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            sigma = other.sigma;
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

class STAP
{
    private:
        Eigen::MatrixXd X;
        Eigen::MatrixXd X_mean; 
        Eigen::MatrixXd X_diff;
        Eigen::MatrixXd X_prime;
        Eigen::MatrixXd X_mean_prime;
        Eigen::MatrixXd X_prime_diff;
        double alpha_grad;
        Eigen::VectorXd beta_grad;
        Eigen::VectorXd beta_bar_grad;
        Eigen::VectorXd theta_grad;
        double sigma_grad;
        Eigen::ArrayXXd dists;
        Eigen::ArrayXXi u_crs;
        Eigen::MatrixXd subj_array;
        Eigen::ArrayXd subj_n;
        Eigen::VectorXd y;
        bool diagnostics;

    public:
        STAP(Eigen::ArrayXXd& input_dists,
             Eigen::ArrayXXi& input_ucrs,
             Eigen::MatrixXd& input_subj_array,
             Eigen::ArrayXd& input_subj_n,
             Eigen::VectorXd& input_y,
             const bool& input_diagnostics);

        double calculate_total_energy(SV& sv);
            
        double sample_u(SV& sv, std::mt19937& rng);

        void calculate_X(double& theta);

        void calculate_X_diff(double& theta);

        void calculate_X_mean();

        void calculate_X_prime(double& theta, double& cur_theta);

        void calculate_X_mean_prime();

        void calculate_X_prime_diff(double& theta,double& cur_theta);

        void calculate_gradient(SV& sv);

        double FindReasonableEpsilon(SV& sv, std::mt19937& rng);

        double get_alpha_grad() const{
            return(alpha_grad);
        }

        Eigen::VectorXd get_beta_grad() const{
            return(beta_grad);
        }

        Eigen::VectorXd get_beta_bar_grad() const{
            return(beta_bar_grad);
        }

        Eigen::VectorXd get_theta_grad() const{
            return(theta_grad);
        }

        double get_sigma_grad() const{
            return(sigma_grad);
        }

};


#include "STAP.inl"
