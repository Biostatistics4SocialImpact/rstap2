#include<random>
#include<RcppEigen.h>
#include<Eigen/Core>
#include "SV_helpers.hpp"

class SG
{
    public:
        double alpha_grad;
        double sigma_grad;
        Eigen::VectorXd delta_grad;
        Eigen::VectorXd beta_grad;
        Eigen::VectorXd beta_bar_grad;
        Eigen::VectorXd theta_grad;
};

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
        Eigen::ArrayXd  var_dm;
        Eigen::VectorXd bm;
        Eigen::ArrayXd  var_bm;
        Eigen::VectorXd bbm;
        Eigen::ArrayXd  var_bbm;
        Eigen::VectorXd tm;
        Eigen::ArrayXd  var_tm;
        Eigen::ArrayXi spc;
        double var_am;
        double var_sm;
        double am;
        double sm;
        bool diagnostics;

        SV(Eigen::ArrayXi& stap_par_code_input,std::mt19937& rng,const bool input_diagnostics){
            diagnostics = input_diagnostics;
            spc = stap_par_code_input;
            sigma = initialize_scalar(rng);
            alpha = 0 ; // spc(0) == 0 ? 0 : initialize_scalar(rng);
            alpha_vec = Eigen::VectorXd::Ones(spc(0)) * alpha;
            delta = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(1),rng);
            beta = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(2),rng); 
            beta_bar = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(3),rng);
            theta = initialize_vec(stap_par_code_input(4),rng);
            if(diagnostics){
                Rcpp::Rcout << " Initialized Parameters" << std::endl;
                print_pars();
            }
        }

        void initialize_var(){

            var_dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : Eigen::VectorXd::Ones(dm.size());
            var_bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : Eigen::VectorXd::Ones(dm.size());
            var_bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : Eigen::VectorXd::Ones(bbm.size());
            var_tm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : Eigen::VectorXd::Ones(bbm.size());
            var_sm = 1.0;
            var_am = 1.0;

        }

        void initialize_momenta(std::mt19937& rng){
            initialize_var();
            sm = GaussianNoise_scalar(rng);
            am = 0.0;//spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1): GaussianNoise(beta_bar.size(),rng);
            tm = GaussianNoise(theta.size(),rng);
        }

        void print_pars(){

            Rcpp::Rcout << "Printing Parameters... " << std::endl;
            Rcpp::Rcout << "------------------------ " << std::endl;
            Rcpp::Rcout << "alpha: " << alpha << std::endl;
            Rcpp::Rcout << "delta: " << delta << std::endl;
            Rcpp::Rcout << "beta: " << beta << std::endl;
            Rcpp::Rcout << "beta_bar: " << beta_bar << std::endl;
            Rcpp::Rcout << "theta: " << theta << std::endl;
            Rcpp::Rcout << "theta_transformed: " << 10 / (1 + exp(-theta(0))) << std::endl;
            Rcpp::Rcout << "sigma: " << sigma << std::endl;
            Rcpp::Rcout << "sigma_transformed: " << exp(sigma) << std::endl;
            Rcpp::Rcout << "------------------------ " << "\n" << std::endl;

        }

        void print_mom(){

            Rcpp::Rcout << "Printing momenta... " << std::endl;
            Rcpp::Rcout << "------------------------ " << std::endl;
            Rcpp::Rcout << "am: " << am << std::endl;
            Rcpp::Rcout << "dm: " << dm << std::endl;
            Rcpp::Rcout << "bm: " << bm << std::endl;
            Rcpp::Rcout << "bbm: " << bbm << std::endl;
            Rcpp::Rcout << "tm: " << tm << std::endl;
            Rcpp::Rcout << "sm: " << sm << std::endl;
            Rcpp::Rcout << "------------------------ " << "\n" << std::endl;

        }

        void momenta_leapfrog_other(SV& sv,double& epsilon, SG& sg){
            am = sv.am + epsilon * sg.alpha_grad / 2.0 ;
            dm = sv.dm + epsilon * sg.delta_grad / 2.0 ;
            bm = sv.bm + epsilon * sg.beta_grad / 2.0 ;
            bbm = sv.bbm + epsilon * sg.beta_bar_grad / 2.0;
            tm = sv.tm + epsilon * sg.theta_grad / 2.0;
            sm = sv.sm + epsilon * sg.sigma_grad / 2.0;
        }

        void momenta_leapfrog_position(double& epsilon, SG& sg){
            alpha = am + epsilon * sg.alpha_grad;
            delta = dm + epsilon * sg.delta_grad;
            beta = bm + epsilon * sg.beta_grad;
            beta_bar = bbm + epsilon * sg.beta_bar_grad;
            theta = tm + epsilon * sg.theta_grad;
            sigma = sm + epsilon * sg.sigma_grad;
        }

        void momenta_leapfrog_self(double& epsilon, SG& sg){
            am = am + epsilon * sg.alpha_grad / 2.0;
            dm = dm + epsilon * sg.delta_grad / 2.0;
            bm = bm + epsilon * sg.beta_grad / 2.0;
            bbm = bbm + epsilon * sg.beta_bar_grad / 2.0;
            tm = tm + epsilon * sg.theta_grad / 2.0;
            sm = sm + epsilon * sg.sigma_grad / 2.0;
        }

        void copy_SV(SV other){
            am = other.am;
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
            return( 10 / (1 + exp(-theta.array())) );
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
        Eigen::ArrayXXd dists;
        Eigen::ArrayXXi u_crs;
        Eigen::MatrixXd subj_array;
        Eigen::ArrayXd subj_n;
        Eigen::VectorXd y;
        bool diagnostics;

    public:
        SG sg;
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

        void print_grads(){

            Rcpp::Rcout << "Printing Gradients: " << std::endl;
            Rcpp::Rcout << "-------------------- " << std::endl;
            Rcpp::Rcout << "alpha_grad: "  << sg.alpha_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << sg.delta_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << sg.beta_grad << std::endl;
            Rcpp::Rcout << "beta_bar_grad: "  << sg.beta_bar_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << sg.theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sg.sigma_grad << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;

        }

        Eigen::MatrixXd get_X_prime_diff() const{
            return(X_prime_diff);
        }

        double get_alpha_grad() const{
            return(sg.alpha_grad);
        }

        Eigen::VectorXd get_beta_grad() const{
            return(sg.beta_grad);
        }

        Eigen::VectorXd get_beta_bar_grad() const{
            return(sg.beta_bar_grad);
        }

        Eigen::VectorXd get_theta_grad() const{
            return(sg.theta_grad);
        }

        double get_sigma_grad() const{
            return(sg.sigma_grad);
        }

};


#include "STAP.inl"
