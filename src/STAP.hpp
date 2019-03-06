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
        void print_grads(){

            Rcpp::Rcout << "Printing Gradients: " << std::endl;
            Rcpp::Rcout << "-------------------- " << std::endl;
            Rcpp::Rcout << "alpha_grad: "  << alpha_grad << std::endl;
            Rcpp::Rcout << "delta_grad: "  << delta_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "beta_bar_grad: "  << beta_bar_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;

        }
};


template<typename T = double>
class Var_Agg
{
    public:
        double count;
        T mean;
        T M2;
        T sd;
        void initialize_first_var(int vec_len){
            count = 0.0;
            mean = Eigen::VectorXd::Zero(vec_len); 
            M2 = Eigen::VectorXd::Zero(vec_len);
            sd = Eigen::VectorXd::Ones(vec_len);
        }
        void initialize_first_var(){
            count = 0.0;
            mean = 0.0;
            M2 = 0.0;
            sd = 1.0;
        }

        void update_var(T new_val){
            count += 1.0;
            T delta = new_val - mean;
            mean = mean + delta / count;
            T delta_two = new_val - mean;
            M2 = M2 + delta * delta_two;
            sd = M2 / (count - 1);
        }

};

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
        Var_Agg<Eigen::ArrayXd> vd;
        Var_Agg<Eigen::ArrayXd> vb;
        Var_Agg<Eigen::ArrayXd> vbb;
        Var_Agg<Eigen::ArrayXd> vt;
        Var_Agg<double> va;
        Var_Agg<double> vs;
        double am;
        double sm;
        bool diagnostics;

        SV(Eigen::ArrayXi& stap_par_code_input,
            std::mt19937& rng,const bool input_diagnostics){
            diagnostics = input_diagnostics;
            spc = stap_par_code_input;
            sigma = initialize_scalar(rng);
            alpha = spc(0) == 0 ? 0 : initialize_scalar(rng);
            delta = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(1),rng);
            beta = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(2),rng); 
            beta_bar = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(stap_par_code_input(3),rng);
            theta = initialize_vec(stap_par_code_input(4),rng);
            if(diagnostics){
                Rcpp::Rcout << " Initialized Parameters" << std::endl;
                print_pars();
            }
            va.initialize_first_var();
            vs.initialize_first_var();
            vd.initialize_first_var(delta.size());
            vb.initialize_first_var(beta.size());
            vbb.initialize_first_var(beta_bar.size());
            vt.initialize_first_var(theta.size());
        }

        void initialize_momenta(std::mt19937& rng){

            sm = GaussianNoise_scalar(rng);
            am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
            tm = GaussianNoise(theta.size(),rng);
        }

        void initialize_momenta(int& iter_ix,std::mt19937& rng){

            if(iter_ix == 1){
                sm = GaussianNoise_scalar(rng);
                am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
                dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
                bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
                bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
                tm = GaussianNoise(theta.size(),rng);
            }else{
                sm = GaussianNoise_scalar(rng);
                am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
                dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng);
                bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
                bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
                tm = GaussianNoise(theta.size(),rng);
                am = am * va.sd;
                dm = dm.array() * vd.sd;
                bm = bm.array() * vb.sd;
                bbm = bbm.array() * vbb.sd;
                tm = tm.array() * vt.sd;
            }
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
            if(diagnostics){
                Rcpp::Rcout << "Printing initial momenta " << std::endl;
                sv.print_mom();
                Rcpp::Rcout << "Printing Gradients" << std::endl;
                sg.print_grads();
            }
            am = sv.am + epsilon * sg.alpha_grad / 2.0 ;
            dm = sv.dm + epsilon * sg.delta_grad / 2.0 ;
            bm = sv.bm + epsilon * sg.beta_grad / 2.0 ;
            bbm = sv.bbm + epsilon * sg.beta_bar_grad / 2.0;
            tm = sv.tm + epsilon * sg.theta_grad / 2.0;
            sm = sv.sm + epsilon * sg.sigma_grad / 2.0;
            if(diagnostics){
                Rcpp::Rcout << "Printing half momenta" << std::endl;
                this->print_mom();
            }
        }

        void momenta_leapfrog_position(SV& sv, double& epsilon){
            if(diagnostics){
                Rcpp::Rcout << "initial positions " << std::endl;
                this->print_pars();
            }
            alpha = sv.alpha + epsilon * am;
            delta = sv.delta + epsilon * dm;
            beta = sv.beta + epsilon * bm;
            beta_bar = sv.beta_bar + epsilon * bbm; 
            theta = sv.theta + epsilon * tm;
            sigma = sv.sigma + epsilon * sm;
            if(diagnostics){
                Rcpp::Rcout << "updated positions: " << std::endl;
                this->print_pars();
            }
        }

        void momenta_leapfrog_self(double& epsilon, SG& sg){
            if(diagnostics){
                Rcpp::Rcout << "final gradients" << std::endl;
                sg.print_grads();
            }
            am = am + epsilon * sg.alpha_grad / 2.0;
            dm = dm + epsilon * sg.delta_grad / 2.0;
            bm = bm + epsilon * sg.beta_grad / 2.0;
            bbm = bbm + epsilon * sg.beta_bar_grad / 2.0;
            tm = tm + epsilon * sg.theta_grad / 2.0;
            sm = sm + epsilon * sg.sigma_grad / 2.0;
            if(diagnostics){
                Rcpp::Rcout << "final momenta" << std::endl;
                this->print_mom();
            }
        }

        void copy_SV(SV other){
            am = other.am;
            dm = other.dm;
            bm = other.bm;
            bbm = other.bbm;
            tm = other.tm;
            sm = other.sm;
            alpha = other.alpha;
            delta = other.delta;
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            sigma = other.sigma;
        }

        void update_var(int iter_ix,SV& sv){

            if(iter_ix <=75){
                if(iter_ix == 1){
                    va.initialize_first_var();
                    vb.initialize_first_var(beta.size());
                    vbb.initialize_first_var(beta_bar.size());
                    vt.initialize_first_var(theta.size());
                    vs.initialize_first_var();
                }else{
                    va.update_var(sv.alpha);
                    vd.update_var(sv.delta);
                    vb.update_var(sv.beta);
                    vbb.update_var(sv.beta_bar);
                    vt.update_var(sv.theta);
                    vs.update_var(sv.sigma);
                }
            }else if( iter_ix < 200){
              if(iter_ix % 25 == 0){
                va.initialize_first_var();
                vb.initialize_first_var(beta.size());
                vbb.initialize_first_var(beta_bar.size());
                vt.initialize_first_var(theta.size());
                vs.initialize_first_var();
            }else{
                va.update_var(sv.alpha);
                vd.update_var(sv.delta);
                vb.update_var(sv.beta);
                vbb.update_var(sv.beta_bar);
                vt.update_var(sv.theta);
                vs.update_var(sv.sigma);
            }
          }else if(iter_ix <= 250){
                if(iter_ix == 200){
                va.initialize_first_var();
                vb.initialize_first_var(beta.size());
                vbb.initialize_first_var(beta_bar.size());
                vt.initialize_first_var(theta.size());
                vs.initialize_first_var();
            }else{
                va.update_var(sv.alpha);
                vd.update_var(sv.delta);
                vb.update_var(sv.beta);
                vbb.update_var(sv.beta_bar);
                vt.update_var(sv.theta);
                vs.update_var(sv.sigma);
            }
        }

        }

        void print_vars(){

            Rcpp::Rcout << "Print variance count parameters" << std::endl;
            Rcpp::Rcout << "alpha count" << va.count << std::endl;
            Rcpp::Rcout << "beta count" << vb.count << std::endl;
            Rcpp::Rcout << "beta bar count" << vbb.count << std::endl;
            Rcpp::Rcout << "theta count" << vt.count << std::endl;
            Rcpp::Rcout << "sigma count" << vs.count << std::endl;
        }
            
        Eigen::VectorXd get_alpha_vector(){
            return(Eigen::VectorXd::Ones(spc(0)) * alpha );
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
            out = dm.transpose() * (1.0 / vd.sd).matrix().asDiagonal() * dm;
            out += bm.transpose() * (1.0 / vb.sd).matrix().asDiagonal() * bm;
            out += bbm.transpose() * (1.0 / vbb.sd).matrix().asDiagonal() * bbm;
            out += tm.transpose() * (1.0 / vt.sd).matrix().asDiagonal() * tm;
            out += (sm * sm) / vs.sd   + (am * am) / va.sd;
            out = out / 2.0;
            return(out);
        }

}; 

bool get_UTI_one(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svl.dm) + (svr.beta - svl.beta).dot(svl.bm) + (svr.beta_bar - svl.beta_bar).dot(svl.bbm) + (svr.theta - svl.theta).dot(svl.tm) + (svr.sigma - svl.sigma) * (svl.sm);
    out += (svr.alpha - svl.alpha) * svl.am;
    out = out / 2.0;

    return((out >=0));
}

bool get_UTI_two(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svr.dm) + (svr.beta - svl.beta).dot(svr.bm);
    out += (svr.beta_bar - svl.beta_bar).dot(svr.bbm) +  (svr.theta - svl.theta).dot(svr.tm);
    out += (svr.sigma - svl.sigma) * (svr.sm);
    out += (svr.alpha - svl.alpha) * svr.am;
    out = out / 2.0;

    return((out >=0));
}


class STAP
{
    public:
        Eigen::MatrixXd X;
        Eigen::MatrixXd X_prime;
        Eigen::ArrayXXd dists;
        Eigen::ArrayXXi u_crs;
        bool diagnostics;
        Eigen::MatrixXd subj_array;
        Eigen::MatrixXd subj_n;
        Eigen::MatrixXd Z;
        Eigen::VectorXd y;
        Eigen::MatrixXd X_mean; 
        Eigen::MatrixXd X_diff;
        Eigen::MatrixXd X_mean_prime;
        Eigen::MatrixXd X_prime_diff;
        SG sg;
        STAP(Eigen::ArrayXXd& input_dists,
             Eigen::ArrayXXi& input_ucrs,
             Eigen::MatrixXd& input_subj_array,
             Eigen::MatrixXd& input_subj_n,
             Eigen::MatrixXd& input_Z,
             Eigen::VectorXd& input_y,
             const bool& input_diagnostics);

        double calculate_ll(SV& sv);

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
            sg.print_grads();
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
