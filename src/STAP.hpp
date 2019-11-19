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
        Eigen::VectorXd theta_t_grad;
        void print_grads(){

            Rcpp::Rcout << "Printing Gradients: " << std::endl;
            Rcpp::Rcout << "-------------------- " << std::endl;
            Rcpp::Rcout << "alpha_grad: "  << alpha_grad << std::endl;
            Rcpp::Rcout << "delta_grad: "  << delta_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "beta_bar_grad: "  << beta_bar_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "theta_t_grad: " << theta_t_grad << std::endl;
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
        T var;
        void initialize_first_var(int vec_len){
            count = 0.0;
            mean = Eigen::VectorXd::Zero(vec_len); 
            M2 = Eigen::VectorXd::Zero(vec_len);
            var = Eigen::VectorXd::Ones(vec_len);
        }
        void initialize_first_var(){
            count = 0.0;
            mean = 0.0;
            M2 = 0.0;
            var = 1.0;
        }
        // from welford's algorithm
        // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        void update_var(T new_val){
            count += 1.0;
            T delta = new_val - mean;
            mean = mean + delta / count;
            T delta_two = new_val - mean;
            M2 = M2 + delta * delta_two;
        }

        void implement_var(){
            var = M2 / (count - 1);
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
        Eigen::VectorXd theta_t;
        double sigma;
        Eigen::VectorXd dm;
        Eigen::VectorXd bm;
        Eigen::VectorXd bbm;
        Eigen::VectorXd tm;
        Eigen::VectorXd ttm;
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
            delta = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(spc(1),rng);
            beta = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(spc(2),rng); 
            beta_bar = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(spc(3),rng);
            theta = initialize_vec(stap_par_code_input(2),rng);
            theta_t = spc(5) == 0 ? Eigen::VectorXd::Zero(1) : initialize_vec(spc(5),rng);
            if(diagnostics){
//                Rcpp::Rcout << " Initialized Parameters" << std::endl;
//                print_pars();
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
            ttm = spc(5) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(theta_t.size(),rng);
        }

        void print_pars(){

            Rcpp::Rcout << "Printing Parameters... " << std::endl;
            Rcpp::Rcout << "-------------------------------------------------------------------------- " << std::endl;
            Rcpp::Rcout << "alpha: " << alpha <<  " | " << "delta: " << delta << " | beta: " << beta << 
              " | beta_bar: " << beta_bar << std::endl;
            Rcpp::Rcout << "theta: " << theta << " |  theta_ " << (10.0/ (1 + exp(-theta(0)))) <<
						 "theta_t: " << theta_t << " |  theta_t_ " << exp(theta_t(0)) <<
                " | sigma: " << sigma << " |  sigma_ " << exp(sigma) << std::endl;
            Rcpp::Rcout << "-------------------------------------------------------------------------- " << std::endl;

        }

        void print_mom(){

            Rcpp::Rcout << "Printing momenta... " << std::endl;
            Rcpp::Rcout << "------------------------ " << std::endl;
            Rcpp::Rcout << "am: " << am << std::endl;
            Rcpp::Rcout << "dm: " << dm << std::endl;
            Rcpp::Rcout << "bm: " << bm << std::endl;
            Rcpp::Rcout << "bbm: " << bbm << std::endl;
            Rcpp::Rcout << "tm: " << tm << std::endl;
            Rcpp::Rcout << "ttm: " << ttm << std::endl;
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
            ttm = sv.ttm + epsilon * sg.theta_t_grad / 2.0;
            sm = sv.sm + epsilon * sg.sigma_grad / 2.0;
            if(diagnostics){
                Rcpp::Rcout << "Printing half momenta" << std::endl;
                this->print_mom();
            }
        }

        void momenta_leapfrog_position(SV& sv, double& epsilon){
            if(diagnostics){
                Rcpp::Rcout << "initial positions " << std::endl;
                sv.print_pars();
            }
            alpha = sv.alpha + epsilon * sv.va.var  * am;
            delta = sv.delta.array() + epsilon * sv.vd.var * dm.array();
            beta = sv.beta.array() + epsilon * sv.vb.var * bm.array();
            beta_bar = sv.beta_bar.array() + epsilon * sv.vbb.var * bbm.array(); 
            theta = sv.theta.array() + epsilon * sv.vt.var * tm.array();
            theta_t = sv.theta_t.array() + epsilon * ttm.array();
            sigma = sv.sigma + epsilon * sv.vs.var * sm;
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
            ttm = ttm + epsilon * sg.theta_t_grad / 2.0;
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
			ttm = other.ttm;
            sm = other.sm;
            alpha = other.alpha;
            delta = other.delta;
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            theta_t = other.theta_t;
            sigma = other.sigma;
        }

        void initialize_vars(){
            va.initialize_first_var();
            vb.initialize_first_var(beta.size());
            vbb.initialize_first_var(beta_bar.size());
            vt.initialize_first_var(theta.size());
            vs.initialize_first_var();
        }

        void update_vars(){
            va.update_var(alpha);
            vd.update_var(delta);
            vb.update_var(beta);
            vbb.update_var(beta_bar);
            vt.update_var(theta);
            vs.update_var(sigma);
        
        }

        void implement_vars(){
            va.implement_var();
            vd.implement_var();
            vb.implement_var();
            vbb.implement_var();
            vt.implement_var();
            vs.implement_var();
        }

        void update_var_scheme(int &iter_ix){

            if(iter_ix == 1)
              initialize_vars();
            /*
            if(iter_ix >75 && iter_ix <=125){
              update_vars();
              if(iter_ix == 125){
                implement_vars();
                print_vars();
              }
            }

            if(iter_ix >=175 && iter_ix <=250 ){
              if(iter_ix == 175 )
                initialize_vars();
              else
                update_vars();
              if(iter_ix == 250){
                implement_vars();
                print_vars();
              }
            }

            if(iter_ix == 325)
              initialize_vars();
            if(iter_ix>325 && iter_ix <500)
              update_vars();
            if(iter_ix==500){
              implement_vars();
              print_vars();
            }
            */
        }

        void print_vars(){

            Rcpp::Rcout << "Print variance count parameters" << std::endl;
            Rcpp::Rcout << "=============================================" << std::endl;
            Rcpp::Rcout << "alpha count " << va.count << " | beta count " << vb.count << " | beta_bar count " <<
              vbb.count << " | theta count " << vt.count << " | sigma count " << vs.count << std::endl;
            Rcpp::Rcout << "Print variance estimates parameters" << std::endl;
            Rcpp::Rcout << "alpha var " << va.var << " | beta var " << vb.var << " | beta_bar var " <<
              vbb.var << " | theta var " << vt.var << " | sigma var " << vs.var << std::endl;
            Rcpp::Rcout << "=============================================" << std::endl;

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
        
        Eigen::VectorXd adjust_delta(double &y_sd){
            return(delta * y_sd);
        }

        Eigen::VectorXd adjust_beta(double &y_sd){
            return(beta * y_sd);
        }

        double adjust_sigma(double &y_sd){
            return(exp(sigma) * y_sd);
        }

        Eigen::VectorXd adjust_beta_bar(double &y_sd){
            return(beta_bar * y_sd);
        }

        Eigen::VectorXd theta_transformed(){
            return( 10 / (1 + exp(-theta.array())) );
        }

        Eigen::VectorXd theta_t_transformed(){
            return(exp(theta_t.array()));
        }

        double kinetic_energy(){
            double out = 0;
            out = dm.transpose() * (vd.var).matrix().asDiagonal() * dm;
            out += bm.transpose() * (vb.var).matrix().asDiagonal() * bm;
            out += bbm.transpose() * (vbb.var).matrix().asDiagonal() * bbm;
            out += tm.transpose() * (vt.var).matrix().asDiagonal() * tm;
            out += ttm.dot(ttm);
            out += (sm * sm) * vs.var   + (am * am) * va.var;
            out = out / 2.0;
            return(out);
        }

}; 

bool get_UTI_one(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svl.dm);
    out += (svr.beta - svl.beta).dot(svl.bm);
    out += (svr.beta_bar - svl.beta_bar).dot(svl.bbm);
    out += (svr.theta - svl.theta).dot(svl.tm);
    out += (svr.theta_t - svl.theta_t).dot(svl.ttm);
    out += (svr.sigma - svl.sigma) * (svl.sm);
    out += (svr.alpha - svl.alpha) * svl.am;
    out = out / 2.0;

    return((out >=0));
}

bool get_UTI_two(SV& svl,SV& svr){

    double out;
    out = (svr.delta - svl.delta).dot(svr.dm);
    out += (svr.beta - svl.beta).dot(svr.bm);
    out += (svr.beta_bar - svl.beta_bar).dot(svr.bbm);
    out +=  (svr.theta - svl.theta).dot(svr.tm);
    out += (svr.theta_t - svl.theta_t).dot(svr.ttm);
    out += (svr.sigma - svl.sigma) * (svr.sm);
    out += (svr.alpha - svl.alpha) * svr.am;
    out = out / 2.0;

    return((out >=0));
}


class STAP
{
    public:
        Eigen::VectorXd eta;
        Eigen::MatrixXd X;
        Eigen::MatrixXd X_prime;
		Eigen::MatrixXd X_tprime;
        Eigen::ArrayXXd dists;
        Eigen::ArrayXXd times;
        Eigen::ArrayXXi u_crs;
        Eigen::ArrayXXi u_tcrs;
        bool diagnostics;
        Eigen::SparseMatrix<double> subj_array ;
        Eigen::MatrixXd subj_n;
        Eigen::MatrixXd Z;
        Eigen::VectorXd y;
        Eigen::MatrixXd X_mean; 
        Eigen::VectorXd X_global_mean;
        Eigen::MatrixXd X_diff;
        Eigen::MatrixXd X_mean_prime;
        Eigen::VectorXd X_mean_prime_global_mean;
		Eigen::MatrixXd X_mean_tprime;
		Eigen::VectorXd X_mean_tprime_global_mean;
        Eigen::MatrixXd X_prime_diff;
		Eigen::MatrixXd X_tprime_diff;
        SG sg;
        STAP(Eigen::ArrayXXd& input_dists,
             Eigen::ArrayXXi& input_ucrs,
	     Eigen::ArrayXXd& input_times,
	     Eigen::ArrayXXi& input_utcrs,
             Eigen::MappedSparseMatrix<double> &input_subj_array,
             Eigen::MatrixXd &input_subj_n,
             Eigen::MatrixXd& input_Z,
             Eigen::VectorXd& input_y,
             const bool& input_diagnostics);

        double calculate_ll(SV& sv);

        double calculate_total_energy(SV& sv);
            
        double sample_u(SV& sv, std::mt19937& rng);

        void calculate_X(double& theta);

        void calculate_X(double& theta_s, double &theta_t);

		/*
        void calculate_X_diff(double& theta);
		*/

        void calculate_X_diff(double& theta, double &theta_t);

        void calculate_X_mean();

		/*
        void calculate_X_prime(double &theta_tilde, double& cur_theta);
		*/
	
		void calculate_X_prime(double &theta_tilde, double &cur_theta, double &theta_t);

        void calculate_X_mean_prime();

		/*
        void calculate_X_prime_diff(double& theta,double& cur_theta);
		*/

        void calculate_X_prime_diff(double& theta,double& cur_theta, double &theta_time);

        void calculate_eta(SV& sv);

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

double adjust_alpha(STAP &stap_object, SV &sv, double &y_bar, double &y_sd){
  
  stap_object.calculate_X_diff(sv.theta(0),sv.theta_t(0));
  
  return(sv.alpha - stap_object.X_global_mean.dot(sv.beta_bar) * y_sd + y_bar);
}



