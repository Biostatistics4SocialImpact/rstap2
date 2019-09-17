#include <random>
Eigen::MatrixXd initialize_matrix(const int& num_rows,const int& num_cols, std::mt19937& rng){

    std::uniform_real_distribution<double> runif(-2,2);
    Eigen::MatrixXd out(num_rows,num_cols); for(int row_ix = 0; row_ix < num_rows; row_ix ++){ 
        for(int col_ix = 0; col_ix < num_cols; col_ix ++)
            out(row_ix,col_ix) = runif(rng);
    }
    return(out);
}

Eigen::MatrixXd GaussianNoise_mat(const int& num_rows, const int& num_cols, std::mt19937& rng){

    Eigen::MatrixXd out(num_rows,num_cols);
    for(int row_ix = 0; row_ix < num_rows; row_ix ++){
        for(int col_ix = 0; col_ix < num_cols; col_ix ++)
            out(row_ix,col_ix) = GaussianNoise_scalar(rng);
    }
    return(out);
}

class SG_glmer: public SG
{
    public:
        Eigen::VectorXd b_grad;
        Eigen::VectorXd b_slope_grad;
        Eigen::VectorXd subj_sig_grad; 
        void print_grads(){

            Rcpp::Rcout << "Printing Gradients: " << std::endl;
            Rcpp::Rcout << "-------------------- " << std::endl;
            Rcpp::Rcout << "alpha_grad: "  << alpha_grad << std::endl;
            Rcpp::Rcout << "delta_grad: "  << delta_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "beta_bar_grad: "  << beta_bar_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
            Rcpp::Rcout << "subj_b_grad: \n" <<b_grad.head(5) << std::endl;
            Rcpp::Rcout << "subj_b_slope_grad: \n" << b_slope_grad.head(5) << std::endl;
            Rcpp::Rcout << "subj_sigma_grad: " << subj_sig_grad << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;
        }
};

class SV_glmer: public SV
{
    public:
        Eigen::VectorXd b;
        Eigen::VectorXd b_slope;
        Eigen::VectorXd b_m;
        Eigen::VectorXd bs_m;
        Eigen::VectorXd Sigma;
        Eigen::VectorXd S_m;
        SV_glmer(Eigen::ArrayXi& stap_par_code_input,
                std::mt19937& rng, const bool input_diagnostics) :
            SV(stap_par_code_input,rng,input_diagnostics)
    {
        b = GaussianNoise(spc(4),rng); 
        b_slope = spc(5) == 2 ? GaussianNoise(spc(4),rng) : Eigen::VectorXd::Zero(spc(4));
        Sigma =  spc(5) == 2 ? GaussianNoise(3,rng) : GaussianNoise(1,rng);

    }
        void print_pars(){

            Rcpp::Rcout << "Printing Parameters... " << std::endl;
            Rcpp::Rcout << "-------------------------------------------------------------------------- " << std::endl;
            Rcpp::Rcout << "alpha: " << alpha <<  " | " << "delta: " << delta << " | beta: " << beta << 
              " | beta_bar: " << beta_bar << std::endl;
            Rcpp::Rcout << "theta: " << theta << " | theta_ " << 10/ (1 + exp(-theta(0))) <<
                " |  sigma: " << sigma << " |  sigma_ " << exp(sigma) << std::endl;
            Rcpp::Rcout << "b : " << b.transpose().head(5) << "| b_slope : " << b_slope.transpose().head(5) << std::endl;
            Rcpp::Rcout << "rho_b : " << get_rho() << std::endl;
            Rcpp::Rcout << "sigma_b :  \n " << Sigma << std::endl;
            Rcpp::Rcout << "sigma_b transformed : " << mer_var_transformed() << std::endl;
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
            Rcpp::Rcout << "sm: " << sm << std::endl;
            Rcpp::Rcout << "S_m: " << S_m << std::endl;
            Rcpp::Rcout << "b_m: \n" << b_m.head(5) << std::endl;
            Rcpp::Rcout << "bs_m: \n" << bs_m.head(5) << std::endl;
            Rcpp::Rcout << "------------------------ " << "\n" << std::endl;

        }

        void initialize_momenta(std::mt19937& rng){

            sm = GaussianNoise_scalar(rng);
            S_m = spc(5) == 2 ? GaussianNoise(3,rng) : GaussianNoise(1,rng);
            am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            b_m = GaussianNoise(spc(4),rng);
            bs_m = spc(5) == 2 ? GaussianNoise(spc(4),rng) : Eigen::VectorXd::Zero(spc(4));
            tm = GaussianNoise(theta.size(),rng);

        }

        double get_rho(){
           if(spc(5) == 1)
             return(1.0);
           else
            return(sigmoid_transform(Sigma(2),-1,1));
        }

        double get_rho_derivative(){
          if(spc(5)==1)
            return(0);
          else
            return(sigmoid_transform_derivative(Sigma(2),-1,1));
        }

        double get_rho_sq_c(){
          if(Sigma.rows() == 1)
            return(0);
          else
            return(1 - pow(sigmoid_transform(Sigma(2),-1,1),2) );
        }

        Eigen::VectorXd adjust_b(){
            return(b * exp(Sigma(0)) );
        }

        Eigen::VectorXd adjust_b_slope(){
            return( exp(Sigma(1)) * (b * get_rho() + b_slope * sqrt(get_rho_sq_c()) ));
        }

        double mer_sd_1(){
            return(exp(Sigma(0)));
        }

        double mer_sd_2(){
            return(exp(Sigma(1)));
        }

        Eigen::MatrixXd mer_var_transformed(){
            if(Sigma.rows() == 1)
                return(Eigen::MatrixXd::Ones(1,1)*Sigma(0));
            else{
                int d = 2;
                Eigen::MatrixXd out(d,d);
                out(0,0) = pow(exp(Sigma(0)),2);
                out(1,1) = pow(exp(Sigma(1)),2);
                out(1,0) = sigmoid_transform(Sigma(2),-1,1) * exp(Sigma(0)) * exp(Sigma(1));
                out(0,1) = sigmoid_transform(Sigma(2),-1,1) * exp(Sigma(0)) * exp(Sigma(1));
                return(out);
            }
        }

        double  mer_L_11(){
          double out; 
          out = exp(Sigma(1)) * pow(get_rho_sq_c(),.5);
          return(out);
        }

        void momenta_leapfrog_other(SV_glmer& svg, double& epsilon, SG_glmer& sgg){

            if(diagnostics){
                Rcpp::Rcout << "Printing initial momenta " << std::endl;
                svg.print_mom();
                Rcpp::Rcout << "Printing Gradients" << std::endl;
                sgg.print_grads();
            }

            am = svg.am + epsilon * sgg.alpha_grad / 2.0 ;
            dm = svg.dm + epsilon * sgg.delta_grad / 2.0 ;
            bm = svg.bm + epsilon * sgg.beta_grad / 2.0 ;
            bbm = svg.bbm + epsilon * sgg.beta_bar_grad / 2.0;
            b_m = svg.b_m + epsilon * sgg.b_grad / 2.0 ;
            bs_m = svg.bs_m + epsilon * sgg.b_slope_grad / 2.0;
            tm = svg.tm + epsilon * sgg.theta_grad / 2.0;
            sm = svg.sm + epsilon * sgg.sigma_grad / 2.0;
            S_m = svg.S_m + epsilon * sgg.subj_sig_grad / 2.0;
            if(diagnostics){
                Rcpp::Rcout << "Printing half momenta" << std::endl;
                this->print_mom();
            }
        }

        void momenta_leapfrog_position(SV_glmer& svg, double& epsilon){
            if(diagnostics){
                Rcpp::Rcout << "initial positions " << std::endl;
                svg.print_pars();
            }
            alpha = svg.alpha + epsilon * am;
            delta = svg.delta + epsilon * dm;
            beta = svg.beta + epsilon * bm;
            beta_bar = svg.beta_bar + epsilon * bbm; 
            b = svg.b + epsilon * b_m;
            b_slope = svg.b_slope + epsilon * bs_m;
            theta = svg.theta + epsilon * tm;
            sigma = svg.sigma + epsilon * sm;
            Sigma = svg.Sigma + epsilon * S_m;
            if(diagnostics){
                Rcpp::Rcout << "updated positions: " << std::endl;
                this->print_pars();
            }
        }

        void momenta_leapfrog_self(double& epsilon,SG_glmer& sgg){
            if(diagnostics){
                Rcpp::Rcout << "final gradients" << std::endl;
                sgg.print_grads();
            }
            am = am + epsilon * sgg.alpha_grad / 2.0;
            dm = dm + epsilon * sgg.delta_grad / 2.0;
            bm = bm + epsilon * sgg.beta_grad / 2.0;
            bbm = bbm + epsilon * sgg.beta_bar_grad / 2.0;
            b_m = b_m + epsilon * sgg.b_grad / 2.0;
            bs_m = bs_m + epsilon * sgg.b_slope_grad / 2.0;
            tm = tm + epsilon * sgg.theta_grad / 2.0;
            sm = sm + epsilon * sgg.sigma_grad / 2.0;
            S_m = S_m + epsilon * sgg.subj_sig_grad / 2.0 ;
            if(diagnostics){
                Rcpp::Rcout << "final momenta" << std::endl;
                this->print_mom();
            }
        }

        void copy_SV_glmer(SV_glmer other){
            am = other.am;
            bm = other.bm;
            bbm = other.bbm;
            dm = other.dm;
            tm = other.tm;
            b_m = other.b_m;
            bs_m = other.bs_m;
            sm = other.sm;
            S_m = other.S_m;
            alpha = other.alpha;
            delta = other.delta;
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            b = other.b;
            b_slope = other.b_slope;
            sigma = other.sigma;
            Sigma = other.Sigma;
        }

        double kinetic_energy_glmer(){
            double out = 0;
            out += (sm * sm)   + (am * am)  ;
            out += dm.transpose() *  dm;
            out += bm.transpose() *  bm;
            out += bbm.transpose() *  bbm;
            out += tm.transpose() * tm;
            out += b_m.dot(b_m);
            out += bs_m.dot(bs_m);
            out += S_m.dot(S_m);
            out = out / 2.0;
            return(out);
        }

};

bool get_UTI_one(SV_glmer& svgl, SV_glmer& svgr){


    double out;
    out = pow((svgr.alpha - svgl.alpha) * svgl.am,2);
    out += pow((svgr.sigma - svgl.sigma) * (svgl.sm),2);
    out += (svgr.delta - svgl.delta).dot(svgl.dm);
    out += (svgr.beta - svgl.beta).dot(svgl.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgl.bbm);
    out += (svgr.theta - svgl.theta).dot(svgl.tm);
    out += (svgr.b - svgl.b).dot(svgl.b_m);
    out += (svgr.b_slope - svgl.b_slope).dot(svgl.bs_m);
    out += (svgr.Sigma - svgl.Sigma).dot(svgl.S_m);
    out = out / 2.0;

    return((out >= 0));

}

bool get_UTI_two(SV_glmer& svgl,SV_glmer& svgr){

    double out = 0;
    out += pow((svgr.alpha - svgl.alpha) * svgr.am,2);
    out += pow((svgr.sigma - svgl.sigma) * (svgr.sm),2);
    out += (svgr.delta - svgl.delta).dot(svgr.dm);
    out += (svgr.beta - svgl.beta).dot(svgr.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgr.bbm);
    out +=  (svgr.theta - svgl.theta).dot(svgr.tm);
    out += (svgr.b - svgl.b).dot(svgr.b_m);
    out += (svgr.b_slope - svgl.b_slope).dot(svgr.bs_m);
    out += (svgr.Sigma - svgl.Sigma).dot(svgr.S_m);
    out = out / 2.0;

    return((out >= 0));
}

class STAP_glmer: public STAP
{
    private: const Eigen::MappedSparseMatrix<double> W;
    public:
        SG_glmer sgg;
        STAP_glmer(Eigen::ArrayXXd &input_dists,
                   Eigen::ArrayXXi &input_ucrs,
                   Eigen::MappedSparseMatrix<double> &input_subj_array,
                   Eigen::MatrixXd &input_subj_n,
                   Eigen::MatrixXd &input_Z,
                   const Eigen::MappedSparseMatrix<double> &input_W,
                   Eigen::VectorXd &input_y,
                   const bool &input_diagnostics) : 

        STAP(input_dists,input_ucrs,input_subj_array,input_subj_n, input_Z, input_y, input_diagnostics), W(input_W){}

        void calculate_glmer_eta(SV_glmer& svg);

        Eigen::VectorXd bdel(SV_glmer& svg);

        Eigen::VectorXd bslope_del(SV_glmer& svg);

        double rho_del(SV_glmer& svg);

        double calculate_glmer_ll(SV_glmer& svg);

        double calculate_glmer_energy(SV_glmer& svg);

        double sample_u(SV_glmer& svg,std::mt19937& rng);

        void calculate_gradient(SV_glmer& svg);

        double FindReasonableEpsilon(SV_glmer& svg,std::mt19937& rng);
};

#include "STAP_glmer.inl"

double adjust_alpha(STAP_glmer &stap_object, SV_glmer &sv,double &y_bar, double &y_sd, Eigen::VectorXd &z_bar){
  
  double out = y_bar;
  stap_object.calculate_X_diff(sv.theta(0));

  out += y_sd * (sv.alpha - stap_object.X_global_mean.dot(sv.beta_bar) - 
      z_bar.dot(sv.delta));
  
  return(out);
}
