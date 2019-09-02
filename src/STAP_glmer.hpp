double sigmoid(double x){
    return (1.0 / (1.0 + exp(-x)));
}

double sigmoid_transform(double x,double a, double b){
    return ( a + (b - a ) * sigmoid(x));
}

double sigmoid_transform_derivative(double x, double a, double b){
    return( (b - a) * sigmoid(x) * (1 - sigmoid(x) ) );
}

double log_sigmoid_transform_derivative(double x, double a, double b){
    return( log((b - a)) - log(1 + exp(-x)) - log(exp(x) +1)  );
}

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
        Eigen::MatrixXd subj_sig_grad; 
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
        Eigen::MatrixXd Sigma;
        Eigen::MatrixXd S_m;
        SV_glmer(Eigen::ArrayXi& stap_par_code_input,
                std::mt19937& rng, const bool input_diagnostics) :
            SV(stap_par_code_input,rng,input_diagnostics)
    {
        b = GaussianNoise(spc(4),rng); 
        b_slope = spc(5) == 2 ? GaussianNoise(spc(4),rng) : Eigen::VectorXd::Zero(spc(4));
        Sigma = initialize_matrix(spc(5),spc(5),rng); 
        if(Sigma.cols()>1)
          Sigma(1,0) = Sigma(0,1);


        if(diagnostics){
            Rcpp::Rcout << "Subj_sigma " << Sigma << std::endl;
            Rcpp::Rcout << "b head " << b.head(5) << std::endl;
            Rcpp::Rcout << "b_slope head " << b_slope.head(5) << std::endl;
        }
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
            S_m = GaussianNoise_mat(Sigma.cols(),Sigma.cols(),rng);
            if(S_m.cols()==2)
              S_m(1,0) = S_m(0,1);
            am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            b_m = GaussianNoise(spc(4),rng);
            bs_m = spc(5) == 2 ? GaussianNoise(spc(4),rng) : Eigen::VectorXd::Zero(spc(4));
            tm = GaussianNoise(theta.size(),rng);

        }

        double get_rho(){
            return(sigmoid_transform(Sigma(0,1),-1,1));
        }

        double get_rho_derivative(){
            return(sigmoid_transform_derivative(Sigma(0,1),-1,1));
        }

        double get_rho_sq_c(){
            return(1 - pow(sigmoid_transform(Sigma(0,1),-1,1),2) );
        }

        double mer_derivative_one(int i){
            double out = 0;
            out = -2.0 / (pow(exp(Sigma(i,i)),2) * get_rho_sq_c() );
            return(out);
        }

        double mer_derivative_two(){
            return (get_rho() / (exp(Sigma(0,0)) * exp(Sigma(1,1))  *get_rho_sq_c()));
        }

        double mer_derivative_three(int i ){
          double out = 0;
          out += exp(-2*Sigma(i,i)) * .25 * exp(- Sigma(0,1)) * (exp( 2 * Sigma(0,1)) + 1);
          return(out);
        }

        double mer_derivative_four(){
          double out = 0;
          out += -.25 * exp(-Sigma(0,0)) * exp(-Sigma(1,1)) *  exp(-Sigma(0,1)) * (exp(2 * Sigma(0,1)) +1 );
          return(out);
        }

        double mer_ssv_1(){
            double out = 0;
            out += - b.rows();
            out += -.5 * b.dot(b) * mer_derivative_one(0);
            out += - (b.dot(b_slope) * mer_derivative_two());
            out += -  mer_sd_transformed()(0,0) + 1;
            return(out);
        }

        double mer_ssv_2(){
            double out = 0;
            out += - b.rows(); 
            out += -.5 * b_slope.dot(b_slope) * mer_derivative_one(1);
            out += - (b.dot(b_slope) * mer_derivative_two()) ;
            out += - mer_sd_transformed()(1,1) + 1;
            return(out);
        }

        double mer_ss_cor(){
            double out = 0;
            out +=  b.rows() * (get_rho() * get_rho_derivative()) / get_rho_sq_c()  ;
            out += (get_rho_sq_c() * get_rho_derivative() + 2*pow(get_rho(),2) * get_rho_derivative()) / (pow(get_rho_sq_c(),2) * exp(Sigma(0,0)) * exp(Sigma(1,1)) ) * b.dot(b_slope)  ;
            Rcpp::Rcout << " mer_ss_cor 2: " << out << std::endl;
            out += - (get_rho() * get_rho_derivative()) / (pow(get_rho_sq_c(),2)) * ( b.dot(b) * exp(-2*Sigma(0,0)) + b_slope.dot(b_slope) * exp(-2 *Sigma(1,1)) ) ;
            Rcpp::Rcout << " mer_ss_cor 3: " << out << std::endl;
            out += log_sigmoid_transform_derivative(Sigma(0,1),-1,1);
            return(out);
        }


        Eigen::MatrixXd mer_precision_transformed(){
            if(Sigma.cols() == 1)
                return (pow(exp(Sigma.array()),-2));
            else{
                Eigen::MatrixXd out(Sigma.rows(),Sigma.cols());
                out = mer_var_transformed();
                return (out.inverse());
            }
        }

        Eigen::MatrixXd mer_sd_transformed(){
            if(Sigma.cols() == 1)
                return(exp(Sigma.array()));
            else{
                Eigen::MatrixXd out(Sigma.rows(),Sigma.cols());
                out(0,0) = exp(Sigma(0,0));
                out(1,1) = exp(Sigma(1,1));
                out(0,1) = sigmoid_transform(Sigma(0,1),-1,1);
                out(1,0) = out(0,1);
                return(out);
            }
        }

        Eigen::MatrixXd mer_var_transformed(){
            if(Sigma.cols() == 1)
                return(pow(exp(Sigma.array()),2));
            else{
                Eigen::MatrixXd out(Sigma.rows(),Sigma.cols());
                out(0,0) = pow(exp(Sigma(0,0)),2);
                out(1,1) = pow(exp(Sigma(1,1)),2);
                out(1,0) = sigmoid_transform(Sigma(0,1),-1,1) * exp(Sigma(0,0)) * exp(Sigma(1,1));
                out(0,1) = sigmoid_transform(Sigma(0,1),-1,1) * exp(Sigma(0,0)) * exp(Sigma(1,1));
                return(out);
            }
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
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            delta = other.delta;
            b = other.b;
            b_slope = other.b_slope;
            sigma = other.sigma;
            Sigma = other.Sigma;
        }

        double kinetic_energy_glmer(){
            double out = 0;
            out = dm.transpose() * (vd.var).matrix().asDiagonal() * dm;
            out += bm.transpose() * (vb.var).matrix().asDiagonal() * bm;
            out += bbm.transpose() * (vbb.var).matrix().asDiagonal() * bbm;
            out += tm.transpose() * (vt.var).matrix().asDiagonal() * tm;
            out += (sm * sm) * vs.var   + (am * am) * va.var;
            out += b_m.dot(b_m);
            out += bs_m.dot(bs_m);
            out += b_m.col(0).dot(b_m.col(0)); 
            out += pow(S_m(0,0),2);
            if(S_m.cols()>1){
                out += pow(S_m(0,1),2);
                out += pow(S_m(1,1),2);
            }
            out = out / 2.0;
            return(out);
        }

};

bool get_UTI_one(SV_glmer& svgl, SV_glmer& svgr){


    double out;
    out = (svgr.delta - svgl.delta).dot(svgl.dm);
    out += (svgr.beta - svgl.beta).dot(svgl.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgl.bbm);
    out += (svgr.theta - svgl.theta).dot(svgl.tm);
    out += pow((svgr.sigma - svgl.sigma) * (svgl.sm),2);
    out += pow((svgr.alpha - svgl.alpha) * svgl.am,2);
    out += pow((svgr.Sigma(0,0) - svgl.Sigma(0,0)) * svgl.S_m(0,0) ,2);
    out += (svgr.b - svgl.b).dot(svgl.b_m);
    out += (svgr.b_slope - svgl.b_slope).dot(svgl.bs_m);
    if(svgr.Sigma.cols() == 2){
        out += pow((svgr.Sigma(0,1) - svgl.Sigma(0,1)) * svgr.S_m(0,1) ,2);
        out += pow((svgr.Sigma(1,1) - svgl.Sigma(1,1)) * svgr.S_m(1,1) ,2);
    }
    out = out / 2.0;

    return((out >= 0));

}

bool get_UTI_two(SV_glmer& svgl,SV_glmer& svgr){

    double out;
    out = (svgr.delta - svgl.delta).dot(svgr.dm);
    out += (svgr.beta - svgl.beta).dot(svgr.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgr.bbm);
    out +=  (svgr.theta - svgl.theta).dot(svgr.tm);
    out += pow((svgr.sigma - svgl.sigma) * (svgr.sm),2);
    out += pow((svgr.alpha - svgl.alpha) * svgr.am,2);
    out += pow((svgr.Sigma(0,0) - svgl.Sigma(0,0)) * svgr.S_m(0,0) ,2);
    out += (svgr.b - svgl.b).dot(svgr.b_m);
    out += (svgr.b_slope - svgl.b_slope).dot(svgr.bs_m);
    if(svgr.b.cols()==2){
        out += pow((svgr.Sigma(0,1) - svgl.Sigma(0,1)) * svgr.S_m(0,1) ,2);
        out += pow((svgr.Sigma(1,1) - svgl.Sigma(1,1)) * svgr.S_m(1,1) ,2);
    }

    out = out / 2.0;

    return((out >= 0));
}

class STAP_glmer: public STAP
{
    private: Eigen::MatrixXd W;
    public:
        SG_glmer sgg;
        STAP_glmer(Eigen::ArrayXXd& input_dists,
                   Eigen::ArrayXXi& input_ucrs,
                   Eigen::MatrixXd& input_subj_array,
                   Eigen::MatrixXd& input_subj_n,
                   Eigen::MatrixXd& input_Z,
                   Eigen::MatrixXd& input_W,
                   Eigen::VectorXd& input_y,
                   const bool& input_diagnostics) : 

        STAP(input_dists,input_ucrs,input_subj_array,input_subj_n, input_Z, input_y, input_diagnostics), W(input_W){}

        void calculate_glmer_eta(SV_glmer& svg);

        double calculate_glmer_ll(SV_glmer& svg);

        double calculate_glmer_energy(SV_glmer& svg);

        double sample_u(SV_glmer& svg,std::mt19937& rng);

        void calculate_gradient(SV_glmer& svg);

        double FindReasonableEpsilon(SV_glmer& svg,std::mt19937& rng);
};

#include "STAP_glmer.inl"
