#include <random>
Eigen::MatrixXd initialize_matrix(const int& num_rows,const int& num_cols, std::mt19937& rng){

    std::uniform_real_distribution<double> runif(-2,2);
    Eigen::MatrixXd out(num_rows,num_cols);
    for(int row_ix = 0; row_ix < num_rows; row_ix ++){ 
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
        Eigen::MatrixXd b_grad;
        double subj_sig_grad; 
        void print_grads(){

            Rcpp::Rcout << "Printing Gradients: " << std::endl;
            Rcpp::Rcout << "-------------------- " << std::endl;
            Rcpp::Rcout << "alpha_grad: "  << alpha_grad << std::endl;
            Rcpp::Rcout << "delta_grad: "  << delta_grad << std::endl;
            Rcpp::Rcout << "beta_grad: "  << beta_grad << std::endl;
            Rcpp::Rcout << "beta_bar_grad: "  << beta_bar_grad << std::endl;
            Rcpp::Rcout << "theta_grad: " << theta_grad << std::endl;
            Rcpp::Rcout << "sigma_grad: " << sigma_grad << std::endl;
            Rcpp::Rcout << "subj_b_grad: " << b_grad.size() << std::endl;
            Rcpp::Rcout << "subj_b_grad: " << b_grad.block(0,0,5,1) << std::endl;
            Rcpp::Rcout << "subj_sigma_grad: " << subj_sig_grad << std::endl;
            Rcpp::Rcout << "-------------------- \n " << std::endl;
        }
};

class SV_glmer: public SV
{
    public:
        Eigen::MatrixXd b;
        Eigen::MatrixXd b_m;
        double Sigma;
        double S_m;
        SV_glmer(Eigen::ArrayXi& stap_par_code_input,
                std::mt19937& rng, const bool input_diagnostics) :
            SV(stap_par_code_input,rng,input_diagnostics)
    {
        b = initialize_matrix(spc(4),1,rng); 
        Sigma = initialize_scalar(rng);
        if(diagnostics){
            Rcpp::Rcout << "Subj_sigma " << Sigma << std::endl;
            Rcpp::Rcout << "b head " << b.block(0,0,5,1) << std::endl;
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
            Rcpp::Rcout << "b : \n" << b.block(0,0,5,1) << std::endl;
            Rcpp::Rcout << "sigma_b : " << Sigma << std::endl;
            Rcpp::Rcout << "sigma_b transformed : " << exp(Sigma) << std::endl;
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
            Rcpp::Rcout << "S_m: " << S_m << std::endl;
            Rcpp::Rcout << "b_m: \n" << b_m.block(0,0,5,1) << std::endl;
            Rcpp::Rcout << "------------------------ " << "\n" << std::endl;

        }

        void initialize_momenta(std::mt19937& rng){

            sm = GaussianNoise_scalar(rng);
            S_m = GaussianNoise_scalar(rng);
            am = spc(0) == 0 ? 0.0 :  GaussianNoise_scalar(rng);
            bm = spc(2) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta.size(),rng);
            bbm = spc(3) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(beta_bar.size(),rng);
            dm = spc(1) == 0 ? Eigen::VectorXd::Zero(1) : GaussianNoise(delta.size(),rng); 
            b_m = GaussianNoise_mat(b.rows(),b.cols(),rng);
            tm = GaussianNoise(theta.size(),rng);

        }

        double mer_precision_transformed(){
            return (pow(exp(Sigma),-2));
        }

        double mer_sd_transformed(){
            return(exp(Sigma));
        }

        double mer_var_transformed(){
            return(pow(exp(Sigma),2));
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
                this->print_pars();
            }
            alpha = svg.alpha + epsilon * am;
            delta = svg.delta + epsilon * dm;
            beta = svg.beta + epsilon * bm;
            beta_bar = svg.beta_bar + epsilon * bbm; 
            b = svg.b + epsilon * b_m;
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
            sm = other.sm;
            S_m = other.S_m;
            alpha = other.alpha;
            beta = other.beta;
            beta_bar = other.beta_bar;
            theta = other.theta;
            delta = other.delta;
            b = other.b;
            sigma = other.sigma;
            Sigma = other.Sigma;
        }

        double kinetic_energy_glmer(){
            double out = 0;
            out = dm.transpose() * (1.0 / vd.sd).matrix().asDiagonal() * dm;
            Rcpp::Rcout << "dm" << out << std::endl;
            out += bm.transpose() * (1.0 / vb.sd).matrix().asDiagonal() * bm;
            Rcpp::Rcout << "bm" << out << std::endl;
            out += bbm.transpose() * (1.0 / vbb.sd).matrix().asDiagonal() * bbm;
            Rcpp::Rcout << "bbm" << out << std::endl;
            out += tm.transpose() * (1.0 / vt.sd).matrix().asDiagonal() * tm;
            Rcpp::Rcout << "tm" << out << std::endl;
            out += (sm * sm) / vs.sd   + (am * am) / va.sd;
            Rcpp::Rcout << "sm" << out << std::endl;
            out += (b_m.transpose() * b_m).sum() ;
            Rcpp::Rcout << "b_m" << out << std::endl;
            out += S_m * (S_m);
            Rcpp::Rcout << "S_m" << out << std::endl;
            out = out / 2.0;
            return(out);
        }

};

bool get_UTI_one(SV_glmer& svgl, SV_glmer& svgr){


    double out;
    out = (svgr.delta - svgl.delta).dot(svgl.dm);
    out += (svgr.beta - svgl.beta).dot(svgl.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgl.bbm);
    out += (svgr.theta - svgl.theta).dot(svgl.tm) + (svgr.sigma - svgl.sigma) * (svgl.sm);
    out += (svgr.alpha - svgl.alpha) * svgl.am;
    out += (svgr.Sigma - svgl.Sigma) * svgl.S_m;
    out += ((svgr.b - svgl.b).transpose() * svgl.b_m).sum();
    out = out / 2.0;

    return((out >=0));

}

bool get_UTI_two(SV_glmer& svgl,SV_glmer& svgr){

    double out;
    out = (svgr.delta - svgl.delta).dot(svgr.dm) + (svgr.beta - svgl.beta).dot(svgr.bm);
    out += (svgr.beta_bar - svgl.beta_bar).dot(svgr.bbm) +  (svgr.theta - svgl.theta).dot(svgr.tm);
    out += (svgr.sigma - svgl.sigma) * (svgr.sm);
    out += (svgr.alpha - svgl.alpha) * svgr.am;
    out += (svgr.Sigma - svgl.Sigma) * svgr.S_m;
    out += ((svgr.b - svgl.b).transpose() * svgr.b_m).sum();
    out = out / 2.0;

    return((out >=0));
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
            STAP(input_dists,input_ucrs,input_subj_array,input_subj_n, input_Z, input_y, input_diagnostics), W(input_W)       
            {
            }

        void calculate_glmer_eta(SV_glmer& svg);

        double calculate_glmer_ll(SV_glmer& svg);

        double calculate_glmer_energy(SV_glmer& svg);

        double sample_u(SV_glmer& svg,std::mt19937& rng);

        void calculate_gradient(SV_glmer& svg);

        double FindReasonableEpsilon(SV_glmer& svg,std::mt19937& rng);
};

#include "STAP_glmer.inl"
