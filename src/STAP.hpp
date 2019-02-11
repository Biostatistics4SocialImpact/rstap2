#include<random>
#include<RcppEigen.h>
#include<Eigen/Core>

class STAP
{
    private:
        Eigen::VectorXd X;
        Eigen::VectorXd X_mean; 
        Eigen::VectorXd X_diff;
        Eigen::VectorXd X_prime;
        Eigen::VectorXd X_mean_prime;
        Eigen::VectorXd X_prime_diff;
        double beta_grad;
        double theta_grad;
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

        double calculate_total_energy(double& cur_beta,  double& cur_theta,  double& cur_sigma, double &cur_bm,  double &cur_tm, double& cur_sm);

        double sample_u(double& cur_beta, double& cur_theta, double& cur_sigma,  double& cur_bm,  double& cur_tm, double& cur_sm,  std::mt19937& rng);

        void calculate_X(double& theta);

        void calculate_X_diff(double& theta);

        void calculate_X_mean();

        void calculate_X_prime(double& theta, double& cur_theta);

        void calculate_X_mean_prime();

        void calculate_X_prime_diff(double& theta, double& cur_theta);

        void calculate_gradient(double& cur_beta, double& cur_theta, double& cur_sigma);

        double FindReasonableEpsilon(double& cur_beta, double& cur_theta,double& cur_sigma, double& bm, double& tm,double& sm, std::mt19937& rng);

        Eigen::VectorXd get_X() const{
            return(X);
        };

        Eigen::VectorXd get_X_diff() const{
            return(X_diff);
        }

        Eigen::VectorXd get_X_mean() const{
            return(X_mean);
        }

        Eigen::VectorXd get_X_prime() const{
            return(X_prime);
        }

        Eigen::VectorXd get_Xp_diff() const{
            return(X_prime_diff);
        }

        double get_beta_grad() const{
            return(beta_grad);
        }

        double get_theta_grad() const{
            return(theta_grad);
        }

        double get_sigma_grad() const{
            return(sigma_grad);
        }

};


#include "STAP.inl"
