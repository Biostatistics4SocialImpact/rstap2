#include<random>
#include <RcppEigen.h>

class STAP
{
    private:
        Eigen::MatrixXd X;
        Eigen::MatrixXd X_mean;
        Eigen::MatrixXd X_diff;
        Eigen::MatrixXd X_prime;
        Eigen::MatrixXd X_mean_prime;
        Eigen::MatrixXd X_prime_diff;
        Eigen::VectorXd delta_grad;
        Eigen::VectorXd beta_grad;
        Eigen::VectorXd theta_grad;
        double sigma_grad;
        Eigen::MatrixXd dists;
        Eigen::MatrixXi u_crs;
        Eigen::VectorXi subj_array;
        Eigen::VectorXi subj_n;
        Eigen::VectorXi diff_coefs;
        Eigen::MatrixXd Z;
        Eigen::VectorXd y;

    public:
        STAP(Eigen::MatrixXd input_dists, Eigen::MatrixXi input_u, Eigen::VectorXi input_subj_array,Eigen::VectorXi input_subj_n, Eigen::VectorXi input_diff_coefs, Eigen::MatrixXd input_Z, Eigen::VectorXd input_y){
            dists = input_dists;
            u_crs = input_u;
            subj_array = input_subj_array;
            subj_n = input_subj_array;
            y = input_y;
            Z = input_Z;
        }

        double calculate_total_energy(STAP_Vars &sv);

        double sample_log_u(STAP_Vars &sv, std::mt19937 &rng);

        void calculate_X_diff(Eigen::VectorXd &theta);

        void calculate_X_mean(Eigen::VectorXd &theta);

        void calculate_X_prime(Eigen::VectorXd &theta);

        void calculate_X_mean_prime(Eigen::VectorXd &theta);

        void calculate_X_prime_diff(Eigen::VectorXd &theta);

        void calculate_gradient(STAP_Vars &sv);

        double FindReasonableEpsilon(STAP_Vars &sv, std::mt19937 &rng);

        Eigen::VectorXd get_delta_grad() const{
            return(delta_grad);
        }
        
        Eigen::VectorXd get_beta_grad() const{
            return(beta_grad);
        }

        Eigen::VectorXd get_theta_grad() const{
            return(theta_grad);
        }

        double get_sigma_grad() const{
            return(sigma_grad);
        }

};


#include "STAPdnd.inl"
