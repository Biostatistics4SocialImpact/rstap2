#include<random>
#include <RcppEigen.h>

class STAP
{
    private:
        Eigen::VectorXd X;
        Eigen::VectorXd X_mean; Eigen::VectorXd X_diff;
        Eigen::VectorXd X_prime;
        Eigen::VectorXd X_mean_prime;
        Eigen::VectorXd X_prime_diff;
        double beta_grad;
        double theta_grad;
        Eigen::MatrixXd dists;
        Eigen::MatrixXd d_one;
        Eigen::MatrixXd d_two;
        Eigen::MatrixXd d_three;
        Eigen::VectorXd y;
        double sigma;

    public:
        STAP(Eigen::MatrixXd &input_dists,
                Eigen::MatrixXd &input_d_one, 
                Eigen::MatrixXd &input_d_two, 
                Eigen::MatrixXd &input_d_three, 
                Eigen::VectorXd &input_y);

        double calculate_total_energy(double cur_beta,  double cur_theta,  double &cur_bm,  double &cur_tm);

        double sample_u( double &cur_beta, double &cur_theta,  double &cur_bm,  double &cur_tm,  std::mt19937 &rng);

        void calculate_X_diff( double &theta);

        void calculate_X_mean( double &theta);

        void calculate_X_prime(double &theta, double &cur_theta);

        void calculate_X_mean_prime(double &theta, double &cur_theta);

        void calculate_X_prime_diff(double &theta, double &cur_theta);

        void calculate_gradient(double &cur_beta, double &cur_theta);

        double FindReasonableEpsilon(double &cur_beta, double &cur_theta, double &bm, double &tm, std::mt19937 &rng);

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

};


#include "STAP.inl"
