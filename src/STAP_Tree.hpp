//#include<cmath>
#include<random>
//#include<iostream>
//#include<vector>


class STAP_Tree
{
    private :
        double br;
        double bl;
        double bml;
        double bmr;
        double bmn;
        double beta_new;
        double tr;
        double tl;
        double tml;
        double tmr;
        double tmn;
        double theta_new;
        double n_prime;
        int s_prime;
        double alpha_prime;
        double n_alpha;

    public:
        void BuildTree(STAP &stap_object,
                double beta_proposed, double theta_proposed,
                double beta_init, double theta_init, 
                double bmp, double tmp, double bmi, double tmi,
                double u,int v, int j, double &epsilon_theta, 
                double &epsilon_beta,std::mt19937 &rng);

        void Leapfrog(STAP &stap_object, double &cur_beta,
                      double &cur_theta, double bm, double tm, 
                      double epsilon_theta, double epsilon_beta);

        const int get_s_prime() const; 

        const double get_n_prime() const;

        const double get_alpha_prime() const;

        const double get_n_alpha() const;

        const double get_beta_new() const;
        
        const double get_bl() const;

        const double get_br() const;

        const double get_bml() const;
        
        const double get_bmr() const;

        const double get_theta_new() const;

        const double get_tr() const;
        
        const double get_tl() const;

        const double get_tml() const;

        const double get_tmr() const;

};


#include "STAP_Tree.inl"
