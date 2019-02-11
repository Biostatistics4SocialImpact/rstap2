#include<random>


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
        double sigma_new;
        double smr;
        double sml;
        double smn;
        double sl;
        double sr;
        double n_prime;
        int s_prime;
        double alpha_prime;
        double n_alpha;
        bool diagnostics;

    public:
        STAP_Tree(const bool& diagnostics_input){
            diagnostics = diagnostics_input;
        }

        void BuildTree(STAP& stap_object,
                double beta_proposed,
                double theta_proposed,
                double sigma_proposed,
                double beta_init, 
                double theta_init, 
                double sigma_init,
                double bmp, double tmp, double smp,
                double bmi, double tmi, double smi,
                double u,int v, int j,
                double &epsilon_beta, double &epsilon_theta,
                std::mt19937 &rng);

        void Leapfrog(STAP& stap_object,
                      double& cur_beta,
                      double& cur_theta, 
                      double& cur_sigma,
                      double bm, double tm,double sm,
                      double epsilon_beta, double epsilon_theta);

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

        const double get_sigma_new() const;

        const double get_sl() const;
        
        const double get_sr() const;

        const double get_sml() const;

        const double get_smr() const;

};


#include "STAP_Tree.inl"
