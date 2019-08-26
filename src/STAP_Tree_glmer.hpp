class STAP_Tree_glmer
{
    private :
        SV_glmer svl;
        SV_glmer svr;
        SV_glmer svn;
        Eigen::ArrayXi spc;
        double n_prime;
        int s_prime;
        double alpha_prime;
        double n_alpha;
        bool diagnostics;

    public:
        STAP_Tree_glmer(Eigen::ArrayXi& input_stap_code,
                       const bool& diagnostics_input,
                       std::mt19937& rng) : 
            svl(input_stap_code,rng,diagnostics_input), 
            svr(input_stap_code,rng,diagnostics_input),
            svn(input_stap_code,rng,diagnostics_input) {
            svl.initialize_momenta(rng);
            svr.initialize_momenta(rng);
            svn.initialize_momenta(rng);
            spc = input_stap_code;
            diagnostics = diagnostics_input;
        }

        void BuildTree(STAP_glmer& stap_object,
                SV_glmer& sv_proposed,
                SV_glmer& sv_init,
                double& u,int v, int j,
                double &epsilon,
                std::mt19937 &rng);

        void Leapfrog(STAP_glmer& stap_object,
                      SV_glmer& sv,
                      double epsilon);

        const int get_s_prime() const{
            return(s_prime);
        }

        const double get_n_prime() const{
            return(n_prime);
        }

        const double get_alpha_prime() const{
            return(alpha_prime);
        }

        const double get_n_alpha() const{
            return(n_alpha);
        }

        SV_glmer get_svl() const{
            return(svl);
        }

        SV_glmer get_svr() const{
            return(svr);
        }

        SV_glmer get_svn() const{
            return(svn);
        }

        double get_alpha_new() const  {
            return(svn.alpha);
        }

        Eigen::VectorXd get_delta_new() const{
            return(svn.delta);
        }

        Eigen::VectorXd get_beta_new() const {
            return(svn.beta);
        }

        Eigen::VectorXd get_beta_bar_new() const {
            return(svn.beta_bar);
        }

        Eigen::VectorXd get_theta_new() {
            return(svn.theta);
        }

        Eigen::VectorXd get_theta_new_transformed() {
            return(svn.theta_transformed());
        }

        const double get_sigma_new_transformed() {
            return(svn.sigma_transformed());
        }
        
        const double get_sigma_new() {
            return(svn.sigma);
        }

        Eigen::VectorXd get_b_new() {
            return(svn.b);
        }

        Eigen::MatrixXd get_Sigma_new() {
            return(svn.Sigma);
        }

        Eigen::RowVectorXd get_Sigma_new_transformed() {
            Eigen::Map<Eigen::RowVectorXd> cov_samp(svn.Sigma.data(),svn.Sigma.size());

            return(cov_samp);

        }

};


#include "STAP_Tree_glmer.inl"
