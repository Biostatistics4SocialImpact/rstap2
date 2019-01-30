#include<cmath>
#include<random>


class STAP_Tree
{
    private:
        STAP_Vars sv_new;
        STAP_Vars svl;
        STAP_Vars svr;
        double n_prime;
        int s_prime;
        double alpha_prime;
        double n_alpha;

    public:
        STAP_Tree(int p, int q, int thq) : sv_new(p,q,thq),svl(p,q,thq), svr(p,q,thq){};

        void BuildTree(STAP &stap_object,
                       STAP_Vars sv,
                       STAP_Vars svi,
                       double u,int v, int j,
                       double &epsilon, std::mt19937 &rng);

        void Leapfrog(STAP &stap_object, STAP_Vars &sv, double epsilon);

        const int get_s_prime() const; 

        const double get_n_prime() const;

        const double get_alpha_prime() const;

        const double get_n_alpha() const;

        const STAP_Vars get_sv_new() const;

        const STAP_Vars get_svl() const;

        const STAP_Vars get_svr() const;
};


#include "STAPdnd_Tree.inl"
