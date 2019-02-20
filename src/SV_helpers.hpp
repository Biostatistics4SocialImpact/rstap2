#include <random>
#include <RcppEigen.h>

double initialize_scalar(std::mt19937 &rng){

    std::uniform_real_distribution<double> par_init(-2.0,2.0);
    return(par_init(rng));

}

Eigen::VectorXd initialize_vec(int dim, std::mt19937& rng){

    Eigen::VectorXd new_vec(dim);
    for(int i = 0 ; i<dim; i++)
        new_vec(i) = initialize_scalar(rng);
    return(new_vec);
}

double normal_draw(std::mt19937& rng){

    std::normal_distribution<double> z(0,1);
    return(z(rng));
}

Eigen::VectorXd normal_draw(int dim, std::mt19937& rng){

    Eigen::VectorXd out(dim);
    for(int i =0 ;i <dim; i++)
        out(i) = normal_draw(rng);
    return(out);
}


