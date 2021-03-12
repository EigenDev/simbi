/**
* Helper Function Template Tool
*/


/**
* Root Finder Template file for a variadic Newton Raphson root Finder
*
*/

#include <vector>

template <typename T, size_t N>
constexpr size_t array_size(T (&)[N]) {
    return N;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] ); 
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

template <typename T, typename... Args>
double newton_raphson(T x1, T (*f)(T, Args... args),  T (*g)(T, Args... args), 
                        double epsilon, Args... args)
{
    /**
     * Newton Raphson Method:
     * 
     * x: The initial guess
     * f:  The key function f(x)
     * g: The derivative of the key function -- f'(x)
     * epsilon: The error tolerance
     * 
     */
      
    
    
    /**
    double x = x1;
    double h = 0;
    double q = (*f)(x, args...);

    
    do {
        h = (*f)(x1, args...)/(*g)(x1, args...);

        x1 = x1 - h; 
    }while (std::abs(h) > epsilon);
    */

    
    double x,xstar,xp, h;
    int ii = 0;
    int maximum_iteration = 100;
    do {
        xp = x;
        x = x1;
        if (ii > 0){
            xstar = x - (*f)(x, args...)/(*g)(0.5*(xp + xstar), args...);
        } else {
            xstar = x1;
        }

        h = (*f)(x, args...)/(*g)(0.5*(x + xstar), args...);

        x1 = x - h;
        

        ii++;

        
        if (ii > maximum_iteration){
            std::cout << "\n Cons2Prim Not Convergent" << std::endl;
            exit(EXIT_FAILURE);
        }
        

    } while(std::abs(x1 - x) >= epsilon);
    
    
    // if (ii > maximum_iteration){
    //     std::cout << "Newton Raphson took: " << ii << " iterations" << std::endl;
    //     std::cin.get();
    // }
    

    return x1; 
};

template <typename T>
std::vector<double> calc_lorentz_gamma(T &v){
    int vsize = v.size();
    std::vector<double> W(vsize); 

    for (int ii=0; ii < vsize; ii++){
        W[ii] = 1/sqrt(1 - v[ii]*v[ii]);
    }

    return W;
};


template<typename T>
std::vector<double>  calc_lorentz_gamma(T &v1, T &v2, int nx, int ny){
    int xgrid_size = nx;
    int ygrid_size = ny;
    std::vector<double> W; 
    W.reserve(ygrid_size * xgrid_size);

    std::transform(v1.begin(), v1.end(), v2.begin(), back_inserter(W), [&](const double& vx, const double& vy){
        return 1.0/sqrt(1.0 - (vx*vx + vy*vy));
    });
    // for (int jj=0; jj < ygrid_size; jj++){
    //     for (int ii=0; ii < xgrid_size; ii++){
    //         vtot = sqrt(v1[ii + xgrid_size * jj]*v1[ii + xgrid_size * jj] + v2[ii + xgrid_size * jj]*v2[ii + xgrid_size * jj]);
// 
    //         W[ii + xgrid_size * jj] = 1/sqrt(1 - vtot*vtot);
    //     }
    // }
    
    return W;
};

template <typename T>
void config_ghosts1D(T &u_state, int grid_size, bool first_order){
    if (first_order){
        u_state.D[0] = u_state.D[1];
        u_state.D[grid_size - 1] = u_state.D[grid_size - 2];

        u_state.S[0] = u_state.S[1];
        u_state.S[grid_size - 1] = u_state.S[grid_size - 2];

        u_state.tau[0] = u_state.tau[1];
        u_state.tau[grid_size - 1] = u_state.tau[grid_size - 2];
    } else {
        u_state.D[0] = u_state.D[2];
        u_state.D[1] = u_state.D[2];
        u_state.D[grid_size - 1] = u_state.D[grid_size - 3];
        u_state.D[grid_size - 2] = u_state.D[grid_size - 3];

        u_state.S[0] = u_state.S[2];
        u_state.S[1] = u_state.S[2];
        u_state.S[grid_size - 1] = u_state.S[grid_size - 3];
        u_state.S[grid_size - 2] = u_state.S[grid_size - 3];

        u_state.tau[0] = u_state.tau[2];
        u_state.tau[1] = u_state.tau[2];
        u_state.tau[grid_size - 1] = u_state.tau[grid_size - 3];
        u_state.tau[grid_size - 2] = u_state.tau[grid_size - 3];
    }
};

template <typename T>
void config_ghosts2D(T &u_state, 
                        int x1grid_size, int x2grid_size, bool first_order,
                        std::string kind){

    if (first_order){
        for (int jj = 0; jj < x2grid_size; jj++){
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 1){
                    u_state.D[ii + x1grid_size * jj] =   u_state.D[ii + x1grid_size];
                    u_state.S1[ii + x1grid_size * jj] =   u_state.S1[ii + x1grid_size];
                    u_state.S2[ii + x1grid_size * jj] = - u_state.S2[ii + x1grid_size];
                    u_state.tau[ii + x1grid_size * jj] =   u_state.tau[ii + x1grid_size];
                    
                } else if (jj > x2grid_size - 2) {
                    u_state.D[ii + x1grid_size * jj]  =   u_state.D[(x2grid_size - 2) * x1grid_size + ii];
                    u_state.S1[ii + x1grid_size * jj]  =   u_state.S1[(x2grid_size - 2) * x1grid_size + ii];
                    u_state.S2[ii + x1grid_size * jj]  = - u_state.S2[(x2grid_size - 2) * x1grid_size + ii];
                    u_state.tau[ii + x1grid_size * jj]  =   u_state.tau[(x2grid_size - 2) * x1grid_size + ii];

                } else {
                    u_state.D[jj * x1grid_size]    = u_state.D[jj * x1grid_size + 1];
                    u_state.D[jj * x1grid_size + x1grid_size - 1] = u_state.D[jj*x1grid_size + x1grid_size - 2];

                    u_state.S1[jj * x1grid_size + 0]               = - u_state.S1[jj * x1grid_size + 1];
                    u_state.S1[jj * x1grid_size + x1grid_size - 1] =   u_state.S1[jj * x1grid_size + x1grid_size - 2];

                    u_state.S2[jj * x1grid_size + 0]                = u_state.S2[jj * x1grid_size + 1];
                    u_state.S2[jj * x1grid_size + x1grid_size - 1]  = u_state.S2[jj * x1grid_size + x1grid_size - 2];

                    u_state.tau[jj * x1grid_size + 0]               = u_state.tau[jj * x1grid_size + 1];
                    u_state.tau[jj * x1grid_size + x1grid_size - 1] = u_state.tau[jj * x1grid_size + x1grid_size - 2];
                }
            }
        }

    } else {
        for (int jj = 0; jj < x2grid_size; jj++){

            // Fix the ghost zones at the radial boundaries
            u_state.D[jj * x1grid_size +  0]               = u_state.D[jj * x1grid_size +  3];
            u_state.D[jj * x1grid_size +  1]               = u_state.D[jj * x1grid_size +  2];
            u_state.D[jj * x1grid_size +  x1grid_size - 1] = u_state.D[jj * x1grid_size +  x1grid_size - 3];
            u_state.D[jj * x1grid_size +  x1grid_size - 2] = u_state.D[jj * x1grid_size +  x1grid_size - 3];

            u_state.S1[jj * x1grid_size + 0]               = - u_state.S1[jj * x1grid_size + 3];
            u_state.S1[jj * x1grid_size + 1]               = - u_state.S1[jj * x1grid_size + 2];
            u_state.S1[jj * x1grid_size + x1grid_size - 1] =   u_state.S1[jj * x1grid_size + x1grid_size - 3];
            u_state.S1[jj * x1grid_size + x1grid_size - 2] =   u_state.S1[jj * x1grid_size + x1grid_size - 3];

            u_state.S2[jj * x1grid_size + 0]               = u_state.S2[jj * x1grid_size + 3];
            u_state.S2[jj * x1grid_size + 1]               = u_state.S2[jj * x1grid_size + 2];
            u_state.S2[jj * x1grid_size + x1grid_size - 1] = u_state.S2[jj * x1grid_size + x1grid_size - 3];
            u_state.S2[jj * x1grid_size + x1grid_size - 2] = u_state.S2[jj * x1grid_size + x1grid_size - 3];

            u_state.tau[jj * x1grid_size + 0]               = u_state.tau[jj * x1grid_size + 3];
            u_state.tau[jj * x1grid_size + 1]               = u_state.tau[jj * x1grid_size + 2];
            u_state.tau[jj * x1grid_size + x1grid_size - 1] = u_state.tau[jj * x1grid_size + x1grid_size - 3];
            u_state.tau[jj * x1grid_size + x1grid_size - 2] = u_state.tau[jj * x1grid_size + x1grid_size - 3];

            // Fix the ghost zones at the angular boundaries
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 2){
                    if (jj == 0){
                        u_state.D[jj * x1grid_size + ii] =   u_state.D[3 * x1grid_size + ii];
                        u_state.S1[jj * x1grid_size + ii] =   u_state.S1[3 * x1grid_size + ii];
                        u_state.S2[jj * x1grid_size + ii] = - u_state.S2[3 * x1grid_size + ii];
                        u_state.tau[jj * x1grid_size + ii] =   u_state.tau[3 * x1grid_size + ii];
                    } else {
                        u_state.D[jj * x1grid_size + ii] =   u_state.D[2 * x1grid_size + ii];
                        u_state.S1[jj * x1grid_size + ii] =   u_state.S1[2 * x1grid_size + ii];
                        u_state.S2[jj * x1grid_size + ii] = - u_state.S2[2 * x1grid_size + ii];
                        u_state.tau[jj * x1grid_size + ii] =   u_state.tau[2 * x1grid_size + ii];
                    }
                    
                } else if (jj > x2grid_size - 3) {
                    if (jj == x2grid_size - 1){
                        u_state.D[jj * x1grid_size + ii] =   u_state.D[(x2grid_size - 4) * x1grid_size + ii];
                        u_state.S1[jj * x1grid_size + ii] =   u_state.S1[(x2grid_size - 4) * x1grid_size + ii];
                        u_state.S2[jj * x1grid_size + ii] = - u_state.S2[(x2grid_size - 4) * x1grid_size + ii];
                        u_state.tau[jj * x1grid_size + ii] =   u_state.tau[(x2grid_size - 4) * x1grid_size + ii];
                    } else {
                        u_state.D[jj * x1grid_size + ii] =   u_state.D[(x2grid_size - 3) * x1grid_size + ii];
                        u_state.S1[jj * x1grid_size + ii] =   u_state.S1[(x2grid_size - 3) * x1grid_size + ii];
                        u_state.S2[jj * x1grid_size + ii] = - u_state.S2[(x2grid_size - 3) * x1grid_size + ii];
                        u_state.tau[jj * x1grid_size + ii] =   u_state.tau[(x2grid_size - 3) * x1grid_size + ii];
                    }

                }
            }
        }

    }
};

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
void toWritePrim(T *from, PrimData *to, int ndim)
{
    /**
    switch (ndim)
    {
    case 1:
        from->rho = to->rho;
        from->v   = to->v;
        from->p   = to->p;
        break;
    
    default:
        from->rho  = to->rho;
        from->v1   = to->v1;
        from->v2   = to->v2;
        from->p    = to->p;
        break;
    }
    */
    to->rho  = from->rho;
    to->v1   = from->v1;
    to->v2   = from->v2;
    to->p    = from->p;
}