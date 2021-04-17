/**
* Helper Function Template Tool
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
        u_state.D[0] = u_state.D[3];
        u_state.D[1] = u_state.D[2];
        u_state.D[grid_size - 1] = u_state.D[grid_size - 3];
        u_state.D[grid_size - 2] = u_state.D[grid_size - 3];

        u_state.S[0] = - u_state.S[3];
        u_state.S[1] = - u_state.S[2];
        u_state.S[grid_size - 1] = u_state.S[grid_size - 3];
        u_state.S[grid_size - 2] = u_state.S[grid_size - 3];

        u_state.tau[0] = u_state.tau[3];
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
                    u_state[ii + x1grid_size * jj].D   =   u_state[ii + x1grid_size].D;
                    u_state[ii + x1grid_size * jj].S1  =   u_state[ii + x1grid_size].S1;
                    u_state[ii + x1grid_size * jj].S2  = - u_state[ii + x1grid_size].S2;
                    u_state[ii + x1grid_size * jj].tau =   u_state[ii + x1grid_size].tau;
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[ii + x1grid_size * jj].D    =   u_state[(x2grid_size - 2) * x1grid_size + ii].D;
                    u_state[ii + x1grid_size * jj].S1   =   u_state[(x2grid_size - 2) * x1grid_size + ii].S1;
                    u_state[ii + x1grid_size * jj].S2   = - u_state[(x2grid_size - 2) * x1grid_size + ii].S2;
                    u_state[ii + x1grid_size * jj].tau  =   u_state[(x2grid_size - 2) * x1grid_size + ii].tau;

                } else {
                    u_state[jj * x1grid_size].D    = u_state[jj * x1grid_size + 1].D;
                    u_state[jj * x1grid_size + x1grid_size - 1].D = u_state[jj*x1grid_size + x1grid_size - 2].D;

                    u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 1].S1;
                    u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 2].S1;

                    u_state[jj * x1grid_size + 0].S2                = u_state[jj * x1grid_size + 1].S2;
                    u_state[jj * x1grid_size + x1grid_size - 1].S2  = u_state[jj * x1grid_size + x1grid_size - 2].S2;

                    u_state[jj * x1grid_size + 0].tau               = u_state[jj * x1grid_size + 1].tau;
                    u_state[jj * x1grid_size + x1grid_size - 1].tau = u_state[jj * x1grid_size + x1grid_size - 2].tau;
                }
            }
        }

    } else {
        for (int jj = 0; jj < x2grid_size; jj++){

            // Fix the ghost zones at the radial boundaries
            u_state[jj * x1grid_size +  0].D               = u_state[jj * x1grid_size +  3].D;
            u_state[jj * x1grid_size +  1].D               = u_state[jj * x1grid_size +  2].D;
            u_state[jj * x1grid_size +  x1grid_size - 1].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;
            u_state[jj * x1grid_size +  x1grid_size - 2].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;

            u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 3].S1;
            u_state[jj * x1grid_size + 1].S1               = - u_state[jj * x1grid_size + 2].S1;
            u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;
            u_state[jj * x1grid_size + x1grid_size - 2].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;

            u_state[jj * x1grid_size + 0].S2               = u_state[jj * x1grid_size + 3].S2;
            u_state[jj * x1grid_size + 1].S2               = u_state[jj * x1grid_size + 2].S2;
            u_state[jj * x1grid_size + x1grid_size - 1].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;
            u_state[jj * x1grid_size + x1grid_size - 2].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;

            u_state[jj * x1grid_size + 0].tau               = u_state[jj * x1grid_size + 3].tau;
            u_state[jj * x1grid_size + 1].tau               = u_state[jj * x1grid_size + 2].tau;
            u_state[jj * x1grid_size + x1grid_size - 1].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;
            u_state[jj * x1grid_size + x1grid_size - 2].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;

            // Fix the ghost zones at the angular boundaries
            if (jj < 2){
                for (int ii = 0; ii < x1grid_size; ii++){
                     if (jj == 0){
                        u_state[jj * x1grid_size + ii].D   =   u_state[3 * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[3 * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  = - u_state[3 * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[3 * x1grid_size + ii].tau;
                    } else {
                        u_state[jj * x1grid_size + ii].D    =   u_state[2 * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1   =   u_state[2 * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2   = - u_state[2 * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau  =   u_state[2 * x1grid_size + ii].tau;
                    }
                }
            } else if (jj > x2grid_size - 3) {
                for (int ii = 0; ii < x1grid_size; ii++){
                    if (jj == x2grid_size - 1){
                        u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 4) * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 4) * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  = - u_state[(x2grid_size - 4) * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 4) * x1grid_size + ii].tau;
                    } else {
                        u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 3) * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 3) * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  = - u_state[(x2grid_size - 3) * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 3) * x1grid_size + ii].tau;
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
    to->rho  = from->rho;
    to->v1   = from->v1;
    to->v2   = from->v2;
    to->p    = from->p;

}