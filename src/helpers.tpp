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

template <typename T>
std::vector<double> calc_lorentz_gamma(T &v){
    int vsize = v.size();
    std::vector<double> W(vsize); 

    for (int ii=0; ii < vsize; ii++){
        W[ii] = 1/sqrt(1 - v[ii]*v[ii]);
    }

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
void toWritePrim(T *from, PrimData *to)
{
    to->rho  = from->rho;
    to->v1   = from->v1;
    to->v2   = from->v2;
    to->p    = from->p;
}