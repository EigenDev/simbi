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
        u_state[0].D = u_state[1].D;
        u_state[grid_size - 1].D = u_state[grid_size - 2].D;

        u_state[0].S = u_state[1].S;
        u_state[grid_size - 1].S = u_state[grid_size - 2].S;

        u_state[0].tau = u_state[1].tau;
        u_state[grid_size - 1].tau = u_state[grid_size - 2].tau;
    } else {
        u_state[0].D = u_state[3].D;
        u_state[1].D = u_state[2].D;
        u_state[grid_size - 1].D = u_state[grid_size - 3].D;
        u_state[grid_size - 2].D = u_state[grid_size - 3].D;

        u_state[0].S = - u_state[3].S;
        u_state[1].S = - u_state[2].S;
        u_state[grid_size - 1].S = u_state[grid_size - 3].S;
        u_state[grid_size - 2].S = u_state[grid_size - 3].S;

        u_state[0].tau = u_state[3].tau;
        u_state[1].tau = u_state[2].tau;
        u_state[grid_size - 1].tau = u_state[grid_size - 3].tau;
        u_state[grid_size - 2].tau = u_state[grid_size - 3].tau;
    }
};

//Handle 2D primitive arrays whether SR or Newtonian
template<typename T, typename N>
typename std::enable_if<is_2D_primitive<N>::value>::type
writeToProd(T *from, PrimData *to){
    to->rho  = from->rho;
    to->v1   = from->v1;
    to->v2   = from->v2;
    to->p    = from->p;
}

template<typename T, typename N>
typename std::enable_if<is_1D_primitive<N>::value>::type
writeToProd(T *from, PrimData *to){
    to->rho  = from->rho;
    to->v    = from->v;
    to->p    = from->p;
}

template<typename T , typename N>
typename std::enable_if<is_2D_primitive<N>::value, T>::type
vec2struct(const std::vector<N> &p){
    T sprims;
    size_t nzones = p.size();

    sprims.rho.reserve(nzones);
    sprims.v1.reserve(nzones);
    sprims.v2.reserve(nzones);
    sprims.p.reserve(nzones);
    for (size_t i = 0; i < nzones; i++)
    {
        sprims.rho.push_back(p[i].rho);
        sprims.v1.push_back(p[i].v1);
        sprims.v2.push_back(p[i].v2);
        sprims.p.push_back(p[i].p);
    }
    
    return sprims;
}

template<typename T , typename N>
typename std::enable_if<is_1D_primitive<N>::value, T>::type
vec2struct(const std::vector<N> &p){
    T sprims;
    size_t nzones = p.size();

    sprims.rho.reserve(nzones);
    sprims.v.reserve(nzones);
    sprims.p.reserve(nzones);
    for (size_t i = 0; i < nzones; i++)
    {
        sprims.rho.push_back(p[i].rho);
        sprims.v.push_back(p[i].v);
        sprims.p.push_back(p[i].p);
    }
    
    return sprims;
}