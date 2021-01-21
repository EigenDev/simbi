/*
* Relativisty Test
*/

#include "helper_functions.h"
#include "helper_functions.cpp"  
#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>

using namespace std;


vector<double> cons2prim(float gamma, double D, double S, double tau, double W){
    vector<double> prims(3);
    double pmin = abs(S - tau - D);

    double p_zero = newton_raphson(pmin, pressure_func, dfdp, 1.e-6, D, tau, W, gamma, S);

    double v = S/(tau + p_zero + D);

    double Wnew = calc_lorentz_gamma(v);

    double rho = D/Wnew;

    prims[0] = rho;
    prims[1] = p_zero;
    prims[2] = v;

    return prims;

}



template <typename T>
T base(T x){
    return x;
}

template <typename T>
T f1(T x, T c){
    return x*c;
}


template <typename T>
T f2(T x, T c, T d){
    return x*x*x*c + d;
}

template <typename T>
T df2dx(T x, T c, T d){
    return 3*x*x*c;
}

template<typename T, typename... Args>
double func(T t, T (*f)(T, Args... args),  Args... args) // recursive variadic function
{
    return f(t, args...);
}


int main()
{
    double energy = 3.0;
    float gamma = 1.4;
    double rho = 1.0;
    double v = 0.5;
    double W = calc_lorentz_gamma(v);
    double p = 0.8;
    double h = calc_enthalpy(gamma, rho, p);
    cout << "P Init: " << p << endl;


    double D = rho*W;
    double S = W*W*rho*v*h;
    double tau = W*W*rho*h - p - rho*W;

    cout << "D: " << D << ", " << "S: " << S << ", " << "Tau: " << tau << endl;

    vector<double> prims = cons2prim(gamma, D, S, tau*1.01, W);
    cout << prims[0] << "\t";
    cout << prims[1] << "\t";
    cout << prims[2] << flush;

    return 0;

} 