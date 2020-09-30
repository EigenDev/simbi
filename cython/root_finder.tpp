/**
* Root Finder Template file for a variadic Newton Raphson root Finder
*
*/
#include <cmath>

template <typename T, typename... Args>
double newton_raphson(T x, T (*f)(T, Args... args),  T (*g)(T, Args... args), 
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
      
     
    
    double h = (*f)(x, args...)/(*g)(x, args...);

    int ii = 0;
    while (std::abs(h) >= epsilon){
        h = (*f)(x, args...)/(*g)(x, args...);

        // x[ii + 1] = x[ii] - f(x)/f'(x)
        
        // increase the iteration count
        ii++;
        x = x - h; 
    }

    // std::cout << "Newton Raphson took: " << ii << " iterations" << std::endl;

    return x; 
}