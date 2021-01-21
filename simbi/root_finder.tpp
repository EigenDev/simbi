/**
* Root Finder Template file for a variadic Newton Raphson root Finder
*
*/
#include <cmath>

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

    
    double x, h;
    int ii = 0;
    int maximum_iteration = 50;
    do {
        x = x1;
        h = (*f)(x, args...)/(*g)(x, args...);

        x1 = x - h;

        ii++;

        
        // if (ii > maximum_iteration){
        //     std::cout << "\n Not Convergent" << std::endl;
        //     exit(EXIT_FAILURE);
        // }
        

    } while(std::abs(x1 - x) >= epsilon);
    
    
    
    // std::cout << "Newton Raphson took: " << ii << " iterations" << std::endl;

    return x1; 
}