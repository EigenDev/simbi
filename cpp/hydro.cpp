/* 
* C++ Library to perform extensive hydro calculations
* to be later wrapped and plotted in Python
* Marcus DuPont
* New York University
* 07/15/2020
* Compressible Hydro Simulation
*/

#include <iostream>
using namespace std;



struct arrWrap1D {
    float arr[3];
};

struct arrWrap1D cons(float rho, float mom, float energy) {
        struct arrWrap1D u;

        u.arr[0] = rho;
        u.arr[1] = mom;
        u.arr[2] = energy;
        // cout << "U looks like: " << u.arr[0] << u.arr[1] << u.arr[2];
        return u;
};

void u_dot(int state, bool first_order = true, float theta = 1.5, bool periodic=false){

    if (periodic == true){ 
        printf("Hello \n");
        };
};
int main()
{
    u_dot(10, true, 1.5,  true);
    float rho, mom, energy;

    cout << "User-Defined Conserved Variable" << endl;
    cout << "Enter The Density: " << flush;
    cin >> rho;

    cout << "Enter The Momentum: " << flush;
    cin >> mom;

    cout << "Enter The Energy: " << flush;
    cin >> energy;

    struct arrWrap1D u = cons(rho, mom, energy);
    std:: cout << "Density: " << u.arr[0] << " Momentum: " << u.arr[1] << " Energy: " << u.arr[2];
    return 0;
}