# This is where we wrap the C++ code with Cython to take advantage of
# the readability of pythonic coding.
# distutils: language = c++
# Cython interface file for wrapping the object

cimport numpy as np 
import numpy as np 

from cython.operator import dereference
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string
from cpython cimport array 

# c++ interface to cython 
cdef extern from "<array>" namespace "std" nogil :
    ctypedef butt "std::array<double, 2>"

    cdef cppclass total "total_zones":
        pass 

    cdef cppclass three "3":
        pass

    cdef cppclass nzones "_NX*_NY":
        pass
    cdef cppclass array2 "std::array<double, 2>":
        array2() except+
        double& operator[](size_t)

    
    cdef cppclass array[T, U]:
        ctypedef T value_type
        cppclass iterator:
            T& operator*()
            iterator operator++()
            iterator operator--()
            iterator operator+(size_t)
            iterator operator-(size_t)
        bint operator==(iterator)
        bint operator!=(iterator)
        bint operator<(iterator)
        bint operator>(iterator)
        bint operator<=(iterator)
        bint operator>=(iterator)
        T& operator[](size_t)
        array() except +
        array(array&) except +

cdef extern from "ustate.h" namespace "states":
    cdef int NX "_NX"
    cdef int NY "_NY"
    cdef int total_zones "total_zones"
    # cdef double[:] mem
    # cdef array2 *arr = <array2 *>(&mem[0])
    ctypedef array[double, nzones] hello

    cdef cppclass Ustate:
        Ustate() except +
        Ustate(vector[vector[double]], float, float, vector[double], string) except + 
        float theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] cons2prim1D(vector[vector[double]])
        vector[vector [double]] simulate1D(float, float, float, bool, bool, bool, bool)

    cdef cppclass UstateSR:
        UstateSR() except +
        UstateSR(vector[vector[double]], float, float, vector[double], string) except + 
        float theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] cons2prim1D(vector[vector[double]], vector[double])
        vector[vector [double]] simulate1D(vector[double], vector[vector[double]], float, float, float, bool, bool, bool, bool)

    cdef cppclass Ustate2D:
        Ustate2D() except +
        Ustate2D(vector[vector[vector[double]]], float, vector[double], vector[double],
                    double, string) except + 
        float theta, gamma
        bool first_order, periodic
        vector[vector[vector[double]]] state
        vector[vector[vector[double]]] cons2prim2D(vector[vector[vector[double]]])
        vector[vector[vector[double]]] simulate2D(vector[vector[vector[double]]], float, bool, double, bool, bool)

    cdef cppclass UstateSR2D:
        UstateSR2D() except +
        UstateSR2D(vector[vector[double]], int, int, float, vector[double], vector[double],
                    double, string) except + 
        float theta, gamma
        int NX, NY
        bool first_order, periodic
        vector[vector[double]] state 

        vector[vector[double]] simulate2D(vector[double],
                                vector[vector[double]], float, 
                                bool, bool, bool, bool,
                                double)


# Creating the cython wrapper class
cdef class PyState:
    cdef Ustate c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[double]] state, float gamma, float CFL=0.4,
                    vector[double] r = [0], string coord_system = "cartesian"):
        self.c_state = Ustate(state, gamma,CFL, r, coord_system)

    def cons2prim1D(self, vector[vector[double]] u_state):

        return np.array(self.c_state.cons2prim1D(u_state))

    def simulate(self, float tend=0.1, float dt=1.e-4, float theta = 1.5, 
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        bool hllc = False):
        return np.array(self.c_state.simulate1D(tend, dt, theta, first_order, periodic, linspace, hllc))

# Relativisitc 1D Class
cdef class PyStateSR:
    cdef UstateSR c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[double]] state, float gamma, float CFL=0.4,
                    vector[double] r = [0], string coord_system = "cartesian"):
        self.c_state = UstateSR(state, gamma,CFL, r, coord_system)

    def cons2prim1D(self, vector[vector[double]] u_state, vector[double] lorentz_gamma):

        return np.array(self.c_state.cons2prim1D(u_state, lorentz_gamma))

    def simulate(self, float tend=0.1, float dt=1.e-4, float theta = 1.5, 
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        vector[double] lorentz_gamma=[1], sources = None, bool hllc=False):
        if not sources:
            source_terms = np.zeros((3, lorentz_gamma.size()), dtype=np.double)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tend, dt, theta, first_order, periodic, linspace, hllc))
        
            return result

        else:
            source_terms = np.array(sources, dtype=np.double)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tend, dt, theta, first_order, periodic, linspace, hllc))
            
            return result


cdef class PyState2D:
    cdef Ustate2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[vector[double]]] state, float gamma,
                     vector[double] x1, vector[double] x2,
                    double cfl=0.4, string coord_system = "cartesian"):

        self.c_state =  Ustate2D(state, gamma, x1, x2, cfl, coord_system)

    def cons2prim2D(self, vector[vector[vector[double]]] u_state):

        return np.array(self.c_state.cons2prim2D(u_state))
    
    def simulate(self, tend=0.1, bool periodic=False, double dt = 1.e-4, 
                        bool linspace=True,
                        bool hllc = False, vector[vector[vector[double]]] sources = [[[0.0]]]):

        source_terms = np.asarray(sources, dtype = np.double)

        result = np.array(self.c_state.simulate2D(source_terms, tend, periodic, dt, linspace, hllc))

        return result

cdef class PyStateSR2D:
    cdef UstateSR2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  np.ndarray[np.float64_t, ndim=3] state, float gamma=1.333, 
                    vector[double] x1 = [0], vector[double] x2 = [0], 
                    double cfl=0.4, string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = UstateSR2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, float tend=0.1, bool periodic=False, double dt = 1.e-4, bool linspace=True,
                    vector[double] lorentz_gamma = [1.0], first_order=True, vector[vector[double]]sources = [[0.0]],
                    hllc = False):
                
        source_terms = np.asarray(sources, dtype = np.double)

        lorentz_gamma = np.asarray(lorentz_gamma)
        
        result = np.asarray( self.c_state.simulate2D(lorentz_gamma, source_terms, tend, first_order,
                             periodic, linspace, hllc,  dt) )

        result = result.reshape(4, self.c_state.NY, self.c_state.NX)
        return result
    
    


    def __dealloc__(self):
        print("Destroying Object in SR")
