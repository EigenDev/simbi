# distutils: language = c++
# distutils: sources = hydro.cpp

# Cython interface file for wrapping the object
#
#
cimport numpy as np 
import numpy as np 

from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string

# c++ interface to cython 
cdef extern from "ustate.h" namespace "states":
    cdef cppclass Ustate:
        Ustate() except +
        Ustate(vector[vector[double]], float, vector[double], string) except + 
        float theta, gamma, tend, dt
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] cons2prim1D(vector[vector[double]])
        vector[vector [double]] simulate1D(float, float, float, bool, bool, bool)

    cdef cppclass Ustate2D:
        Ustate2D() except +
        Ustate2D(vector[vector[vector[double]]], float) except + 
        float theta, gamma
        bool first_order, periodic
        vector[vector[vector[double]]] state
        vector[vector[vector[double]]] cons2prim2D(vector[vector[vector[double]]])
        vector[vector[vector[double]]] simulate2D(float, bool)
    

# Creating the cython wrapper class
cdef class PyState:
    cdef Ustate*c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[double]] state, float gamma, 
                    vector[double] r = [0], string coord_system = "cartesian"):
        self.c_state = new Ustate(state, gamma, r, coord_system)

    def cons2prim1D(self, vector[vector[double]] u_state):

        return np.array(self.c_state.cons2prim1D(u_state))

    def simulate(self, float tend=0.1, float dt=1.e-4, float theta = 1.5, 
                        bool first_order=True, bool periodic = False, bool linspace = True):
        return np.array(self.c_state.simulate1D(tend, dt, theta, first_order, periodic, linspace))

cdef class PyState2D:
    cdef Ustate2D*c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[vector[double]]] state, float gamma):
        self.c_state = new Ustate2D(state, gamma)

    def cons2prim2D(self, vector[vector[vector[double]]] u_state):

        return np.array(self.c_state.cons2prim2D(u_state))
    
    def simulate(self, tend=0.1, bool periodic=False):

        return np.array(self.c_state.simulate2D(tend, periodic))
