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

cdef extern from "classical_1d.h" namespace "simbi":
    cdef int total_zones "total_zones"

    cdef cppclass Newtonian1D:
        Newtonian1D() except +
        Newtonian1D(vector[vector[double]], float, float, vector[double], string) except + 
        float theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] simulate1D(float, float, float, bool, bool, bool, bool)

cdef extern from "classical_2d.h" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[double]], int NX, int NY, float, vector[double], vector[double],
                    double, string) except + 
        double theta, gamma
        bool first_order, periodic
        int NX, NY
        vector[vector[double]] state
        vector[vector[double]] simulate2D(vector[vector[double]], double, bool, double, bool, bool, double theta)


cdef extern from "srhd_1d.h" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[double]], float, float, vector[double], string) except + 
        float theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] simulate1D(vector[double], vector[vector[double]], float, 
                                            float, float, double ,double, double, string,
                                            bool, bool, bool, bool)

cdef extern from "srhd_2d.h" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[double]], int, int, float, vector[double], vector[double],
                    double, string) except + 
        float theta, gamma
        int NX, NY
        bool first_order, periodic
        vector[vector[double]] state 

        vector[vector[double]] simulate2D(vector[double],
                                vector[vector[double]], float, float,
                                double, double, double,double,
                                string, bool, bool, bool, bool)


# Creating the cython wrapper class
cdef class PyState:
    cdef Newtonian1D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[double]] state, float gamma, float CFL=0.4,
                    vector[double] r = [0], string coord_system = "cartesian"):
        self.c_state = Newtonian1D(state, gamma,CFL, r, coord_system)

    def simulate(self, float tend=0.1, float dt=1.e-4, float theta = 1.5, 
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        bool hllc = False):
                        
        return np.array(self.c_state.simulate1D(tend, dt, theta, first_order, periodic, linspace, hllc))


# Relativisitc 1D Class
cdef class PyStateSR:
    cdef SRHD c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[double]] state, float gamma, float CFL=0.4,
                    vector[double] r = [0], string coord_system = "cartesian"):
        self.c_state = SRHD(state, gamma,CFL, r, coord_system)
        

    def simulate(self, float tstart = 0, float tend=0.1, float dt=1.e-4, double theta = 1.5, 
                        double engine_duration = 10, double chkpt_interval = 0.1, string data_directory = "data/",
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        vector[double] lorentz_gamma=[1], sources = None, bool hllc=False):
        if not sources:
            source_terms = np.zeros((3, lorentz_gamma.size()), dtype=np.double)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tstart, tend, dt, theta, 
                                                        engine_duration, chkpt_interval, data_directory, 
                                                        first_order, periodic, linspace, hllc))
        
            return result

        else:
            source_terms = np.array(sources, dtype=np.double)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tstart, tend, dt, theta,
                                                        engine_duration, chkpt_interval,
                                                        data_directory, first_order, periodic, linspace, hllc))
            
            return result


cdef class PyState2D:
    cdef Newtonian2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] state, float gamma,
                     vector[double] x1, vector[double] x2,
                    double cfl=0.4, string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
                    tend=0.1,  bool periodic=False, 
                    double dt = 1.e-4, bool linspace=True, 
                    bool hllc = False, double theta = 1.5,
                    vector[vector[double]] sources = [[0.0]]):

        source_terms = np.asarray(sources, dtype = np.double)

        result = np.array(self.c_state.simulate2D(source_terms, tend, periodic, dt, linspace, hllc, theta))
        
        result = result.reshape(4, self.c_state.NY, self.c_state.NX)

        return result

cdef class PyStateSR2D:
    cdef SRHD2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  np.ndarray[np.float64_t, ndim=3] state, double gamma=1.333, 
                    vector[double] x1 = [0], vector[double] x2 = [0], 
                    double cfl=0.4, string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = SRHD2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, float tstart = 0.0, 
                       float tend=0.1, 
                       bool periodic=False, 
                       double dt = 1.e-4,
                       double theta = 1.5,
                       double engine_duration = 10,
                       double chkpt_interval = 0.1, 
                       string data_directory = "data/",
                       bool linspace=True,
                       vector[double] lorentz_gamma = [1.0], 
                       first_order=True, 
                       vector[vector[double]]sources = [[0.0]],
                       bool hllc = False
                       ):
                
        source_terms = np.asarray(sources, dtype = np.double)

        lorentz_gamma = np.asarray(lorentz_gamma)
        
        result = np.asarray( self.c_state.simulate2D(lorentz_gamma, 
                                                     source_terms, 
                                                     tstart, tend, 
                                                     dt, theta, 
                                                     engine_duration,
                                                     chkpt_interval,
                                                     data_directory,
                                                     first_order,
                                                     periodic, linspace, 
                                                     hllc) )

        result = result.reshape(4, self.c_state.NY, self.c_state.NX)
        return result
    
    


    def __dealloc__(self):
        print("Destroying Object in SR")
