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

# DEF FLOAT_PRECISION = 0
# IF FLOAT_PRECISION == 1:
#     ctypedef np.float32 real
# ELSE:
#     ctypedef double real


cdef extern from "config.hpp":
    cdef bool _FLOAT_PRECISION "FLOAT_PRECISION"
    ctypedef float real

# if (_FLOAT_PRECISION):
#     ctypedef np.float32 real 
# else:
#     ctypedef double real

cdef extern from "euler1D.hpp" namespace "simbi":
    cdef int total_zones "total_zones"

    cdef cppclass Newtonian1D:
        Newtonian1D() except +
        Newtonian1D(vector[vector[real]], float, float, vector[real], string) except + 
        float plm_theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(float, float, float, bool, bool, bool, bool)

cdef extern from "euler2D.hpp" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[real]], int NX, int NY, float, vector[real], vector[real],
                    real, string) except + 
        real plm_theta, gamma
        bool first_order, periodic
        int NX, NY
        vector[vector[real]] state
        vector[vector[real]] simulate2D(vector[vector[real]], real, bool, real, bool, bool, real plm_theta)


cdef extern from "srhydro1D.hpp" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[real]], float, float, vector[real], string) except + 
        float plm_theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(vector[real], vector[vector[real]], float, 
                                            float, float, real ,real, real, string,
                                            bool, bool, bool, bool)

cdef extern from "srhydro2D.hpp" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[real]], int, int, float, vector[real], vector[real],
                    real, string) except + 
        float plm_theta, gamma
        int NX, NY
        bool first_order, periodic
        vector[vector[real]] state 

        vector[vector[real]] simulate2D(vector[real],
                                vector[vector[real]], float, float,
                                real, real, real,real,
                                string, bool, bool, bool, bool)


# Creating the cython wrapper class
cdef class PyState:
    cdef Newtonian1D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[real]] state, float gamma, float CFL=0.4,
                    vector[real] r = [0], string coord_system = "cartesian"):
        self.c_state = Newtonian1D(state, gamma,CFL, r, coord_system)

    def simulate(self, float tend=0.1, float dt=1.e-4, float plm_theta = 1.5, 
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        bool hllc = False):
                        
        return np.array(self.c_state.simulate1D(tend, dt, plm_theta, first_order, periodic, linspace, hllc))


# Relativisitc 1D Class
cdef class PyStateSR:
    cdef SRHD c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, vector[vector[real]] state, float gamma, float CFL=0.4,
                    vector[real] r = [0], string coord_system = "cartesian"):
        self.c_state = SRHD(state, gamma,CFL, r, coord_system)
        

    def simulate(self, float tstart = 0, float tend=0.1, float dt=1.e-4, real plm_theta = 1.5, 
                        real engine_duration = 10, real chkpt_interval = 0.1, string data_directory = "data/",
                        bool first_order=True, bool periodic = False, bool linspace = True,
                        vector[real] lorentz_gamma=[1], sources = None, bool hllc=False):
        if not sources:
            source_terms = np.zeros((3, lorentz_gamma.size()), dtype=float)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tstart, tend, dt, plm_theta, 
                                                        engine_duration, chkpt_interval, data_directory, 
                                                        first_order, periodic, linspace, hllc))
        
            return result

        else:
            source_terms = np.array(sources, dtype=float)
            result = np.array(self.c_state.simulate1D(lorentz_gamma, source_terms, tstart, tend, dt, plm_theta,
                                                        engine_duration, chkpt_interval,
                                                        data_directory, first_order, periodic, linspace, hllc))
            
            return result


cdef class PyState2D:
    cdef Newtonian2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, np.ndarray[np.float64_t, ndim=3] state, float gamma,
                     vector[real] x1, vector[real] x2,
                    real cfl=0.4, string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
                    tend=0.1,  bool periodic=False, 
                    real dt = 1.e-4, bool linspace=True, 
                    bool hllc = False, real plm_theta = 1.5,
                    vector[vector[real]] sources = [[0.0]]):

        source_terms = np.asarray(sources, dtype = float)

        result = np.array(self.c_state.simulate2D(source_terms, tend, periodic, dt, linspace, hllc, plm_theta))
        
        result = result.reshape(4, self.c_state.NY, self.c_state.NX)

        return result

cdef class PyStateSR2D:
    cdef SRHD2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  np.ndarray[np.float64_t, ndim=3] state, 
        real gamma=1.333, 
        vector[real] x1 = [0], 
        vector[real] x2 = [0], 
        real cfl=0.4, 
        string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = SRHD2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, float tstart = 0.0, 
                       float tend=0.1, 
                       bool periodic=False, 
                       real dt = 1.e-4,
                       real plm_theta = 1.5,
                       real engine_duration = 10,
                       real chkpt_interval = 0.1, 
                       string data_directory = "data/",
                       bool linspace=True,
                       vector[real] lorentz_gamma = [1.0], 
                       first_order=True, 
                       vector[vector[real]]sources = [[0.0]],
                       bool hllc = False
                       ):
                
        source_terms = np.asarray(sources, dtype = float)

        lorentz_gamma = np.asarray(lorentz_gamma)
        
        result = np.asarray( self.c_state.simulate2D(lorentz_gamma, 
                                                     source_terms, 
                                                     tstart, tend, 
                                                     dt, plm_theta, 
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
