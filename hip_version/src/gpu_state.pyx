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

cdef extern from "config.hpp":
    cdef int FLOAT_PRECISION "FLOAT_PRECISION"
    ctypedef double real 

cdef extern from "euler1D.hpp" namespace "simbi":
    cdef int total_zones "total_zones"

    cdef cppclass Newtonian1D:
        Newtonian1D() except +
        Newtonian1D(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)

cdef extern from "euler2D.hpp" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[real]], int NX, int NY, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        bool first_order, periodic
        int NX, NY
        vector[vector[real]] state
        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            real tstart, 
            real tend, 
            real dt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_interval ,
            string data_directory, 
            bool first_order,
            bool periodic, 
            bool linspace, 
            bool hllc)


cdef extern from "srhydro1D.hpp" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources, 
            real tstart,
            real tend, 
            real dt, 
            real plm_theta, 
            real engine_duration,
            real chkpt_interval, 
            string data_directory,
            bool first_order, 
            bool periodic, 
            bool linspace, 
            bool hllc)

cdef extern from "srhydro2D.hpp" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[real]], int, int, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        int NX, NY
        bool first_order, periodic
        vector[vector[real]] state 

        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)

cdef extern from "srhydro3D.hpp" namespace "simbi":
    cdef cppclass SRHD3D:
        SRHD3D() except +
        SRHD3D(
            vector[vector[real]] state, 
            int NX, 
            int NY,
            int NZ,
            real ad_gamma,
            vector[real] x1, 
            vector[real] x2,
            vector[real] x3,
            real CFL,
            string coord_system) except + 
        real theta, gamma
        int NX, NY, NZ
        bool first_order, periodic
        vector[vector[real]] state 

        vector[vector[real]] simulate3D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)


# Creating the cython wrapper class
cdef class PyState:
    cdef Newtonian1D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real CFL=0.4,
        vector[real] r = [0], 
        string coord_system = "cartesian"):

        self.c_state = Newtonian1D(state, gamma,CFL, r, coord_system)

    def simulate(self, 
        vector[vector[real]] sources,
        real tstart = 0.0,
        real tend = 0.1,
        real dt = 1e-4,
        real plm_theta = 1.5,
        real engine_duration = 10.0,
        real chkpt_interval = 1.0,
        string data_directory = "data/",
        bool first_order = True,
        bool periodic = False,
        bool linspace = True,
        bool hllc = True):

        result = self.c_state.simulate1D(
            sources,
            tstart,
            tend,
            dt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            data_directory,
            first_order,
            periodic,
            linspace,
            hllc)

        return np.asarray(result)


# Relativisitc 1D Class
cdef class PyStateSR:
    cdef SRHD c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real CFL=0.4,
        vector[real] r = [0], 
        string coord_system = "cartesian"):

        self.c_state = SRHD(state, gamma,CFL, r, coord_system)
        

    def simulate(self,
        vector[vector[real]] sources,
        real tstart = 0.0,
        real tend   = 0.1, 
        real dt     = 1e-4, 
        real plm_theta = 1.5, 
        real engine_duration = 10.0,
        real chkpt_interval  = 1.0, 
        string data_directory  = "data/",
        bool first_order = True, 
        bool periodic = False, 
        bool linspace = True, 
        bool hllc     = False):

        result = self.c_state.simulate1D(
            sources, 
            tstart,
            tend, 
            dt, 
            plm_theta, 
            engine_duration,
            chkpt_interval, 
            data_directory,
            first_order, 
            periodic, 
            linspace, 
            hllc)
        
        return np.asarray(result)

cdef class PyState2D:
    cdef Newtonian2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=3] state, 
        real gamma,
        vector[real] x1, 
        vector[real] x2,
        real cfl=0.4, 
        string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
        vector[vector[real]] sources,
        real tstart = 0.0, 
        real tend   = 0.1, 
        real dt     = 1.e-4, 
        real plm_theta = 1.5,
        real engine_duration = 10.0, 
        real chkpt_interval  = 1.0 ,
        string data_directory  = "data/", 
        bool first_order       = True,
        bool periodic          = False, 
        bool linspace          = True, 
        bool hllc              = False):

        result = self.c_state.simulate2D(
            sources,
            tstart, 
            tend, 
            dt, 
            plm_theta,
            engine_duration, 
            chkpt_interval ,
            data_directory, 
            first_order,
            periodic, 
            linspace, 
            hllc)

        result = np.asarray(result)
        result = result.reshape(4, self.c_state.NY, self.c_state.NX)

        return result

cdef class PyStateSR2D:
    cdef SRHD2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  
        np.ndarray[np.double_t, ndim=3] state, 
        real gamma, 
        vector[real] x1, 
        vector[real] x2, 
        real cfl, 
        string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = SRHD2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, 
        vector[vector[real]] sources,
        real tstart,
        real tend,
        real dt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        string data_directory,
        bool first_order,
        bool periodic,
        bool linspace,
        bool hllc):
        
        result = self.c_state.simulate2D(
            sources,
            tstart,
            tend,
            dt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            data_directory,
            first_order,
            periodic,
            linspace,
            hllc)
        result = np.asarray(result)
        result = result.reshape(4, self.c_state.NY, self.c_state.NX)
        return result

cdef class PyStateSR3D:
    cdef SRHD3D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  
        np.ndarray[np.double_t, ndim=4] state, 
        real gamma, 
        vector[real] x1, 
        vector[real] x2, 
        vector[real] x3,
        real cfl, 
        string coord_system = "cartesian"):

        nz, ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = SRHD3D(state_contig, nx, ny, nz, gamma, x1, x2, x3, cfl, coord_system)
    

    def simulate(self, 
        vector[vector[real]] sources,
        real tstart,
        real tend,
        real dt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        string data_directory,
        bool first_order,
        bool periodic,
        bool linspace,
        bool hllc):
        
        result = self.c_state.simulate3D(
            sources,
            tstart,
            tend,
            dt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            data_directory,
            first_order,
            periodic,
            linspace,
            hllc)
        result = np.asarray(result)
        result = result.reshape(5, self.c_state.NZ, self.c_state.NY, self.c_state.NX)
        return result