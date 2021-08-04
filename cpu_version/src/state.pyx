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
        Newtonian1D(vector[vector[double]], double, double, vector[double], string) except + 
        double theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] simulate1D(
            vector[vector[double]] sources,
            double tstart,
            double tend,
            double dt,
            double plm_theta,
            double engine_duration,
            double chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)

cdef extern from "classical_2d.h" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[double]], int NX, int NY, double, vector[double], vector[double],
                    double, string) except + 
        double theta, gamma
        bool first_order, periodic
        int NX, NY
        vector[vector[double]] state
        vector[vector[double]] simulate2D(
            vector[vector[double]] sources,
            double tstart, 
            double tend, 
            double dt, 
            double plm_theta,
            double engine_duration, 
            double chkpt_interval ,
            string data_directory, 
            bool first_order,
            bool periodic, 
            bool linspace, 
            bool hllc)


cdef extern from "srhd_1d.h" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[double]], double, double, vector[double], string) except + 
        double theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[double] r
        vector[vector[double]] state
        vector[vector [double]] simulate1D(
            vector[vector[double]] sources, 
            double tstart,
            double tend, 
            double dt, 
            double plm_theta, 
            double engine_duration,
            double chkpt_interval, 
            string data_directory,
            bool first_order, 
            bool periodic, 
            bool linspace, 
            bool hllc)

cdef extern from "srhd_2d.h" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[double]], int, int, double, vector[double], vector[double],
                    double, string) except + 
        double theta, gamma
        int NX, NY
        bool first_order, periodic
        vector[vector[double]] state 

        vector[vector[double]] simulate2D(
            vector[vector[double]] sources,
            double tstart,
            double tend,
            double dt,
            double plm_theta,
            double engine_duration,
            double chkpt_interval,
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
        double gamma, 
        double CFL=0.4,
        vector[double] r = [0], 
        string coord_system = "cartesian"):

        self.c_state = Newtonian1D(state, gamma,CFL, r, coord_system)

    def simulate(self, 
        vector[vector[double]] sources,
        double tstart = 0.0,
        double tend = 0.1,
        double dt = 1e-4,
        double plm_theta = 1.5,
        double engine_duration = 10.0,
        double chkpt_interval = 1.0,
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
        double gamma, 
        double CFL=0.4,
        vector[double] r = [0], 
        string coord_system = "cartesian"):

        self.c_state = SRHD(state, gamma,CFL, r, coord_system)
        

    def simulate(self,
        vector[vector[double]] sources,
        double tstart = 0.0,
        double tend   = 0.1, 
        double dt     = 1e-4, 
        double plm_theta = 1.5, 
        double engine_duration = 10.0,
        double chkpt_interval  = 1.0, 
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
        double gamma,
        vector[double] x1, 
        vector[double] x2,
        double cfl=0.4, 
        string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
        vector[vector[double]] sources,
        double tstart = 0.0, 
        double tend   = 0.1, 
        double dt     = 1.e-4, 
        double plm_theta = 1.5,
        double engine_duration = 10.0, 
        double chkpt_interval  = 1.0 ,
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
        double gamma, 
        vector[double] x1, 
        vector[double] x2, 
        double cfl, 
        string coord_system = "cartesian"):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)

        self.c_state = SRHD2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, 
        vector[vector[double]] sources,
        double tstart,
        double tend,
        double dt,
        double plm_theta,
        double engine_duration,
        double chkpt_interval,
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
