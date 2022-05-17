# This is where we wrap the C++ code with Cython to take advantage of
# the readability of pythonic coding.
# distutils: language = c++
# Cython interface file for wrapping the object

cimport numpy as np 

import numpy as np 

from hydro_classes cimport *
from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 


# Creating the cython wrapper class
cdef class PyState:
    cdef Newtonian1D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real cfl=0.4,
        vector[real] r = [0], 
        string coord_system = "cartesian".encode('ascii')):

        self.c_state = Newtonian1D(state, gamma,cfl, r, coord_system)

    def simulate(self, 
        vector[vector[real]] sources,
        real tstart,
        real tend,
        real dt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        string data_directory,
        string boundary_condition,
        bool first_order,
        bool linspace,
        bool hllc):

        result = self.c_state.simulate1D(
            sources,
            tstart,
            tend,
            dt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            data_directory,
            boundary_condition,
            first_order,
            linspace,
            hllc)

        return np.asarray(result)


# Relativisitc 1D Class
cdef class PyStateSR:
    cdef SRHD c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real cfl=0.4,
        vector[real] r = [0], 
        string coord_system = "cartesian".encode('ascii')):
        self.c_state = SRHD(state, gamma,cfl, r, coord_system)
        
    def simulate(self,
        vector[vector[real]] sources,
        real tstart,
        real tend, 
        real dt, 
        real plm_theta, 
        real engine_duration,
        real chkpt_interval, 
        string data_directory,
        string boundary_condition,
        bool first_order, 
        bool linspace, 
        bool hllc,
        a,
        adot):

        cdef PyObjWrapper a_cpp    = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)
        result = self.c_state.simulate1D(
            sources, 
            tstart,
            tend, 
            dt, 
            plm_theta, 
            engine_duration,
            chkpt_interval, 
            data_directory,
            boundary_condition,
            first_order, 
            linspace, 
            hllc,
            a_cpp,
            adot_cpp)
        
        return np.asarray(result)

cdef class PyState2D:
    cdef Newtonian2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=3] state, 
        real gamma,
        vector[real] x1, 
        vector[real] x2,
        real cfl=0.4, 
        string coord_system = "cartesian".encode('ascii')):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
        vector[vector[real]] sources,
        real tstart, 
        real tend, 
        real dt, 
        real plm_theta,
        real engine_duration, 
        real chkpt_interval,
        string data_directory, 
        string boundary_condition,
        bool first_order,
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
            boundary_condition, 
            first_order,
            linspace, 
            hllc)

        result = np.asarray(result)
        result = result.reshape(5, self.c_state.ny, self.c_state.nx)

        return result

cdef class PyStateSR2D:
    cdef SRHD2D c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self,  
        np.ndarray[np.double_t, ndim=3] state, 
        real gamma, 
        vector[real] x1, 
        vector[real] x2, 
        real cfl, 
        string coord_system = "cartesian".encode('ascii')):


        ny, nx = state[0].shape
        if  col_maj:
            state = np.transpose(state, axes=(0,2,1))
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
        string boundary_condition,
        bool first_order,
        bool linspace,
        bool hllc,
        bool quirk_smoothing):
        
        result = self.c_state.simulate2D(
            sources,
            tstart,
            tend,
            dt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            data_directory,
            boundary_condition,
            first_order,
            linspace,
            hllc,
            quirk_smoothing)

        result = np.asarray(result)
        if col_maj:
            result = result.reshape(5, self.c_state.nx, self.c_state.ny)
            result = np.transpose(result, axes=(0, 2, 1))
        else:
            result = result.reshape(5, self.c_state.ny, self.c_state.nx)
        
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
        string coord_system = "cartesian".encode('ascii')):

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
        string boundary_condition,
        bool first_order,
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
            boundary_condition,
            first_order,
            linspace,
            hllc)
        result = np.asarray(result)
        result = result.reshape(5, self.c_state.nz, self.c_state.ny, self.c_state.nx)
        return result
