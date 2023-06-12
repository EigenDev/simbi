# Cython file to expose key hydro classes to Python while hiding
# many of the internal functions expressed in the C++ implementation.
# This is a key file which has a soft symlink connected it to it since 
# in Cython the extension name and file name need to match, but the gpu
# implementation is identical for the cpu / gpu extensions, so instead of 
# continually maintaining two pieces of identical code, we create a symlink 
# instead. 
#
# Marcus DuPont 
# New York University 
# Update on: 2022 / 12 / 06
#

cimport numpy as np 
import numpy as np 
import sys
from hydro_classes cimport *
from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 

# Creating the cython wrapper class
cdef class PyState:
    cdef Newtonian1D *c_state             # hold a c++ instance that we're wrapping           
    def __init__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real cfl=0.4,
        vector[real] x1 = [0], 
        string coord_system = "cartesian".encode('ascii')):
        self.c_state =  new Newtonian1D(state, gamma,cfl, x1, coord_system)

    def simulate(self, 
        vector[vector[real]] sources,
        real tstart,
        real tend,
        real dlogt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        int  chkpt_idx,
        string data_directory,
        vector[string] boundary_conditions,
        bool first_order,
        bool linspace,
        bool hllc,
        bool constant_sources,
        vector[vector[real]] boundary_sources):

        result = self.c_state.simulate1D(
            sources,
            tstart,
            tend,
            dlogt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            chkpt_idx,
            data_directory,
            boundary_conditions,
            first_order,
            linspace,
            hllc,
            constant_sources,
            boundary_sources)

        return np.asanyarray(result)

    def __dealloc__(self):
        del self.c_state

# Relativisitc 1D Class
cdef class PyStateSR:
    cdef SRHD *c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=2] state, 
        real gamma, 
        real cfl=0.4,
        vector[real] x1 = [0], 
        string coord_system = "cartesian".encode('ascii')):
        self.c_state = new SRHD(state, gamma,cfl, x1, coord_system)
        
    def simulate(self,
        vector[vector[real]] sources,
        vector[real] gravity_sources,
        real tstart,
        real tend, 
        real dlogt, 
        real plm_theta, 
        real engine_duration,
        real chkpt_interval, 
        int  chkpt_idx,
        string data_directory,
        vector[string] boundary_conditions,
        bool first_order, 
        bool linspace, 
        bool hllc,
        bool constant_sources,
        vector[vector[real]] boundary_sources,
        a,
        adot,
        d_outer = None,
        s_outer = None,
        e_outer = None):

        cdef PyObjWrapper a_cpp    = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)
        cdef PyObjWrapper d_cpp = PyObjWrapper(d_outer)
        cdef PyObjWrapper s_cpp = PyObjWrapper(s_outer)
        cdef PyObjWrapper e_cpp = PyObjWrapper(e_outer)

        if d_outer and s_outer and e_outer:
            result = self.c_state.simulate1D(
                sources, 
                gravity_sources,
                tstart,
                tend, 
                dlogt, 
                plm_theta, 
                engine_duration,
                chkpt_interval, 
                chkpt_idx,
                data_directory,
                boundary_conditions,
                first_order, 
                linspace, 
                hllc,
                constant_sources,
                boundary_sources,
                a_cpp,
                adot_cpp,
                d_cpp,
                s_cpp,
                e_cpp)
        else:
            result = self.c_state.simulate1D(
                sources, 
                gravity_sources,
                tstart,
                tend, 
                dlogt, 
                plm_theta, 
                engine_duration,
                chkpt_interval, 
                chkpt_idx,
                data_directory,
                boundary_conditions,
                first_order, 
                linspace, 
                hllc,
                constant_sources,
                boundary_sources,
                a_cpp,
                adot_cpp)
            
        return np.asanyarray(result)

    def __dealloc__(self):
        del self.c_state
cdef class PyState2D:
    cdef Newtonian2D *c_state             # hold a c++ instance that we're wrapping           

    def __cinit__(self, 
        np.ndarray[np.double_t, ndim=3] state, 
        real gamma,
        vector[real] x1, 
        vector[real] x2,
        real cfl=0.4, 
        string coord_system = "cartesian".encode('ascii')):

        ny, nx = state[0].shape
        state_contig = state.reshape(state.shape[0], -1)
        self.c_state =  new Newtonian2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    
    def simulate(self, 
        vector[vector[real]] sources,
        real tstart, 
        real tend, 
        real dlogt, 
        real plm_theta,
        real engine_duration, 
        real chkpt_interval,
        int  chkpt_idx,
        string data_directory, 
        vector[string] boundary_conditions,
        bool first_order,
        bool linspace, 
        bool hllc,
        bool constant_sources,
        vector[vector[real]] boundary_sources):

        result = self.c_state.simulate2D(
            sources,
            tstart, 
            tend, 
            dlogt, 
            plm_theta,
            engine_duration, 
            chkpt_interval,
            chkpt_idx,
            data_directory,
            boundary_conditions, 
            first_order,
            linspace, 
            hllc,
            constant_sources,
            boundary_sources)
            
        result = np.asanyarray(result)
        result = result.reshape(5, self.c_state.ny, self.c_state.nx)

        return result

    def __dealloc__(self):
        del self.c_state

cdef class PyStateSR2D:
    cdef SRHD2D *c_state             # hold a c++ instance that we're wrapping           

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
        self.c_state = new SRHD2D(state_contig, nx, ny, gamma, x1, x2, cfl, coord_system)
    

    def simulate(self, 
        vector[vector[real]] sources,
        np.ndarray[bool, ndim=2] object_cells,
        real tstart,
        real tend,
        real dlogt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        int  chkpt_idx,
        string data_directory,
        vector[string] boundary_conditions,
        bool first_order,
        bool linspace,
        bool hllc,
        bool quirk_smoothing,
        bool constant_sources,
        vector[vector[real]] boundary_sources,
        a,
        adot,
        d_outer  = None,
        s1_outer = None,
        s2_outer = None,
        e_outer  = None):
        
        cdef PyObjWrapper a_cpp    = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)
        cdef PyObjWrapper d_cpp    = PyObjWrapper(d_outer)
        cdef PyObjWrapper s1_cpp   = PyObjWrapper(s1_outer)
        cdef PyObjWrapper s2_cpp   = PyObjWrapper(s2_outer)
        cdef PyObjWrapper e_cpp    = PyObjWrapper(e_outer)

        object_contig = object_cells.flatten()
        if d_outer and s1_outer and s2_outer and e_outer:
            result = self.c_state.simulate2D(
                sources,
                object_contig,
                tstart,
                tend,
                dlogt,
                plm_theta,
                engine_duration,
                chkpt_interval,
                chkpt_idx,
                data_directory,
                boundary_conditions,
                first_order,
                linspace,
                hllc,
                quirk_smoothing,
                constant_sources,
                boundary_sources,
                a_cpp,
                adot_cpp,
                d_cpp,
                s1_cpp,
                s2_cpp,
                e_cpp)
        else:
            result = self.c_state.simulate2D(
                sources,
                object_contig,
                tstart,
                tend,
                dlogt,
                plm_theta,
                engine_duration,
                chkpt_interval,
                chkpt_idx,
                data_directory,
                boundary_conditions,
                first_order,
                linspace,
                hllc,
                quirk_smoothing,
                constant_sources,
                boundary_sources,
                a_cpp,
                adot_cpp)

        result = np.asanyarray(result)
        if col_maj:
            result = result.reshape(5, self.c_state.nx, self.c_state.ny)
            result = np.transpose(result, axes=(0, 2, 1))
        else:
            result = result.reshape(5, self.c_state.ny, self.c_state.nx)
        
        return result

    def __dealloc__(self):
        del self.c_state

cdef class PyStateSR3D:
    cdef SRHD3D *c_state             # hold a c++ instance that we're wrapping           

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

        self.c_state = new SRHD3D(state_contig, nx, ny, nz, gamma, x1, x2, x3, cfl, coord_system)
    

    def simulate(self, 
        vector[vector[real]] sources,
        np.ndarray[bool, ndim=3] object_cells,
        real tstart,
        real tend,
        real dlogt,
        real plm_theta,
        real engine_duration,
        real chkpt_interval,
        int  chkpt_idx,
        string data_directory,
        vector[string] boundary_conditions,
        bool first_order,
        bool linspace,
        bool hllc,
        bool constant_sources,
        vector[vector[real]] boundary_sources):
        
        object_contig = object_cells.flatten()
        result = self.c_state.simulate3D(
            sources,
            object_contig,
            tstart,
            tend,
            dlogt,
            plm_theta,
            engine_duration,
            chkpt_interval,
            chkpt_idx,
            data_directory,
            boundary_conditions,
            first_order,
            linspace,
            hllc,
            constant_sources,
            boundary_sources)
        result = np.asanyarray(result)
        result = result.reshape(5, self.c_state.nz, self.c_state.ny, self.c_state.nx)
        return result

    def __dealloc__(self):
        del self.c_state