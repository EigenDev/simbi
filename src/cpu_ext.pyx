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

cdef class SimState:
    cdef Driver driver_state

    def __cinit__(self):
        self.driver_state = Driver()

    def run(
        self,
        *,
        np.ndarray[np.double_t, ndim=2] state,
        int dim,
        string regime,
        dict sim_info,
        a,
        adot,
        boundary_sources = [None],
        hydro_sources = [None],
        gravity_sources = [None],
    ):
        cdef InitialConditions sim_cond 
        sim_cond.tend             = sim_info['tend']
        sim_cond.tstart           = sim_info['tstart']
        sim_cond.chkpt_interval   = sim_info['chkpt_interval']
        sim_cond.dlogt            = sim_info['dlogt']
        sim_cond.plm_theta        = sim_info['plm_theta']
        sim_cond.engine_duration  = sim_info['engine_duration']
        sim_cond.nx               = sim_info['nx']
        sim_cond.ny               = sim_info['ny']
        sim_cond.nz               = sim_info['nz']
        sim_cond.spatial_order    = sim_info['spatial_order']
        sim_cond.time_order       = sim_info['time_order']
        sim_cond.x1_cell_spacing  = sim_info['x1_cell_spacing']
        sim_cond.x2_cell_spacing  = sim_info['x2_cell_spacing']
        sim_cond.x3_cell_spacing  = sim_info['x3_cell_spacing']
        sim_cond.object_cells     = sim_info['object_cells']
        sim_cond.data_directory   = sim_info['data_directory']
        sim_cond.coord_system     = sim_info['coord_system']
        sim_cond.solver           = sim_info['solver']
        sim_cond.gamma            = sim_info['gamma']
        sim_cond.x1               = sim_info['x1']
        sim_cond.cfl              = sim_info['cfl']
        sim_cond.boundary_conditions = sim_info['boundary_conditions']
        sim_cond.chkpt_idx           = sim_info['chkpt_idx']
        sim_cond.constant_sources    = sim_info['constant_sources']
        sim_cond.quirk_smoothing     = sim_info['quirk_smoothing']
        sim_cond.bfield              = sim_info["bfield"]
        mhd: bool = False 
        if sim_info["bfield"] is not None:
            mhd = True
            nvar = 9

        if dim > 1:
            sim_cond.x2 = sim_info['x2']
        if dim > 2:
            sim_cond.x3 = sim_info['x3']

        # convert python lambdas to cpp lambdas
        cdef PyObjWrapper a_cpp    = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)

        # push the vector of lambdas into a c++ compliant vector of functors
        cdef vector[PyObjWrapper] bsource_vec 
        cdef vector[PyObjWrapper] hsource_vec
        cdef vector[PyObjWrapper] gsource_vec

        for qq in boundary_sources:
            bsource_vec.push_back(PyObjWrapper(qq) if qq else PyObjWrapper())
        
        for qq in boundary_sources:
            hsource_vec.push_back(PyObjWrapper(qq) if qq else PyObjWrapper())
        
        for qq in gravity_sources:
            gsource_vec.push_back(PyObjWrapper(qq) if qq else PyObjWrapper())

        self.driver_state.run(
            state, 
            dim, 
            regime, 
            sim_cond,
            a_cpp,
            adot_cpp,
            bsource_vec,
            hsource_vec,
            gsource_vec
        )