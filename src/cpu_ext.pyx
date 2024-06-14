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
        dens_lambda = None,
        mom1_lambda = None,
        mom2_lambda = None,
        mom3_lambda = None,
        enrg_lambda = None,
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
        sim_cond.sources          = sim_info['sources']
        sim_cond.gsources         = sim_info['gsource']
        sim_cond.data_directory   = sim_info['data_directory']
        sim_cond.coord_system     = sim_info['coord_system']
        sim_cond.solver           = sim_info['solver']
        sim_cond.gamma            = sim_info['gamma']
        sim_cond.x1               = sim_info['x1']
        sim_cond.boundary_sources = sim_info['boundary_sources']
        sim_cond.cfl              = sim_info['cfl']
        sim_cond.boundary_conditions = sim_info['boundary_conditions']
        sim_cond.chkpt_idx           = sim_info['chkpt_idx']
        sim_cond.constant_sources    = sim_info['constant_sources']
        sim_cond.quirk_smoothing     = sim_info['quirk_smoothing']
        sim_cond.bfield              = sim_info["bfield"]

        if dim > 1:
            sim_cond.x2 = sim_info['x2']
        if dim > 2:
            sim_cond.x3 = sim_info['x3']

        # convert python lambdas to cpp lambdas
        cdef PyObjWrapper a_cpp    = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)
        cdef PyObjWrapper d_cpp    = PyObjWrapper(dens_lambda) if dens_lambda else PyObjWrapper()
        cdef PyObjWrapper m1_cpp   = PyObjWrapper(mom1_lambda) if mom1_lambda else PyObjWrapper()
        cdef PyObjWrapper m2_cpp   = PyObjWrapper(mom2_lambda) if mom2_lambda else PyObjWrapper()
        cdef PyObjWrapper m3_cpp   = PyObjWrapper(mom3_lambda) if mom3_lambda else PyObjWrapper()
        cdef PyObjWrapper e_cpp    = PyObjWrapper(enrg_lambda) if enrg_lambda else PyObjWrapper()

        self.driver_state.run(
            state, 
            dim, 
            regime, 
            sim_cond,
            a_cpp,
            adot_cpp,
            d_cpp,
            m1_cpp,
            m2_cpp,
            m3_cpp,
            e_cpp
        )