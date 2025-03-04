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
        dict sim_info,
        a: callable[[float], float],
        adot: callable[[float], float]
    ):

        cdef InitialConditions sim_cond 

        # Directly assign dictionary values to sim_cond members
        sim_cond.tend                = <double>sim_info['tend']
        sim_cond.time                = <double>sim_info['tstart']
        sim_cond.checkpoint_interval = <double>sim_info['checkpoint_interval']
        sim_cond.dlogt               = <double>sim_info['dlogt']
        sim_cond.plm_theta           = <double>sim_info['plm_theta']
        sim_cond.nx                  = <int>sim_info['nx']
        sim_cond.ny                  = <int>sim_info['ny']
        sim_cond.nz                  = <int>sim_info['nz']
        sim_cond.spatial_order       = <string>sim_info['spatial_order']
        sim_cond.temporal_order      = <string>sim_info['temporal_order']
        sim_cond.x1_spacing          = <string>sim_info['x1_spacing']
        sim_cond.x2_spacing          = <string>sim_info['x2_spacing']
        sim_cond.x3_spacing          = <string>sim_info['x3_spacing']
        sim_cond.data_directory      = <string>sim_info['data_directory']
        sim_cond.coord_system        = <string>sim_info['coord_system']
        sim_cond.solver              = <string>sim_info['solver']
        sim_cond.gamma               = <double>sim_info['gamma']
        sim_cond.x1bounds            = <pair[real, real]>sim_info['x1bounds']
        sim_cond.x2bounds            = <pair[real, real]>sim_info['x2bounds']
        sim_cond.x3bounds            = <pair[real, real]>sim_info['x3bounds']
        sim_cond.cfl                 = <double>sim_info['cfl']
        sim_cond.boundary_conditions = <vector[string]>sim_info['boundary_conditions']
        sim_cond.checkpoint_idx      = <int>sim_info['checkpoint_idx']
        sim_cond.quirk_smoothing     = <bool>sim_info['quirk_smoothing']
        sim_cond.bfield              = <vector[vector[real]]>sim_info["bfield"]
        sim_cond.hydro_source_lib    = <string>sim_info['hydro_source_lib']
        sim_cond.gravity_source_lib  = <string>sim_info['gravity_source_lib']
        sim_cond.boundary_source_lib = <string>sim_info['boundary_source_lib']
        sim_cond.mesh_motion         = <bool>sim_info['mesh_motion']
        sim_cond.homologous          = <bool>sim_info['homologous']

        cdef PyObjWrapper a_cpp = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)

        self.driver_state.run(
            state, 
            sim_info["dimensionality"], 
            sim_info["regime"], 
            sim_cond,
            a_cpp,
            adot_cpp
        )