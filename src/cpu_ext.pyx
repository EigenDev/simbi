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
        a: callable[[float], float],
        adot: callable[[float], float],
        boundary_sources=None,
        hydro_sources=None,
        gravity_sources=None,
    ):
        if boundary_sources is None:
            boundary_sources = []
        if hydro_sources is None:
            hydro_sources = []
        if gravity_sources is None:
            gravity_sources = []

        cdef InitialConditions sim_cond 

        # Directly assign dictionary values to sim_cond members
        sim_cond.tend             = <double>sim_info.get('tend', 0.0)
        sim_cond.time             = <double>sim_info.get('tstart', 0.0)
        sim_cond.checkpoint_interval   = <double>sim_info.get('checkpoint_interval', 0.0)
        sim_cond.dlogt            = <double>sim_info.get('dlogt', 0.0)
        sim_cond.plm_theta        = <double>sim_info.get('plm_theta', 0.0)
        sim_cond.engine_duration  = <double>sim_info.get('engine_duration', 0.0)
        sim_cond.nx               = <int>sim_info.get('nx', 0)
        sim_cond.ny               = <int>sim_info.get('ny', 0)
        sim_cond.nz               = <int>sim_info.get('nz', 0)
        sim_cond.spatial_order    = <string>sim_info.get('spatial_order', 0)
        sim_cond.time_order       = <string>sim_info.get('time_order', 0)
        sim_cond.x1_cell_spacing  = <string>sim_info.get('x1_cell_spacing', 0.0)
        sim_cond.x2_cell_spacing  = <string>sim_info.get('x2_cell_spacing', 0.0)
        sim_cond.x3_cell_spacing  = <string>sim_info.get('x3_cell_spacing', 0.0)
        sim_cond.data_directory   = <string>sim_info.get('data_directory', "".encode("utf-8"))
        sim_cond.coord_system     = <string>sim_info.get('coord_system', "".encode("utf-8"))
        sim_cond.solver           = <string>sim_info.get('solver', "".encode("utf-8"))
        sim_cond.gamma            = <double>sim_info.get('gamma', 0.0)
        sim_cond.x1bounds         = <pair[real, real]>sim_info.get('x1bounds', [0.0, 1.0])
        sim_cond.x2bounds         = <pair[real, real]>sim_info.get('x2bounds', [0.0, 1.0])
        sim_cond.x3bounds         = <pair[real, real]>sim_info.get('x3bounds', [0.0, 1.0])
        sim_cond.cfl              = <double>sim_info.get('cfl', 0.0)
        sim_cond.boundary_conditions = <vector[string]>sim_info.get('boundary_conditions', "".encode("utf-8"))
        sim_cond.checkpoint_idx           = <int>sim_info.get('checkpoint_idx', 0)
        sim_cond.quirk_smoothing     = <bool>sim_info.get('quirk_smoothing', False)
        sim_cond.bfield              = <vector[vector[real]]>sim_info.get("bfield", [[0.0], [0.0], [0.0]])
        sim_cond.hydro_source_lib    = <string>sim_info.get('hydro_source_lib', "".encode("utf-8"))
        sim_cond.gravity_source_lib  = <string>sim_info.get('gravity_source_lib', "".encode("utf-8"))
        sim_cond.boundary_source_lib = <string>sim_info.get('boundary_source_lib', "".encode("utf-8"))
        # if dim > 1:
        #     sim_cond.x2 = <vector[real]>sim_info.get('x2', [0.0, 1.0])
        # if dim > 2:
        #     sim_cond.x3 = <vector[real]>sim_info.get('x3', [0.0, 1.0])

        cdef PyObjWrapper a_cpp = PyObjWrapper(a)
        cdef PyObjWrapper adot_cpp = PyObjWrapper(adot)

        self.driver_state.run(
            state, 
            dim, 
            regime, 
            sim_cond,
            a_cpp,
            adot_cpp
        )