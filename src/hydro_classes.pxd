# distutils: language = c++

from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 

# adapted from: https://stackoverflow.com/a/39052204/13874039
cdef extern from "util/pyobj_wrapper.hpp":
    cdef cppclass PyObjWrapper:
        PyObjWrapper()
        PyObjWrapper(object) # define a constructor that takes a Python object
                             # note - doesn't match c++ signature - that's fine!

cdef extern from "build_options.hpp":
    cdef bool col_maj "COLUMN_MAJOR"
    ctypedef double real 

cdef extern from "common/hydro_structs.hpp":
    cdef cppclass InitialConditions:
        real tstart, chkpt_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend
        int nx, ny, nz, chkpt_idx
        bool quirk_smoothing, constant_sources
        vector[vector[real]] sources, gsources, bfield
        vector[bool] object_cells
        string data_directory, coord_system, solver
        string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing
        string spatial_order, time_order
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources
        vector[real] x1, x2, x3

cdef extern from "hydro/driver.hpp" namespace "simbi":
    cdef cppclass Driver:
        Driver() except +

        vector[vector[real]] run(
            vector[vector[real]] state,
            int dim, 
            string regime, 
            InitialConditions sim_cond,
            PyObjWrapper a,
            PyObjWrapper adot,
            PyObjWrapper dens_lambda,
            PyObjWrapper mom1_lambda,
            PyObjWrapper mom2_lambda,
            PyObjWrapper mom3_lambda,
            PyObjWrapper enrg_lambda
        ) except +
        