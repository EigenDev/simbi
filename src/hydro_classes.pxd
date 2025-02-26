from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 
from libcpp.pair cimport pair

cdef extern from "core/cython/pyobj_wrapper.hpp":
    cdef cppclass PyObjWrapper:
        PyObjWrapper() except +
        PyObjWrapper(object py_func) except +
        
cdef extern from "build_options.hpp":
    cdef bool col_major "COLUMN_MAJOR"
    ctypedef double real 

cdef extern from "core/types/utility/init_conditions.hpp":
    cdef cppclass InitialConditions:
        real time, checkpoint_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend
        int nx, ny, nz, checkpoint_idx
        bool quirk_smoothing, constant_sources
        vector[vector[real]] sources, bfield
        string data_directory, coord_system, solver
        string x1_cell_spacing, x2_cell_spacing, x3_cell_spacing
        string spatial_order, time_order
        string hydro_source_lib, gravity_source_lib, boundary_source_lib
        pair[real, real] x1bounds, x2bounds, x3bounds
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources

cdef extern from "core/cython/driver.hpp" namespace "simbi":
    cdef cppclass Driver:
        Driver() except +

        vector[vector[real]] run(
            vector[vector[real]] state,
            int dim, 
            string regime, 
            InitialConditions sim_cond,
            PyObjWrapper a,
            PyObjWrapper adot
        ) except +
        