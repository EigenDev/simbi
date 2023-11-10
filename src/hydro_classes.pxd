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
    # a few cname hacks 
    ctypedef int dim1 "1" 
    ctypedef int dim2 "2"
    ctypedef int dim3 "3"


cdef extern from "common/enums.hpp":
    cdef int FLOAT_PRECISION "FLOAT_PRECISION"

cdef extern from "common/hydro_structs.hpp":
    cdef cppclass InitialConditions:
        real tstart, chkpt_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend
        int nx, ny, nz, chkpt_idx
        bool first_order, quirk_smoothing, constant_sources
        vector[vector[real]] sources, gsource
        vector[bool] object_cells
        string data_directory, coord_system, solver, x1_cell_spacing, x2_cell_spacing, x3_cell_spacing
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources
        vector[real] x1, x2, x3

cdef extern from "hydro/driver.hpp" namespace "simbi":
    ctypedef void* void_ptr
    ctypedef fused null_or_lambda:
        PyObjWrapper
        void_ptr

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
        