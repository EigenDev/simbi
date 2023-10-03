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
    # a few cname hacks 
    ctypedef int dim1 "1" 
    ctypedef int dim2 "2"
    ctypedef int dim3 "3"
    ctypedef int build_mode "BuildPlatform"
    cdef bool col_maj "COLUMN_MAJOR"

cdef extern from "common/enums.hpp":
    cdef int FLOAT_PRECISION "FLOAT_PRECISION"
    ctypedef double real 

cdef extern from "common/hydro_structs.hpp":
    cdef cppclass InitialConditions:
        real tstart, chkpt_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend
        int nx, ny, nz
        bool first_order, linspace
        vector[vector[real]] sources, gsource
        vector[bool] object_cells
        string data_directory, coord_system, solver
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources
        vector[real] x1, x2, x3

cdef extern from "hydro/euler1D.hpp" namespace "simbi":
    cdef cppclass Newtonian1D:
        Newtonian1D() except + 
        Newtonian1D(vector[vector[real]], real, real, vector[real], string) except +  
        vector[vector [real]] simulate1D(
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
            string solver,
            bool constant_sources,
            vector[vector[real]] boundary_sources) except +

cdef extern from "hydro/euler2D.hpp" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[real]], int nx, int ny, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        bool first_order
        int nx, ny
        vector[vector[real]] state
        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            real tstart, 
            real tend, 
            real dlogt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_interval ,
            int  chkpt_idx,
            string data_directory, 
            vector[string] boundary_conditions,
            bool first_order,
            bool linspace, 
            string solver,
            bool constant_sources,
            vector[vector[real]] boundary_sources) except +


cdef extern from "hydro/srhydro1D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD1D:
        SRHD1D() except +
        SRHD1D(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dlogt, cfl
        bool first_order, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources, 
            vector[real] gsources,
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
            string solver,
            bool constant_sources,
            vector[vector[real]] boundary_sources,
            PyObjWrapper a,
            PyObjWrapper adot) except + 
            
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources, 
            vector[real] gsources,
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
            string solver,
            bool constant_sources,
            vector[vector[real]] boundary_sources,
            PyObjWrapper a,
            PyObjWrapper adot,
            PyObjWrapper d_outer,
            PyObjWrapper s_outer,
            PyObjWrapper e_outer) except + 

cdef extern from "hydro/srhydro2D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[real]], int, int, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        int nx, ny
        bool first_order
        vector[vector[real]] state 

        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            vector[bool] object_cells,
            vector[real] gsources,
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
            string solver,
            bool quirk_smoothing,
            bool constant_sources,
            vector[vector[real]] boundary_sources,
            PyObjWrapper a,
            PyObjWrapper adot)

        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            vector[bool] object_cells,
            vector[real] gsources,
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
            string solver,
            bool quirk_smoothing,
            bool constant_sources,
            vector[vector[real]] boundary_sources,
            PyObjWrapper a,
            PyObjWrapper adot,
            PyObjWrapper d_outer,
            PyObjWrapper s1_outer,
            PyObjWrapper s2_outer,
            PyObjWrapper e_outer)

cdef extern from "hydro/srhydro3D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD3D:
        SRHD3D() except +
        SRHD3D(
            vector[vector[real]] state, 
            int nx, 
            int ny,
            int nz,
            real ad_gamma,
            vector[real] x1, 
            vector[real] x2,
            vector[real] x3,
            real cfl,
            string coord_system) except + 
        real theta, gamma
        int nx, ny, nz
        bool first_order
        vector[vector[real]] state 

        vector[vector[real]] simulate3D(
            vector[vector[real]] sources,
            vector[bool] object_cells,
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
            string solver,
            bool constant_sources,
            vector[vector[real]] boundary_sources)

# cdef extern from "hydro/srhd.hpp" namespace "simbi":
#     cdef cppclass SRHD[T]:
#         SRHD() except +
#         SRHD(
#             vector[vector[real]] state, 
#             InitialConditions sim_cond) except +
             
#         real theta, gamma
#         int nx, ny, nz
#         bool first_order
#         vector[vector[real]] state 

#         vector[vector[real]] simulate(
#             PyObjWrapper a,
#             PyObjWrapper adot)

#         vector[vector[real]] simulate (
#             PyObjWrapper a,
#             PyObjWrapper adot,
#             PyObjWrapper d_outer,
#             PyObjWrapper s1_outer,
#             PyObjWrapper s2_outer,
#             PyObjWrapper s3_outer,
#             PyObjWrapper e_outer)

cdef extern from "hydro/driver.hpp" namespace "simbi":
    cdef cppclass Driver:
        Driver() except +

        vector[vector[real]] run(
            vector[vector[real]] state,
            int dim, 
            string regime, 
            InitialConditions sim_cond
        )
        