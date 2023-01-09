# distutils: language = c++

from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 


cdef extern from "util/pyobj_wrapper.hpp":
    cdef cppclass PyObjWrapper:
        PyObjWrapper()
        PyObjWrapper(object) # define a constructor that takes a Python object
                             # note - doesn't match c++ signature - that's fine!
        
cdef extern from "build_options.hpp":
    cdef bool col_maj "COLUMN_MAJOR"

cdef extern from "common/enums.hpp":
    cdef int FLOAT_PRECISION "FLOAT_PRECISION"
    ctypedef double real 

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
            string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc,
            bool constant_sources)

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
            string boundary_condition,
            bool first_order,
            bool linspace, 
            bool hllc,
            bool constant_sources) except +


cdef extern from "hydro/srhydro1D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dlogt, cfl
        bool first_order, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
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
            string boundary_condition,
            bool first_order, 
            bool linspace, 
            bool hllc,
            bool constant_sources,
            PyObjWrapper a,
            PyObjWrapper adot) except + 
            
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
            string boundary_condition,
            bool first_order, 
            bool linspace, 
            bool hllc,
            bool constant_sources,
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
            real tstart,
            real tend,
            real dlogt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            int  chkpt_idx,
            string data_directory,
            string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc,
            bool quirk_smoothing,
            bool constant_sources,
            PyObjWrapper a,
            PyObjWrapper adot)

        vector[vector[real]] simulate2D(
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
            string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc,
            bool quirk_smoothing,
            bool constant_sources,
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
            real tstart,
            real tend,
            real dlogt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            int  chkpt_idx,
            string data_directory,
            string boundary_condition,
            bool first_order,
            bool linspace,
            bool hllc,
            bool constant_sources)