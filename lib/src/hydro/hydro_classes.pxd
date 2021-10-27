# distutils: language = c++

from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 

cdef extern from "build_options.hpp":
    cdef bool col_maj "CYTHON_COL_MAJOR"

cdef extern from "common/config.hpp":
    cdef int FLOAT_PRECISION "FLOAT_PRECISION"
    ctypedef double real 

cdef extern from "euler1D.hpp" namespace "simbi":
    cdef int total_zones "total_zones"

    cdef cppclass Newtonian1D:
        Newtonian1D() except +
        Newtonian1D(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)

cdef extern from "euler2D.hpp" namespace "simbi":
    cdef cppclass Newtonian2D:
        Newtonian2D() except +
        Newtonian2D(vector[vector[real]], int nx, int ny, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        bool first_order, periodic
        int nx, ny
        vector[vector[real]] state
        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            real tstart, 
            real tend, 
            real dt, 
            real plm_theta,
            real engine_duration, 
            real chkpt_interval ,
            string data_directory, 
            bool first_order,
            bool periodic, 
            bool linspace, 
            bool hllc)


cdef extern from "srhydro1D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD:
        SRHD() except +
        SRHD(vector[vector[real]], real, real, vector[real], string) except + 
        real theta, gamma, tend, dt, CFL
        bool first_order, periodic, linspace
        string coord_system
        vector[real] r
        vector[vector[real]] state
        vector[vector [real]] simulate1D(
            vector[vector[real]] sources, 
            real tstart,
            real tend, 
            real dt, 
            real plm_theta, 
            real engine_duration,
            real chkpt_interval, 
            string data_directory,
            bool first_order, 
            bool periodic, 
            bool linspace, 
            bool hllc)

cdef extern from "srhydro2D.hip.hpp" namespace "simbi":
    cdef cppclass SRHD2D:
        SRHD2D() except +
        SRHD2D(vector[vector[real]], int, int, real, vector[real], vector[real],
                    real, string) except + 
        real theta, gamma
        int nx, ny
        bool first_order, periodic
        vector[vector[real]] state 

        vector[vector[real]] simulate2D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)

cdef extern from "srhydro3D.hip.hpp" namespace "simbi":
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
            real CFL,
            string coord_system) except + 
        real theta, gamma
        int nx, ny, nz
        bool first_order, periodic
        vector[vector[real]] state 

        vector[vector[real]] simulate3D(
            vector[vector[real]] sources,
            real tstart,
            real tend,
            real dt,
            real plm_theta,
            real engine_duration,
            real chkpt_interval,
            string data_directory,
            bool first_order,
            bool periodic,
            bool linspace,
            bool hllc)