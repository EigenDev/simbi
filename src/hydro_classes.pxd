from libcpp.vector cimport vector 
from libcpp cimport bool
from libcpp.string cimport string 
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map

cdef extern from "<variant>" namespace "std" nogil:
    cdef cppclass variant[T1, T2]:
        variant() except +
        variant(T1) except +
        variant(T2) except +
        T1& get[T1]() except +
        T2& get[T2]() except +

cdef extern from "core/cython/pyobj_wrapper.hpp":
    cdef cppclass PyObjWrapper:
        PyObjWrapper() except +
        PyObjWrapper(object py_func) except +
        
cdef extern from "build_options.hpp":
    cdef bool col_major "COLUMN_MAJOR"
    ctypedef double real 

cdef extern from "physics/hydro/schemes/ib/bodies/immersed_boundary.hpp" namespace "simbi::ib":
    cdef enum class BodyType:
        GRAVITATIONAL "simbi::ib::BodyType::GRAVITATIONAL"
        GRAVITATIONAL_SINK "simbi::ib::BodyType::GRAVITATIONAL_SINK"
        ELASTIC "simbi::ib::BodyType::ELASTIC"
        RIGID "simbi::ib::BodyType::RIGID"
        VISCOUS "simbi::ib::BodyType::VISCOUS"
        SINK "simbi::ib::BodyType::SINK"
        SOURCE "simbi::ib::BodyType::SOURCE"

cdef extern from "core/types/utility/init_conditions.hpp":
    ctypedef variant[real, vector[real]] PropertyValue "InitialConditions::PropertyValue"

cdef extern from "core/types/utility/init_conditions.hpp":
    cdef cppclass InitialConditions:
        real time, checkpoint_interval, dlogt, plm_theta, engine_duration, gamma, cfl, tend
        int nx, ny, nz, checkpoint_idx
        bool quirk_smoothing, mesh_motion, homologous
        vector[vector[real]] sources, bfield
        string data_directory, coord_system, solver
        string x1_spacing, x2_spacing, x3_spacing
        string spatial_order, temporal_order
        string hydro_source_lib, gravity_source_lib, boundary_source_lib
        pair[real, real] x1bounds, x2bounds, x3bounds
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources
        vector[pair[BodyType, unordered_map[string, PropertyValue]]] immersed_bodies

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
        