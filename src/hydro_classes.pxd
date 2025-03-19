from libcpp.vector cimport vector
from libcpp cimport bool as cbool
from libcpp.string cimport string
from libcpp.list cimport list as cpplist
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
    cdef cbool col_major "COLUMN_MAJOR"
    ctypedef double real

    cdef extern from "core/types/utility/config_dict.hpp" namespace "simbi":
        cdef cppclass ConfigDict:
            ConfigDict() except +
            # Add the indexing operator
            ConfigValue& operator[](const string&) except +
            # Add method to check if key exists
            bint find "find"(const string&) const
            # Add method to retrieve a value
            ConfigValue& at(const string&) except +
            # Add iterators
            cppclass iterator:
                pair[string, ConfigValue]& operator*()
                iterator operator++()
                bint operator==(iterator)
                bint operator!=(iterator)
            iterator begin()
            iterator end()

        cdef cppclass ConfigValue:
            ConfigValue() except +
            ConfigValue(cbool value) except +
            ConfigValue(int value) except +
            ConfigValue(double value) except +
            ConfigValue(string value) except +
            ConfigValue(vector[double] value) except +
            ConfigValue(ConfigDict value) except +
            ConfigValue(cpplist[ConfigDict] value) except +
            # Add type checking methods
            bint is_bool() const
            bint is_int() const
            bint is_double() const
            bint is_string() const
            bint is_array() const
            bint is_dict() const
            bint is_list() const


cdef extern from "core/types/utility/enums.hpp" namespace "simbi":
    cdef enum class BodyType:
        GRAVITATIONAL "simbi::BodyType::GRAVITATIONAL"
        GRAVITATIONAL_SINK "simbi::BodyType::GRAVITATIONAL_SINK"
        ELASTIC "simbi::BodyType::ELASTIC"
        RIGID "simbi::BodyType::RIGID"
        VISCOUS "simbi::BodyType::VISCOUS"
        SINK "simbi::BodyType::SINK"
        SOURCE "simbi::BodyType::SOURCE"

cdef extern from "core/types/utility/init_conditions.hpp":
    ctypedef variant[real, vector[real]] PropertyValue "InitialConditions::PropertyValue"

cdef extern from "core/types/utility/init_conditions.hpp":
    cdef cppclass InitialConditions:
        real time, checkpoint_interval, dlogt, plm_theta, gamma, cfl, tend, sound_speed_squared
        int nx, ny, nz, checkpoint_idx
        cbool quirk_smoothing, mesh_motion, homologous, isothermal
        vector[vector[real]] sources, bfield
        string data_directory, coord_system, solver
        string x1_spacing, x2_spacing, x3_spacing
        string spatial_order, temporal_order
        string hydro_source_lib, gravity_source_lib, boundary_source_lib
        pair[real, real] x1bounds, x2bounds, x3bounds
        vector[string] boundary_conditions
        vector[vector[real]] boundary_sources
        vector[pair[BodyType, unordered_map[string, PropertyValue]]] immersed_bodies
        ConfigDict config

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
