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
            ConfigValue(vector[vector[double]] value) except +
            ConfigValue(vector[string] value) except +
            ConfigValue(ConfigDict value) except +
            ConfigValue(cpplist[ConfigDict] value) except +
            ConfigValue(pair[double, double] value) except +


cdef extern from "core/types/utility/init_conditions.hpp":
    ctypedef variant[real, vector[real]] PropertyValue "InitialConditions::PropertyValue"

cdef extern from "core/types/utility/init_conditions.hpp":
    cdef cppclass InitialConditions:
        @staticmethod
        InitialConditions create(ConfigDict sim_dict) except +

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
