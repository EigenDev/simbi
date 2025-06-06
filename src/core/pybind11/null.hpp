#include "core/types/containers/ndarray.hpp"   // for ndarray
#include "core/types/utility/enums.hpp"        // for Regime
#include "driver.hpp"                          // for InitialConditions
#include <pybind11/numpy.h>                    // for py::array_t
#include <pybind11/pybind11.h>                 // for pybind11 module
#include <pybind11/stl.h>                      // for STL support in pybind11
namespace simbi {
    template <size_t Dims, Regime R>
    void run_simulation_impl(
        py::array_t<real, py::array::c_style> state_array,
        InitialConditions& init_cond,
        std::function<real(real)> scale_factor,
        std::function<real(real)> scale_factor_derivative
    )
    {
        // get buffer info without copying
        py::buffer_info buffer = state_array.request();

        // create shape vector
        std::vector<size_type> shape(buffer.shape.begin(), buffer.shape.end());

        // directly interpret the numpy array memory as our anyConserved type
        // Using zero-copy approach
        using conserved_t = anyConserved<Dims, R>;

        // Create an ndarray view of the data (no ownership transfer)
        ndarray<conserved_t, Dims> hydro_array(
            reinterpret_cast<conserved_t*>(buffer.ptr),
            shape,
            false   // Don't take ownership
        );

        driver::run<Dims, R>(
            hydro_array,
            init_cond,
            scale_factor,
            scale_factor_derivative
        );
    }

    // explicitly instantiate all needed specializations
    template <>
    void run_simulation_impl<1, Regime::NEWTONIAN>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<1, Regime::SRHD>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<2, Regime::NEWTONIAN>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<2, Regime::SRHD>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<3, Regime::NEWTONIAN>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<3, Regime::SRHD>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );

    template <>
    void run_simulation_impl<3, Regime::RMHD>(
        py::array_t<real, py::array::c_style>,
        InitialConditions&,
        std::function<real(real)>,
        std::function<real(real)>
    );
}   // namespace simbi
