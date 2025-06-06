#include "../src/rad_units.hpp"
#include <map>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

// python wrapper for the calc_fnu function
void py_calc_fnu(
    std::map<std::string, py::array_t<double>> fields,
    py::array_t<double> tbin_edges,
    py::array_t<double> flux_array,
    std::map<std::string, py::array_t<double>> mesh,
    std::map<std::string, double> qscales,
    std::map<std::string, py::object> sim_info,
    int checkpoint_index,
    int data_dim
)
{
    // extract field arrays
    auto rho_array        = fields["rho"].request();
    auto gamma_beta_array = fields["gamma_beta"].request();
    auto pressure_array   = fields["p"].request();

    // extract tbin_edges and flux arrays
    auto tbin_req = tbin_edges.request();
    auto flux_req = flux_array.request();

    // extract mesh arrays
    auto x1_req = mesh["x1"].request();
    auto x2_req = mesh["x2"].request();
    auto x3_req = mesh["x3"].request();

    // convert numpy arrays to vectors
    std::vector<double> rho(
        static_cast<double*>(rho_array.ptr),
        static_cast<double*>(rho_array.ptr) + rho_array.size
    );
    std::vector<double> gamma_beta(
        static_cast<double*>(gamma_beta_array.ptr),
        static_cast<double*>(gamma_beta_array.ptr) + gamma_beta_array.size
    );
    std::vector<double> pressure(
        static_cast<double*>(pressure_array.ptr),
        static_cast<double*>(pressure_array.ptr) + pressure_array.size
    );

    std::vector<double> tbin_vec(
        static_cast<double*>(tbin_req.ptr),
        static_cast<double*>(tbin_req.ptr) + tbin_req.size
    );

    std::vector<std::vector<double>> flattened_mesh = {
      std::vector<double>(
          static_cast<double*>(x1_req.ptr),
          static_cast<double*>(x1_req.ptr) + x1_req.size
      ),
      std::vector<double>(
          static_cast<double*>(x2_req.ptr),
          static_cast<double*>(x2_req.ptr) + x2_req.size
      ),
      std::vector<double>(
          static_cast<double*>(x3_req.ptr),
          static_cast<double*>(x3_req.ptr) + x3_req.size
      )
    };

    // set up the sim_conditions struct
    sogbo_rad::sim_conditions sim_cond;
    sim_cond.dt              = py::cast<double>(sim_info["dt"]);
    sim_cond.current_time    = py::cast<double>(sim_info["current_time"]);
    sim_cond.theta_obs       = py::cast<double>(sim_info["theta_obs"]);
    sim_cond.adiabatic_index = py::cast<double>(sim_info["adiabatic_index"]);
    sim_cond.z               = py::cast<double>(sim_info["z"]);
    sim_cond.p               = py::cast<double>(sim_info["p"]);
    sim_cond.eps_e           = py::cast<double>(sim_info["eps_e"]);
    sim_cond.eps_b           = py::cast<double>(sim_info["eps_b"]);
    sim_cond.d_L             = py::cast<double>(sim_info["d_L"]);

    // extract frequencies
    py::array freq_array     = py::cast<py::array>(sim_info["nus"]);
    py::buffer_info freq_buf = freq_array.request();
    double* freq_ptr         = static_cast<double*>(freq_buf.ptr);
    sim_cond.nus = std::vector<double>(freq_ptr, freq_ptr + freq_buf.size);

    // set up the quant_scales struct
    sogbo_rad::quant_scales quant_scales;
    quant_scales.time_scale   = qscales["time_scale"];
    quant_scales.pre_scale    = qscales["pre_scale"];
    quant_scales.rho_scale    = qscales["rho_scale"];
    quant_scales.v_scale      = qscales["v_scale"];
    quant_scales.length_scale = qscales["length_scale"];

    // extract flux array for modification
    double* flux_ptr = static_cast<double*>(flux_req.ptr);
    std::vector<double> flux_vec(flux_ptr, flux_ptr + flux_req.size);

    // call the C++ function
    sogbo_rad::calc_fnu(
        sim_cond,
        quant_scales,
        rho,
        gamma_beta,
        pressure,
        flattened_mesh,
        tbin_vec,
        flux_vec,
        checkpoint_index,
        data_dim
    );

    // copy back the results
    std::copy(flux_vec.begin(), flux_vec.end(), flux_ptr);
}

// python wrapper for the log_events function
void py_log_events(
    std::map<std::string, py::array_t<double>> fields,
    py::array_t<double> photon_distro,
    py::array_t<double> x_mu,
    std::map<std::string, py::array_t<double>> mesh,
    std::map<std::string, double> qscales,
    std::map<std::string, py::object> sim_info,
    int data_dim
)
{
    // extract field arrays
    auto rho_array        = fields["rho"].request();
    auto gamma_beta_array = fields["gamma_beta"].request();
    auto pressure_array   = fields["p"].request();

    // extract photon distro and four position arrays
    auto photon_req = photon_distro.request();
    auto xmu_req    = x_mu.request();

    // extract mesh arrays
    auto x1_req = mesh["x1"].request();
    auto x2_req = mesh["x2"].request();
    auto x3_req = mesh["x3"].request();

    // convert numpy arrays to vectors
    std::vector<std::vector<double>> flattened_fields = {
      std::vector<double>(
          static_cast<double*>(rho_array.ptr),
          static_cast<double*>(rho_array.ptr) + rho_array.size
      ),
      std::vector<double>(
          static_cast<double*>(gamma_beta_array.ptr),
          static_cast<double*>(gamma_beta_array.ptr) + gamma_beta_array.size
      ),
      std::vector<double>(
          static_cast<double*>(pressure_array.ptr),
          static_cast<double*>(pressure_array.ptr) + pressure_array.size
      )
    };

    std::vector<std::vector<double>> flattened_mesh = {
      std::vector<double>(
          static_cast<double*>(x1_req.ptr),
          static_cast<double*>(x1_req.ptr) + x1_req.size
      ),
      std::vector<double>(
          static_cast<double*>(x2_req.ptr),
          static_cast<double*>(x2_req.ptr) + x2_req.size
      ),
      std::vector<double>(
          static_cast<double*>(x3_req.ptr),
          static_cast<double*>(x3_req.ptr) + x3_req.size
      )
    };

    // set up the sim_conditions struct
    sogbo_rad::sim_conditions sim_cond;
    sim_cond.dt              = py::cast<double>(sim_info["dt"]);
    sim_cond.current_time    = py::cast<double>(sim_info["current_time"]);
    sim_cond.theta_obs       = py::cast<double>(sim_info["theta_obs"]);
    sim_cond.adiabatic_index = py::cast<double>(sim_info["adiabatic_index"]);
    sim_cond.z               = py::cast<double>(sim_info["z"]);
    sim_cond.p               = py::cast<double>(sim_info["p"]);
    sim_cond.eps_e           = py::cast<double>(sim_info["eps_e"]);
    sim_cond.eps_b           = py::cast<double>(sim_info["eps_b"]);
    sim_cond.d_L             = py::cast<double>(sim_info["d_L"]);

    // extract frequencies
    py::array freq_array     = py::cast<py::array>(sim_info["nus"]);
    py::buffer_info freq_buf = freq_array.request();
    double* freq_ptr         = static_cast<double*>(freq_buf.ptr);
    sim_cond.nus = std::vector<double>(freq_ptr, freq_ptr + freq_buf.size);

    // set up the quant_scales struct
    sogbo_rad::quant_scales quant_scales;
    quant_scales.time_scale   = qscales["time_scale"];
    quant_scales.pre_scale    = qscales["pre_scale"];
    quant_scales.rho_scale    = qscales["rho_scale"];
    quant_scales.v_scale      = qscales["v_scale"];
    quant_scales.length_scale = qscales["length_scale"];

    // extract arrays for modification
    double* photon_ptr = static_cast<double*>(photon_req.ptr);
    double* xmu_ptr    = static_cast<double*>(xmu_req.ptr);
    std::vector<double> photon_vec(photon_ptr, photon_ptr + photon_req.size);
    std::vector<double> xmu_vec(xmu_ptr, xmu_ptr + xmu_req.size);

    // call the C++ function
    sogbo_rad::log_events(
        sim_cond,
        quant_scales,
        flattened_fields,
        flattened_mesh,
        photon_vec,
        xmu_vec,
        data_dim
    );

    // copy back the results
    std::copy(photon_vec.begin(), photon_vec.end(), photon_ptr);
    std::copy(xmu_vec.begin(), xmu_vec.end(), xmu_ptr);
}

PYBIND11_MODULE(rad_hydro, m)
{
    m.doc() = "Synchrotron radiation hydrodynamics module for post processing "
              "simbi simulations";

    m.def(
        "py_calc_fnu",
        &py_calc_fnu,
        "Calculate spectral flux from hydro data assuming synchotron spectrum",
        py::arg("fields"),
        py::arg("tbin_edges"),
        py::arg("flux_array"),
        py::arg("mesh"),
        py::arg("qscales"),
        py::arg("sim_info"),
        py::arg("checkpoint_index"),
        py::arg("data_dim")
    );

    m.def(
        "py_log_events",
        &py_log_events,
        "Log photon emission events from hydro data",
        py::arg("fields"),
        py::arg("photon_distro"),
        py::arg("x_mu"),
        py::arg("mesh"),
        py::arg("qscales"),
        py::arg("sim_info"),
        py::arg("data_dim")
    );
}
