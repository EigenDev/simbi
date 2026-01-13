// =============================================================================
// binding.cpp
//
// pybind11 bindings for photon event generation and analysis.
// provides python interface to modern event-based radiation system.
//
// bindings:
//   - photon_event_t: event structure
//   - sim_conditions_t: simulation parameters
//   - quant_scales_t: quantity scales
//   - generate_photon_events: event generation
//   - monte_carlo_radiative_transfer: MCRT processing
//   - write_photon_events: HDF5 output
//   - read_photon_events: HDF5 input
// =============================================================================

#include "../src/photon_event_io.hpp"
#include "../src/rad.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
using namespace simbi::afterglow;

// =============================================================================
// helper functions: python dict/array → c++ struct/vector conversion
// =============================================================================

std::vector<double> array_to_vector(const py::array_t<double>& arr)
{
    auto buf = arr.request();
    auto ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.size);
}

std::vector<std::vector<double>> dict_of_arrays_to_vectors(const py::dict& dict)
{
    std::vector<std::vector<double>> result;

    // extract in order: x1, x2, x3
    if (dict.contains("x1")) {
        result.push_back(array_to_vector(dict["x1"].cast<py::array_t<double>>()));
    }
    if (dict.contains("x2")) {
        result.push_back(array_to_vector(dict["x2"].cast<py::array_t<double>>()));
    }
    if (dict.contains("x3")) {
        result.push_back(array_to_vector(dict["x3"].cast<py::array_t<double>>()));
    }

    return result;
}

sim_conditions_t dict_to_sim_conditions(const py::dict& d)
{
    sim_conditions_t s;
    s.dt              = d["dt"].cast<double>();
    s.theta_obs       = d["theta_obs"].cast<double>();
    s.adiabatic_index = d["adiabatic_index"].cast<double>();
    s.current_time    = d["current_time"].cast<double>();
    s.p               = d["p"].cast<double>();
    s.z               = d["z"].cast<double>();
    s.eps_e           = d["eps_e"].cast<double>();
    s.eps_b           = d["eps_b"].cast<double>();
    s.d_L             = d["d_L"].cast<double>();

    if (d.contains("nus")) {
        py::list nus_list = d["nus"];
        for (auto item : nus_list) {
            s.nus.push_back(item.cast<double>());
        }
    }

    if (d.contains("hydro_type")) {
        std::string ht = d["hydro_type"].cast<std::string>();
        s.hydro_type   = (ht == "SRMHD") ? hydro_type_t::SRMHD : hydro_type_t::SRHD;
    }
    else {
        s.hydro_type = hydro_type_t::SRHD;
    }

    return s;
}

quant_scales_t dict_to_quant_scales(const py::dict& d)
{
    quant_scales_t q;
    q.time_scale   = d["time_scale"].cast<double>();
    q.pre_scale    = d["pre_scale"].cast<double>();
    q.rho_scale    = d["rho_scale"].cast<double>();
    q.v_scale      = d["v_scale"].cast<double>();
    q.length_scale = d["length_scale"].cast<double>();
    return q;
}

py::dict metadata_to_dict(const photon_event_metadata_t& meta)
{
    py::dict d;
    d["dt"]              = meta.dt;
    d["theta_obs"]       = meta.theta_obs;
    d["adiabatic_index"] = meta.adiabatic_index;
    d["current_time"]    = meta.current_time;
    d["p"]               = meta.p;
    d["z"]               = meta.z;
    d["eps_e"]           = meta.eps_e;
    d["eps_b"]           = meta.eps_b;
    d["d_L"]             = meta.d_L;
    d["time_scale"]      = meta.time_scale;
    d["pre_scale"]       = meta.pre_scale;
    d["rho_scale"]       = meta.rho_scale;
    d["v_scale"]         = meta.v_scale;
    d["length_scale"]    = meta.length_scale;
    d["n_events"]        = meta.n_events;
    d["data_dim"]        = meta.data_dim;
    d["hydro_type"]      = static_cast<int>(meta.hydro_type);

    py::list freq_list;
    for (double nu : meta.frequencies) {
        freq_list.append(nu);
    }
    d["frequencies"] = freq_list;

    return d;
}

// =============================================================================
// python wrapper functions
// =============================================================================

std::vector<std::vector<double>> extract_fields(const py::dict& fields_dict)
{
    std::vector<std::vector<double>> result;

    // extract in order: rho, gamma_beta (or vr), pre (pressure)
    if (fields_dict.contains("rho")) {
        result.push_back(array_to_vector(fields_dict["rho"].cast<py::array_t<double>>()));
    }
    if (fields_dict.contains("gamma_beta")) {
        result.push_back(array_to_vector(fields_dict["gamma_beta"].cast<py::array_t<double>>()));
    }
    else if (fields_dict.contains("vr")) {
        result.push_back(array_to_vector(fields_dict["vr"].cast<py::array_t<double>>()));
    }
    if (fields_dict.contains("pre") || fields_dict.contains("p")) {
        auto key = fields_dict.contains("pre") ? "pre" : "p";
        result.push_back(array_to_vector(fields_dict[key].cast<py::array_t<double>>()));
    }

    return result;
}

py::list generate_photon_events_wrapper(
    py::dict      sim_cond_dict,
    py::dict      qscales_dict,
    py::dict      fields_dict,
    py::dict      mesh_dict,
    std::int64_t  data_dim,
    std::uint64_t max_events,
    std::uint64_t photons_per_cell
)
{
    auto sim_cond = dict_to_sim_conditions(sim_cond_dict);
    auto qscales  = dict_to_quant_scales(qscales_dict);
    auto fields   = extract_fields(fields_dict);
    auto mesh     = dict_of_arrays_to_vectors(mesh_dict);

    auto events = generate_photon_events(
        sim_cond,
        qscales,
        fields,
        mesh,
        data_dim,
        max_events,
        photons_per_cell
    );

    py::list result;
    for (const auto& evt : events) {
        result.append(evt);
    }
    return result;
}

void monte_carlo_radiative_transfer_wrapper(
    py::list     events_list,
    py::dict     sim_cond_dict,
    py::dict     qscales_dict,
    py::dict     fields_dict,
    py::dict     mesh_dict,
    std::int64_t data_dim,
    bool         include_scattering,
    bool         include_pair_production
)
{
    std::vector<photon_event_t> events;
    for (auto item : events_list) {
        events.push_back(item.cast<photon_event_t>());
    }

    auto sim_cond = dict_to_sim_conditions(sim_cond_dict);
    auto qscales  = dict_to_quant_scales(qscales_dict);
    auto fields   = extract_fields(fields_dict);
    auto mesh     = dict_of_arrays_to_vectors(mesh_dict);

    monte_carlo_radiative_transfer(
        events,
        sim_cond,
        qscales,
        fields,
        mesh,
        data_dim,
        include_scattering,
        include_pair_production
    );

    for (size_t ii = 0; ii < events.size(); ++ii) {
        events_list[ii] = events[ii];
    }
}

void write_photon_events_wrapper(
    const std::string& filename,
    py::list           events_list,
    py::dict           sim_cond_dict,
    py::dict           qscales_dict
)
{
    std::vector<photon_event_t> events;
    for (auto item : events_list) {
        events.push_back(item.cast<photon_event_t>());
    }

    auto sim_cond = dict_to_sim_conditions(sim_cond_dict);
    auto qscales  = dict_to_quant_scales(qscales_dict);

    write_photon_events(filename, events, sim_cond, qscales);
}

py::tuple read_photon_events_wrapper(const std::string& filename)
{
    auto [events, meta] = read_photon_events(filename);

    py::list events_list;
    for (const auto& evt : events) {
        events_list.append(evt);
    }

    py::dict meta_dict = metadata_to_dict(meta);

    return py::make_tuple(events_list, meta_dict);
}

py::dict read_photon_event_metadata_wrapper(const std::string& filename)
{
    auto meta = read_photon_event_metadata(filename);
    return metadata_to_dict(meta);
}

// =============================================================================
// module definition
// =============================================================================

PYBIND11_MODULE(rad_hydro, m)
{
    m.doc() = "photon event generation and monte carlo radiative transfer for afterglow modeling";

    // bind hydro_type enum
    py::enum_<hydro_type_t>(m, "HydroType")
        .value("SRHD", hydro_type_t::SRHD)
        .value("SRMHD", hydro_type_t::SRMHD)
        .export_values();

    // bind photon_event_t structure
    py::class_<photon_event_t>(m, "PhotonEvent")
        .def(py::init<>())
        .def_readwrite("t_emission", &photon_event_t::t_emission, "emission time [s]")
        .def_readwrite("x", &photon_event_t::x, "x position [cm]")
        .def_readwrite("y", &photon_event_t::y, "y position [cm]")
        .def_readwrite("z", &photon_event_t::z, "z position [cm]")
        .def_readwrite("energy", &photon_event_t::energy, "photon energy [erg]")
        .def_readwrite("px", &photon_event_t::px, "x component of direction")
        .def_readwrite("py", &photon_event_t::py, "y component of direction")
        .def_readwrite("pz", &photon_event_t::pz, "z component of direction")
        .def_readwrite("stokes_I", &photon_event_t::stokes_I, "stokes I parameter")
        .def_readwrite("stokes_Q", &photon_event_t::stokes_Q, "stokes Q parameter")
        .def_readwrite("stokes_U", &photon_event_t::stokes_U, "stokes U parameter")
        .def_readwrite("stokes_V", &photon_event_t::stokes_V, "stokes V parameter")
        .def_readwrite("doppler_factor", &photon_event_t::doppler_factor, "doppler boost factor")
        .def_readwrite("lorentz_factor", &photon_event_t::lorentz_factor, "fluid lorentz factor")
        .def_readwrite("optical_depth", &photon_event_t::optical_depth, "integrated optical depth")
        .def_readwrite("cell_id", &photon_event_t::cell_id, "cell index")
        .def_readwrite("absorbed", &photon_event_t::absorbed, "absorption flag")
        .def_readwrite("n_scatter", &photon_event_t::n_scatter, "number of scattering events")
        .def("__repr__", [](const photon_event_t& e) {
            return "<PhotonEvent energy=" + std::to_string(e.energy) +
                   " absorbed=" + std::to_string(e.absorbed) + ">";
        });

    // bind sim_conditions_t structure
    py::class_<sim_conditions_t>(m, "SimConditions")
        .def(py::init<>())
        .def(py::init([](py::dict d) { return dict_to_sim_conditions(d); }))
        .def_readwrite("dt", &sim_conditions_t::dt)
        .def_readwrite("theta_obs", &sim_conditions_t::theta_obs)
        .def_readwrite("adiabatic_index", &sim_conditions_t::adiabatic_index)
        .def_readwrite("current_time", &sim_conditions_t::current_time)
        .def_readwrite("p", &sim_conditions_t::p)
        .def_readwrite("z", &sim_conditions_t::z)
        .def_readwrite("eps_e", &sim_conditions_t::eps_e)
        .def_readwrite("eps_b", &sim_conditions_t::eps_b)
        .def_readwrite("d_L", &sim_conditions_t::d_L)
        .def_readwrite("nus", &sim_conditions_t::nus)
        .def_readwrite("hydro_type", &sim_conditions_t::hydro_type);

    // bind quant_scales_t structure
    py::class_<quant_scales_t>(m, "QuantScales")
        .def(py::init<>())
        .def(py::init([](py::dict d) { return dict_to_quant_scales(d); }))
        .def_readwrite("time_scale", &quant_scales_t::time_scale)
        .def_readwrite("pre_scale", &quant_scales_t::pre_scale)
        .def_readwrite("rho_scale", &quant_scales_t::rho_scale)
        .def_readwrite("v_scale", &quant_scales_t::v_scale)
        .def_readwrite("length_scale", &quant_scales_t::length_scale);

    // bind photon event generation function
    m.def(
        "generate_photon_events",
        &generate_photon_events_wrapper,
        py::arg("sim_cond"),
        py::arg("qscales"),
        py::arg("fields"),
        py::arg("mesh"),
        py::arg("data_dim"),
        py::arg("max_events")       = 1000000,
        py::arg("photons_per_cell") = 0,
        "generate photon events from hydrodynamic simulation data"
    );

    // bind monte carlo radiative transfer function
    m.def(
        "monte_carlo_radiative_transfer",
        &monte_carlo_radiative_transfer_wrapper,
        py::arg("events"),
        py::arg("sim_cond"),
        py::arg("qscales"),
        py::arg("fields"),
        py::arg("mesh"),
        py::arg("data_dim"),
        py::arg("include_scattering")      = true,
        py::arg("include_pair_production") = false,
        "apply monte carlo radiative transfer to photon events (modifies in place)"
    );

    // bind HDF5 write function
    m.def(
        "write_photon_events",
        &write_photon_events_wrapper,
        py::arg("filename"),
        py::arg("events"),
        py::arg("sim_cond"),
        py::arg("qscales"),
        "write photon events to HDF5 file with compression"
    );

    // bind HDF5 read function
    m.def(
        "read_photon_events",
        &read_photon_events_wrapper,
        py::arg("filename"),
        "read photon events from HDF5 file, returns (events, metadata)"
    );

    // bind metadata-only read function
    m.def(
        "read_photon_event_metadata",
        &read_photon_event_metadata_wrapper,
        py::arg("filename"),
        "read metadata from HDF5 file without loading events"
    );
}
