// =============================================================================
// photon_event_io.hpp
//
// serialization/deserialization for photon_event_t collections.
// uses HDF5 for efficient chunked storage with compression.
//
// design:
//   - each field stored as separate dataset for columnar access
//   - metadata stored as HDF5 attributes
//   - compression enabled by default (gzip level 6)
//   - supports partial reads (coming soon)
//
// usage:
//   std::vector<photon_event_t> events = generate_photon_events(...);
//   write_photon_events("output.h5", events, sim_cond, qscales);
//
//   auto [loaded_events, meta] = read_photon_events("output.h5");
// =============================================================================
#pragma once

#include "rad.hpp"

#include <H5Cpp.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace simbi::afterglow {

    // metadata stored with photon events
    struct photon_event_metadata_t
    {
        // simulation parameters
        double dt              = 0.0;
        double theta_obs       = 0.0;
        double adiabatic_index = 4.0 / 3.0;
        double current_time    = 0.0;
        double p               = 2.5;
        double z               = 0.0;
        double eps_e           = 0.1;
        double eps_b           = 0.01;
        double d_L             = 1e28;

        // scales
        double time_scale   = 1.0;
        double pre_scale    = 1.0;
        double rho_scale    = 1.0;
        double v_scale      = 1.0;
        double length_scale = 1.0;

        // data info
        std::uint64_t n_events   = 0;
        hydro_type_t  hydro_type = hydro_type_t::SRHD;

        // frequencies (stored separately as array)
        std::vector<double> frequencies;
    };

    // write photon events to HDF5 file
    // creates datasets for each field with compression
    void write_photon_events(
        const std::string&                 filename,
        const std::vector<photon_event_t>& events,
        const sim_conditions_t&            sim_cond,
        const quant_scales_t&              qscales
    );

    // read photon events from HDF5 file
    // returns events + metadata
    std::pair<std::vector<photon_event_t>, photon_event_metadata_t>
    read_photon_events(const std::string& filename);

    // query metadata without loading full dataset
    photon_event_metadata_t read_photon_event_metadata(const std::string& filename);

} // namespace simbi::afterglow
