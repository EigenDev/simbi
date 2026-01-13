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

#ifndef SIMBI_AFTERGLOW_PHOTON_EVENT_IO_HPP
#define SIMBI_AFTERGLOW_PHOTON_EVENT_IO_HPP

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
        double dt;
        double theta_obs;
        double adiabatic_index;
        double current_time;
        double p;
        double z;
        double eps_e;
        double eps_b;
        double d_L;

        // scales
        double time_scale;
        double pre_scale;
        double rho_scale;
        double v_scale;
        double length_scale;

        // data info
        std::uint64_t n_events;
        std::int64_t  data_dim;
        hydro_type_t  hydro_type;

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

    // append events to existing HDF5 file
    // useful for streaming large simulations
    void append_photon_events(
        const std::string&                 filename,
        const std::vector<photon_event_t>& new_events
    );

    // query metadata without loading full dataset
    photon_event_metadata_t read_photon_event_metadata(const std::string& filename);

    // read subset of events matching filter criteria
    // filter function: bool(const photon_event_t&)
    template <typename FilterFunc>
    std::vector<photon_event_t>
    read_photon_events_filtered(const std::string& filename, FilterFunc filter);

} // namespace simbi::afterglow

#endif // SIMBI_AFTERGLOW_PHOTON_EVENT_IO_HPP
