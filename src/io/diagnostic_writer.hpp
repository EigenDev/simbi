// =============================================================================
// diagnostic_writer.hpp
//
// lightweight hdf5 writer for body diagnostics only.
// writes metadata + body state + diagnostic deltas without any hydro fields,
// grid hierarchy, or skeleton data. produces small files suitable for
// high-cadence diagnostic output.
//
// usage:
//   io::write_diagnostic_checkpoint(sim, "diagnostics/128.diag.0.01.h5");
// =============================================================================
#pragma once

#include "build_config.hpp"
#include "checkpoint.hpp"
#include "h5_serializable.hpp"
#include "write_policy.hpp"

#include <H5Cpp.h>
#include <filesystem>
#include <string>

namespace simbi::io {

    template <typename Sim>
    void write_diagnostic_checkpoint(
        const Sim&            sim,
        const std::string&    filename,
        const write_policy_t& policy = {}
    )
    {
        // ensure directory exists
        std::filesystem::path path(filename);
        if (path.has_parent_path()) {
            std::filesystem::create_directories(path.parent_path());
        }

        H5::H5File file(filename, H5F_ACC_TRUNC);

        write_attribute(file, "format_version", std::string("2.0"));
        write_attribute(file, "type", std::string("diagnostic"));

        // metadata (for time, iteration, etc.)
        h5_serializable<ecs::simulation_metadata_t<Sim::rank>>::write(file, sim.metadata(), policy);

        // body state + diagnostic deltas (reuse checkpoint_writer_t logic)
        checkpoint_writer_t<Sim>{sim, policy}.write_bodies_to(file);
    }

} // namespace simbi::io
