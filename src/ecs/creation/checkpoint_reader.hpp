#ifndef ECS_CREATION_CHECKPOINT_READER_HPP
#define ECS_CREATION_CHECKPOINT_READER_HPP

// =============================================================================
// checkpoint_reader.hpp
//
// production-grade checkpoint reconstruction utilities.
//
// responsibilities:
//   1. read partition topology from hdf5
//   2. validate domain consistency (owned ⊂ allocated, non-empty, etc.)
//   3. reconstruct partition_t with proper initialization
//   4. allocate auxiliary fields (flux, efield) using existing allocation logic
//
// design principles:
//   - reuse battle-tested allocation logic from decomposition builder
//   - validate early, fail fast with clear error messages
//   - separate concerns: i/o vs validation vs reconstruction
//   - make checkpoint format changes localized to this file
//
// usage:
//   auto [part, fields] = checkpoint_reader_t<R, C, P, Rank>::read_partition(
//       part_group, block, locality
//   );
// =============================================================================

#include "compat.hpp"
#include "ecs/components.hpp"
#include "grid/block_info.hpp"
#include "grid/domain.hpp"
#include "grid/field.hpp"
#include "io/h5_serializable.hpp"
#include "io/serialization/all.hpp"
#include "utility/enums.hpp"
#include "xpu/xpu.hpp"

#include <H5Cpp.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simbi::ecs::creation {

    // =============================================================================
    // checkpoint format version
    // =============================================================================

    // current checkpoint format version this code understands
    constexpr const char* CHECKPOINT_FORMAT_VERSION = "2.0";

    // compatible versions (can read these older formats)
    constexpr const char* COMPATIBLE_VERSIONS[] = {"2.0"};

    inline void validate_checkpoint_version(const H5::H5File& file)
    {
        std::string version;
        try {
            version = io::read_attribute<std::string>(file, "format_version");
        }
        catch (...) {
            throw std::runtime_error(
                "checkpoint validation failed: missing format_version attribute. "
                "this file may be from an older version of simbi."
            );
        }

        // check if version is compatible
        bool compatible = false;
        for (const auto& compat_ver : COMPATIBLE_VERSIONS) {
            if (version == compat_ver) {
                compatible = true;
                break;
            }
        }

        if (!compatible) {
            std::ostringstream oss;
            oss << "checkpoint validation failed: incompatible format version '" << version
                << "'. this code supports: ";
            for (std::size_t ii = 0; ii < std::size(COMPATIBLE_VERSIONS); ++ii) {
                oss << "'" << COMPATIBLE_VERSIONS[ii] << "'";
                if (ii + 1 < std::size(COMPATIBLE_VERSIONS)) {
                    oss << ", ";
                }
            }
            throw std::runtime_error(oss.str());
        }
    }

    // =============================================================================
    // validation helpers
    // =============================================================================

    template <std::uint64_t Rank>
    void validate_partition_domains(
        const grid::domain_t<Rank>& owned,
        const grid::domain_t<Rank>& allocated,
        std::uint64_t               partition_id
    )
    {
        // owned must be non-empty
        if (owned.empty()) {
            std::ostringstream oss;
            oss << "checkpoint validation failed for partition " << partition_id << ": "
                << "owned domain is empty " << owned;
            throw std::runtime_error(oss.str());
        }

        // allocated must be non-empty
        if (allocated.empty()) {
            std::ostringstream oss;
            oss << "checkpoint validation failed for partition " << partition_id << ": "
                << "allocated domain is empty " << allocated;
            throw std::runtime_error(oss.str());
        }

        // owned must be subset of allocated
        for (std::uint64_t dd = 0; dd < Rank; ++dd) {
            if (owned.start[dd] < allocated.start[dd] || owned.fin[dd] > allocated.fin[dd]) {
                std::ostringstream oss;
                oss << "checkpoint validation failed for partition " << partition_id << ": "
                    << "owned domain " << owned << " not subset of allocated " << allocated
                    << " (dimension " << dd << ")";
                throw std::runtime_error(oss.str());
            }
        }
    }

    // =============================================================================
    // checkpoint_reader_t
    // =============================================================================

    template <regime_t R, typename Conserved, typename Primitive, std::uint64_t Rank>
    struct checkpoint_reader_t
    {
        using partition_fields_t = ecs::partition_fields_t<Conserved, Primitive, Rank>;

        // -------------------------------------------------------------------------
        // read_domain
        //
        // reads domain_t from hdf5 group using start/fin datasets.
        // -------------------------------------------------------------------------
        static grid::domain_t<Rank> read_domain(const H5::Group& group, const std::string& prefix)
        {
            auto start_data = io::read_dataset<std::int64_t>(group, prefix + "_start");
            auto fin_data   = io::read_dataset<std::int64_t>(group, prefix + "_fin");

            grid::domain_t<Rank> domain;
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                domain.start[dd] = start_data[dd];
                domain.fin[dd]   = fin_data[dd];
            }

            return domain;
        }

        // -------------------------------------------------------------------------
        // compute_face_domains
        //
        // derives face-centered domains from owned domain.
        // face_domains[d] has one extra cell in dimension d (for face-centered flux).
        // for mhd, also extends transverse dimensions by 1 for ct stencil.
        // -------------------------------------------------------------------------
        static void compute_face_domains(partition_t<Rank>& part)
        {
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                part.face_domains[dd] = part.owned_domain;
                part.face_domains[dd].fin[dd] += 1;

                // mhd: extend transverse dims for constrained transport stencil
                if constexpr (R == regime_t::MHD || R == regime_t::RMHD) {
                    for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                        if (tt != dd) {
                            part.face_domains[dd].start[tt] -= 1;
                            part.face_domains[dd].fin[tt] += 1;
                        }
                    }
                }
            }
        }

        // -------------------------------------------------------------------------
        // compute_edge_domains
        //
        // derives edge-centered domains from owned domain.
        // edge_domains[d] has one extra cell in transverse dimensions (for efield).
        // -------------------------------------------------------------------------
        static void compute_edge_domains(partition_t<Rank>& part)
        {
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                part.edge_domains[dd] = part.owned_domain;
                for (std::uint64_t tt = 0; tt < Rank; ++tt) {
                    if (tt != dd) {
                        part.edge_domains[dd].fin[tt] += 1;
                    }
                }
            }
        }

        // -------------------------------------------------------------------------
        // allocate_flux_arrays
        //
        // allocates face-centered flux storage for each dimension.
        // flux arrays are always empty at checkpoint time (transient state).
        // -------------------------------------------------------------------------
        static void allocate_flux_arrays(const partition_t<Rank>& part, partition_fields_t& fields)
        {
            // use unified memory by default for field allocation
            for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                auto flux_domain = part.face_domains[dd];
                fields.flux[dd]  = grid::field_t<Conserved, Rank>(flux_domain);
            }
        }

        // -------------------------------------------------------------------------
        // allocate_efield_arrays
        //
        // allocates edge-centered electric field storage for mhd.
        // efield arrays are derived from bfield during evolution.
        // -------------------------------------------------------------------------
        static void
        allocate_efield_arrays(const partition_t<Rank>& part, partition_fields_t& fields)
        {
            if constexpr (R == regime_t::MHD || R == regime_t::RMHD) {
                // use unified memory by default for field allocation
                for (std::uint64_t dd = 0; dd < Rank; ++dd) {
                    auto efield_domain = part.edge_domains[dd];
                    fields.efield[dd]  = grid::field_t<real, Rank>(efield_domain);
                }
            }
        }

        // -------------------------------------------------------------------------
        // read_partition
        //
        // main entry point: reads partition from checkpoint and reconstructs
        // complete partition_t + partition_fields_t.
        //
        // returns: {partition, fields} ready to insert into decomposition.
        // -------------------------------------------------------------------------
        static std::pair<partition_t<Rank>, partition_fields_t> read_partition(
            const H5::Group&                part_group,
            const grid::block_info_t<Rank>& block,
            std::int64_t                    device_id,
            std::uint64_t                   partition_id
        )
        {
            // phase 1: read owned domain from checkpoint
            auto owned_domain = read_domain(part_group, "owned");

            // phase 2: load hydro fields (prim, cons, bfield)
            auto fields = io::h5_serializable<partition_fields_t>::read(part_group);

            // phase 3: allocated domain comes from loaded field
            auto allocated_domain = fields.prim.domain();

            // phase 4: validate consistency
            validate_partition_domains(owned_domain, allocated_domain, partition_id);

            // phase 5: construct partition_t with proper initialization
            partition_t<Rank> part;
            part.block            = block;
            part.owned_domain     = owned_domain;
            part.allocated_domain = allocated_domain;
            part.executor         = xpu::executor_t<xpu::default_space>(device_id);
            part.rank_id          = xpu::comm::rank_id_t{0, device_id}; // single-rank default

            // phase 6: compute derived domains (face, edge)
            compute_face_domains(part);
            compute_edge_domains(part);

            // phase 7: allocate auxiliary fields (flux, efield)
            allocate_flux_arrays(part, fields);
            allocate_efield_arrays(part, fields);

            return {std::move(part), std::move(fields)};
        }
    };

} // namespace simbi::ecs::creation

#endif // ECS_CREATION_CHECKPOINT_READER_HPP
