#ifndef SERIALIZER2_HPP
#define SERIALIZER2_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "functional/fp.hpp"
#include "functional/monad/result.hpp"
#include "physics/ib/body.hpp"
#include "utility/bimap.hpp"   // for simbi::helpers::serialize
#include "utility/helpers.hpp"

#include <H5Cpp.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace simbi::io {

    struct serialization_context_t {
        H5::H5File file;
        std::string filename;
    };

    namespace h5 {
        using namespace simbi::helpers;
        // core h5 operations
        inline result_t<H5::Group>
        create_group(H5::H5File& file, const std::string& name)
        {
            try {
                return result_t<H5::Group>::ok(file.createGroup(name));
            }
            catch (const H5::Exception& e) {
                return result_t<H5::Group>::error(e.getDetailMsg());
            }
        }

        template <typename Sim>
        std::string compute_checkpoint_filename(const Sim& sim)
        {
            using namespace helpers;
            static std::int64_t tchunk_order_of_mag = 2;

            const auto meta                = sim.metadata();
            const auto data_directory      = meta.data_dir;
            const auto step                = meta.checkpoint_index;
            const auto timestepping_of_mag = std::floor(std::log10(meta.time));

            if (timestepping_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }

            std::string tnow;
            if (meta.dlogt != 0) {
                const auto timestepping_of_mag = std::floor(std::log10(step));
                if (timestepping_of_mag > tchunk_order_of_mag) {
                    tchunk_order_of_mag += 1;
                }
                tnow = format_real(step);
            }
            else if (!sim.in_failure_state) {
                tnow = format_real(meta.checkpoint_identifier());
            }
            else {
                if (sim.was_interrupted) {
                    tnow = "interrupted";
                }
                else {
                    tnow = "crashed";
                }
            }

            return data_directory + string_format(
                                        "%d.chkpt." + tnow + ".h5",
                                        meta.checkpoint_zones
                                    );
        }

        template <typename T, std::uint64_t Dims>
        result_t<void> write_field(
            const T* data,
            iarray<Dims> shape,
            H5::Group& group,
            const std::string& name
        )
        {
            try {
                std::vector<hsize_t> dims(Dims);
                for (std::uint64_t ii = 0; ii < Dims; ++ii) {
                    dims[ii] = shape[ii];
                }
                auto space   = H5::DataSpace(Dims, dims.data());
                auto dataset = group.createDataSet(
                    name,
                    H5::PredType::NATIVE_DOUBLE,
                    space
                );
                dataset.write(data, H5::PredType::NATIVE_DOUBLE);
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename T, typename U = real>
        result_t<void>
        write_scalar(H5::Group& group, const std::string& name, const T& value)
        {
            try {

                auto scalar_space = H5::DataSpace(H5S_SCALAR);
                auto attr         = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_DOUBLE,
                    scalar_space
                );
                attr.write(H5::PredType::NATIVE_DOUBLE, &value);
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }
        template <typename T>
        result_t<void>
        write_integer(H5::Group& group, const std::string& name, const T& value)
        {
            try {
                auto scalar_space = H5::DataSpace(H5S_SCALAR);
                auto attr         = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_INT,
                    scalar_space
                );
                attr.write(H5::PredType::NATIVE_INT, &value);
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename T>
        result_t<void>
        write_boolean(H5::Group& group, const std::string& name, const T& value)
        {
            try {
                auto scalar_space = H5::DataSpace(H5S_SCALAR);
                auto attr         = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_HBOOL,
                    scalar_space
                );
                hbool_t h5_value = static_cast<hbool_t>(value);
                attr.write(H5::PredType::NATIVE_HBOOL, &h5_value);
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename T, std::uint64_t N>
        result_t<void> write_array(
            H5::Group& group,
            const std::string& name,
            const vector_t<T, N>& vec
        )
        {
            try {
                hsize_t dims[1] = {N};
                auto dataspace  = H5::DataSpace(1, dims);
                auto dataset    = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_DOUBLE,   // assuming T is real/double
                    dataspace
                );
                dataset.write(H5::PredType::NATIVE_DOUBLE, vec.data());
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename T, std::uint64_t N>
        result_t<void> write_int_array(
            H5::Group& group,
            const std::string& name,
            const vector_t<T, N>& vec
        )
        {
            try {
                hsize_t dims[] = {N};
                auto dataspace = H5::DataSpace(1, dims);
                auto dataset   = group.createAttribute(
                    name,
                    H5::PredType::NATIVE_INT64,
                    dataspace
                );
                dataset.write(H5::PredType::NATIVE_INT64, vec.data());
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename T>
        result_t<void>
        write_string(H5::Group& group, const std::string& name, const T& value)
        {
            try {
                H5::StrType str_type(H5::PredType::C_S1, 256);
                // H5::StrType strdatatype(
                //     H5::PredType::C_S1,
                //     H5T_VARIABLE
                // );   // variable-length string
                auto dataspace = H5::DataSpace(H5S_SCALAR);
                auto dataset = group.createAttribute(name, str_type, dataspace);
                dataset.write(str_type, value.c_str());
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }

        template <typename Sim>
        result_t<void>
        write_mesh(const Sim& sim, H5::Group& group, std::size_t level)
        {
            try {
                auto& mesh = sim.mesh(level);

                h5::write_int_array(group, "shape", mesh.shape);
                h5::write_array(group, "bounds_min", mesh.bounds_min);
                h5::write_array(group, "bounds_max", mesh.bounds_max);
                h5::write_integer(group, "halo_radius", mesh.halo_radius);
                h5::write_scalar(
                    group,
                    "expansion_factor",
                    mesh.expansion_factor
                );
                h5::write_boolean(group, "mesh_motion", mesh.mesh_motion);

                // spacing types as comma-separated string
                auto spacing_str =
                    mesh.spacing_types |
                    fp::map([](auto s) { return serialize(s); }) |
                    fp::fold(
                        [](auto a, auto b) {
                            if (a == "") {
                                return b;
                            }

                            return a + "," + b;
                        },
                        std::string{}
                    );
                h5::write_string(group, "spacing_types", spacing_str);

                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(e.getDetailMsg());
            }
        }
    }   // namespace h5

    // core serialization functions
    template <typename Sim>
    result_t<void>
    write_primitives(const Sim& sim, H5::Group& group, size_t level)
    {
        constexpr auto Dims = Sim::dimensions;
        auto& hydro         = sim.hydro(level);

        const auto size  = hydro.prim.domain().size();
        const auto shape = hydro.prim.domain().shape();

        std::vector<real> component_data(size);

        try {
            for (size_t ii = 0; ii < size; ++ii) {
                component_data[ii] = hydro.prim[ii].rho;
            }
            {
                auto result =
                    h5::write_field(component_data.data(), shape, group, "rho");
                if (!result.is_ok()) {
                    return result;
                }
            }

            for (size_t dd = 0; dd < Dims; ++dd) {
                for (size_t ii = 0; ii < size; ++ii) {
                    component_data[ii] = hydro.prim[ii].vel[dd];
                }
                {
                    auto result = h5::write_field(
                        component_data.data(),
                        shape,
                        group,
                        "v" + std::to_string(dd + 1)
                    );
                    if (!result.is_ok()) {
                        return result;
                    }
                }
            }

            for (size_t ii = 0; ii < size; ++ii) {
                component_data[ii] = hydro.prim[ii].pre;
            }
            {
                auto result =
                    h5::write_field(component_data.data(), shape, group, "p");
                if (!result.is_ok()) {
                    return result;
                }
            }

            if constexpr (Sim::is_mhd) {
                for (size_t dd = 0; dd < Dims; ++dd) {
                    for (size_t ii = 0; ii < size; ++ii) {
                        component_data[ii] = hydro.prim[ii].mag[dd];
                    }
                    {
                        auto result = h5::write_field(
                            component_data.data(),
                            shape,
                            group,
                            "b" + std::to_string(dd + 1) + "_mean"
                        );
                        if (!result.is_ok()) {
                            return result;
                        }
                    }
                }
            }

            for (size_t ii = 0; ii < size; ++ii) {
                component_data[ii] = hydro.prim[ii].chi;
            }
            return h5::write_field(component_data.data(), shape, group, "chi");
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

    template <typename Sim>
    result_t<void>
    write_magnetic(const Sim& sim, H5::Group& group, std::size_t level)
    {
        if constexpr (!Sim::is_mhd) {
            return result_t<void>::ok();
        }

        const auto& bfield = sim.hydro(level).bfield;
        auto result        = h5::write_field(
            bfield[2].data(),
            bfield[2].domain().shape(),
            group,
            "b1"
        );
        if (!result.is_ok()) {
            return result;
        }

        result = h5::write_field(
            bfield[1].data(),
            bfield[1].domain().shape(),
            group,
            "b2"
        );
        if (!result.is_ok()) {
            return result;
        }

        return h5::write_field(
            bfield[0].data(),
            bfield[0].domain().shape(),
            group,
            "b3"
        );
    }

    template <typename Sim>
    result_t<void>
    write_level(Sim& sim, H5::Group& level_group, std::size_t level)
    {
        auto result = write_primitives(sim, level_group, level);
        if (!result.is_ok()) {
            return result;
        }

        result = write_magnetic(sim, level_group, level);
        if (!result.is_ok()) {
            return result;
        }

        auto mesh_group = level_group.createGroup("mesh");
        return h5::write_mesh(sim, mesh_group, level);
    }

    template <typename Sim>
    result_t<void> write_metadata(const Sim& sim, H5::Group& group)
    {
        try {
            const auto& meta = sim.metadata();

            h5::write_scalar(group, "time", meta.time);
            h5::write_scalar(group, "dt", meta.dt);
            h5::write_integer(group, "iteration", meta.iteration);
            h5::write_integer(group, "dimensions", meta.dimensions);
            h5::write_string(group, "regime", serialize(meta.regime));
            h5::write_string(group, "solver", serialize(meta.solver));
            h5::write_string(
                group,
                "shock_smoother",
                serialize(meta.shock_smoother)
            );
            h5::write_string(
                group,
                "reconstruction",
                serialize(meta.reconstruction)
            );
            h5::write_string(
                group,
                "timestepping",
                serialize(meta.timestepping)
            );
            h5::write_string(
                group,
                "coord_system",
                serialize(meta.coord_system)
            );
            h5::write_scalar(group, "plm_theta", meta.plm_theta);
            h5::write_scalar(group, "cfl_number", meta.cfl);
            h5::write_boolean(group, "is_mhd", meta.is_mhd);
            h5::write_boolean(group, "is_relativistic", meta.is_relativistic);
            h5::write_scalar(group, "adiabatic_index", meta.gamma);
            h5::write_scalar(group, "end_time", meta.tend);
            h5::write_string(group, "x1_spacing", serialize(meta.x1_spacing));
            h5::write_string(group, "x2_spacing", serialize(meta.x2_spacing));
            h5::write_string(group, "x3_spacing", serialize(meta.x3_spacing));
            h5::write_integer(group, "checkpoint_index", meta.checkpoint_index);
            h5::write_scalar(
                group,
                "checkpoint_interval",
                meta.checkpoint_interval
            );
            h5::write_integer(group, "halo_radius", meta.halo_radius);

            // bcs as comma-separated string
            auto bc_str = meta.boundary_conditions |
                          fp::map([](auto bc) { return serialize(bc); }) |
                          fp::fold(
                              [](auto a, auto b) {
                                  if (a == "") {
                                      return b;
                                  }
                                  return a + "," + b;
                              },
                              std::string{}
                          );
            h5::write_string(group, "boundary_conditions", bc_str);

            // resolution as comma-separated string
            auto res_str = meta.resolution |
                           fp::map([](auto x) { return std::to_string(x); }) |
                           fp::fold(
                               [](auto a, auto b) {
                                   if (a == "") {
                                       return b;
                                   }
                                   return a + "," + b;
                               },
                               std::string{}
                           );
            h5::write_string(group, "resolution", res_str);

            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

    // write hierarchy data for fmr
    template <typename H>
    result_t<void> write_hierarchy(const H& hierarchy, H5::Group& group)
    {
        try {
            const auto n_levels = hierarchy.num_levels;
            h5::write_integer(group, "num_levels", n_levels);

            if (n_levels > 1) {
                const auto ratios =
                    hierarchy.levels |
                    fp::map([&](const auto& lvl) { return lvl.ref_ratio; }) |
                    fp::collect<>;
                std::vector<hsize_t> dims{n_levels - 1};
                auto space = H5::DataSpace(1, dims.data());

                auto dataset = group.createDataSet(
                    "refinement_ratios",
                    H5::PredType::NATIVE_UINT64,
                    space
                );
                dataset.write(ratios.data(), H5::PredType::NATIVE_UINT64);
            }

            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

    // optional body system serialization
    template <typename T, std::uint64_t MaxBodies = 2>
    result_t<void>
    write_diagnostics(H5::Group& group, const T& diagnostics, real dt)
    {
        try {
            static auto prev_masses = diagnostics |
                                      fp::map([](const auto& body) -> real {
                                          return body.prev_mass_delta;
                                      }) |
                                      fp::collect<vector_t<real, MaxBodies>>;

            // extract and serialize force components
            for (std::uint64_t body_idx = 0; body_idx < MaxBodies; ++body_idx) {
                const auto& delta = diagnostics[body_idx];
                const auto& pmass = prev_masses[body_idx];

                // serialize force components for this body
                h5::write_array(
                    group,
                    "force_" + std::to_string(body_idx),
                    delta.force_delta
                );

                // serialize torque
                h5::write_array(
                    group,
                    "torque_" + std::to_string(body_idx),
                    delta.torque_delta
                );

                // serialize mass delta
                h5::write_scalar(
                    group,
                    "cumulative_mass_delta_" + std::to_string(body_idx),
                    delta.mass_delta
                );

                const auto dm   = delta.mass_delta - pmass;
                const auto mdot = (dt > 0) ? dm / dt : 0.0;

                h5::write_scalar(
                    group,
                    "accretion_rate_" + std::to_string(body_idx),
                    mdot
                );
                prev_masses[body_idx] = delta.mass_delta;
            }
            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

    template <typename Sim>
    result_t<void> write_bodies(const Sim& sim, H5::Group& group)
    {
        if (!sim.has_bodies()) {
            return result_t<void>::ok();
        }
        const auto& meta            = sim.metadata();
        static auto last_chkpt_time = meta.prev_checkpoint_time;
        const auto delta_time       = meta.checkpoint_time - last_chkpt_time;
        try {
            const auto& bodies      = sim.bodies();
            const auto& diagnostics = sim.diagnostics()->consolidate();
            auto diag_group         = group.createGroup("diagnostics");
            write_diagnostics(diag_group, diagnostics, delta_time);

            h5::write_integer(group, "body_count", bodies.size());
            h5::write_string(group, "system_name", bodies.name());
            h5::write_string(
                group,
                "reference_frame",
                bodies.reference_frame()
            );

            // serialize binary parameters if present
            if (bodies.binary_params_) {
                auto binary_group  = group.createGroup("binary_params");
                const auto& params = bodies.binary_params();
                h5::write_scalar(binary_group, "total_mass", params.total_mass);
                h5::write_scalar(binary_group, "semi_major", params.semi_major);
                h5::write_scalar(
                    binary_group,
                    "eccentricity",
                    params.eccentricity
                );
                h5::write_scalar(binary_group, "mass_ratio", params.mass_ratio);
                h5::write_scalar(
                    binary_group,
                    "orbital_period",
                    params.orbital_period
                );
                h5::write_boolean(
                    binary_group,
                    "is_circular_orbit",
                    params.is_circular_orbit
                );
            }

            // write each body
            bodies.visit_all([&](const auto& body) {
                using body_t = std::decay_t<decltype(body)>;
                auto body_group =
                    group.createGroup("body_" + std::to_string(body.idx));

                h5::write_scalar(body_group, "mass", body.mass);
                h5::write_scalar(body_group, "radius", body.radius);
                h5::write_array(body_group, "position", body.position);
                h5::write_array(body_group, "velocity", body.velocity);

                // capabilities
                auto caps = body.caps();
                h5::write_integer(body_group, "capabilities", caps);

                if constexpr (body::has_gravitational_capability_c<body_t>) {
                    h5::write_scalar(
                        body_group,
                        "softening_length",
                        body::softening_length(body)
                    );
                }

                if constexpr (body::has_accretion_capability_c<body_t>) {
                    h5::write_scalar(body_group, "sink_rate", sink_rate(body));
                    h5::write_scalar(
                        body_group,
                        "sink_delta",
                        body::sink_delta(body)
                    );
                    h5::write_scalar(
                        body_group,
                        "accretion_radius",
                        body::accretion_radius(body)
                    );
                }
            });
            last_chkpt_time = meta.checkpoint_time;
            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

    template <typename Sim>
    result_t<void> serialize_sim_state(Sim& sim, const std::string& filename)
    {
        try {
            auto file = H5::H5File(filename, H5F_ACC_TRUNC);

            // write metadata
            return h5::create_group(file, "metadata")
                .and_then([&](auto meta_group) {
                    return write_metadata(sim, meta_group);
                })
                .and_then([&]() {
                    // write bodies if present
                    return h5::create_group(file, "bodies")
                        .and_then([&](auto body_group) {
                            return write_bodies(sim, body_group);
                        });
                })
                .and_then([&]() {
                    if (sim.has_refinement()) {
                        // write hierarchy and levels
                        return h5::create_group(file, "hierarchy")
                            .and_then([&](auto hierarchy_group) {
                                return write_hierarchy(
                                    sim.hierarchy(),
                                    hierarchy_group
                                );
                            })
                            .and_then([&]() {
                                // write each level
                                for (std::size_t lvl = 0;
                                     lvl < sim.num_levels();
                                     lvl++) {
                                    auto level_result =
                                        h5::create_group(
                                            file,
                                            "level_" + std::to_string(lvl)
                                        )
                                            .and_then([&](auto level_group) {
                                                return write_level(
                                                    sim,
                                                    level_group,
                                                    lvl
                                                );
                                            });

                                    if (!level_result.is_ok()) {
                                        return level_result;
                                    }
                                }
                                return result_t<void>::ok();
                            });
                    }
                    else {
                        // single level case
                        return h5::create_group(file, "level_0")
                            .and_then([&](auto level_group) {
                                return write_level(sim, level_group, 0);
                            });
                    }
                });
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(e.getDetailMsg());
        }
    }

}   // namespace simbi::io

#endif
