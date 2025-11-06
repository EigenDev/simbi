#ifndef LOADER_HPP
#define LOADER_HPP

#include "compat.hpp"
#include "containers/vector.hpp"
#include "ecs/builders.hpp"
#include "ecs/simulation.hpp"
#include "functional/monad/result.hpp"
#include "physics/hydro/physics.hpp"
#include "physics/ib/factory.hpp"
#include "utility/bimap.hpp"
#include "utility/config_dict.hpp"
#include "utility/enums.hpp"
#include "utility/init_conditions.hpp"
#include "write_traits.hpp"

#include <H5Cpp.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simbi::io {

    namespace h5 {
        // core read operations
        template <typename T, std::uint64_t Dims>
        result_t<void> read_field(
            T* data,
            const iarray<Dims>&,
            H5::Group& group,
            const std::string& name
        )
        {
            try {
                auto dataset = group.openDataSet(name);
                dataset.read(data, H5::PredType::NATIVE_DOUBLE);
                return result_t<void>::ok();
            }
            catch (const H5::Exception& e) {
                return result_t<void>::error(
                    "Failed to read field '" + name + "': " + e.getDetailMsg()
                );
            }
        }

        template <typename T>
        result_t<T> read_attribute(H5::Group& group, const std::string& name)
        {
            try {
                auto attr = group.openAttribute(name);
                T value;

                if constexpr (std::is_same_v<T, bool>) {
                    hbool_t h5_value;
                    attr.read(h5_pred_type<bool>::value(), &h5_value);
                    value = static_cast<bool>(h5_value);
                }
                else {
                    attr.read(h5_pred_type<T>::value(), &value);
                }
                return result_t<T>::ok(value);
            }
            catch (const H5::Exception& e) {
                return result_t<T>::error(
                    "Failed to read attribute '" + name +
                    "': " + e.getDetailMsg()
                );
            }
        }

        result_t<std::string> inline read_string(
            H5::Group& group,
            const std::string& name
        )
        {
            try {
                auto attr = group.openAttribute(name);
                H5::StrType str_type(H5::PredType::C_S1, 256);
                std::string value;
                value.resize(256);
                attr.read(str_type, value.data());
                // trim null terminators
                value = value.c_str();
                return result_t<std::string>::ok(value);
            }
            catch (const H5::Exception& e) {
                return result_t<std::string>::error(
                    "Failed to read string '" + name + "': " + e.getDetailMsg()
                );
            }
        }

        template <typename T, std::uint64_t N>
        result_t<vector_t<T, N>>
        read_array(H5::Group& group, const std::string& name)
        {
            try {
                auto attr = group.openAttribute(name);
                vector_t<T, N> vec;
                attr.read(H5::PredType::NATIVE_DOUBLE, vec.data());
                return result_t<vector_t<T, N>>::ok(vec);
            }
            catch (const H5::Exception& e) {
                return result_t<vector_t<T, N>>::error(
                    "Failed to read array '" + name + "': " + e.getDetailMsg()
                );
            }
        }

        template <typename T, std::uint64_t N>
        result_t<vector_t<T, N>>
        read_int_array(H5::Group& group, const std::string& name)
        {
            try {
                auto attr = group.openAttribute(name);
                vector_t<T, N> vec;
                attr.read(H5::PredType::NATIVE_INT64, vec.data());
                return result_t<vector_t<T, N>>::ok(vec);
            }
            catch (const H5::Exception& e) {
                return result_t<vector_t<T, N>>::error(
                    "Failed to read int array '" + name +
                    "': " + e.getDetailMsg()
                );
            }
        }

        // parse comma-separated strings (for boundary conditions, resolution,
        // etc.)
        inline std::vector<std::string> parse_csv_string(const std::string& csv)
        {
            std::vector<std::string> result;
            std::string current;
            for (char c : csv) {
                if (c == ',') {
                    if (!current.empty()) {
                        result.push_back(current);
                        current.clear();
                    }
                }
                else {
                    current += c;
                }
            }
            if (!current.empty()) {
                result.push_back(current);
            }
            return result;
        }

    }   // namespace h5

    template <typename primitive_t, std::uint64_t Dims>
    result_t<field_t<primitive_t, Dims>>
    read_primitives(H5::Group& level_group, const iarray<Dims>& shape)
    {
        try {
            auto domain     = make_domain(shape);
            const auto size = domain.size();

            // allocate all temporary SoA buffers
            std::vector<real> rho_data(size);
            std::vector<real> p_data(size);
            std::vector<real> chi_data(size);

            vector_t<std::vector<real>, Dims> v_data;
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                v_data[dd].resize(size);
            }

            vector_t<std::vector<real>, Dims> b_data;   // for MHD
            if constexpr (is_mhd_primitive_c<primitive_t>) {
                for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                    b_data[dd].resize(size);
                }
            }

            // --- read rho ---
            auto result =
                h5::read_field(rho_data.data(), shape, level_group, "rho");
            if (!result.is_ok()) {
                return result_t<field_t<primitive_t, Dims>>::error(
                    result.error()
                );
            }

            // --- read velocity components ---
            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                result = h5::read_field(
                    v_data[dd].data(),
                    shape,
                    level_group,
                    "v" + std::to_string(dd + 1)
                );
                if (!result.is_ok()) {
                    return result_t<field_t<primitive_t, Dims>>::error(
                        result.error()
                    );
                }
            }

            // --- read pressure ---
            result = h5::read_field(p_data.data(), shape, level_group, "p");
            if (!result.is_ok()) {
                return result_t<field_t<primitive_t, Dims>>::error(
                    result.error()
                );
            }

            // --- read magnetic fields (if MHD) ---
            if constexpr (is_mhd_primitive_c<primitive_t>) {
                for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                    result = h5::read_field(
                        b_data[dd].data(),
                        shape,
                        level_group,
                        "b" + std::to_string(dd + 1) + "_mean"
                    );
                    if (!result.is_ok()) {
                        return result_t<field_t<primitive_t, Dims>>::error(
                            result.error()
                        );
                    }
                }
            }

            // --- read chi (tracer) ---
            result = h5::read_field(chi_data.data(), shape, level_group, "chi");
            if (!result.is_ok()) {
                return result_t<field_t<primitive_t, Dims>>::error(
                    result.error()
                );
            }

            // allocate the final AoS field
            auto field = stored_field<primitive_t, Dims>(domain);

            // perform the single-pass SoA-to-AoS flip
            field = field.enum_map(
                [&](auto coord,
                    primitive_t elem) {   // elem is just a default struct
                    auto ii = domain.coord_to_linear(coord);

                    // aseemble the prim struct for this coordinate
                    elem.rho = rho_data[ii];
                    elem.pre = p_data[ii];
                    elem.chi = chi_data[ii];

                    for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                        elem.vel[dd] = v_data[dd][ii];
                    }

                    if constexpr (is_mhd_primitive_c<primitive_t>) {
                        for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                            elem.mag[dd] = b_data[dd][ii];
                        }
                    }
                    return elem;
                }
            );

            return result_t<field_t<primitive_t, Dims>>::ok(std::move(field));
        }
        catch (const H5::Exception& e) {
            return result_t<field_t<primitive_t, Dims>>::error(
                "Failed to read primitives: " + std::string(e.getDetailMsg())
            );
        }
    }

    // read staggered magnetic fields
    template <std::uint64_t Dims>
    result_t<vector_t<field_t<real, Dims>, Dims>>
    read_magnetic_fields(H5::Group& level_group, const iarray<Dims>& base_shape)
    {
        try {
            vector_t<field_t<real, Dims>, Dims> bfields;

            for (std::uint64_t dd = 0; dd < Dims; ++dd) {
                // compute staggered shape for this direction
                iarray<Dims> staggered_shape = base_shape;
                staggered_shape[dd] += 1;

                auto domain = make_domain(staggered_shape);
                auto field  = stored_field<real, Dims>(domain);

                std::vector<real> data(domain.size());
                auto result = h5::read_field(
                    data.data(),
                    staggered_shape,
                    level_group,
                    "b" + std::to_string(dd + 1)
                );

                if (!result.is_ok()) {
                    return result_t<vector_t<field_t<real, Dims>, Dims>>::error(
                        result.error()
                    );
                }
                field = field.coord_map([&data, domain](auto coord) {
                    auto ii = domain.coord_to_linear(coord);
                    return data[ii];
                });

                bfields[dd] = std::move(field);
            }

            return result_t<vector_t<field_t<real, Dims>, Dims>>::ok(
                std::move(bfields)
            );
        }
        catch (const H5::Exception& e) {
            return result_t<vector_t<field_t<real, Dims>, Dims>>::error(
                "Failed to read magnetic fields: " +
                std::string(e.getDetailMsg())
            );
        }
    }

    namespace {   // anonymous namespace for internal-linkage helpers

        /**
         * @brief translates a single hdf5 body group (e.g., /bodies/body_0)
         * into a config_dict_t.
         *
         * this function reads all known required and optional body
         * properties. the body factory will then infer the body type
         * from the *presence* of these optional properties.
         */
        template <std::uint64_t Dims>
        result_t<config_dict_t> read_body_as_config(H5::Group& body_group)
        {
            config_dict_t props;

            try {
                // --- read required attributes ---
                // if any of these fail, the whole function fails.
                auto mass_res = h5::read_attribute<real>(body_group, "mass");
                if (!mass_res.is_ok()) {
                    return result_t<config_dict_t>::error(
                        "failed to read 'mass': " + mass_res.error()
                    );
                }
                props["mass"] = mass_res.value();

                auto radius_res =
                    h5::read_attribute<real>(body_group, "radius");
                if (!radius_res.is_ok()) {
                    return result_t<config_dict_t>::error(
                        "failed to read 'radius': " + radius_res.error()
                    );
                }
                props["radius"] = radius_res.value();

                auto pos_res =
                    h5::read_array<real, Dims>(body_group, "position");
                if (!pos_res.is_ok()) {
                    return result_t<config_dict_t>::error(
                        "failed to read 'position': " + pos_res.error()
                    );
                }
                props["position"] = pos_res.value();

                auto vel_res =
                    h5::read_array<real, Dims>(body_group, "velocity");
                if (!vel_res.is_ok()) {
                    return result_t<config_dict_t>::error(
                        "failed to read 'velocity': " + vel_res.error()
                    );
                }
                props["velocity"] = vel_res.value();

                // --- read optional attributes ---
                // if these fail, we just ignore them, as they may not exist
                // for this body type.
                auto soft_res =
                    h5::read_attribute<real>(body_group, "softening_length");
                if (soft_res.is_ok()) {
                    props["softening_length"] = soft_res.value();
                }

                auto sink_r_res =
                    h5::read_attribute<real>(body_group, "sink_rate");
                if (sink_r_res.is_ok()) {
                    props["sink_rate"] = sink_r_res.value();
                }

                auto sink_d_res =
                    h5::read_attribute<real>(body_group, "sink_delta");
                if (sink_d_res.is_ok()) {
                    props["sink_delta"] = sink_d_res.value();
                }

                auto accr_r_res =
                    h5::read_attribute<real>(body_group, "accretion_radius");
                if (accr_r_res.is_ok()) {
                    props["accretion_radius"] = accr_r_res.value();
                }

                auto total_acc_res =
                    h5::read_attribute<real>(body_group, "total_accreted_mass");
                if (total_acc_res.is_ok()) {
                    props["total_accreted_mass"] = total_acc_res.value();
                }

                auto inertia_res =
                    h5::read_attribute<real>(body_group, "inertia");
                if (inertia_res.is_ok()) {
                    props["inertia"] = inertia_res.value();
                }

                auto no_slip_res =
                    h5::read_attribute<bool>(body_group, "apply_no_slip");
                if (no_slip_res.is_ok()) {
                    props["apply_no_slip"] = no_slip_res.value();
                }

                return result_t<config_dict_t>::ok(std::move(props));
            }
            catch (const H5::Exception& e) {
                return result_t<config_dict_t>::error(
                    "hdf5 exception in read_body_as_config: " +
                    std::string(e.getDetailMsg())
                );
            }
        }

    }   // end anonymous namespace

    /**
     * @brief reads the /bodies hdf5 group and populates the
     * initial_conditions_t object.
     *
     * this function acts as an "anti-serializer" by reconstructing
     * the config_dict_t structure that the body_factory expects.
     * it handles both individual bodies and binary systems.
     */
    template <std::uint64_t Dims>
    result_t<void> read_body_config(
        H5::Group& bodies_group,
        initial_conditions_t& init   // non-const: this is the output
    )
    {
        try {
            auto count_res =
                h5::read_attribute<std::uint64_t>(bodies_group, "body_count");
            if (!count_res.is_ok()) {
                return result_t<void>::error(
                    "failed to read 'body_count': " + count_res.error()
                );
            }

            auto body_count = count_res.value();
            if (body_count == 0) {
                return result_t<void>::ok();   // no bodies to load
            }

            // system_name is optional; default to "" (individual)
            auto system_name_res = h5::read_string(bodies_group, "system_name");
            auto system_name =
                system_name_res.is_ok() ? system_name_res.value() : "";

            // --- case 1: load a "binary_system" ---
            if (system_name == "binary_system") {
                config_dict_t sys_props;
                sys_props["system_type"] = "binary";

                auto frame_res =
                    h5::read_string(bodies_group, "reference_frame");
                if (!frame_res.is_ok()) {
                    return result_t<void>::error(
                        "failed to read 'reference_frame': " + frame_res.error()
                    );
                }
                sys_props["reference_frame"] = frame_res.value();

                // rebuild the "binary_config" dictionary
                config_dict_t binary_config;
                auto binary_group = bodies_group.openGroup("binary_params");

                // read all binary parameters
#define READ_BINARY_PARAM(name, type)                                          \
    auto name##_res = h5::read_attribute<type>(binary_group, #name);           \
    if (!name##_res.is_ok())                                                   \
        return result_t<void>::error(                                          \
            "failed to read binary param '" #name "': " + name##_res.error()   \
        );                                                                     \
    binary_config[#name] = name##_res.value();

                READ_BINARY_PARAM(total_mass, real);
                READ_BINARY_PARAM(semi_major, real);
                READ_BINARY_PARAM(eccentricity, real);
                READ_BINARY_PARAM(mass_ratio, real);
                READ_BINARY_PARAM(orbital_period, real);
                READ_BINARY_PARAM(is_circular_orbit, bool);
#undef READ_BINARY_PARAM

                // load the component bodies
                std::list<config_dict_t> components;
                for (std::uint64_t i = 0; i < body_count; ++i) {
                    auto body_group =
                        bodies_group.openGroup("body_" + std::to_string(i));
                    auto body_cfg_res = read_body_as_config<Dims>(body_group);

                    if (!body_cfg_res.is_ok()) {
                        return result_t<void>::error(
                            "failed to load body " + std::to_string(i) + ": " +
                            body_cfg_res.error()
                        );
                    }
                    components.push_back(body_cfg_res.value());
                }
                binary_config["components"] = std::move(components);

                sys_props["binary_config"] = std::move(binary_config);

                // set the "body_system" property in the main init object
                init.config["body_system"] = std::move(sys_props);

                // --- case 2: load "individual bodies" ---
            }
            else {
                // clear any existing default bodies and load from checkpoint
                init.immersed_bodies.clear();
                for (std::uint64_t i = 0; i < body_count; ++i) {
                    auto body_group =
                        bodies_group.openGroup("body_" + std::to_string(i));
                    auto body_cfg_res = read_body_as_config<Dims>(body_group);

                    if (!body_cfg_res.is_ok()) {
                        return result_t<void>::error(
                            "failed to load body " + std::to_string(i) + ": " +
                            body_cfg_res.error()
                        );
                    }
                    init.immersed_bodies.push_back(body_cfg_res.value());
                }
            }

            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(
                "hdf5 exception in read_body_config: " +
                std::string(e.getDetailMsg())
            );
        }
    }

    // read metadata and update initial_conditions_t
    inline result_t<void>
    read_metadata(H5::Group& meta_group, initial_conditions_t& init)
    {
        try {
            auto time_result = h5::read_attribute<real>(meta_group, "time");
            if (!time_result.is_ok()) {
                return result_t<void>::error(time_result.error());
            }
            init.time = time_result.value();

            auto dt_result = h5::read_attribute<real>(meta_group, "dt");
            if (dt_result.is_ok()) {
                // dt is informational, not critical
            }

            auto iter_result =
                h5::read_attribute<std::uint64_t>(meta_group, "iteration");
            if (iter_result.is_ok()) {
                // iteration is informational
            }

            auto regime_result = h5::read_string(meta_group, "regime");
            if (!regime_result.is_ok()) {
                return result_t<void>::error(regime_result.error());
            }
            init.regime = regime_result.value();

            auto solver_result = h5::read_string(meta_group, "solver");
            if (!solver_result.is_ok()) {
                return result_t<void>::error(solver_result.error());
            }
            init.solver = solver_result.value();

            auto reconstruct_result =
                h5::read_string(meta_group, "reconstruction");
            if (!reconstruct_result.is_ok()) {
                return result_t<void>::error(reconstruct_result.error());
            }
            init.reconstruct = reconstruct_result.value();

            auto coord_result = h5::read_string(meta_group, "coord_system");
            if (!coord_result.is_ok()) {
                return result_t<void>::error(coord_result.error());
            }
            init.coord_system = coord_result.value();

            auto gamma_result =
                h5::read_attribute<real>(meta_group, "adiabatic_index");
            if (!gamma_result.is_ok()) {
                return result_t<void>::error(gamma_result.error());
            }
            init.gamma = gamma_result.value();

            auto cfl_result =
                h5::read_attribute<real>(meta_group, "cfl_number");
            if (!cfl_result.is_ok()) {
                return result_t<void>::error(cfl_result.error());
            }
            init.cfl = cfl_result.value();

            auto plm_result = h5::read_attribute<real>(meta_group, "plm_theta");
            if (!plm_result.is_ok()) {
                return result_t<void>::error(plm_result.error());
            }
            init.plm_theta = plm_result.value();

            auto tend_result = h5::read_attribute<real>(meta_group, "end_time");
            if (!tend_result.is_ok()) {
                return result_t<void>::error(tend_result.error());
            }
            init.tend = tend_result.value();

            auto mhd_result = h5::read_attribute<bool>(meta_group, "is_mhd");
            if (!mhd_result.is_ok()) {
                return result_t<void>::error(mhd_result.error());
            }
            init.is_mhd = mhd_result.value();

            auto rel_result =
                h5::read_attribute<bool>(meta_group, "is_relativistic");
            if (!rel_result.is_ok()) {
                return result_t<void>::error(rel_result.error());
            }
            init.is_relativistic = rel_result.value();

            auto bc_result = h5::read_string(meta_group, "boundary_conditions");
            if (!bc_result.is_ok()) {
                return result_t<void>::error(bc_result.error());
            }
            init.boundary_conditions = h5::parse_csv_string(bc_result.value());

            auto res_result = h5::read_string(meta_group, "resolution");
            if (!res_result.is_ok()) {
                return result_t<void>::error(res_result.error());
            }
            auto res_strings = h5::parse_csv_string(res_result.value());
            if (res_strings.size() >= 1) {
                init.nx = std::stoll(res_strings[2]);
            }
            if (res_strings.size() >= 2) {
                init.ny = std::stoll(res_strings[1]);
            }
            if (res_strings.size() >= 3) {
                init.nz = std::stoll(res_strings[0]);
            }

            auto x1_spacing_result = h5::read_string(meta_group, "x1_spacing");
            if (x1_spacing_result.is_ok()) {
                init.x1_spacing = x1_spacing_result.value();
            }

            auto x2_spacing_result = h5::read_string(meta_group, "x2_spacing");
            if (x2_spacing_result.is_ok()) {
                init.x2_spacing = x2_spacing_result.value();
            }

            auto x3_spacing_result = h5::read_string(meta_group, "x3_spacing");
            if (x3_spacing_result.is_ok()) {
                init.x3_spacing = x3_spacing_result.value();
            }

            auto halo_result =
                h5::read_attribute<std::uint64_t>(meta_group, "halo_radius");
            if (!halo_result.is_ok()) {
                return result_t<void>::error(halo_result.error());
            }
            init.halo_radius = halo_result.value();

            auto chkpt_idx_result = h5::read_attribute<std::uint64_t>(
                meta_group,
                "checkpoint_index"
            );
            if (chkpt_idx_result.is_ok()) {
                init.checkpoint_index = chkpt_idx_result.value();
            }

            auto chkpt_int_result =
                h5::read_attribute<real>(meta_group, "checkpoint_interval");
            if (chkpt_int_result.is_ok()) {
                init.checkpoint_interval = chkpt_int_result.value();
            }

            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(
                "Failed to read metadata: " + std::string(e.getDetailMsg())
            );
        }
    }

    template <std::uint64_t Dims, Geometry G>
    result_t<mesh::mesh_config_t<Dims, G>>
    read_mesh_config(H5::Group& mesh_group, bool is_mhd)
    {
        try {
            auto halo_result =
                h5::read_attribute<std::uint64_t>(mesh_group, "halo_radius");
            if (!halo_result.is_ok()) {
                return result_t<mesh::mesh_config_t<Dims, G>>::error(
                    halo_result.error()
                );
            }
            auto halo_radius = static_cast<std::int64_t>(halo_result.value());

            auto shape_result =
                h5::read_int_array<std::int64_t, Dims>(mesh_group, "shape");
            if (!shape_result.is_ok()) {
                return result_t<mesh::mesh_config_t<Dims, G>>::error(
                    shape_result.error()
                );
            }
            auto shape = shape_result.value();
            iarray<Dims> full_shape;
            for (std::size_t ii = 0; ii < shape.size(); ii++) {
                full_shape[ii] = shape[ii] + 2 * halo_radius;
            }
            auto full_domain = make_domain(full_shape);
            auto domain      = domain_algebra::contract(
                full_domain,
                ones<Dims, std::int64_t>() * halo_radius
            );

            auto bounds_min_result =
                h5::read_array<real, Dims>(mesh_group, "bounds_min");
            if (!bounds_min_result.is_ok()) {
                return result_t<mesh::mesh_config_t<Dims, G>>::error(
                    bounds_min_result.error()
                );
            }
            auto bounds_min = bounds_min_result.value();

            auto bounds_max_result =
                h5::read_array<real, Dims>(mesh_group, "bounds_max");
            if (!bounds_max_result.is_ok()) {
                return result_t<mesh::mesh_config_t<Dims, G>>::error(
                    bounds_max_result.error()
                );
            }
            auto bounds_max = bounds_max_result.value();

            // auto expansion_result =
            //     h5::read_attribute<real>(mesh_group, "expansion_factor");
            // if (!expansion_result.is_ok()) {
            //     return result_t<mesh::mesh_config_t<Dims, G>>::error(
            //         expansion_result.error()
            //     );
            // }
            // auto expansion_factor = expansion_result.value();

            // auto motion_result = h5::read_attribute<bool>(mesh_group,
            // "mesh_motion"); if (!motion_result.is_ok()) {
            //     return result_t<mesh::mesh_config_t<Dims, G>>::error(
            //         motion_result.error()
            //     );
            // }
            // auto mesh_motion = motion_result.value();

            auto spacing_result = h5::read_string(mesh_group, "spacing_types");
            if (!spacing_result.is_ok()) {
                return result_t<mesh::mesh_config_t<Dims, G>>::error(
                    spacing_result.error()
                );
            }
            auto spacing_strings = h5::parse_csv_string(spacing_result.value());

            vector_t<Cellspacing, Dims> spacing_types;
            for (std::uint64_t d = 0; d < Dims && d < spacing_strings.size();
                 ++d) {
                spacing_types[d] = deserialize<Cellspacing>(spacing_strings[d]);
            }

            vector_t<domain_t<Dims>, Dims> face_domain;
            for (std::uint64_t ii = 0; ii < Dims; ii++) {
                auto amount     = iarray<Dims>{0};
                amount[ii]      = 1;
                face_domain[ii] = domain_algebra::expand_end(
                    make_domain(domain.shape()),
                    amount
                );

                if (is_mhd) {
                    face_domain[ii].start[(ii + 1) % Dims] += 1;
                    face_domain[ii].fin[(ii + 1) % Dims] += 1;
                    face_domain[ii].start[(ii + 2) % Dims] += 1;
                    face_domain[ii].fin[(ii + 2) % Dims] += 1;
                }
            }

            // construct mesh config
            mesh::mesh_config_t<Dims, G> config{
              .shape         = shape,
              .full_shape    = full_shape,
              .halo_radius   = halo_radius,
              .full_domain   = make_domain(full_shape),
              .domain        = domain,
              .face_domain   = face_domain,
              .bounds_min    = bounds_min,
              .bounds_max    = bounds_max,
              .spacing_types = spacing_types,
              .dx            = {(bounds_max[0] - bounds_min[0]) / shape[0]}
            };

            return result_t<mesh::mesh_config_t<Dims, G>>::ok(
                std::move(config)
            );
        }
        catch (const H5::Exception& e) {
            return result_t<mesh::mesh_config_t<Dims, G>>::error(
                "failed to read mesh config: " + std::string(e.getDetailMsg())
            );
        }
    }

    // load a single level from checkpoint

    template <Regime R, std::uint64_t Dims, Geometry G, typename EoS>
    result_t<void> load_level(
        ecs::simulation_t<R, Dims, G, EoS>& sim,
        H5::H5File& file,
        std::size_t level,
        const initial_conditions_t& init
    )
    {
        using namespace ecs;
        using primitive_t = typename simulation_t<R, Dims, G, EoS>::primitive_t;
        using conserved_t = typename simulation_t<R, Dims, G, EoS>::conserved_t;

        try {
            auto level_group = file.openGroup("level_" + std::to_string(level));
            auto mesh_group  = level_group.openGroup("mesh");

            // read and construct mesh
            auto mesh_result =
                read_mesh_config<Dims, G>(mesh_group, init.is_mhd);
            if (!mesh_result.is_ok()) {
                return result_t<void>::error(mesh_result.error());
            }
            auto mesh  = mesh_result.value();
            auto shape = mesh.shape;

            // read primitives
            auto prim_result = read_primitives<primitive_t, Dims>(
                level_group,
                mesh.full_shape
            );
            if (!prim_result.is_ok()) {
                return result_t<void>::error(prim_result.error());
            }
            auto prim_field = prim_result.value();

            // convert primitives to conserved
            auto domain     = make_domain(mesh.full_shape);
            auto cons_field = stored_field<conserved_t, Dims>(domain);
            cons_field =
                cons_field.coord_map([prim_field, g = init.gamma](auto coord) {
                    return hydro::to_conserved(prim_field[coord], g);
                });

            // read magnetic fields if MHD
            vector_t<field_t<real, Dims>, Dims> bfields;
            if constexpr (R == Regime::MHD || R == Regime::RMHD) {
                auto bfield_result =
                    read_magnetic_fields<Dims>(level_group, shape);
                if (!bfield_result.is_ok()) {
                    return result_t<void>::error(bfield_result.error());
                }
                bfields = bfield_result.value();
            }

            // create flux fields
            auto flux_fields =
                fp::range(Dims) | fp::map([&](std::uint64_t dir) {
                    return field(
                        mesh.face_domain[dir],
                        fp::default_t<conserved_t>{}
                    );
                }) |
                fp::collect<vector_t<field_t<conserved_t, Dims>, Dims>>;

            // add hydro fields to level
            sim.registry.add(
                sim.levels[level],
                hydro_fields_t<conserved_t, primitive_t, Dims>{
                  .cons   = std::move(cons_field),
                  .prim   = std::move(prim_field),
                  .flux   = std::move(flux_fields),
                  .bfield = std::move(bfields)
                }
            );

            // add mesh geometry
            sim.registry.add(
                sim.levels[level],
                mesh_geometry_t<Dims, G>{.config = std::move(mesh)}
            );

            // add level info (refinement ratio handled separately for FMR)
            // sim.registry.add(sim.levels[level],
            // build_level_info<Dims>(level));

            return result_t<void>::ok();
        }
        catch (const H5::Exception& e) {
            return result_t<void>::error(
                "failed to load level " + std::to_string(level) + ": " +
                std::string(e.getDetailMsg())
            );
        }
    }

    template <Regime R, std::uint64_t Dims, Geometry G, typename EoS>
    result_t<ecs::simulation_t<R, Dims, G, EoS>> load_checkpoint(
        const std::string& filename,
        initial_conditions_t init,
        std::function<real(real)> const& scale_factor,
        std::function<real(real)> const& scale_factor_derivative
    )
    {
        using namespace ecs;
        using namespace body;
        using namespace body::factory;
        try {
            auto file = H5::H5File(filename, H5F_ACC_RDONLY);

            // read and update metadata
            auto meta_group  = file.openGroup("metadata");
            auto meta_result = read_metadata(meta_group, init);
            if (!meta_result.is_ok()) {
                return result_t<simulation_t<R, Dims, G, EoS>>::error(
                    meta_result.error()
                );
            }

            // deternube dimensionality from resolution
            init.dimensionality = (init.nz > 1) ? 3 : ((init.ny > 1) ? 2 : 1);
            if (init.dimensionality != Dims) {
                return result_t<simulation_t<R, Dims, G, EoS>>::error(
                    "Dimension mismatch: checkpoint has " +
                    std::to_string(init.dimensionality) + "D data but " +
                    std::to_string(Dims) + "D was requested"
                );
            }

            // check if FMR is present
            bool has_fmr             = false;
            std::uint64_t num_levels = 1;
            std::vector<std::uint64_t> ref_ratios;
            try {
                auto hierarchy_group = file.openGroup("hierarchy");
                has_fmr              = true;

                // read num_levels
                auto num_levels_result = h5::read_attribute<std::uint64_t>(
                    hierarchy_group,
                    "num_levels"
                );
                if (!num_levels_result.is_ok()) {
                    return result_t<simulation_t<R, Dims, G, EoS>>::error(
                        "loading the number of levels failed"
                    );
                }
                num_levels = num_levels_result.value();

                // read refinement ratios
                if (num_levels > 1) {
                    ref_ratios.resize(num_levels - 1);
                    try {
                        auto dataset =
                            hierarchy_group.openDataSet("refinement_ratios");
                        dataset.read(
                            ref_ratios.data(),
                            H5::PredType::NATIVE_UINT64
                        );
                    }
                    catch (const H5::Exception& e) {
                        return result_t<simulation_t<R, Dims, G, EoS>>::error(
                            "failed to read refinement ratios: " +
                            std::string(e.getDetailMsg())
                        );
                    }
                }
            }
            catch (const H5::Exception&) {
                has_fmr = false;   // "hierarchy" group doesn't exist
            }

            // create simulation structure (without data initially)
            simulation_t<R, Dims, G, EoS> sim;

            // create global metadata entity
            sim.global = sim.registry.create();
            sim.registry.add(
                sim.global,
                ecs::build_metadata_component<Dims>(init)
            );
            sim.registry.add(
                sim.global,
                ecs::build_sources_component<Dims>(init)
            );

            // load level 0
            auto level_0 = sim.registry.create();
            sim.levels.push_back(level_0);

            auto mesh = mesh::mesh_config_t<Dims, G>::from_init_conditions(
                init,
                scale_factor,
                scale_factor_derivative
            );
            sim.registry.add(
                level_0,
                ecs::mesh_geometry_t<Dims, G>{.config = std::move(mesh)}
            );
            auto load_result = load_level(sim, file, 0, init);
            if (!load_result.is_ok()) {
                return result_t<simulation_t<R, Dims, G, EoS>>::error(
                    load_result.error()
                );
            }
            sim.registry.add(
                level_0,
                level_info_t{.level_id = 0, .refinement_ratio = 1}
            );

            try {
                auto bodies_group = file.openGroup("bodies");
                auto body_cfg_res = read_body_config<Dims>(bodies_group, init);
                if (!body_cfg_res.is_ok()) {
                    return result_t<simulation_t<R, Dims, G, EoS>>::error(
                        body_cfg_res.error()
                    );
                }
            }
            catch (const H5::Exception&) {
                // no "bodies" group, which is fine.
            }

            auto bodies      = create_body_collection_from_init<Dims>(init);
            auto diagnostics = body::create_diagnostics_accumulator<Dims>();

            // add bodies if enabled
            if (bodies) {
                sim.registry.add(
                    sim.global,
                    immersed_bodies_t<Dims>{.bodies = std::move(bodies.value())}
                );

                bodies->visit_all([&](const auto& body) {
                    using body_type = std::decay_t<decltype(body)>;
                    auto delta      = body::body_delta_t<Dims>{
                           .idx          = body.idx,
                           .force_delta  = body.force,
                           .torque_delta = body.torque,
                           .mass_delta   = 0.0
                    };
                    if constexpr (body::has_accretion_capability_c<body_type>) {
                        delta.prev_mass_delta = body::total_accreted_mass(body);
                    }
                    diagnostics->accumulate_delta(delta);
                });

                sim.registry.add(
                    sim.global,
                    body_info_t<Dims>{.diagnostics = std::move(diagnostics)}
                );
            }

            if (has_fmr) {
                for (std::uint64_t lvl = 1; lvl < num_levels; ++lvl) {
                    auto level_entity = sim.registry.create();
                    sim.levels.push_back(level_entity);

                    // looad this level's data (mesh, prims, etc.)
                    // load_level will use sim.levels[lvl] to get the entity
                    auto load_result = load_level(sim, file, lvl, init);
                    if (!load_result.is_ok()) {
                        result_t<simulation_t<R, Dims, G, EoS>>::error(
                            "Failed to load level: " + load_result.error()
                        );
                    }

                    // read the parent_coverage domain we saved in Step 1
                    domain_t<Dims> parent_coverage;
                    try {
                        auto level_group =
                            file.openGroup("level_" + std::to_string(lvl));

                        auto start_res = h5::read_int_array<std::int64_t, Dims>(
                            level_group,
                            "parent_coverage_start"
                        );
                        auto fin_res = h5::read_int_array<std::int64_t, Dims>(
                            level_group,
                            "parent_coverage_fin"
                        );

                        if (!start_res.is_ok()) {
                            result_t<simulation_t<R, Dims, G, EoS>>::error(
                                "Failed to read starting parent coverage: " +
                                start_res.error()
                            );
                        }
                        if (!fin_res.is_ok()) {
                            result_t<simulation_t<R, Dims, G, EoS>>::error(
                                "Failed to read ending parent coverage: " +
                                fin_res.error()
                            );
                        }

                        parent_coverage.start = start_res.value();
                        parent_coverage.fin   = fin_res.value();
                    }
                    catch (const H5::Exception& e) {
                        result_t<simulation_t<R, Dims, G, EoS>>::error(
                            "Failed to read paren coverage: " +
                            std::string(e.getDetailMsg())
                        );
                    }

                    // add the FMR components to link the levels
                    sim.registry.add(
                        level_entity,
                        level_info_t{
                          .level_id         = lvl,
                          .refinement_ratio = ref_ratios[lvl - 1]
                        }
                    );

                    sim.registry.add(
                        level_entity,
                        refinement_child_t<Dims>{
                          .parent = sim.levels[lvl - 1],   // link to parent
                          .parent_coverage = parent_coverage
                        }
                    );
                }

                // --- rebuild the global hierarchy component
                mesh::fmr::mesh_hierarchy_t<Dims> hierarchy;
                hierarchy.num_levels = num_levels;

                for (std::uint64_t lvl = 0; lvl < num_levels; ++lvl) {
                    const auto& mesh = sim.mesh(lvl);
                    const auto& info = sim.level_info(lvl);

                    domain_t<Dims> p_cov =
                        (lvl == 0)
                            ? mesh.domain
                            : sim.registry
                                  .template get<refinement_child_t<Dims>>(
                                      sim.levels[lvl]
                                  )
                                  .parent_coverage;

                    if (lvl > 0) {
                        hierarchy.levels[lvl].ref_ratio =
                            hierarchy.levels[lvl - 1].ref_ratio *
                            info.refinement_ratio;
                    }
                    else {
                        hierarchy.levels[lvl].ref_ratio = 1;
                    }

                    hierarchy.levels[lvl].level_id    = lvl;
                    hierarchy.levels[lvl].domain      = mesh.domain;
                    hierarchy.levels[lvl].full_domain = mesh.full_domain;
                    hierarchy.levels[lvl].dx          = mesh.dx;
                    hierarchy.levels[lvl].parent_level_id =
                        (lvl == 0) ? 0 : lvl - 1;
                    hierarchy.levels[lvl].parent_coverage = p_cov;
                    hierarchy.levels[lvl].physical_min    = mesh.bounds_min;
                    hierarchy.levels[lvl].physical_max    = mesh.bounds_max;
                    hierarchy.levels[lvl].face_domains    = mesh.face_domain;
                }

                sim.registry.add(
                    sim.global,
                    fmr_hierarchy_t<Dims>{std::move(hierarchy)}
                );
            }
            // TODO: Load body system if present

            return result_t<simulation_t<R, Dims, G, EoS>>::ok(std::move(sim));
        }
        catch (const H5::Exception& e) {
            return result_t<simulation_t<R, Dims, G, EoS>>::error(
                "Failed to load checkpoint: " + std::string(e.getDetailMsg())
            );
        }
    }

}   // namespace simbi::io

#endif   // LOADER_HPP
