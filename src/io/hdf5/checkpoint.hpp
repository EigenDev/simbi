/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            checkpoint.hpp
 *  * @brief
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-26
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-26      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "config.hpp"
#include "physics/hydro/schemes/ib/serialization/body_serialization.hpp"
#include "physics/hydro/schemes/ib/serialization/system_serialization.hpp"
#include <string>
#include <unordered_map>

using namespace simbi::ibsystem;
std::unordered_map<simbi::Cellspacing, std::string> const cell2str = {
  {simbi::Cellspacing::LOG, "log"},
  {simbi::Cellspacing::LINEAR, "linear"}
  // {"log-linear",Cellspacing},
  // {"linear-log",Cellspacing},
};
namespace simbi {
    namespace io {
        // forward declaration
        template <typename T>
        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const T& state
        );

        template <typename Sim_type>
        void write_to_file(Sim_type& sim_state, auto& table)
        {
            sim_state.sync_to_host();
            if constexpr (Sim_type::regime == "srmhd") {
                sim_state.bstag1.sync_to_host();
                sim_state.bstag2.sync_to_host();
                sim_state.bstag3.sync_to_host();
            }
            auto data_directory             = sim_state.data_directory();
            auto step                       = sim_state.checkpoint_index();
            static lint tchunk_order_of_mag = 2;
            const auto temporal_order_of_mag =
                std::floor(std::log10(sim_state.time()));
            if (temporal_order_of_mag > tchunk_order_of_mag) {
                tchunk_order_of_mag += 1;
            }

            std::string tnow;
            if (sim_state.dlogt() != 0) {
                const auto temporal_order_of_mag = std::floor(std::log10(step));
                if (temporal_order_of_mag > tchunk_order_of_mag) {
                    tchunk_order_of_mag += 1;
                }
                tnow = format_real(step);
            }
            else if (!sim_state.is_in_failure_state()) {
                tnow = format_real(sim_state.checkpoint_identifier());
            }
            else {
                if (sim_state.has_been_interrupted()) {
                    tnow = "interrupted";
                }
                else {
                    tnow = "crashed";
                }
            }
            const auto filename = string_format(
                "%d.chkpt." + tnow + ".h5",
                sim_state.checkpoint_zones()
            );
            sim_state.update_next_checkpoint_location();
            write_hdf5(data_directory, filename, sim_state);
            table.post_info(
                "Checkpoint written to: " + data_directory + filename
            );
            table.refresh();
        }

        template <typename T>
        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const T& state
        )
        {
            const auto full_filename = data_directory + filename;

            // Create a new file using the default property list.
            H5::H5File file(full_filename, H5F_ACC_TRUNC);

            // Create the data space for the dataset.
            hsize_t dims[1]  = {state.total_zones()};
            hsize_t dimxv[1] = {
              state.full_xvertex_policy().get_active_extent()
            };
            hsize_t dimyv[1] = {
              state.full_yvertex_policy().get_active_extent()
            };
            hsize_t dimzv[1] = {
              state.full_zvertex_policy().get_active_extent()
            };
            hsize_t dim_bc[1] = {
              state.solver_config().boundary_conditions().size()
            };

            H5::DataSpace hdataspace(1, dims);
            H5::DataSpace hdataspace_bc(1, dim_bc);
            H5::DataSpace b1dataspace(1, dimxv);
            H5::DataSpace b2dataspace(1, dimyv);
            H5::DataSpace b3dataspace(1, dimzv);

            // Create empty dataspace for attributes
            H5::DataSpace attr_dataspace(H5S_NULL);
            H5::DataType attr_type(H5::PredType::NATIVE_INT);
            // create attribute data space that houses scalar type
            H5::DataSpace scalar_dataspace(H5S_SCALAR);

            //==================================================================
            // DATA TYPES
            //==================================================================
            // Define the real-type for the data in the file.
            H5::DataType real_type = H5::PredType::NATIVE_DOUBLE;

            // int-type
            H5::DataType int_type = H5::PredType::NATIVE_INT;

            // bool-type
            H5::DataType bool_type = H5::PredType::NATIVE_HBOOL;

            // scalar-type
            H5::DataType scalar_type = H5::PredType::NATIVE_DOUBLE;

            // Define the string type for variable string length
            H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

            //==================================================================
            //  BOUNDARY CONDITIONS
            //==================================================================
            // convert the string to a char array
            auto bc_s = state.solver_config().boundary_conditions_c_str();

            // Write the boundary conditions to the file
            H5::DataSet bc_dataset = file.createDataSet(
                "boundary_conditions",
                str_type,
                hdataspace_bc
            );
            bc_dataset.write(bc_s.data(), str_type);
            bc_dataset.close();

            //==================================================================
            //  PRIMITIVE DATA
            //==================================================================
            H5::DataSet dataset;
            // the regime is a  constexprstring view, so convert to
            // std::string
            const auto regime = std::string(state.regime);
            // helper lambda for writing the prim data using a for 1D loop
            // and hyperslab selection
            auto write_prims = [&](const std::string& name,
                                   const auto& dataspace,
                                   const auto member) {
                // Write the data using a for loop
                dataset = file.createDataSet(name, real_type, dataspace);
                for (hsize_t i = 0; i < state.primitives().size(); ++i) {
                    hsize_t offset[1] = {i};
                    hsize_t count[1]  = {1};
                    H5::DataSpace memspace(1, count);
                    dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
                    // unwrap the Maybe<primitive> type
                    dataset.write(
                        &state.primitives()[i].value()[member],
                        real_type,
                        memspace,
                        dataspace
                    );
                }
                dataset.close();
            };

            auto write_fields = [&](const std::string& name,
                                    const auto& dataspace,
                                    const auto member) {
                if constexpr (T::regime == "srmhd") {
                    // Write the data using a for loop
                    dataset = file.createDataSet(name, real_type, dataspace);
                    if (member == 1) {
                        for (hsize_t i = 0; i < state.bstag1.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);
                            dataset.write(
                                &const_cast<T&>(state).bstag1[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    else if (member == 2) {
                        for (hsize_t i = 0; i < state.bstag2.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);

                            dataset.write(
                                &const_cast<T&>(state).bstag2[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    else {
                        for (hsize_t i = 0; i < state.bstag3.size(); ++i) {
                            hsize_t offset[1] = {i};
                            hsize_t count[1]  = {1};
                            H5::DataSpace memspace(1, count);
                            dataspace
                                .selectHyperslab(H5S_SELECT_SET, count, offset);

                            dataset.write(
                                &const_cast<T&>(state).bstag3[i],
                                real_type,
                                memspace,
                                dataspace
                            );
                        }
                    }
                    dataset.close();
                }
            };
            write_prims("rho", hdataspace, 0);
            write_prims("v1", hdataspace, 1);
            if constexpr (T::dimensions > 1) {
                write_prims("v2", hdataspace, 2);
            }
            if constexpr (T::dimensions > 2) {
                write_prims("v3", hdataspace, 3);
            }
            write_prims("p", hdataspace, state.dimensions + 1);
            if constexpr (T::regime == "srmhd") {
                write_fields("b1", b1dataspace, 1);
                write_fields("b2", b2dataspace, 2);
                write_fields("b3", b3dataspace, 3);
                write_prims("chi", hdataspace, state.dimensions + 5);
            }
            else {
                write_prims("chi", hdataspace, state.dimensions + 2);
            }

            //==================================================================
            //  ATTRIBUTE DATA
            //==================================================================
            // create dataset for simulation information
            H5::DataSet sim_info =
                file.createDataSet("sim_info", attr_type, attr_dataspace);

            // write simulation information in attributes and then close the
            // file
            auto current_time          = state.time();
            auto end_time              = state.tend();
            auto cfl_number            = state.cfl_number();
            auto plm_theta             = state.plm_theta();
            auto viscosity             = state.viscosity();
            auto shakura_sunyaev_alpha = state.shakura_sunyaev_alpha();
            auto dt_val                = state.dt();
            auto nx_val                = state.nx();
            auto ny_val                = state.ny();
            auto nz_val                = state.nz();
            bool using_fourvel         = global::using_four_velocity;
            bool using_quirk_smoothing = state.quirk_smoothing();
            bool mesh_moving           = state.mesh().mesh_is_moving();
            auto x1min            = state.mesh().geometry_state().min_bound(0);
            auto x1max            = state.mesh().geometry_state().max_bound(0);
            auto x2min            = state.mesh().geometry_state().min_bound(1);
            auto x2max            = state.mesh().geometry_state().max_bound(1);
            auto x3min            = state.mesh().geometry_state().min_bound(2);
            auto x3max            = state.mesh().geometry_state().max_bound(2);
            auto checkpoint_index = state.io().checkpoint_index();
            auto checkpoint_interval =
                state.time_manager().checkpoint_interval();
            auto gamma_val      = state.adiabatic_index();
            auto spatial_order  = state.spatial_order();
            auto temporal_order = state.temporal_order();
            auto geometry       = state.mesh().geometry_to_c_str();
            auto x1_spacing =
                cell2str.at(state.mesh().geometry_state().spacing_type(0));
            auto x2_spacing =
                cell2str.at(state.mesh().geometry_state().spacing_type(1));
            auto x3_spacing =
                cell2str.at(state.mesh().geometry_state().spacing_type(2));

            const std::vector<std::pair<std::string, const void*>> attributes =
                {
                  {"current_time", &current_time},
                  {"end_time", &end_time},
                  {"cfl_number", &cfl_number},
                  {"time_step", &dt_val},
                  {"plm_theta", &plm_theta},
                  {"viscosity", &viscosity},
                  {"shakura_sunyaev_alpha", &shakura_sunyaev_alpha},
                  {"spatial_order", spatial_order.c_str()},
                  {"temporal_order", temporal_order.c_str()},
                  {"using_gamma_beta", &using_fourvel},
                  {"using_quirk_smoothing", &using_quirk_smoothing},
                  {"mesh_motion", &mesh_moving},
                  {"x1max", &x1max},
                  {"x1min", &x1min},
                  {"x2max", &x2max},
                  {"x2min", &x2min},
                  {"x3max", &x3max},
                  {"x3min", &x3min},
                  {"adiabatic_index", &gamma_val},
                  {"nx", &nx_val},
                  {"ny", &ny_val},
                  {"nz", &nz_val},
                  {"checkpoint_index", &checkpoint_index},
                  {"checkpoint_interval", &checkpoint_interval},
                  {"geometry", geometry},
                  {"regime", regime.c_str()},
                  {"dimensions", &state.dimensions},
                  {"x1_spacing", x1_spacing.c_str()},
                  {"x2_spacing", x2_spacing.c_str()},
                  {"x3_spacing", x3_spacing.c_str()},
                  {"data_directory", data_directory.c_str()},
                  {"solver", state.solver_config().solver_name().data()},
                };

            for (const auto& [name, value] : attributes) {
                H5::DataType type;
                if (name == "spatial_order" || name == "temporal_order" ||
                    name == "geometry" || name == "regime" ||
                    name.find("_spacing") != std::string::npos ||
                    name == "solver" || name == "data_directory") {

                    type = H5::StrType(H5::PredType::C_S1, 256);
                }
                else if (name == "using_gamma_beta" || name == "mesh_motion") {
                    type = bool_type;
                }
                else if (name == "nx" || name == "ny" || name == "nz" ||
                         name == "checkpoint_index" || name == "dimensions") {
                    type = int_type;
                }
                else {
                    type = real_type;
                }

                auto att =
                    sim_info.createAttribute(name, type, scalar_dataspace);
                att.write(type, value);
                att.close();
            }
            sim_info.close();

            // write the immersed boundary data if it exists
            if (state.has_immersed_bodies()) {
                // Create group for immersed bodies
                H5::Group ib_group = file.createGroup("immersed_bodies");

                // Write body count
                auto& body_system = *state.body_system();
                auto body_count   = body_system.size();

                auto bodies_count_attr = ib_group.createAttribute(
                    "count",
                    int_type,
                    scalar_dataspace
                );
                bodies_count_attr.write(int_type, &body_count);
                bodies_count_attr.close();

                auto ref_attr = ib_group.createAttribute(
                    "reference_frame",
                    H5::StrType(H5::PredType::C_S1, 256),
                    scalar_dataspace
                );
                const auto ref_frame = body_system.reference_frame();
                ref_attr.write(
                    H5::StrType(H5::PredType::C_S1, 256),
                    ref_frame.c_str()
                );
                ref_attr.close();

                // Write each body's data
                for (size_type body_idx = 0; body_idx < body_count;
                     ++body_idx) {
                    const auto& body = body_system.get_body(body_idx).value();
                    // Create a subgroup for this specific body
                    std::string body_group_name =
                        "body_" + std::to_string(body_idx);
                    H5::Group body_group =
                        ib_group.createGroup(body_group_name);

                    // Get all serializable properties for this body
                    auto properties =
                        body_system.get_serializable_properties(body_idx);

                    // Write each property using the appropriate
                    // serialization trait
                    for (const auto& prop_variant : properties) {
                        std::visit(
                            [&body_group](const auto& prop) {
                                using PropType =
                                    std::decay_t<decltype(prop.extractor(0))>;

                                // Extract the property value for this body
                                PropType value = prop.extractor(0);

                                // Use the appropriate serialization trait to
                                // write the value
                                PropertySerializationTrait<PropType>::
                                    write_to_h5(body_group, prop.name, value);
                            },
                            prop_variant
                        );
                    }

                    // Add capability flags as attributes
                    auto caps         = body.capabilities();
                    uint32_t caps_int = static_cast<uint32_t>(caps);

                    auto caps_attr = body_group.createAttribute(
                        "capabilities",
                        int_type,
                        scalar_dataspace
                    );
                    caps_attr.write(int_type, &caps_int);
                    caps_attr.close();

                    // close this body's group
                    body_group.close();
                }

                if (body_system.has_system_config()) {
                    serialize_system_config(
                        body_system.system_config().get(),
                        ib_group
                    );
                }

                // Close the immersed_bodies group
                ib_group.close();
            }

            // close the file
            file.close();
        }
    }   // namespace io

}   // namespace simbi

#endif
