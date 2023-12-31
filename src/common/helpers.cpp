#include "common/helpers.hpp"
#include "H5Cpp.h"
#include <thread>

//==================================
//              GPU HELPERS
//==================================
real gpu_theoretical_bw = 1;
using namespace H5;

namespace simbi {
    namespace helpers {
        // Flag that detects whether program was terminated by external forces
        std::atomic<bool> killsig_received = false;

        InterruptException::InterruptException(int s) : status(s) {}

        const char* InterruptException::what() const noexcept
        {
            return "Simulation interrupted. Saving last checkpoint...";
        }

        void catch_signals()
        {
            const static auto signal_handler = [](int sig) {
                killsig_received = true;
            };
            std::signal(SIGTERM, signal_handler);
            std::signal(SIGINT, signal_handler);
            std::signal(SIGKILL, signal_handler);
            if (killsig_received) {
                killsig_received = false;
                throw helpers::InterruptException(1);
            }
        }

        SimulationFailureException::SimulationFailureException(
            const char* reason,
            const char* details
        )
            : reason(reason), details(details)
        {
        }

        const char* SimulationFailureException::what() const noexcept
        {
            return "Simulation Crashed";
        }

        //====================================================================================================
        //                                  WRITE DATA TO FILE
        //====================================================================================================
        std::string
        create_step_str(const real current_time, const int max_order_of_mag)
        {
            if (current_time == 0) {
                return "000_000";
            }
            // Convert the time interval into an int with 2 decimal
            // displacements
            const int current_time_int  = std::round(1e3 * current_time);
            const int time_order_of_mag = std::floor(
                std::log10(std::round(1000.0 * current_time) / 1000.0)
            );
            const int num_zeros         = max_order_of_mag - time_order_of_mag;
            const std::string pad_zeros = std::string(num_zeros, '0');
            auto time_string            = std::to_string(current_time_int);
            time_string.insert(0, pad_zeros);
            separate<3, '_'>(time_string);
            return time_string;
        }

        void write_hdf5(
            const std::string data_directory,
            const std::string filename,
            const PrimData prims,
            const DataWriteMembers setup,
            const int dim,
            const int size
        )
        {
            std::string filePath = data_directory;
            std::cout << "\n"
                      << "[Writing File...: " << filePath + filename << "]"
                      << "\n";

            H5::H5File file(filePath + filename, H5F_ACC_TRUNC);

            // Dataset dims
            hsize_t dimsf[1], dimsf1[1], dimsf2[1], dimsf3[1];
            dimsf[0]  = size;
            dimsf1[0] = setup.x1.capacity();
            dimsf2[0] = setup.x2.capacity();
            dimsf3[0] = setup.x3.capacity();
            int rank  = 1;
            H5::DataSpace hydro_dataspace(rank, dimsf);
            H5::DataSpace dataspacex1(rank, dimsf1);
            H5::DataSpace dataspacex2(rank, dimsf2);
            H5::DataSpace dataspacex3(rank, dimsf3);

            hid_t dtype_str = H5Tcopy(H5T_C_S1);
            size_t size_str = 100;
            H5Tset_size(dtype_str, size_str);

            // HDF5 only understands vector of char* :-(
            std::vector<const char*> arr_c_str;
            for (size_t ii = 0; ii < setup.boundary_conditions.size(); ++ii) {
                arr_c_str.push_back(setup.boundary_conditions[ii].c_str());
            }

            //
            //  one dimension
            //
            hsize_t str_dimsf[1]{arr_c_str.size()};
            H5::DataSpace bc_dataspace(rank, str_dimsf);

            // Variable length string
            H5::StrType str_datatype(H5::PredType::C_S1, H5T_VARIABLE);
            H5::DataSet str_dataset = file.createDataSet(
                "boundary_conditions",
                str_datatype,
                bc_dataspace
            );
            str_dataset.write(arr_c_str.data(), str_datatype);
            str_dataset.close();

            H5::DataType real_type;
            if (typeid(real) == typeid(double)) {
                real_type = H5::PredType::NATIVE_DOUBLE;
            }
            else {
                real_type = H5::PredType::NATIVE_FLOAT;
            }

            // Write the Primitives
            H5::DataSet dataset =
                file.createDataSet("rho", real_type, hydro_dataspace);

            dataset.write(prims.rho.data(), real_type);
            dataset.close();

            dataset = file.createDataSet("v1", real_type, hydro_dataspace);
            dataset.write(prims.v1.data(), real_type);
            dataset.close();

            if (dim > 1) {
                dataset = file.createDataSet("v2", real_type, hydro_dataspace);
                dataset.write(prims.v2.data(), real_type);
                dataset.close();

                dataset = file.createDataSet("x2", real_type, dataspacex2);
                dataset.write(setup.x2.data(), real_type);
                dataset.close();
            }
            if (dim > 2) {
                dataset = file.createDataSet("v3", real_type, hydro_dataspace);
                dataset.write(prims.v3.data(), real_type);
                dataset.close();

                dataset = file.createDataSet("x3", real_type, dataspacex3);
                dataset.write(setup.x3.data(), real_type);
                dataset.close();
            }

            dataset = file.createDataSet("p", real_type, hydro_dataspace);
            dataset.write(prims.p.data(), real_type);
            dataset.close();

            dataset = file.createDataSet("chi", real_type, hydro_dataspace);
            dataset.write(prims.chi.data(), real_type);
            dataset.close();

            dataset = file.createDataSet("x1", real_type, dataspacex1);
            dataset.write(setup.x1.data(), real_type);
            dataset.close();

            if (setup.regime == "srmhd") {
                dataset = file.createDataSet("b1", real_type, hydro_dataspace);
                dataset.write(prims.b1.data(), real_type);
                dataset.close();

                dataset = file.createDataSet("b2", real_type, hydro_dataspace);
                dataset.write(prims.b2.data(), real_type);
                dataset.close();

                dataset = file.createDataSet("b3", real_type, hydro_dataspace);
                dataset.write(prims.b3.data(), real_type);
                dataset.close();
            }

            // Write Dataset Attribute
            H5::DataType int_type(H5::PredType::NATIVE_INT);

            H5::DataType bool_type(H5::PredType::NATIVE_HBOOL);
            H5::DataSpace att_space(H5S_SCALAR);

            H5::DataSpace empty_dspace(1, dimsf);
            H5::DataType empty_dtype(H5::PredType::NATIVE_INT);
            H5::DataSet sim_info =
                file.createDataSet("sim_info", empty_dtype, empty_dspace);

            H5::Attribute att =
                sim_info.createAttribute("current_time", real_type, att_space);
            att.write(real_type, &setup.t);
            att.close();

            att = sim_info.createAttribute("time_step", real_type, att_space);
            att.write(real_type, &setup.dt);
            att.close();

            att = sim_info.createAttribute("first_order", bool_type, att_space);
            att.write(bool_type, &setup.first_order);
            att.close();

            att = sim_info.createAttribute(
                "using_gamma_beta",
                bool_type,
                att_space
            );
            att.write(bool_type, &setup.using_fourvelocity);
            att.close();

            att = sim_info.createAttribute("mesh_motion", bool_type, att_space);
            att.write(bool_type, &setup.mesh_motion);
            att.close();

            att = sim_info.createAttribute("x1max", real_type, att_space);
            att.write(real_type, &setup.x1max);
            att.close();

            att = sim_info.createAttribute("x1min", real_type, att_space);
            att.write(real_type, &setup.x1min);
            att.close();

            att = sim_info.createAttribute("x2max", real_type, att_space);
            att.write(real_type, &setup.x2max);
            att.close();

            att = sim_info.createAttribute("x2min", real_type, att_space);
            att.write(real_type, &setup.x2min);
            att.close();

            att = sim_info.createAttribute("x3max", real_type, att_space);
            att.write(real_type, &setup.x3max);
            att.close();

            att = sim_info.createAttribute("x3min", real_type, att_space);
            att.write(real_type, &setup.x3min);
            att.close();

            att = sim_info
                      .createAttribute("adiabatic_gamma", real_type, att_space);
            att.write(real_type, &setup.ad_gamma);
            att.close();

            att = sim_info.createAttribute("nx", int_type, att_space);
            att.write(int_type, &setup.nx);
            att.close();

            att = sim_info.createAttribute("ny", int_type, att_space);
            att.write(int_type, &setup.ny);
            att.close();

            att = sim_info.createAttribute("nz", int_type, att_space);
            att.write(int_type, &setup.nz);
            att.close();

            att = sim_info.createAttribute("chkpt_idx", int_type, att_space);
            att.write(int_type, &setup.chkpt_idx);
            att.close();

            att =
                sim_info.createAttribute("xactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            att =
                sim_info.createAttribute("yactive_zones", int_type, att_space);
            att.write(int_type, &setup.yactive_zones);
            att.close();

            att =
                sim_info.createAttribute("zactive_zones", int_type, att_space);
            att.write(int_type, &setup.zactive_zones);
            att.close();

            att = sim_info.createAttribute("geometry", dtype_str, att_space);
            att.write(dtype_str, setup.coord_system.c_str());
            att.close();

            att = sim_info.createAttribute("regime", dtype_str, att_space);
            att.write(dtype_str, setup.regime.c_str());
            att.close();

            att = sim_info.createAttribute("dimensions", int_type, att_space);
            att.write(int_type, &setup.dimensions);
            att.close();

            att = sim_info
                      .createAttribute("x1_cell_spacing", dtype_str, att_space);
            att.write(dtype_str, setup.x1_cell_spacing.c_str());
            att.close();

            att = sim_info
                      .createAttribute("x2_cell_spacing", dtype_str, att_space);
            att.write(dtype_str, setup.x2_cell_spacing.c_str());
            att.close();

            att = sim_info
                      .createAttribute("x3_cell_spacing", dtype_str, att_space);
            att.write(dtype_str, setup.x3_cell_spacing.c_str());
            att.close();

            sim_info.close();
        }

        void anyDisplayProps()
        {
// Adapted from:
// https://stackoverflow.com/questions/5689028/how-to-get-card-specs-programmatically-in-cuda
#if GPU_CODE
            const int kb = 1024;
            const int mb = kb * kb;
            int devCount;
            gpu::api::getDeviceCount(&devCount);
            std::cout << std::string(80, '=') << "\n";
            std::cout << "GPU Device(s): " << std::endl << std::endl;

            for (int i = 0; i < devCount; ++i) {
                anyGpuProp_t props;
                gpu::api::getDeviceProperties(&props, i);
                std::cout << "  Device number:   " << i << std::endl;
                std::cout << "  Device name:     " << props.name << ": "
                          << props.major << "." << props.minor << std::endl;
                std::cout << "  Global memory:   " << props.totalGlobalMem / mb
                          << "mb" << std::endl;
                std::cout << "  Shared memory:   "
                          << props.sharedMemPerBlock / kb << "kb" << std::endl;
                std::cout << "  Constant memory: " << props.totalConstMem / kb
                          << "kb" << std::endl;
                std::cout << "  Block registers: " << props.regsPerBlock
                          << std::endl
                          << std::endl;

                std::cout << "  Warp size:         " << props.warpSize
                          << std::endl;
                std::cout << "  Threads per block: " << props.maxThreadsPerBlock
                          << std::endl;
                std::cout << "  Max block dimensions: [ "
                          << props.maxThreadsDim[0] << ", "
                          << props.maxThreadsDim[1] << ", "
                          << props.maxThreadsDim[2] << " ]" << std::endl;
                std::cout << "  Max grid dimensions:  [ "
                          << props.maxGridSize[0] << ", "
                          << props.maxGridSize[1] << ", "
                          << props.maxGridSize[2] << " ]" << std::endl;
                std::cout << "  Memory Clock Rate (KHz): "
                          << props.memoryClockRate << std::endl;
                std::cout << "  Memory Bus Width (bits): "
                          << props.memoryBusWidth << std::endl;
                std::cout << "  Peak Memory Bandwidth (GB/s): "
                          << 2.0 * props.memoryClockRate *
                                 (props.memoryBusWidth / 8) / 1.0e6
                          << std::endl;
                std::cout << std::endl;
                gpu_theoretical_bw = 2.0 * props.memoryClockRate *
                                     (props.memoryBusWidth / 8) / 1.0e6;
            }
#else
            const auto processor_count = std::thread::hardware_concurrency();
            std::cout << std::string(80, '=') << "\n";
            std::cout << "CPU Compute Thread(s): " << processor_count
                      << std::endl;
#endif
        }
    }   // namespace helpers
}   // namespace simbi
