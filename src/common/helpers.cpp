
/*
* helpers.cpp is where all of the universal functions that can be used
* for all N-Dim hydro calculations
*/

#include "helpers.hpp" 
#include "hydro_structs.hpp"
#include <atomic>
using namespace H5;
namespace simbi
{
    namespace helpers
    {
        // Flag that detects whether program was terminated by external forces
        std::atomic<bool> killsig_received = false;
        
        InterruptException::InterruptException(int s)
        : status(s)
        {
        }

        const char* InterruptException::what() const noexcept {
            return "Simulation interrupted. Saving last checkpoint...";
        }

        void catch_signals() {
            const static auto signal_handler = [](int sig) {killsig_received = true;};
            std::signal(SIGTERM, signal_handler);
            std::signal(SIGINT,  signal_handler);
            std::signal(SIGKILL, signal_handler);
            if (killsig_received) {
                killsig_received = false;
                throw helpers::InterruptException(1);
            }
        }

        SimulationFailureException::SimulationFailureException(const char* reason, const char* details)
        : reason(reason), details(details)
        {
        }

        const char* SimulationFailureException::what() const noexcept {
            const auto err_ms = "Simulation failed\n reason: " + reason + "\n details: " + details;
            return err_ms.c_str();
        }

        //====================================================================================================
        //                                  WRITE DATA TO FILE
        //====================================================================================================
        std::string create_step_str(real t_interval, std::string &tnow){

            // Convert the time interval into an int with 2 decimal displacements
            int t_interval_int = round( 1.e3 * t_interval );
            int a, b;

            std::string s = std::to_string(t_interval_int);

            // Pad the file string if size less than tnow_size
            if (s.size() < tnow.size()) {

                int num_zeros = tnow.size() - s.size();
                std::string pad_zeros = std::string(num_zeros, '0');
                s.insert(0, pad_zeros);
            }

            // insert underscore to signify decimal placement
            s.insert(s.length() - 3, "_");

            int label_size = tnow.size();
            for (int i = 0; i < label_size; i++){
                a = tnow[i] - '0';
                b = s[i] - '0';
                s[i] = a + b + '0';
            }

            return s;


        }
        void write_hdf5(
            const std::string data_directory, 
            const std::string filename, 
            const PrimData prims, 
            const DataWriteMembers setup, 
            const int dim = 2,
            const int size = 1)
        {
            std::string filePath = data_directory;
            std::cout << "\n" <<  "[Writing File...: " << filePath + filename << "]" << "\n";

            H5::H5File file(filePath + filename, H5F_ACC_TRUNC );

            // Dataset dims
            hsize_t dimsf[1], dimsf1[1], dimsf2[1], dimsf3[1];
            dimsf[0]  = size;      
            dimsf1[0] = setup.x1.capacity();
            dimsf2[0] = setup.x2.capacity();
            dimsf3[0] = setup.x3.capacity();         
            int rank = 1;
            H5::DataSpace dataspace(rank, dimsf);
            H5::DataSpace dataspacex1(rank, dimsf1);
            H5::DataSpace dataspacex2(rank, dimsf2);
            H5::DataSpace dataspacex3(rank, dimsf3);
            H5::DataType  datatype(H5::PredType::NATIVE_DOUBLE);
            
            hid_t dtype_str = H5Tcopy(H5T_C_S1);
            size_t size_str = 100;                    
            H5Tset_size(dtype_str, size_str);

            // HDF5 only understands vector of char* :-(
            std::vector<const char*> arr_c_str;
            for (size_t ii = 0; ii < setup.boundary_conditions.size(); ++ii) 
                arr_c_str.push_back(setup.boundary_conditions[ii].c_str());

            //
            //  one dimension
            // 
            hsize_t     str_dimsf[1] {arr_c_str.size()};
            H5::DataSpace bc_dataspace(rank, str_dimsf);

            // Variable length string
            H5::StrType str_datatype(H5::PredType::C_S1, H5T_VARIABLE); 
            H5::DataSet str_dataset = file.createDataSet("boundary_conditions", str_datatype, bc_dataspace);
            str_dataset.write(arr_c_str.data(), str_datatype);
            str_dataset.close();

            H5::DataType real_type;
            if (typeid(real) == typeid(double)) {
                real_type = H5::PredType::NATIVE_DOUBLE;
            } else {
                real_type = H5::PredType::NATIVE_FLOAT;
            }
            switch (dim)
            {
                case 1:
                {
                    auto rho = std::unique_ptr<real>(new real[size]);
                    auto v   = std::unique_ptr<real>(new real[size]);
                    auto p   = std::unique_ptr<real>(new real[size]);
                    auto x1  = std::unique_ptr<real>(new real[setup.x1.size()]);

                    std::copy(prims.rho.begin(), prims.rho.begin() + size, rho.get());
                    std::copy(prims.v.begin(), prims.v.begin() + size, v.get());
                    std::copy(prims.p.begin(), prims.p.begin() + size, p.get());
                    std::copy(setup.x1.begin(),  setup.x1.begin() + setup.x1.size(), x1.get());
                    H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

                    // Write the Primitives 
                    dataset.write(rho.get(), real_type);
                    dataset.close();
                    
                    dataset = file.createDataSet("v1", datatype, dataspace);
                    dataset.write(v.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("p", datatype, dataspace);
                    dataset.write(p.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x1", datatype, dataspacex1);
                    dataset.write(x1.get(), real_type);
                    dataset.close();
                    break;
                }
                case 2:
                    {
                    // Write the Primitives 
                    auto rho = std::unique_ptr<real>(new real[size]);
                    auto v1  = std::unique_ptr<real>(new real[size]);
                    auto v2  = std::unique_ptr<real>(new real[size]);
                    auto p   = std::unique_ptr<real>(new real[size]);
                    auto chi = std::unique_ptr<real>(new real[size]);
                    auto x1  = std::unique_ptr<real>(new real[setup.x1.size()]);
                    auto x2  = std::unique_ptr<real>(new real[setup.x2.size()]);

                    std::copy(prims.rho.begin(),  prims.rho.begin() + size, rho.get());
                    std::copy(prims.v1.begin(),   prims.v1.begin()  + size, v1.get());
                    std::copy(prims.v2.begin(),   prims.v2.begin()  + size, v2.get());
                    std::copy(prims.p.begin(),    prims.p.begin()   + size, p.get());
                    std::copy(prims.chi.begin(),  prims.chi.begin() + size, chi.get());
                    std::copy(setup.x1.begin(),  setup.x1.begin() + setup.x1.size(), x1.get());
                    std::copy(setup.x2.begin(),  setup.x2.begin() + setup.x2.size(), x2.get());
                    H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

                    // Write the Primitives 
                    dataset.write(rho.get(), real_type);
                    dataset.close();
                    
                    dataset = file.createDataSet("v1", datatype, dataspace);
                    dataset.write(v1.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("v2", datatype, dataspace);
                    dataset.write(v2.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("p", datatype, dataspace);
                    dataset.write(p.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("chi", datatype, dataspace);
                    dataset.write(chi.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x1", datatype, dataspacex1);
                    dataset.write(x1.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x2", datatype, dataspacex2);
                    dataset.write(x2.get(), real_type);
                    dataset.close();
                break;
                }
                case 3:
                    {
                    // Write the Primitives 
                    auto rho = std::unique_ptr<real>(new real[size]);
                    auto v1  = std::unique_ptr<real>(new real[size]);
                    auto v2  = std::unique_ptr<real>(new real[size]);
                    auto v3  = std::unique_ptr<real>(new real[size]);
                    auto p   = std::unique_ptr<real>(new real[size]);
                    auto chi = std::unique_ptr<real>(new real[size]);
                    auto x1  = std::unique_ptr<real>(new real[setup.x1.size()]);
                    auto x2  = std::unique_ptr<real>(new real[setup.x2.size()]);
                    auto x3  = std::unique_ptr<real>(new real[setup.x3.size()]);

                    std::copy(prims.rho.begin(), prims.rho.begin() + size, rho.get());
                    std::copy(prims.v1.begin(), prims.v1.begin() + size, v1.get());
                    std::copy(prims.v2.begin(), prims.v2.begin() + size, v2.get());
                    std::copy(prims.v3.begin(), prims.v3.begin() + size, v3.get());
                    std::copy(prims.p.begin(),  prims.p.begin() + size, p.get());
                    std::copy(prims.chi.begin(),  prims.chi.begin() + size, chi.get());
                    std::copy(setup.x1.begin(),  setup.x1.begin() + setup.x1.size(), x1.get());
                    std::copy(setup.x2.begin(),  setup.x2.begin() + setup.x2.size(), x2.get());
                    std::copy(setup.x3.begin(),  setup.x3.begin() + setup.x3.size(), x3.get());

                    H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

                    // Write the Primitives 
                    dataset.write(rho.get(), real_type);
                    dataset.close();
                    
                    dataset = file.createDataSet("v1", datatype, dataspace);
                    dataset.write(v1.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("v2", datatype, dataspace);
                    dataset.write(v2.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("v3", datatype, dataspace);
                    dataset.write(v3.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("p", datatype, dataspace);
                    dataset.write(p.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("chi", datatype, dataspace);
                    dataset.write(chi.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x1", datatype, dataspacex1);
                    dataset.write(x1.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x2", datatype, dataspacex2);
                    dataset.write(x2.get(), real_type);
                    dataset.close();

                    dataset = file.createDataSet("x3", datatype, dataspacex3);
                    dataset.write(x3.get(), real_type);
                    dataset.close();
                break;
                }
            } // end switch
        
            // Write Datset Attributesauto real_type(real_type);
            H5::DataType int_type(H5::PredType::NATIVE_INT);
            
            H5::DataType bool_type(H5::PredType::NATIVE_HBOOL);
            H5::DataSpace att_space(H5S_SCALAR);

            H5::DataSpace empty_dspace(1, dimsf);
            H5::DataType  empty_dtype(H5::PredType::NATIVE_INT);
            H5::DataSet   sim_info = file.createDataSet("sim_info", empty_dtype, empty_dspace);
            
            H5::Attribute att = sim_info.createAttribute("current_time", real_type, att_space);
            att.write(real_type, &setup.t);
            att.close();

            att = sim_info.createAttribute("time_step", real_type, att_space);
            att.write(real_type, &setup.dt);
            att.close();

            att = sim_info.createAttribute("linspace", bool_type, att_space);
            att.write(bool_type, &setup.linspace);
            att.close();

            att = sim_info.createAttribute("first_order", bool_type, att_space);
            att.write(bool_type, &setup.first_order);
            att.close();

            att = sim_info.createAttribute("using_gamma_beta", bool_type, att_space);
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

            att = sim_info.createAttribute("adiabatic_gamma", real_type, att_space);
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

            att = sim_info.createAttribute("xactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            att = sim_info.createAttribute("yactive_zones", int_type, att_space);
            att.write(int_type, &setup.yactive_zones);
            att.close();

            att = sim_info.createAttribute("zactive_zones", int_type, att_space);
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

            sim_info.close();
        }
        
    } // namespace helpers
    
} // namespace simbi
