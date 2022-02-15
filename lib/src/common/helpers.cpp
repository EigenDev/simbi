
/*
* helpers.cpp is where all of the universal functions that can be used
* for all N-Dim hydro calculations
*/

#include "helpers.hpp" 
#include "hydro_structs.hpp"
#include <cstdarg>

using namespace H5;
// =========================================================================================================
//        HELPER FUNCTIONS FOR COMPUTATION
// =========================================================================================================
void pause_program()
{
    std::cin.get();
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
    hsize_t dimsf[1];
    dimsf[0] = size;               
    int rank = 1;
    H5::DataSpace dataspace(rank, dimsf);
    H5::DataType  datatype(H5::PredType::NATIVE_DOUBLE);

    switch (dim)
    {
        case 1:
        {
            real* rho = new real[size];
            real* v   = new real[size];
            real* p   = new real[size];

            std::copy(prims.rho.begin(), prims.rho.begin() + size, rho);
            std::copy(prims.v.begin(), prims.v.begin() + size, v);
            std::copy(prims.p.begin(), prims.p.begin() + size, p);
            H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

            // Write the Primitives 
            dataset.write(rho, H5::PredType::NATIVE_DOUBLE);
            dataset.close();
            
            dataset = file.createDataSet("v", datatype, dataspace);
            dataset.write(v, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            dataset = file.createDataSet("p", datatype, dataspace);
            dataset.write(p, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            // Free the heap
            delete[]rho;
            delete[]v;
            delete[]p;

            // Write Datset Attributes
            H5::DataType real_type(H5::PredType::NATIVE_DOUBLE);
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

            att = sim_info.createAttribute("adiabatic_gamma", real_type, att_space);
            att.write(real_type, &setup.ad_gamma);
            att.close();

            att = sim_info.createAttribute("x1max", real_type, att_space);
            att.write(real_type, &setup.x1max);
            att.close();

            att = sim_info.createAttribute("x1min", real_type, att_space);
            att.write(real_type, &setup.x1min);
            att.close();

            att = sim_info.createAttribute("Nx", int_type, att_space);
            att.write(int_type, &setup.nx);
            att.close();

            att = sim_info.createAttribute("xactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            

            sim_info.close();
            break;
        }
        case 2:
            {
            // Write the Primitives 
            real* rho = new real[size];
            real* v1  = new real[size];
            real* v2  = new real[size];
            real* p   = new real[size];
            real* chi = new real[size];

            std::copy(prims.rho.begin(),  prims.rho.begin() + size, rho);
            std::copy(prims.v1.begin(),   prims.v1.begin()  + size, v1);
            std::copy(prims.v2.begin(),   prims.v2.begin()  + size, v2);
            std::copy(prims.p.begin(),    prims.p.begin()   + size, p);
            std::copy(prims.chi.begin(),  prims.chi.begin() + size, chi);
            H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

            // Write the Primitives 
            dataset.write(rho, H5::PredType::NATIVE_DOUBLE);
            dataset.close();
            
            dataset = file.createDataSet("v1", datatype, dataspace);
            dataset.write(v1, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            dataset = file.createDataSet("v2", datatype, dataspace);
            dataset.write(v2, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            dataset = file.createDataSet("p", datatype, dataspace);
            dataset.write(p, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            dataset = file.createDataSet("chi", datatype, dataspace);
            dataset.write(chi, H5::PredType::NATIVE_DOUBLE);
            dataset.close();

            // Free the heap
            delete[] rho;
            delete[] v1;
            delete[] v2;
            delete[] p;
            delete[] chi;

            // Write Datset Attributesauto real_type(H5::PredType::NATIVE_DOUBLE);
            H5::DataType int_type(H5::PredType::NATIVE_INT);
            H5::DataType real_type;
            if (typeid(real) == typeid(double))
            {
                real_type = H5::PredType::NATIVE_DOUBLE;
            } else {
                real_type = H5::PredType::NATIVE_FLOAT;
            }
            
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

            att = sim_info.createAttribute("adiabatic_gamma", real_type, att_space);
            att.write(real_type, &setup.ad_gamma);
            att.close();

            att = sim_info.createAttribute("nx", int_type, att_space);
            att.write(int_type, &setup.nx);
            att.close();

            att = sim_info.createAttribute("ny", int_type, att_space);
            att.write(int_type, &setup.ny);
            att.close();

            att = sim_info.createAttribute("xactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            att = sim_info.createAttribute("yactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            sim_info.close();
        break;
        }
        case 3:
            {
            // Write the Primitives 
            real* rho = new real[size];
            real* v1  = new real[size];
            real* v2  = new real[size];
            real* v3  = new real[size];
            real* p   = new real[size];

            std::copy(prims.rho.begin(), prims.rho.begin() + size, rho);
            std::copy(prims.v1.begin(), prims.v1.begin() + size, v1);
            std::copy(prims.v2.begin(), prims.v2.begin() + size, v2);
            std::copy(prims.v3.begin(), prims.v3.begin() + size, v3);
            std::copy(prims.p.begin(),  prims.p.begin() + size, p);
            H5::DataSet dataset = file.createDataSet("rho", datatype, dataspace);

            H5::DataType real_type;
            if (typeid(real) == typeid(double))
            {
                real_type = H5::PredType::NATIVE_DOUBLE;
            } else {
                real_type = H5::PredType::NATIVE_FLOAT;
            }

            // Write the Primitives 
            dataset.write(rho, real_type);
            dataset.close();
            
            dataset = file.createDataSet("v1", datatype, dataspace);
            dataset.write(v1, real_type);
            dataset.close();

            dataset = file.createDataSet("v2", datatype, dataspace);
            dataset.write(v2, real_type);
            dataset.close();

            dataset = file.createDataSet("v3", datatype, dataspace);
            dataset.write(v3, real_type);
            dataset.close();

            dataset = file.createDataSet("p", datatype, dataspace);
            dataset.write(p, real_type);
            dataset.close();

            // Free the heap
            delete[] rho;
            delete[] v1;
            delete[] v2;
            delete[] p;

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

            att = sim_info.createAttribute("zmax", real_type, att_space);
            att.write(real_type, &setup.zmax);
            att.close();

            att = sim_info.createAttribute("zmin", real_type, att_space);
            att.write(real_type, &setup.zmin);
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

            att = sim_info.createAttribute("xactive_zones", int_type, att_space);
            att.write(int_type, &setup.xactive_zones);
            att.close();

            att = sim_info.createAttribute("yactive_zones", int_type, att_space);
            att.write(int_type, &setup.yactive_zones);
            att.close();

            att = sim_info.createAttribute("zactive_zones", int_type, att_space);
            att.write(int_type, &setup.zactive_zones);
            att.close();

            sim_info.close();
        break;
        }
    
    }
    
}
