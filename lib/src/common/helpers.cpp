
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

bool strong_shock(real pl, real pr)
{
    return abs(pr - pl) / my_min(pl, pr) > QUIRK_THERSHOLD;
}

std::map<std::string, simbi::Geometry> geometry;
void config_system() {
    geometry["cartesian"] = simbi::Geometry::CARTESIAN;
    geometry["spherical"] = simbi::Geometry::SPHERICAL;
}

void config_ghosts1D(std::vector<hydro1d::Conserved> &u_state, int grid_size, bool first_order){
    if (first_order){
        u_state[0].rho = u_state[1].rho;
        u_state[grid_size - 1].rho = u_state[grid_size - 2].rho;

        u_state[0].m = - u_state[1].m;
        u_state[grid_size - 1].m = u_state[grid_size - 2].m;

        u_state[0].e_dens = u_state[1].e_dens;
        u_state[grid_size - 1].e_dens = u_state[grid_size - 2].e_dens;
    } else {
        u_state[0].rho = u_state[3].rho;
        u_state[1].rho = u_state[2].rho;
        u_state[grid_size - 1].rho = u_state[grid_size - 3].rho;
        u_state[grid_size - 2].rho = u_state[grid_size - 3].rho;

        u_state[0].m = - u_state[3].m;
        u_state[1].m = - u_state[2].m;
        u_state[grid_size - 1].m = u_state[grid_size - 3].m;
        u_state[grid_size - 2].m = u_state[grid_size - 3].m;

        u_state[0].e_dens = u_state[3].e_dens;
        u_state[1].e_dens = u_state[2].e_dens;
        u_state[grid_size - 1].e_dens = u_state[grid_size - 3].e_dens;
        u_state[grid_size - 2].e_dens = u_state[grid_size - 3].e_dens;
    }
};


void config_ghosts2D(
    std::vector<hydro2d::Conserved> &u_state, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order)
{

    if (first_order){
        for (int jj = 0; jj < x2grid_size; jj++){
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 1){
                    u_state[ii + x1grid_size * jj].rho    =   u_state[ii + x1grid_size].rho;
                    u_state[ii + x1grid_size * jj].m1     =   u_state[ii + x1grid_size].m1;
                    u_state[ii + x1grid_size * jj].m2     = - u_state[ii + x1grid_size].m2;
                    u_state[ii + x1grid_size * jj].e_dens =   u_state[ii + x1grid_size].e_dens;
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[ii + x1grid_size * jj].rho     =   u_state[(x2grid_size - 2) * x1grid_size + ii].rho;
                    u_state[ii + x1grid_size * jj].m1      =   u_state[(x2grid_size - 2) * x1grid_size + ii].m1;
                    u_state[ii + x1grid_size * jj].m2      = - u_state[(x2grid_size - 2) * x1grid_size + ii].m2;
                    u_state[ii + x1grid_size * jj].e_dens  =   u_state[(x2grid_size - 2) * x1grid_size + ii].e_dens;

                } else {
                    u_state[jj * x1grid_size].rho         =   u_state[jj * x1grid_size + 1].rho;
                    u_state[jj * x1grid_size + 0].m1      = - u_state[jj * x1grid_size + 1].m1;
                    u_state[jj * x1grid_size + 0].m2      =   u_state[jj * x1grid_size + 1].m2;
                    u_state[jj * x1grid_size + 0].e_dens  =   u_state[jj * x1grid_size + 1].e_dens;

                    u_state[jj * x1grid_size + x1grid_size - 1].rho    =  u_state[jj*x1grid_size + x1grid_size - 2].rho;
                    u_state[jj * x1grid_size + x1grid_size - 1].m1     =  u_state[jj * x1grid_size + x1grid_size - 2].m1;
                    u_state[jj * x1grid_size + x1grid_size - 1].m2     =  u_state[jj * x1grid_size + x1grid_size - 2].m2;
                    u_state[jj * x1grid_size + x1grid_size - 1].e_dens =  u_state[jj * x1grid_size + x1grid_size - 2].e_dens;
                }
            }
        }

    } else {
        for (int jj = 0; jj < x2grid_size; jj++){

            // Fix the ghost zones at the radial boundaries
            u_state[jj * x1grid_size +  0].rho               = u_state[jj * x1grid_size +  3].rho;
            u_state[jj * x1grid_size +  1].rho               = u_state[jj * x1grid_size +  2].rho;
            u_state[jj * x1grid_size +  x1grid_size - 1].rho = u_state[jj * x1grid_size +  x1grid_size - 3].rho;
            u_state[jj * x1grid_size +  x1grid_size - 2].rho = u_state[jj * x1grid_size +  x1grid_size - 3].rho;

            u_state[jj * x1grid_size + 0].m1               = - u_state[jj * x1grid_size + 3].m1;
            u_state[jj * x1grid_size + 1].m1               = - u_state[jj * x1grid_size + 2].m1;
            u_state[jj * x1grid_size + x1grid_size - 1].m1 =   u_state[jj * x1grid_size + x1grid_size - 3].m1;
            u_state[jj * x1grid_size + x1grid_size - 2].m1 =   u_state[jj * x1grid_size + x1grid_size - 3].m1;

            u_state[jj * x1grid_size + 0].m2               = u_state[jj * x1grid_size + 3].m2;
            u_state[jj * x1grid_size + 1].m2               = u_state[jj * x1grid_size + 2].m2;
            u_state[jj * x1grid_size + x1grid_size - 1].m2 = u_state[jj * x1grid_size + x1grid_size - 3].m2;
            u_state[jj * x1grid_size + x1grid_size - 2].m2 = u_state[jj * x1grid_size + x1grid_size - 3].m2;

            u_state[jj * x1grid_size + 0].e_dens                = u_state[jj * x1grid_size + 3].e_dens;
            u_state[jj * x1grid_size + 1].e_dens                = u_state[jj * x1grid_size + 2].e_dens;
            u_state[jj * x1grid_size + x1grid_size - 1].e_dens  = u_state[jj * x1grid_size + x1grid_size - 3].e_dens;
            u_state[jj * x1grid_size + x1grid_size - 2].e_dens  = u_state[jj * x1grid_size + x1grid_size - 3].e_dens;

            // Fix the ghost zones at the angular boundaries
            /**
            if (jj < 2){
                for (int ii = 0; ii < x1grid_size; ii++){
                     if (jj == 0){
                        u_state[jj * x1grid_size + ii].rho    =   u_state[3 * x1grid_size + ii].rho;
                        u_state[jj * x1grid_size + ii].m1     =   u_state[3 * x1grid_size + ii].m1;
                        u_state[jj * x1grid_size + ii].m2     = - u_state[3 * x1grid_size + ii].m2;
                        u_state[jj * x1grid_size + ii].e_dens =   u_state[3 * x1grid_size + ii].e_dens;
                    } else {
                        u_state[jj * x1grid_size + ii].rho     =   u_state[2 * x1grid_size + ii].rho;
                        u_state[jj * x1grid_size + ii].m1      =   u_state[2 * x1grid_size + ii].m1;
                        u_state[jj * x1grid_size + ii].m2      = - u_state[2 * x1grid_size + ii].m2;
                        u_state[jj * x1grid_size + ii].e_dens  =   u_state[2 * x1grid_size + ii].e_dens;
                    }
                }
            } else if (jj > x2grid_size - 3) {
                for (int ii = 0; ii < x1grid_size; ii++){
                    if (jj == x2grid_size - 1){
                        u_state[jj * x1grid_size + ii].rho    =   u_state[(x2grid_size - 4) * x1grid_size + ii].rho;
                        u_state[jj * x1grid_size + ii].m1     =   u_state[(x2grid_size - 4) * x1grid_size + ii].m1;
                        u_state[jj * x1grid_size + ii].m2     = - u_state[(x2grid_size - 4) * x1grid_size + ii].m2;
                        u_state[jj * x1grid_size + ii].e_dens =   u_state[(x2grid_size - 4) * x1grid_size + ii].e_dens;
                    } else {
                        u_state[jj * x1grid_size + ii].rho    =   u_state[(x2grid_size - 3) * x1grid_size + ii].rho;
                        u_state[jj * x1grid_size + ii].m1     =   u_state[(x2grid_size - 3) * x1grid_size + ii].m1;
                        u_state[jj * x1grid_size + ii].m2     = - u_state[(x2grid_size - 3) * x1grid_size + ii].m2;
                        u_state[jj * x1grid_size + ii].e_dens =   u_state[(x2grid_size - 3) * x1grid_size + ii].e_dens;
                    }
                }
            }
            */
            
        }

    }
};


void config_ghosts2D(
    std::vector<sr2d::Conserved> &u_state, 
    int x1grid_size, 
    int x2grid_size, 
    bool first_order,
    bool bipolar){

    if (first_order){
        for (int jj = 0; jj < x2grid_size; jj++){
            for (int ii = 0; ii < x1grid_size; ii++){
                if (jj < 1){
                    u_state[ii + x1grid_size * jj].D   =   u_state[ii + x1grid_size].D;
                    u_state[ii + x1grid_size * jj].S1  =   u_state[ii + x1grid_size].S1;
                    u_state[ii + x1grid_size * jj].S2  = - u_state[ii + x1grid_size].S2;
                    u_state[ii + x1grid_size * jj].tau =   u_state[ii + x1grid_size].tau;
                    
                } else if (jj > x2grid_size - 2) {
                    u_state[ii + x1grid_size * jj].D    =   u_state[(x2grid_size - 2) * x1grid_size + ii].D;
                    u_state[ii + x1grid_size * jj].S1   =   u_state[(x2grid_size - 2) * x1grid_size + ii].S1;
                    u_state[ii + x1grid_size * jj].S2   = - u_state[(x2grid_size - 2) * x1grid_size + ii].S2;
                    u_state[ii + x1grid_size * jj].tau  =   u_state[(x2grid_size - 2) * x1grid_size + ii].tau;

                } else {
                    u_state[jj * x1grid_size].D    = u_state[jj * x1grid_size + 1].D;
                    u_state[jj * x1grid_size + x1grid_size - 1].D = u_state[jj*x1grid_size + x1grid_size - 2].D;

                    u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 1].S1;
                    u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 2].S1;

                    u_state[jj * x1grid_size + 0].S2                = u_state[jj * x1grid_size + 1].S2;
                    u_state[jj * x1grid_size + x1grid_size - 1].S2  = u_state[jj * x1grid_size + x1grid_size - 2].S2;

                    u_state[jj * x1grid_size + 0].tau               = u_state[jj * x1grid_size + 1].tau;
                    u_state[jj * x1grid_size + x1grid_size - 1].tau = u_state[jj * x1grid_size + x1grid_size - 2].tau;
                }
            }
        }

    } else {
        for (int jj = 0; jj < x2grid_size; jj++){

            // Fix the ghost zones at the radial boundaries
            u_state[jj * x1grid_size +  0].D               = u_state[jj * x1grid_size +  3].D;
            u_state[jj * x1grid_size +  1].D               = u_state[jj * x1grid_size +  2].D;
            u_state[jj * x1grid_size +  x1grid_size - 1].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;
            u_state[jj * x1grid_size +  x1grid_size - 2].D = u_state[jj * x1grid_size +  x1grid_size - 3].D;

            u_state[jj * x1grid_size + 0].S1               = - u_state[jj * x1grid_size + 3].S1;
            u_state[jj * x1grid_size + 1].S1               = - u_state[jj * x1grid_size + 2].S1;
            u_state[jj * x1grid_size + x1grid_size - 1].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;
            u_state[jj * x1grid_size + x1grid_size - 2].S1 =   u_state[jj * x1grid_size + x1grid_size - 3].S1;

            u_state[jj * x1grid_size + 0].S2               = u_state[jj * x1grid_size + 3].S2;
            u_state[jj * x1grid_size + 1].S2               = u_state[jj * x1grid_size + 2].S2;
            u_state[jj * x1grid_size + x1grid_size - 1].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;
            u_state[jj * x1grid_size + x1grid_size - 2].S2 = u_state[jj * x1grid_size + x1grid_size - 3].S2;

            u_state[jj * x1grid_size + 0].tau                = u_state[jj * x1grid_size + 3].tau;
            u_state[jj * x1grid_size + 1].tau                = u_state[jj * x1grid_size + 2].tau;
            u_state[jj * x1grid_size + x1grid_size - 1].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;
            u_state[jj * x1grid_size + x1grid_size - 2].tau  = u_state[jj * x1grid_size + x1grid_size - 3].tau;

            // Fix the ghost zones at the angular boundaries
            
            if (jj < 2){
                for (int ii = 0; ii < x1grid_size; ii++){
                     if (jj == 0){
                        u_state[jj * x1grid_size + ii].D   =   u_state[3 * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[3 * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  =   u_state[3 * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[3 * x1grid_size + ii].tau;
                    } else {
                        u_state[jj * x1grid_size + ii].D    =   u_state[2 * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1   =   u_state[2 * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2   =   u_state[2 * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau  =   u_state[2 * x1grid_size + ii].tau;
                    }
                }
            } else if (jj > x2grid_size - 3) {
                for (int ii = 0; ii < x1grid_size; ii++){
                    if (jj == x2grid_size - 1){
                        u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 4) * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 4) * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  =   u_state[(x2grid_size - 4) * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 4) * x1grid_size + ii].tau;
                    } else {
                        u_state[jj * x1grid_size + ii].D   =   u_state[(x2grid_size - 3) * x1grid_size + ii].D;
                        u_state[jj * x1grid_size + ii].S1  =   u_state[(x2grid_size - 3) * x1grid_size + ii].S1;
                        u_state[jj * x1grid_size + ii].S2  =   u_state[(x2grid_size - 3) * x1grid_size + ii].S2;
                        u_state[jj * x1grid_size + ii].tau =   u_state[(x2grid_size - 3) * x1grid_size + ii].tau;
                    }
                }
            }
            
        }

    }
};

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

            att = sim_info.createAttribute("xmax", real_type, att_space);
            att.write(real_type, &setup.xmax);
            att.close();

            att = sim_info.createAttribute("xmin", real_type, att_space);
            att.write(real_type, &setup.xmin);
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

            std::copy(prims.rho.begin(), prims.rho.begin() + size, rho);
            std::copy(prims.v1.begin(), prims.v1.begin() + size, v1);
            std::copy(prims.v2.begin(), prims.v2.begin() + size, v2);
            std::copy(prims.p.begin(),  prims.p.begin() + size, p);
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

            // Free the heap
            delete[] rho;
            delete[] v1;
            delete[] v2;
            delete[] p;

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
            
            att = sim_info.createAttribute("xmax", real_type, att_space);
            att.write(real_type, &setup.xmax);
            att.close();

            att = sim_info.createAttribute("xmin", real_type, att_space);
            att.write(real_type, &setup.xmin);
            att.close();

            att = sim_info.createAttribute("ymax", real_type, att_space);
            att.write(real_type, &setup.ymax);
            att.close();

            att = sim_info.createAttribute("ymin", real_type, att_space);
            att.write(real_type, &setup.ymin);
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
            
            att = sim_info.createAttribute("xmax", real_type, att_space);
            att.write(real_type, &setup.xmax);
            att.close();

            att = sim_info.createAttribute("xmin", real_type, att_space);
            att.write(real_type, &setup.xmin);
            att.close();

            att = sim_info.createAttribute("ymax", real_type, att_space);
            att.write(real_type, &setup.ymax);
            att.close();

            att = sim_info.createAttribute("ymin", real_type, att_space);
            att.write(real_type, &setup.ymin);
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
//=======================================================================================================
//                                      RELATIVISITC HYDRO
//=======================================================================================================
real calc_intermed_wave(real energy_density, real momentum_density, 
                            real flux_momentum_density, 
                            real flux_energy_density)
{
    real a = flux_energy_density;
    real b = - (energy_density + flux_momentum_density);
    real c = momentum_density;
    real disc = sqrt( b*b - 4*a*c);
    real quad = -0.5*(b + sgn(b)*disc);
    return c/quad;
}

real calc_intermed_pressure(real a,real aStar, real energy, real norm_mom, real u, real p){

    real e, f, g;
    e = (a*energy - norm_mom)*aStar;
    f = norm_mom*(a - u) - p;
    g = 1 + a*aStar;

    return (e - f)/g;
}
//------------------------------------------------------------------------------------------------------------
//  F-FUNCTION FOR ROOT FINDING: F(P)
//------------------------------------------------------------------------------------------------------------
real pressure_func(real pressure, real D, real tau, real lorentz_gamma, float gamma, real S){

    real v       = S / (tau + pressure + D);
    real W_s     = 1.0 / sqrt(1.0 - v * v);
    real rho     = D / W_s; 
    real epsilon = ( tau + D*(1. - W_s) + (1. - W_s*W_s)*pressure )/(D * W_s);

    return (gamma - 1.)*rho*epsilon - pressure;
}

real dfdp(real pressure, real D, real tau, real lorentz_gamma, float gamma, real S){

    real v       = S/(tau + D + pressure);
    real W_s     = 1.0 / sqrt(1.0 - v*v);
    real rho     = D / W_s; 
    real h       = 1 + pressure*gamma/(rho*(gamma - 1.));
    real c2      = gamma*pressure/(h*rho);
    

    return v*v*c2 - 1.;
}
