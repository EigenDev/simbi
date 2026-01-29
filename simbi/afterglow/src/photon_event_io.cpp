// =============================================================================
// photon_event_io.cpp
//
// HDF5 serialization for photon events.
// columnar storage with compression for efficient I/O.
// =============================================================================

#include "photon_event_io.hpp"

#include <H5Cpp.h>
#include <stdexcept>

namespace simbi::afterglow {

    namespace {
        // helper: create compressed dataset
        H5::DataSet create_compressed_dataset(
            H5::H5File&         file,
            const std::string&  name,
            hsize_t             size,
            const H5::DataType& dtype
        )
        {
            H5::DataSpace         space(1, &size);
            H5::DSetCreatPropList plist;

            // enable chunking (required for compression)
            hsize_t chunk_size = std::min(size, hsize_t(8192));
            plist.setChunk(1, &chunk_size);

            // gzip compression level 6
            plist.setDeflate(6);

            return file.createDataSet(name, dtype, space, plist);
        }

        // helper: write scalar attribute
        template <typename T>
        void write_attribute(H5::H5Object& obj, const std::string& name, T value)
        {
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::DataType  dtype = H5::PredType::NATIVE_DOUBLE;

            if constexpr (std::is_same_v<T, double>) {
                dtype = H5::PredType::NATIVE_DOUBLE;
            }
            else if constexpr (std::is_same_v<T, std::uint64_t>) {
                dtype = H5::PredType::NATIVE_UINT64;
            }
            else if constexpr (std::is_same_v<T, std::int64_t>) {
                dtype = H5::PredType::NATIVE_INT64;
            }
            else if constexpr (std::is_same_v<T, std::uint32_t>) {
                dtype = H5::PredType::NATIVE_UINT32;
            }

            auto attr = obj.createAttribute(name, dtype, attr_space);
            attr.write(dtype, &value);
        }

        // helper: read scalar attribute
        template <typename T>
        T read_attribute(const H5::H5Object& obj, const std::string& name)
        {
            T            value;
            H5::DataType dtype = H5::PredType::NATIVE_DOUBLE;

            if constexpr (std::is_same_v<T, double>) {
                dtype = H5::PredType::NATIVE_DOUBLE;
            }
            else if constexpr (std::is_same_v<T, std::uint64_t>) {
                dtype = H5::PredType::NATIVE_UINT64;
            }
            else if constexpr (std::is_same_v<T, std::int64_t>) {
                dtype = H5::PredType::NATIVE_INT64;
            }
            else if constexpr (std::is_same_v<T, std::uint32_t>) {
                dtype = H5::PredType::NATIVE_UINT32;
            }

            auto attr = obj.openAttribute(name);
            attr.read(dtype, &value);
            return value;
        }

        // helper: write vector as dataset
        void write_vector_dataset(
            H5::H5File&                file,
            const std::string&         name,
            const std::vector<double>& data
        )
        {
            if (data.empty()) {
                return;
            }

            hsize_t size = data.size();
            auto    dset = create_compressed_dataset(file, name, size, H5::PredType::NATIVE_DOUBLE);
            dset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
        }

        // helper: read vector dataset
        std::vector<double> read_vector_dataset(const H5::H5File& file, const std::string& name)
        {
            auto    dset  = file.openDataSet(name);
            auto    space = dset.getSpace();
            hsize_t size;
            space.getSimpleExtentDims(&size);

            std::vector<double> data(size);
            dset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
            return data;
        }

        // helper: write uint32 vector as dataset
        void write_uint32_dataset(
            H5::H5File&                       file,
            const std::string&                name,
            const std::vector<std::uint32_t>& data
        )
        {
            if (data.empty()) {
                return;
            }

            hsize_t size = data.size();
            auto    dset = create_compressed_dataset(file, name, size, H5::PredType::NATIVE_UINT32);
            dset.write(data.data(), H5::PredType::NATIVE_UINT32);
        }

        // helper: read uint32 vector dataset
        std::vector<std::uint32_t>
        read_uint32_dataset(const H5::H5File& file, const std::string& name)
        {
            auto    dset  = file.openDataSet(name);
            auto    space = dset.getSpace();
            hsize_t size;
            space.getSimpleExtentDims(&size);

            std::vector<std::uint32_t> data(size);
            dset.read(data.data(), H5::PredType::NATIVE_UINT32);
            return data;
        }

        // helper: write bool vector as uint8 dataset
        void
        write_bool_dataset(H5::H5File& file, const std::string& name, const std::vector<bool>& data)
        {
            if (data.empty()) {
                return;
            }

            std::vector<std::uint8_t> uint_data(data.size());
            for (size_t ii = 0; ii < data.size(); ++ii) {
                uint_data[ii] = data[ii] ? 1 : 0;
            }

            hsize_t size = uint_data.size();
            auto    dset = create_compressed_dataset(file, name, size, H5::PredType::NATIVE_UINT8);
            dset.write(uint_data.data(), H5::PredType::NATIVE_UINT8);
        }

        // helper: read bool vector dataset
        std::vector<bool> read_bool_dataset(const H5::H5File& file, const std::string& name)
        {
            auto    dset  = file.openDataSet(name);
            auto    space = dset.getSpace();
            hsize_t size;
            space.getSimpleExtentDims(&size);

            std::vector<std::uint8_t> uint_data(size);
            dset.read(uint_data.data(), H5::PredType::NATIVE_UINT8);

            std::vector<bool> data(size);
            for (size_t ii = 0; ii < size; ++ii) {
                data[ii] = uint_data[ii] != 0;
            }
            return data;
        }

    } // anonymous namespace

    void write_photon_events(
        const std::string&                 filename,
        const std::vector<photon_event_t>& events,
        const sim_conditions_t&            sim_cond,
        const quant_scales_t&              qscales
    )
    {
        if (events.empty()) {
            throw std::invalid_argument("cannot write empty event list");
        }

        H5::H5File file(filename, H5F_ACC_TRUNC);
        hsize_t    n_events = events.size();

        // write metadata as attributes
        write_attribute(file, "dt", sim_cond.dt);
        write_attribute(file, "theta_obs", sim_cond.theta_obs);
        write_attribute(file, "adiabatic_index", sim_cond.adiabatic_index);
        write_attribute(file, "current_time", sim_cond.current_time);
        write_attribute(file, "p", sim_cond.p);
        write_attribute(file, "z", sim_cond.z);
        write_attribute(file, "eps_e", sim_cond.eps_e);
        write_attribute(file, "eps_b", sim_cond.eps_b);
        write_attribute(file, "d_L", sim_cond.d_L);
        write_attribute(file, "time_scale", qscales.time_scale);
        write_attribute(file, "pre_scale", qscales.pre_scale);
        write_attribute(file, "rho_scale", qscales.rho_scale);
        write_attribute(file, "v_scale", qscales.v_scale);
        write_attribute(file, "length_scale", qscales.length_scale);
        write_attribute(file, "n_events", static_cast<std::uint64_t>(n_events));
        write_attribute(file, "hydro_type", static_cast<std::uint32_t>(sim_cond.hydro_type));

        // write frequencies as dataset
        if (!sim_cond.nus.empty()) {
            write_vector_dataset(file, "frequencies", sim_cond.nus);
        }

        // extract fields into columnar arrays
        std::vector<double>        t_emission(n_events);
        std::vector<double>        x(n_events), y(n_events), z(n_events);
        std::vector<double>        energy(n_events);
        std::vector<double>        px(n_events), py(n_events), pz(n_events);
        std::vector<double>        stokes_I(n_events), stokes_Q(n_events);
        std::vector<double>        stokes_U(n_events), stokes_V(n_events);
        std::vector<double>        doppler_factor(n_events);
        std::vector<double>        lorentz_factor(n_events);
        std::vector<double>        optical_depth(n_events);
        std::vector<std::uint32_t> cell_id(n_events);
        std::vector<bool>          absorbed(n_events);
        std::vector<std::uint32_t> n_scatter(n_events);

        for (size_t ii = 0; ii < n_events; ++ii) {
            const auto& evt    = events[ii];
            t_emission[ii]     = evt.t_emission;
            x[ii]              = evt.x;
            y[ii]              = evt.y;
            z[ii]              = evt.z;
            energy[ii]         = evt.energy;
            px[ii]             = evt.px;
            py[ii]             = evt.py;
            pz[ii]             = evt.pz;
            stokes_I[ii]       = evt.stokes_I;
            stokes_Q[ii]       = evt.stokes_Q;
            stokes_U[ii]       = evt.stokes_U;
            stokes_V[ii]       = evt.stokes_V;
            doppler_factor[ii] = evt.doppler_factor;
            lorentz_factor[ii] = evt.lorentz_factor;
            optical_depth[ii]  = evt.optical_depth;
            cell_id[ii]        = evt.cell_id;
            absorbed[ii]       = evt.absorbed;
            n_scatter[ii]      = evt.n_scatter;
        }

        // write columnar datasets
        write_vector_dataset(file, "t_emission", t_emission);
        write_vector_dataset(file, "x", x);
        write_vector_dataset(file, "y", y);
        write_vector_dataset(file, "z", z);
        write_vector_dataset(file, "energy", energy);
        write_vector_dataset(file, "px", px);
        write_vector_dataset(file, "py", py);
        write_vector_dataset(file, "pz", pz);
        write_vector_dataset(file, "stokes_I", stokes_I);
        write_vector_dataset(file, "stokes_Q", stokes_Q);
        write_vector_dataset(file, "stokes_U", stokes_U);
        write_vector_dataset(file, "stokes_V", stokes_V);
        write_vector_dataset(file, "doppler_factor", doppler_factor);
        write_vector_dataset(file, "lorentz_factor", lorentz_factor);
        write_vector_dataset(file, "optical_depth", optical_depth);
        write_uint32_dataset(file, "cell_id", cell_id);
        write_bool_dataset(file, "absorbed", absorbed);
        write_uint32_dataset(file, "n_scatter", n_scatter);

        file.close();
    }

    std::pair<std::vector<photon_event_t>, photon_event_metadata_t>
    read_photon_events(const std::string& filename)
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // read metadata
        photon_event_metadata_t meta;
        meta.dt              = read_attribute<double>(file, "dt");
        meta.theta_obs       = read_attribute<double>(file, "theta_obs");
        meta.adiabatic_index = read_attribute<double>(file, "adiabatic_index");
        meta.current_time    = read_attribute<double>(file, "current_time");
        meta.p               = read_attribute<double>(file, "p");
        meta.z               = read_attribute<double>(file, "z");
        meta.eps_e           = read_attribute<double>(file, "eps_e");
        meta.eps_b           = read_attribute<double>(file, "eps_b");
        meta.d_L             = read_attribute<double>(file, "d_L");
        meta.time_scale      = read_attribute<double>(file, "time_scale");
        meta.pre_scale       = read_attribute<double>(file, "pre_scale");
        meta.rho_scale       = read_attribute<double>(file, "rho_scale");
        meta.v_scale         = read_attribute<double>(file, "v_scale");
        meta.length_scale    = read_attribute<double>(file, "length_scale");
        meta.n_events        = read_attribute<std::uint64_t>(file, "n_events");
        meta.hydro_type =
            static_cast<hydro_type_t>(read_attribute<std::uint32_t>(file, "hydro_type"));

        // read frequencies if present
        try {
            meta.frequencies = read_vector_dataset(file, "frequencies");
        }
        catch (...) {
            // no frequencies stored
        }

        // read columnar data
        auto t_emission     = read_vector_dataset(file, "t_emission");
        auto x              = read_vector_dataset(file, "x");
        auto y              = read_vector_dataset(file, "y");
        auto z              = read_vector_dataset(file, "z");
        auto energy         = read_vector_dataset(file, "energy");
        auto px             = read_vector_dataset(file, "px");
        auto py             = read_vector_dataset(file, "py");
        auto pz             = read_vector_dataset(file, "pz");
        auto stokes_I       = read_vector_dataset(file, "stokes_I");
        auto stokes_Q       = read_vector_dataset(file, "stokes_Q");
        auto stokes_U       = read_vector_dataset(file, "stokes_U");
        auto stokes_V       = read_vector_dataset(file, "stokes_V");
        auto doppler_factor = read_vector_dataset(file, "doppler_factor");
        auto lorentz_factor = read_vector_dataset(file, "lorentz_factor");
        auto optical_depth  = read_vector_dataset(file, "optical_depth");
        auto cell_id        = read_uint32_dataset(file, "cell_id");
        auto absorbed       = read_bool_dataset(file, "absorbed");
        auto n_scatter      = read_uint32_dataset(file, "n_scatter");

        file.close();

        // reconstruct events
        std::vector<photon_event_t> events(meta.n_events);
        for (size_t ii = 0; ii < meta.n_events; ++ii) {
            events[ii].t_emission     = t_emission[ii];
            events[ii].x              = x[ii];
            events[ii].y              = y[ii];
            events[ii].z              = z[ii];
            events[ii].energy         = energy[ii];
            events[ii].px             = px[ii];
            events[ii].py             = py[ii];
            events[ii].pz             = pz[ii];
            events[ii].stokes_I       = stokes_I[ii];
            events[ii].stokes_Q       = stokes_Q[ii];
            events[ii].stokes_U       = stokes_U[ii];
            events[ii].stokes_V       = stokes_V[ii];
            events[ii].doppler_factor = doppler_factor[ii];
            events[ii].lorentz_factor = lorentz_factor[ii];
            events[ii].optical_depth  = optical_depth[ii];
            events[ii].cell_id        = cell_id[ii];
            events[ii].absorbed       = absorbed[ii];
            events[ii].n_scatter      = n_scatter[ii];
        }

        return {events, meta};
    }

    photon_event_metadata_t read_photon_event_metadata(const std::string& filename)
    {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        photon_event_metadata_t meta;
        meta.dt              = read_attribute<double>(file, "dt");
        meta.theta_obs       = read_attribute<double>(file, "theta_obs");
        meta.adiabatic_index = read_attribute<double>(file, "adiabatic_index");
        meta.current_time    = read_attribute<double>(file, "current_time");
        meta.p               = read_attribute<double>(file, "p");
        meta.z               = read_attribute<double>(file, "z");
        meta.eps_e           = read_attribute<double>(file, "eps_e");
        meta.eps_b           = read_attribute<double>(file, "eps_b");
        meta.d_L             = read_attribute<double>(file, "d_L");
        meta.time_scale      = read_attribute<double>(file, "time_scale");
        meta.pre_scale       = read_attribute<double>(file, "pre_scale");
        meta.rho_scale       = read_attribute<double>(file, "rho_scale");
        meta.v_scale         = read_attribute<double>(file, "v_scale");
        meta.length_scale    = read_attribute<double>(file, "length_scale");
        meta.n_events        = read_attribute<std::uint64_t>(file, "n_events");
        meta.hydro_type =
            static_cast<hydro_type_t>(read_attribute<std::uint32_t>(file, "hydro_type"));

        try {
            meta.frequencies = read_vector_dataset(file, "frequencies");
        }
        catch (...) {
            // no frequencies
        }

        file.close();
        return meta;
    }

} // namespace simbi::afterglow
