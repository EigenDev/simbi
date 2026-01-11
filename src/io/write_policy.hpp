#ifndef IO_WRITE_POLICY_HPP
#define IO_WRITE_POLICY_HPP

#include "build_config.hpp"
#include "write_traits.hpp"

#include <H5Cpp.h>
#include <vector>

namespace simbi::io {

    enum class precision_t {
        native,
        float32,
        float64
    };

    struct write_policy_t {
        precision_t precision = precision_t::native;
        bool compress         = false;
        int compression_level = 6;   // 0-9, only used if compress=true
        bool chunked          = true;

        // get hdf5 data type based on precision setting
        H5::DataType data_type() const
        {
            switch (precision) {
                case precision_t::float32: return H5::PredType::NATIVE_FLOAT;
                case precision_t::float64: return H5::PredType::NATIVE_DOUBLE;
                default: return h5_pred_type<real>::value();
            }
        }

        // configure dataset creation property list
        H5::DSetCreatPropList
        creation_props(const std::vector<hsize_t>& dims) const
        {
            H5::DSetCreatPropList plist;

            if (chunked || compress) {
                // for chunking, use the dataset dims as chunk dims
                // (could be smarter about this for very large datasets)
                plist.setChunk(static_cast<int>(dims.size()), dims.data());
            }

            if (compress) {
                plist.setDeflate(compression_level);
            }

            return plist;
        }
    };

}   // namespace simbi::io

#endif   // IO_WRITE_POLICY_HPP
