#include "serializer.hpp"   // for serialization_context_t
#include "result.hpp"       // for result_t<T> monad
#include <H5Cpp.h>          // for H5::H5File
#include <functional>       // for std::function
#include <string>           // for std::string
#include <utility>          // for std::move
#include <vector>           // for std::vector

namespace simbi::io {

    auto create_file(const std::string& filename)
        -> result_t<serialization_context_t>
    {
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);
            return result_t<serialization_context_t>::ok(
                serialization_context_t{std::move(file), filename, {}, {}}
            );
        }
        catch (const H5::Exception& e) {
            return result_t<serialization_context_t>::error(
                "failed to create file " + filename + ": " + e.getDetailMsg()
            );
        }
    }

    auto close_file()
        -> std::function<result_t<std::string>(serialization_context_t)>
    {
        return [](serialization_context_t ctx) -> result_t<std::string> {
            try {
                ctx.file.close();
                return result_t<std::string>::ok(
                    "successfully wrote " +
                    std::to_string(ctx.written_datasets.size()) +
                    " datasets to " + ctx.filename
                );
            }
            catch (const H5::Exception& e) {
                return result_t<std::string>::error(
                    "error finalizing file: " + e.getDetailMsg()
                );
            }
        };
    }

}   // namespace simbi::io
