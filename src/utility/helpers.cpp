#include "utility/helpers.hpp"

#include "build_config.hpp"
#include "io/exceptions.hpp"
#include "utility/enums.hpp"

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <sys/signal.h>

//==================================
//              GPU HELPERS
//==================================
using namespace H5;

namespace simbi {
    namespace helpers {
        // Flag that detects whether program was terminated by external forces
        std::atomic<bool> killsig_received = false;

        std::string get_color_code(color_t color)
        {
            switch (color) {
                case color_t::RED:
                    return "\033[31m";
                case color_t::GREEN:
                    return "\033[32m";
                case color_t::YELLOW:
                    return "\033[33m";
                case color_t::BLUE:
                    return "\033[34m";
                case color_t::MAGENTA:
                    return "\033[35m";
                case color_t::CYAN:
                    return "\033[36m";
                case color_t::WHITE:
                    return "\033[37m";
                case color_t::LIGHT_BLUE:
                    return "\033[0;94m";
                case color_t::LIGHT_CYAN:
                    return "\033[0;96m";
                case color_t::LIGHT_GREEN:
                    return "\033[0;92m";
                case color_t::LIGHT_GREY:
                    return "\033[0;37m";
                case color_t::LIGHT_MAGENTA:
                    return "\033[0;95m";
                case color_t::LIGHT_RED:
                    return "\033[0;91m";
                case color_t::LIGHT_YELLOW:
                    return "\033[0;93m";
                case color_t::BLACK:
                    return "\033[0;30m";
                case color_t::DARK_GREY:
                    return "\033[0;90m";
                case color_t::BOLD:
                    return "\033[1m";
                default:
                    return "\033[0m";
            }
        }

        void catch_signals()
        {
            const static auto signal_handler = [](int) { killsig_received = true; };
            std::signal(SIGTERM, signal_handler);
            std::signal(SIGINT, signal_handler);
            std::signal(SIGABRT, signal_handler);
            std::signal(SIGSEGV, signal_handler);
            std::signal(SIGQUIT, signal_handler);
            if (killsig_received) {
                killsig_received = false;
                throw exception::InterruptException(1);
            }
        }

        std::string format_real(real value)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << value;
            std::string str = oss.str();

            // Replace decimal postd::int64_twith underscore
            std::replace(str.begin(), str.end(), '.', '_');

            // Pad with zeros if necessary
            if (str.find('_') == std::string::npos) {
                str += "_000";
            }
            else {
                while (str.substr(str.find('_') + 1).length() < 3) {
                    str += "0";
                }
            }

            // Ensure the string is at least in the hundreds place
            if (str[0] == '-') {
                while (str.find('_') < 4) {
                    str.insert(1, "0");
                }
            }
            else {
                while (str.find('_') < 3) {
                    str.insert(0, "0");
                }
            }

            // Insert underscores for thousands, millions, etc.
            std::int64_t insert_position = str.find('_') - 3;
            while (insert_position > 0) {
                str.insert(insert_position, "_");
                insert_position -= 3;
            }

            return str;
        }

        std::string error_code_to_string(ErrorCode code)
        {
            // handle bit-field error codes
            if (code == ErrorCode::NONE) {
                return "No error";
            }

            std::string result;
            // Check each error code and append the corresponding message
            // to the result string. For each error code after the first one,
            // we add 'or' to the result string.
            if (has_error(code, ErrorCode::NEGATIVE_PRESSURE)) {
                result += "negative pressure or ";
            }
            if (has_error(code, ErrorCode::NON_FINITE_PRESSURE)) {
                result += "non-finite pressure or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_DENSITY)) {
                result += "negative density or ";
            }
            if (has_error(code, ErrorCode::SUPERLUMINAL_VELOCITY)) {
                result += "superluminal velocity or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_ENERGY)) {
                result += "negative energy or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_ENTROPY)) {
                result += "negative entropy or ";
            }
            if (has_error(code, ErrorCode::NEGATIVE_MASS)) {
                result += "negative mass or ";
            }
            if (has_error(code, ErrorCode::NON_FINITE_ROOT)) {
                result += "non-finite root or ";
            }
            if (has_error(code, ErrorCode::MAX_ITER)) {
                result += "maximum iterations reached or ";
            }
            if (has_error(code, ErrorCode::UNDEFINED)) {
                result += "undefined error or ";
            }
            if (result.empty()) {
                return "Unknown error";
            }
            result.pop_back(); // Remove the trailing space
            result.pop_back(); // Remove the trailing r
            result.pop_back(); // Remove the trailing o
            result.pop_back(); // Remove the trailing space
            return result;
        }

    } // namespace helpers
} // namespace simbi
