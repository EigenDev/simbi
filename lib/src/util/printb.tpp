
namespace simbi
{
    namespace util
    {
        template <typename ...ARGS>
        void write (std::string const & fmt, ARGS... args) {
                const std::string argss[] = {[](const auto &x){
                        std::stringstream ss;
                        // Set Fixed -Point Notation
                        ss << std::fixed;
                        // Set precision to 3 digits
                        ss << std::setprecision(3);
                        ss << std::scientific;
                        ss << x;
                        return ss.str();
                } (args) ... }; 

                auto argss_len = sizeof(argss) / sizeof(argss[0]);
                static int index = 0;
                for (auto &ch: fmt) {
                        if (ch == '{') {
                                auto const left  = ch;
                                auto const right = *(&ch +1);
                                if (right != '}') {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "missing closing brace"
                                        );
                                }
                                std::cout << std::setw(10) << argss[index];
                                index++;
                                index %= argss_len;
                        } else if (ch != '}') {
                                std::cout << ch;
                        }
                }
        }

        template <typename ...ARGS> void writeln(std::string const & fmt, ARGS... args) {
                write (fmt, args...);
                std::cout << '\n';
        }

        template <typename ...ARGS> void writefl(std::string const & fmt, ARGS... args) {
                write (fmt, args...);
                std::cout << std::flush;
        }
    } // namespace util
    
} // namespace simbi
