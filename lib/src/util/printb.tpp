namespace simbi
{
    namespace util
    {

        // lambdas not yet
        template <typename T> std::string to_string (T val) {
                std::stringstream ss;
                ss << val;
                return ss.str();
        }

        template <typename ...ARGS>
        void write (std::string const & fmt, ARGS... args) {
                const std::string argss[] = {to_string (args)...}; // <- indeed
                enum {argss_len = sizeof (argss) / sizeof(argss[0])};
        
                // no range based for loops yet ("for (auto it : fmt)")
                for (auto it: fmt) {
                        std::cout << it << "\n";
                        if (it == '{') {
                                auto const left = ++it;
                                if (&it != &fmt.back()) {
                                        // closing brace: fine
                                        if (it == '}')
                                                break;
                                        // check if numeric. if not, throw.
                                        switch (it) {
                                        case '0':case '1':case '2':case '3':case '4':
                                        case '5':case '6':case '7':case '8':case '9':;

                                        default:
                                                throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "only numeric digits allowed between "
                                                "braces"
                                                );
                                        
                                        };
                                }
                                if (it != '}') {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "missing closing brace"
                                        );
                                }
                                auto const right = it;
        
                                if (left == right) {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "no index given inside braces"
                                        );
                                }
        
                                std::stringstream ss;
                                ss << std::string(left,right);
                                size_t index;
                                ss >> index;
                                if (index >= argss_len) {
                                        throw std::invalid_argument (
                                                "syntax error in format string, "
                                                "index too big"
                                        );
                                }
                                std::cout << argss[index];
                        } else {
                                std::cout << it;
                        }
                }
        }

        template <typename ...ARGS> void writeln (std::string const & fmt, ARGS... args) {
                write (fmt, args...);
                std::cout << '\n';
        }
    } // namespace util
    
} // namespace simbi
