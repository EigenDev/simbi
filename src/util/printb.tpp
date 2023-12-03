
namespace simbi {
    namespace util {
        template <Color C, typename... ARGS>
        void write(std::string const& fmt, ARGS... args)
        {
            const std::string argss[] = {[](const auto& x) {
                std::stringstream ss;
                ss << x;
                return ss.str();
            }(args)...};

            auto argss_len        = sizeof(argss) / sizeof(argss[0]);
            std::string width_str = "";
            int width             = 1;
            int index             = 0;
            bool open_brace       = false;
            bool inserted         = false;
            unsigned cidx         = 0;
            for (const auto& ch : fmt) {
                if (ch == '{') {
                    open_brace = true;
                    inserted   = false;
                }
                else if (ch == '}') {
                    open_brace = false;
                    continue;
                }

                if (open_brace) {
                    if (!inserted) {
                        auto const right = *(&ch + 1);
                        if (right == ':') {
                            bool scientific = false;
                            int precision   = 0;
                            for (auto ii = cidx; ii < fmt.size(); ii++) {
                                const auto fmt_char = fmt[ii];
                                if (fmt_char == '>') {
                                    const auto left_num  = fmt[ii + 1];
                                    const auto right_num = fmt[ii + 2];
                                    if (isdigit(left_num)) {
                                        width_str.push_back(left_num);
                                    }

                                    if (isdigit(right_num)) {
                                        width_str.push_back(right_num);
                                    }

                                    if (width_str.size() > 0) {
                                        width = std::stoi(width_str);
                                    }
                                }
                                else if (fmt_char == '.') {
                                    const auto left_num       = fmt[ii + 1];
                                    const auto right_num      = fmt[ii + 2];
                                    std::string precision_str = "";
                                    if (isdigit(left_num)) {
                                        precision_str.push_back(left_num);
                                    }

                                    if (isdigit(right_num)) {
                                        precision_str.push_back(right_num);
                                    }

                                    if (precision_str.size() > 0) {
                                        precision = std::stoi(precision_str);
                                    }
                                    scientific = fmt[ii + 2] == 'e' ||
                                                 fmt[ii + 3] == 'e';
                                }
                                else if (fmt_char == '}') {
                                    break;
                                }
                            }
                            const bool numeric = is_number(argss[index]);
                            if (scientific) {
                                if (numeric) {
                                    std::cout << std::fixed << std::scientific
                                              << std::setprecision(precision)
                                              << std::setw(width)
                                              << std::stod(argss[index]);
                                }
                                else {
                                    std::cout << argss[index];
                                }
                            }
                            else {
                                if (numeric) {
                                    std::cout << std::fixed
                                              << std::setprecision(precision)
                                              << std::setw(width)
                                              << std::stod(argss[index]);
                                }
                                else {
                                    std::cout << argss[index];
                                }
                            }

                            index++;
                            index %= argss_len;
                        }
                        else if (right == '}') {
                            std::cout << argss[index];
                            index++;
                            index %= argss_len;
                        }
                        else {
                            if (right != '}' && right != '>') {
                                throw std::invalid_argument(
                                    "syntax error in format string, "
                                    "missing closing brace");
                            }
                            else {
                                throw std::invalid_argument(
                                    "syntax error in format string, "
                                    "missing format signifier (:)");
                            }
                        }
                        inserted = true;
                    }
                }
                else {
                    width_str = "";
                    std::cout << color_map.at(C) << ch
                              << color_map.at(Color::RESET);
                }
                cidx++;
            }
        }

        template <Color C, typename... ARGS>
        void writeln(std::string const& fmt, ARGS... args)
        {
            std::cout << "\n";
            write<C>(fmt, args...);
            std::cout << '\n';
        }

        template <Color C, typename... ARGS>
        void writefl(std::string const& fmt, ARGS... args)
        {
            write<C>(fmt, args...);
            std::cout << std::flush;
        }
    }   // namespace util

}   // namespace simbi
