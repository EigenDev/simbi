
#include "H5Cpp.h"
#include "adapter/device_adapter_api.hpp"
#include "util/parallel/parallel_for.hpp"

namespace simbi {
    namespace helpers {
        template <typename... Args>
        std::string string_format(const std::string& format, Args... args)
        {
            size_t size = snprintf(nullptr, 0, format.c_str(), args...) +
                          1;   // Extra space for '\0'
            if (size <= 0) {
                throw std::runtime_error("Error during formatting.");
            }
            std::unique_ptr<char[]> buf(new char[size]);
            snprintf(buf.get(), size, format.c_str(), args...);
            return std::string(
                buf.get(),
                buf.get() + size - 1
            );   // We don't want the '\0' inside
        }

        /***
         * takes a string and adds a separator character every n-th steps
         * through the string
         * @param input input string
         * @return none
         */
        template <const unsigned num, const char separator>
        void separate(std::string& input)
        {
            for (auto it = input.rbegin() + 1;
                 (num + 0) <= std::distance(it, input.rend());
                 ++it) {
                std::advance(it, num - 1);
                it = std::make_reverse_iterator(
                    input.insert(it.base(), separator)
                );
            }
        }

        template <typename T>
        DUAL T solve_cubic(T b, T c, T d)
        {
            T p = c - b * b / 3.0;
            T q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;

            if (p == 0.0) {
                return std::pow(q, 1.0 / 3.0);
            }
            if (q == 0.0) {
                return 0.0;
            }

            T t = std::sqrt(std::abs(p) / 3.0);
            T g = 1.5 * q / (p * t);
            if (p > 0.0) {
                return -2.0 * t * std::sinh(std::asinh(g) / 3.0) - b / 3.0;
            }
            else if (4.0 * p * p * p + 27.0 * q * q < 0.0) {
                return 2.0 * t * std::cos(std::acos(g) / 3.0) - b / 3.0;
            }
            else if (q > 0.0) {
                return -2.0 * t * std::cosh(std::acosh(-g) / 3.0) - b / 3.0;
            }

            return 2.0 * t * std::cosh(std::acosh(g) / 3.0) - b / 3.0;
        }

        /*--------------------------------------------
            quartic solver adapted from:
            https://stackoverflow.com/a/50747781/13874039
        --------------------------------------------*/
        template <typename T>
        DUAL int solve_quartic(T b, T c, T d, T e, T res[4])
        {
            T p = c - 0.375 * b * b;
            T q = 0.125 * b * b * b - 0.5 * b * c + d;
            T m = solve_cubic<real>(
                p,
                0.25 * p * p + 0.01171875 * b * b * b * b - e + 0.25 * b * d -
                    0.0625 * b * b * c,
                -0.125 * q * q
            );
            if (q == 0.0) {
                if (m < 0.0) {
                    return 0;
                };

                int nroots = 0;
                T sqrt_2m  = std::sqrt(2.0 * m);
                if (-m - p > 0.0) {
                    T delta       = std::sqrt(2.0 * (-m - p));
                    res[nroots++] = -0.25 * b + 0.5 * (sqrt_2m - delta);
                    res[nroots++] = -0.25 * b - 0.5 * (sqrt_2m - delta);
                    res[nroots++] = -0.25 * b + 0.5 * (sqrt_2m + delta);
                    res[nroots++] = -0.25 * b - 0.5 * (sqrt_2m + delta);
                }

                if (-m - p == 0.0) {
                    res[nroots++] = -0.25 * b - 0.5 * sqrt_2m;
                    res[nroots++] = -0.25 * b + 0.5 * sqrt_2m;
                }

                if constexpr (platform::is_gpu) {
                    iterativeQuickSort(res, 0, nroots - 1);
                }
                else {
                    recursiveQuickSort(res, 0, nroots - 1);
                }
                return nroots;
            }

            if (m < 0.0) {
                return 0;
            };
            T sqrt_2m  = std::sqrt(2.0 * m);
            int nroots = 0;
            if (-m - p + q / sqrt_2m >= 0.0) {
                T delta       = std::sqrt(2.0 * (-m - p + q / sqrt_2m));
                res[nroots++] = 0.5 * (-sqrt_2m + delta) - 0.25 * b;
                res[nroots++] = 0.5 * (-sqrt_2m - delta) - 0.25 * b;
            }

            if (-m - p - q / sqrt_2m >= 0.0) {
                T delta       = std::sqrt(2.0 * (-m - p - q / sqrt_2m));
                res[nroots++] = 0.5 * (sqrt_2m + delta) - 0.25 * b;
                res[nroots++] = 0.5 * (sqrt_2m - delta) - 0.25 * b;
            }

            if constexpr (platform::is_gpu) {
                iterativeQuickSort(res, 0, 3);
            }
            else {
                recursiveQuickSort(res, 0, nroots - 1);
            }
            return nroots;
        }

        // Function to swap two elements
        template <typename T>
        DUAL void myswap(T& a, T& b)
        {
            T temp = a;
            a      = b;
            b      = temp;
        }

        // Partition the array and return the pivot index
        template <typename T, typename index_type>
        DUAL index_type partition(T arr[], index_type low, index_type high)
        {
            T pivot = arr[high];   // Choose the rightmost element as the pivot
            index_type i = low - 1;   // Index of the smaller element

            for (index_type j = low; j <= high - 1; j++) {
                if (arr[j] <= pivot) {
                    i++;
                    myswap(arr[i], arr[j]);
                }
            }
            myswap(arr[i + 1], arr[high]);
            return i + 1;   // Return the pivot index
        }

        // Quick sort implementation
        template <typename T, typename index_type>
        DUAL void recursiveQuickSort(T arr[], index_type low, index_type high)
        {
            if (low < high) {
                index_type pivotIndex = partition(arr, low, high);

                // Recursively sort the left and right subarrays
                recursiveQuickSort(arr, low, pivotIndex - 1);
                recursiveQuickSort(arr, pivotIndex + 1, high);
            }
        }

        template <typename T, typename index_type>
        DUAL void iterativeQuickSort(T arr[], index_type low, index_type high)
        {
            // Create an auxiliary stack
            T stack[4];

            // initialize top of stack
            index_type top = -1;

            // push initial values of l and h to stack
            stack[++top] = low;
            stack[++top] = high;

            // Keep popping from stack while is not empty
            while (top >= 0) {
                // Pop h and l
                high = stack[top--];
                low  = stack[top--];

                // Set pivot element at its correct position
                // in sorted array
                index_type pivotIndex = partition(arr, low, high);

                // If there are elements on left side of pivot,
                // then push left side to stack
                if (pivotIndex - 1 > low) {
                    stack[++top] = low;
                    stack[++top] = pivotIndex - 1;
                }

                // If there are elements on right side of pivot,
                // then push right side to stack
                if (pivotIndex + 1 < high) {
                    stack[++top] = pivotIndex + 1;
                    stack[++top] = high;
                }
            }
        }

        template <typename T, typename U>
        inline real get_flops(
            const luint dim,
            const luint radius,
            const luint total_zones,
            const luint real_zones,
            const float delta_t
        )
        {
            // the advance step does one write plus 1.0 + dim * 2 * hr reads
            const float advance_contr =
                real_zones * sizeof(T) * (1.0 + (1.0 + dim * 2 * radius));
            const float cons2prim_contr = total_zones * sizeof(U);
            const float ghost_conf_contr =
                (total_zones - real_zones) * sizeof(T);
            return (advance_contr + cons2prim_contr + ghost_conf_contr) /
                   (delta_t * 1e9);
        }

        template <size_type Dims>
        DEV static auto
        memory_layout_coordinates(auto idx, const uarray<Dims>& shape)
            -> uarray<Dims>
        {
            uarray<Dims> coords;
            auto stride = 1;
            if constexpr (global::col_major) {
                // Column major: shape=(nk,nj,ni)
                // Want [k,j,i] where k is fastest
                for (size_type ii = 0; ii < Dims; ++ii) {
                    coords[Dims - 1 - ii] =
                        (idx / stride) % shape[Dims - 1 - ii];
                    stride *= shape[Dims - 1 - ii];
                }
            }
            else {
                // Row major: shape=(nk,nj,ni)
                // Want [i,j,k] where i is fastest
                for (size_type ii = Dims - 1; ii < Dims; --ii) {
                    coords[Dims - 1 - ii] = (idx / stride) % shape[ii];
                    stride *= shape[ii];
                }
            }
            return coords;
        }
    }   // namespace helpers
}   // namespace simbi
