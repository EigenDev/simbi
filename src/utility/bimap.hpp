#ifndef UTILITY_BIMAP_HPP
#define UTILITY_BIMAP_HPP

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace simbi {
    template <typename T1, typename T2, size_t N>
    class bi_map_t
    {
      private:
        std::array<std::pair<T1, T2>, N> forward_map_;

      public:
        constexpr bi_map_t(std::array<std::pair<T1, T2>, N> init)
            : forward_map_(init)
        {
        }

        //  initializer list constructor
        constexpr bi_map_t(std::initializer_list<std::pair<T1, T2>> init)
        {
            if (init.size() != N) {
                throw std::length_error(
                    "Initializer list size must match template parameter N"
                );
            }
            std::copy(init.begin(), init.end(), forward_map_.begin());
        }

        constexpr T2 forward(const T1& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.first == key; }
            );
            if (it == forward_map_.end()) {
                throw std::runtime_error("Key not found in forward map");
            }
            return it->second;
        }

        constexpr T1 reverse(const T2& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.second == key; }
            );
            if (it == forward_map_.end()) {
                throw std::runtime_error("Key not found in reverse map");
            }
            return it->first;
        }

        //  optional versions that don't throw
        constexpr std::optional<T2> try_forward(const T1& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.first == key; }
            );
            return (it != forward_map_.end()) ? std::optional<T2>(it->second)
                                              : std::nullopt;
        }

        constexpr std::optional<T1> try_reverse(const T2& key) const
        {
            auto it = std::find_if(
                forward_map_.begin(),
                forward_map_.end(),
                [&key](const auto& pair) { return pair.second == key; }
            );
            return (it != forward_map_.end()) ? std::optional<T1>(it->first)
                                              : std::nullopt;
        }
    };

    // register to store BiMaps for each enum type
    template <typename EnumType>
    struct enum_bimap {
        // must be specialized for each enum
        static_assert(
            sizeof(EnumType) == 0,
            "enum_bimap must be specialized for this enum type"
        );
    };

    // generic serialize function using BiMap
    template <typename EnumType>
    std::string serialize(EnumType value)
    {
        auto name = enum_bimap<EnumType>::map.forward(value);
        std::string result(name);
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

    // generic deserialize function using BiMap
    template <typename EnumType>
    std::optional<EnumType> raw_deserialize(std::string_view name)
    {
        // convert to lowercase for lookup
        std::string lower_name(name);
        std::transform(
            lower_name.begin(),
            lower_name.end(),
            lower_name.begin(),
            ::tolower
        );

        return enum_bimap<EnumType>::map.try_reverse(
            std::string_view(lower_name)
        );
    }

    template <typename EnumType>
    EnumType deserialize(std::string_view name)
    {
        auto result = raw_deserialize<EnumType>(name);
        if (!result) {
            throw std::runtime_error(
                "Failed to deserialize enum: " + std::string(name)
            );
        }
        return *result;
    }

    template <typename EnumType, size_t N>
    using EnumBiMap = bi_map_t<EnumType, std::string_view, N>;
}   // namespace simbi

// convenience macro for registration
#define REGISTER_ENUM_BIMAP(EnumType, ...)                                     \
    template <>                                                                \
    struct enum_bimap<EnumType> {                                              \
        static constexpr std::array<                                           \
            std::pair<EnumType, std::string_view>,                             \
            std::initializer_list<std::pair<EnumType, std::string_view>>{      \
              __VA_ARGS__                                                      \
            }                                                                  \
                .size()>                                                       \
            data{{__VA_ARGS__}};                                               \
        static constexpr auto map = EnumBiMap<EnumType, data.size()>{data};    \
    };

#endif
