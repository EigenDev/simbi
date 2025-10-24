#ifndef ENTITY_HPP
#define ENTITY_HPP

#include <any>             // for std::any
#include <cstdint>         // for std::uint64_t
#include <tuple>           // for std::tuple
#include <typeindex>       // for std::type_index
#include <unordered_map>   // for std::unordered_map
#include <utility>         // for std::pair, std::move
#include <vector>          // for std::vector

namespace simbi::ecs {
    /**
     * Here lies an extremely minimal Entity-Component-System (ECS) framework.
     * Entities are represented by unique IDs (entity_t), and components
     * are stored in a type-erased manner using std::any within a registry.
     * The registry allows for adding, retrieving, checking, and removing
     * components associated with entities. It also provides simple
     * viewing capabilities to iterate over entities possessing specific
     * components.
     */

    using entity_t = std::uint64_t;

    // minimal registry: stores components by type
    class registry_t
    {
        std::uint64_t next_id_{0};

        // type_index -> (entity_id -> component)
        std::unordered_map<
            std::type_index,
            std::unordered_map<entity_t, std::any>>
            storage_;

      public:
        entity_t create() { return next_id_++; }

        template <typename T>
        void add(entity_t entity, T component)
        {
            auto type              = std::type_index(typeid(T));
            storage_[type][entity] = std::move(component);
        }

        template <typename T>
        T& get(entity_t entity)
        {
            auto type = std::type_index(typeid(T));
            return std::any_cast<T&>(storage_[type].at(entity));
        }

        template <typename T>
        const T& get(entity_t entity) const
        {
            auto type = std::type_index(typeid(T));
            return std::any_cast<const T&>(storage_.at(type).at(entity));
        }

        template <typename T>
        bool has(entity_t entity) const
        {
            auto type = std::type_index(typeid(T));
            auto it   = storage_.find(type);
            if (it == storage_.end()) {
                return false;
            }
            return it->second.contains(entity);
        }

        template <typename T>
        void remove(entity_t entity)
        {
            auto type = std::type_index(typeid(T));
            storage_[type].erase(entity);
        }

        // iterate over entities with component T
        template <typename T>
        auto view()
        {
            auto type = std::type_index(typeid(T));
            std::vector<std::pair<entity_t, T*>> result;

            if (storage_.contains(type)) {
                for (auto& [entity, component_any] : storage_[type]) {
                    result.emplace_back(
                        entity,
                        &std::any_cast<T&>(component_any)
                    );
                }
            }

            return result;
        }

        // iterate over entities with components T and U
        template <typename T, typename U>
        auto view()
        {
            std::vector<std::tuple<entity_t, T*, U*>> result;

            for (auto [entity, t_ptr] : view<T>()) {
                if (has<U>(entity)) {
                    result.emplace_back(entity, t_ptr, &get<U>(entity));
                }
            }

            return result;
        }
    };

}   // namespace simbi::ecs

#endif
