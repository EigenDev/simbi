/**
 *  *=============================================================================
 *  *           SIMBI - Special Relativistic Magnetohydrodynamics Code
 *  *=============================================================================
 *  *
 *  * @file            time_manager.hpp
 *  * @brief           a helper struct to manage time info in the simulation
 *  * @details
 *  *
 *  * @version         0.8.0
 *  * @date            2025-02-23
 *  * @author          Marcus DuPont
 *  * @email           marcus.dupont@princeton.edu
 *  *
 *  *==============================================================================
 *  * @build           Requirements & Dependencies
 *  *==============================================================================
 *  * @requires        C++20
 *  * @depends         CUDA >= 11.0, HDF5 >= 1.12, OpenMP >= 4.5
 *  * @platform        Linux, MacOS
 *  * @parallel        GPU (CUDA, HIP), CPU (OpenMP)
 *  *
 *  *==============================================================================
 *  * @documentation   Reference & Notes
 *  *==============================================================================
 *  * @usage
 *  * @note
 *  * @warning
 *  * @todo
 *  * @bug
 *  * @performance
 *  *
 *  *==============================================================================
 *  * @testing        Quality Assurance
 *  *==============================================================================
 *  * @test
 *  * @benchmark
 *  * @validation
 *  *
 *  *==============================================================================
 *  * @history        Version History
 *  *==============================================================================
 *  * 2025-02-23      v0.8.0      Initial implementation
 *  *
 *  *==============================================================================
 *  * @copyright (C) 2025 Marcus DuPont. All rights reserved.
 *  *==============================================================================
 */
#ifndef TIME_MANAGER_HPP
#define TIME_MANAGER_HPP

#include "build_options.hpp"
#include "core/types/utility/init_conditions.hpp"
namespace simbi {
    class TimeManager
    {
      private:
        // Time variables
        real t_, dt_, tend_, dlogt_;
        real checkpoint_interval_;
        real next_checkpoint_time_{0.0};

      public:
        TimeManager(const InitialConditions& init)
            : t_(init.time),
              tend_(init.tend),
              dlogt_(init.dlogt),
              checkpoint_interval_(init.checkpoint_interval)
        {
        }

        // simple accessors
        real time() const { return t_; }
        real dt() const { return dt_; }
        real tend() const { return tend_; }
        real dlogt() const { return dlogt_; }
        real checkpoint_interval() const { return checkpoint_interval_; }

        // simple mutators
        void advance(const real step) { t_ += step * dt_; }
        void set_dt(real dt) { dt_ = dt; }
        void update_next_checkpoint_time()
        {
            // Set the initial time interval
            // based on the current time, advanced
            // by the checkpoint interval to the nearest
            // place in the log10 scale. If dlogt is 0
            // then the interval is set to the current time
            // shifted towards the nearest checkpoint interval
            // if the checkpoint interval is 0 then the interval
            // is set to the current time
            if (dlogt_ != 0) {
                next_checkpoint_time_ =
                    t_ * std::pow(10.0, std::floor(std::log10(t_) + dlogt_));
            }
            else {
                static auto round_place = 1.0 / checkpoint_interval_;
                next_checkpoint_time_ =
                    checkpoint_interval_ +
                    std::floor(t_ * round_place + 0.5) / round_place;
            }
        }

        // checkpointing
        bool time_to_write_checkpoint() const
        {
            return t_ >= next_checkpoint_time_;
        }
        bool log_time_enabled() const { return dlogt_ != 0; }
        auto checkpoint_time() const { return next_checkpoint_time_; }
    };
}   // namespace simbi

#endif