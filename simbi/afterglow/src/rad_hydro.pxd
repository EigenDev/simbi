# distutils: language = c++

from libcpp.vector cimport vector
cdef extern from "rad_units.hpp" namespace "sogbo_rad":
    cdef struct sim_conditions:
        double dt, theta_obs, ad_gamma, current_time, p, z, eps_e, eps_b, d_L
        vector[double] nus

    cdef struct quant_scales:
        double time_scale, pre_scale, rho_scale, v_scale, length_scale

    cdef void calc_fnu(
        sim_conditions args,
        quant_scales  qscales,
        vector[vector[double]] &fields, 
        vector[vector[double]] &mesh,
        vector[double] &tbin_edges,
        vector[double] &flux_array, 
        int chkpt_idx,
        int data_dim
    )

    cdef void log_events(
        sim_conditions args,
        quant_scales   qscales,
        vector[vector[double]] &fields,
        vector[vector[double]] &mesh,
        vector[double] &photon_distribution,
        vector[double] &four_position,
        int data_dim
    )