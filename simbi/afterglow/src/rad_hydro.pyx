import numpy as np 
import astropy.units as units 
cimport numpy as np 
cimport rad_hydro

def py_calc_fnu(
    fields:         dict, 
    tbin_edges:     np.ndarray,
    flux_array:     np.ndarray,
    mesh:           dict, 
    qscales:        dict, 
    sim_info:       dict,
    checkpoint_idx:      int,
    data_dim:       int,
):
    """
    Calculate the spectral flux from hydro data assuming a synchotron spectrum

    Params:
    ---------------------------------------
    fields: Dictionary of the hydro variables
    tbin_edges: a numpy ndarray of the time bin edges
    flux_array: a dictionary for the flux storage
    mesh:       a dictionary for the mesh central zones
    qscales:    a dictionary for the physical quantitiy scales in the problem
    sim_info:   a dictionary for the importnat simulation information like time, dt_chckpt, etc
    checkpoint_idx:  the checkpoint index value
    data_dim:   the dimensions of the checkpoint data
    """
    flattened_mesh = np.asanyarray(
        [mesh['x1'],
         mesh['x2'],
         mesh['x3']], dtype=object
    )

    cdef sim_conditions sim_cond 
    cdef quant_scales quant_scales

    # Set the sim conditions
    sim_cond.dt           = sim_info['dt']
    sim_cond.current_time = sim_info['current_time']
    sim_cond.theta_obs    = sim_info['theta_obs']
    sim_cond.ad_gamma     = sim_info['adiabatic_gamma']
    sim_cond.nus          = sim_info['nus'] * (1 + sim_info['z'])
    sim_cond.z            = sim_info['z']
    sim_cond.p            = sim_info['p']
    sim_cond.eps_e        = sim_info['eps_e']
    sim_cond.eps_b        = sim_info['eps_b']
    sim_cond.d_L          = sim_info['d_L']
   
    # set the dimensional scales
    quant_scales.time_scale    = qscales['time_scale']
    quant_scales.pre_scale     = qscales['pre_scale']
    quant_scales.rho_scale     = qscales['rho_scale']
    quant_scales.v_scale       = 1.0 
    quant_scales.length_scale  = qscales['length_scale']

    cdef vector[double] fnu = flux_array[:]
    calc_fnu(
        sim_cond,
        quant_scales,
        fields['rho'][:].flat,
        fields['gamma_beta'][:].flat,
        fields['p'][:].flat,
        flattened_mesh,
        tbin_edges / (1 + sim_info['z']),
        fnu, 
        checkpoint_idx,
        data_dim
    )
    flux_array[:] = np.ascontiguousarray(fnu)

def py_log_events(
    fields:         dict, 
    photon_distro:  np.ndarray,
    x_mu:           np.ndarray,
    mesh:           dict, 
    qscales:        dict, 
    sim_info:       dict,
    data_dim:       int):

    flattened_fields = np.array(
        [fields['rho'].flat,
        fields['gamma_beta'].flat,
        fields['p'].flat], dtype=float
    )

    flattened_mesh = np.asanyarray(
        [mesh['x1'].flat,
         mesh['x2'].flat,
         mesh['x3'].flat], dtype=object
    )

    cdef sim_conditions sim_cond 
    cdef quant_scales quant_scales

    # Set the sim conditions
    sim_cond.dt           = sim_info['dt']
    sim_cond.current_time = sim_info['current_time']
    sim_cond.theta_obs    = sim_info['theta_obs']
    sim_cond.ad_gamma     = sim_info['adiabatic_gamma']
    sim_cond.nus          = sim_info['nus'] * (1 + sim_info['z'])
    sim_cond.z            = sim_info['z']
    sim_cond.p            = sim_info['p']
    sim_cond.eps_e        = sim_info['eps_e']
    sim_cond.eps_b        = sim_info['eps_b']
    sim_cond.d_L          = sim_info['d_L']

    # set the dimensional scales
    quant_scales.time_scale    = qscales['time_scale']
    quant_scales.pre_scale     = qscales['pre_scale']
    quant_scales.rho_scale     = qscales['rho_scale']
    quant_scales.v_scale       = 1.0 
    quant_scales.length_scale  = qscales['length_scale']

    photon_flat   = photon_distro[:].flat
    x_mu_flat     = x_mu[:].flat
    log_events(
        sim_cond,
        quant_scales,
        flattened_fields,
        flattened_mesh,
        photon_flat,
        x_mu_flat,
        data_dim
    )

