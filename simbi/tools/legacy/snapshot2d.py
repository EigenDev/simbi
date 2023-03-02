#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import argparse 
import astropy.units as u 
import utility as util 
import sys 
from utility import DEFAULT_SIZE, SMALL_SIZE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import cycle
from visual import lin_fields, derived
from simbi._detail import *
try:
    import cmasher as cmr 
except:
    print('No Cmasher module, so defaulting to matplotlib colors')
    
def place_annotation(args: argparse.ArgumentParser, fields: dict, ax: plt.Axes, etot: float) -> None:
    order_of_mag = np.floor(np.log10(etot))
    front_factor = int(etot / 10**order_of_mag)
    if front_factor != 1:
        anchor_text = r"$E_{\rm exp} = %i \times 10^{%i}$ erg"%(front_factor, order_of_mag)     
    else:
        anchor_text = r"$E_{\rm exp} = 10^{%i}$ erg"%(order_of_mag)
    
    if args.anot_text is not None:
        extra_text   = args.anot_text
        anchor_text += "\n     %s"%(extra_text)
    if args.print:
        size = SMALL_SIZE
    else:
        size = 15
    at = AnchoredText(
    anchor_text, prop=dict(size=size), frameon=False, loc=args.anot_loc)
    at.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2")
    ax.add_artist(at)

def equipotential_surfaces(file: str = None):
    from scipy import optimize as scopt
    def eggleton_radius(q: float) -> float:
        """Approximate effective dimensionless Roche lobe radius according to Eggelton 1983 (ApJ 268, 368)."""
        q23 = q**(2./3)
        return 0.49 * q23 / (0.6 * q23 + np.log(1 + q**(1./3)))

    def calc_phi_n(
        x1_prime: float, 
        x2_prime: float, 
        x3_prime: float, 
        q: float) -> float:
        """
        Dimensionless Roche potential
        
        Parameter
        --------------------
        x1_prime: 
            generalized coordinate 1
        x2_prime:
            idem 2 
        x3_prime:
            idem 3 
        q:
            mass ratio M_2/M_1
            
        Return
        ---------------------
        phi_n: float
            dimensionless roch lobe potential as a function of the coordinate system
        """
        coefficient = 2.0 / (1.0 + q)
        xi    = - np.sin(x2_prime) * np.cos(x3_prime)
        mu    =   np.sin(x2_prime)
        term1 = 1.0 / x1_prime 
        term2 = q * (1 / (1.0 - 2 * x1_prime * xi + x1_prime**2)**0.5 - x1_prime * xi )
        term3 = (q + 1.0) / 2.0 * x1_prime ** 2 * mu**2 + q**2 / (2.0 * (1 + q))
        return coefficient * (term1 + term2 + term3)

    def calc_radius(
        x2_prime:   float,
        x3_prime:   float,
        phi_const:  float, 
        r_endpoint: float,
        q:          float,
        r_array:    np.ndarray = None
    ) -> float:
        """
        Calculate the radius of a point in space knowing the equipotential, theta, and phi
        
        Parameters
        ---------------------------
        x2_prime: 
            generalized coordinate 2
        x3_prime:
            idem 3 
        phi_const:
            potential at a given level surface 
        r_endpoint:
            radius of endpoint 
        q:
            mass ratio
            
        Return
        ---------------------------
        radius: float
            the radius at the given coordinates
        """
        # numerically find the geometric radius for a given potential, theta, phi and q
        potential = lambda x1_prime: calc_phi_n(x1_prime,x2_prime,x3_prime, q) - phi_const
        radius    = scopt.brentq(potential, a=0.9*args.xlims[0], b=r_endpoint)  
        return radius

    def calc_volume(
        q: float, 
        equi_pot: float, 
        r_endpoint: float, 
        r_arr: np.ndarray,
        phi_arr: np.ndarray,
        dV: np.ndarray,
        theta_arr: np.ndarray = None) -> float:
        """ get the volume of the star
        Parameters
        ----------
        q:
            the mass ratio of the binary 
        equi_pot:
            the potential level at the surface of a star
        r_endpoint:
            the radius of the endpoint
        tolerance:
            the tolerance for scipy's root finder
        Returns
        -------
        vol : float
            the volume of the star
        """
        if args.ndim == 2:
            rmask   = r_arr[0] < r_endpoint
            phi_mod = calc_phi_n(r_arr[:,rmask],np.pi/2, phi_arr[:,rmask], q) - equi_pot
            min_idx = find_nearest(phi_mod, 0.0)
            volume_ray  = np.array([dV[phi_idx,:cutoff].sum() for phi_idx,cutoff in enumerate(min_idx)])
            return volume_ray.sum()
        else:
            rmask   = r_arr[0,0] < r_endpoint
            phi_mod = calc_phi_n(r_arr[:,:,rmask],theta_arr[:,:,rmask], phi_arr[:,:,rmask], q) - equi_pot
            min_ids = np.argmin(np.abs(phi_mod), axis=args.ndim-1)
            volume_ray  = np.array([dV[coords[0],coords[1],:cutoff].sum() for coords,cutoff in np.ndenumerate(min_ids)])
            return volume_ray.sum()

    def dphi_dx1(
        x1_prime: float, 
        x2_prime: float, 
        x3_prime: float, 
        q: float) -> float:
        """
        Derivate of dimensionless roche potential
        
        Parameters
        ----------------------
        x1_prime:
            generalized coordinate 1 
        x2_prime:
            idem 2 
        x3_prime:
            idem 3 
        q:
            mass ratio M_2/M_1
        
        Returns
        -----------------------
        del_phi dot n_hat: float
            the dimensionless derivative of the Roche potential along some nhat
        """
        coefficient = 2.0 / (1.0 + q)
        xi    = - np.sin(x2_prime) * np.cos(x3_prime)
        mu    =   np.sin(x2_prime)
        term1 = -1.0 /  x1_prime ** 2 
        term2 = q * (-0.5*(1 - 2.0 * x1_prime * xi + x1_prime**2)**(-1.5)*(-2.0 * xi + 2.0 * x1_prime) - 2.0 * xi)
        term3 = (q + 1.0) * x1_prime * mu**2 
        return coefficient * (term1 + term2 + term3)
        
    def find_lagrange_points_and_phi(a: float, b: float, *args):
        """
        Calculate the Lagrange points and roche potential at those Lagrange points
        """
        rsolution = scopt.brentq(dphi_dx1, a, b, args=(args[0], args[1], args[2]))
        phi = calc_phi_n(rsolution, args[0], args[1], args[2])
        return np.array([rsolution, phi])

    def find_nearest(arr: np.ndarray, val: float) -> int:
        if arr.ndim > 1:
            ids = np.argmin(np.abs(arr - val), axis=args.ndim-1)
            return ids
        else:
            idx = np.argmin(np.abs(arr - val))
            return idx, arr[idx]
        
    data = {}
    with open('../ccn/ccsn_c12/b_cc_45.profile', 'r') as f:
        data['index']        = f.readline()
        data['info']         = f.readline()
        data['val_for_info'] = f.readline()
        empty                = f.readline()
        data['param_idx']    = f.readline()
        param_names               = f.readline().split()
        
        nice_info = data['info'].split()
        nice_vals = data['val_for_info'].split()
        for index, key_name in enumerate(nice_info):
            data[key_name] = nice_vals[index]
            
        for name in param_names:
            data[name] = []
        
        contents = f.readlines()
        for x in contents:
            a = x.split()
            for index, value in enumerate(param_names):
                data[value] += [a[index]]

    r              = 10**np.asanyarray(data['logR'],   dtype=float)[::-1]
    rho_data       = 10**np.asanyarray(data['logRho'], dtype=float)[::-1]
    min_density    = min(enumerate(rho_data), key=lambda x: x[1] if x[1] > 0 else float('inf'))[1]
    cutoff, _      = find_nearest(rho_data, min_density)
    stellar_radius = r[cutoff]
    nr_data        = int(data['num_zones'])
    ndec           = np.log10(r[-1] / r[0])
    zpd_data       = int(nr_data // ndec)
    rinit          = 1e-3 
    rmax           = 10
    zpd            = 1024
    new_ndec       = np.log10(rmax / rinit)
    nr_new         = round(zpd * new_ndec)
    rnew           = np.geomspace(rinit, rmax, nr_new)
    
    qinit       = 0.8 
    m1_init     = float(data['initial_mass'])
    m2_init     = qinit * m1_init
    mtotal      = m2_init + m1_init
    m1_final    = float(data['mass'][0])
    m2_final    = mtotal - m1_final
    q           = m2_final / m1_final
    a           = 1.5 * stellar_radius / eggleton_radius(q)
    theta_min   = 0.0
    theta_max   = np.pi
    npolar      = round(1 + zpd * theta_max / np.log(10.0))
    theta       = np.linspace(theta_min, theta_max, npolar)
    rr, thetta  = np.meshgrid(rnew, theta)
    potential   = calc_phi_n(rr / a, np.pi * 0.5, thetta, q)    
    lagrange_pts = []
    rl      = eggleton_radius(q) * a
    eps     = 1e-6
    x2_pos  = np.pi * 0.5
    x3_pos  = 0.0 
    i       = 0.0
    guesses = [(eps, 1-eps),(eps, 1-eps), (1+eps, 10*rl)]
    
    for aa, bb in guesses:
        if  i >= 1:
            x3_pos = np.pi 
        roots = find_lagrange_points_and_phi(aa, bb, x2_pos, x3_pos, q)
        roots[0] = roots[0] * a
        lagrange_pts.append(roots)
        i += 1
    
    lagrange_pts = np.asanyarray(lagrange_pts)
    lagrange_cor = lagrange_pts[:, 0]
    r_L1         = lagrange_cor.min()
    
    radial_cutoff, _ = find_nearest(rnew, r_L1)
    phi_0            = potential[-1][radial_cutoff]
    stellar_radius   = np.zeros_like(theta)
    for tidx, _ in enumerate(theta):
        pot_ray      = potential[tidx][: radial_cutoff]
        phi_mod      = abs(pot_ray - phi_0)
        equi_idx, _  = find_nearest(phi_mod, 0.0)
        stellar_radius[tidx] = rr[tidx, equi_idx] 
        
    return stellar_radius    
    
def plot_polar_plot(
    fields:     dict,                            # Field dict
    args:       argparse.ArgumentParser,         # argparse object
    mesh:       dict,                            # Mesh dict
    dset:       dict,                            # Sim Params dict
    subplots:   bool = False,                    # If true, don;'t generate own
    fig:        Union[None,plt.figure] = None,   # Figure object
    axs:        Union[None, plt.Axes]  = None    # Axes object
    ) -> Union[None, plt.figure]:
    '''
    Plot the given data on a polar projection plot. 
    '''
    num_fields = len(args.fields)
    is_wedge   = args.nwedge > 0
    rr, tt = mesh['x1'], mesh['x2']
    t2     = - tt[::-1]
    x1max  = dset['x1max']
    x1min  = dset['x1min']
    x2max  = dset['x2max']
    if not subplots:
        if is_wedge:
            nplots = args.nwedge + 1
            fig, axes = plt.subplots(1, nplots, subplot_kw={'projection': 'polar'},
                                figsize=(15, 12), constrained_layout=True)
            ax    = axes[0]
            wedge = axes[1]
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                                figsize=args.fig_dims, constrained_layout=False)
    else:
        if is_wedge:
            ax    = axs[0]
            wedge = axs[1]
        else:
            ax = axs
    
    vmin,vmax = args.cbar[:2]
    ax.grid(False)
    unit_scale = np.ones(num_fields)
    if args.units:
        for idx, field in enumerate(args.fields):
            if field == 'rho' or field == 'D':
                unit_scale[idx] = util.rho_scale.value
            elif field == 'p' or field == 'energy' or field == 'energy_rst':
                unit_scale[idx] = util.edens_scale.value
    
    units = unit_scale if args.units else np.ones(num_fields)
     
    if args.rcmap:
        color_map = (plt.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.get_cmap(args.cmap)
        
    tend = dset['time'] * (util.time_scale if args.units else 1.0)
    # If plotting multiple fields on single polar projection, split the 
    # field projections into their own quadrants
    if num_fields > 1:
        cs  = np.zeros(4, dtype=object)
        var = []
        kwargs = []
        for field in args.fields:
            if field in derived:
                if x2max == np.pi:
                    var += np.array_split(util.prims2var(fields, field), 2)
                else:
                    var.append(util.prims2var(fields, field))
            else:
                if x2max == np.pi:
                    var += np.array_split(fields[field], 2)
                else:
                    var.append(fields[field])
                
        if x2max == np.pi: 
            units  = np.repeat(units, 2)
            
        var    = np.asanyarray(var, dtype=float)
        var    = np.array([units[idx] * var[idx] for idx in range(var.shape[0])], dtype=float)
        
        tchop  = np.array_split(tt, 2)
        trchop = np.array_split(t2, 2)
        rchop  = np.array_split(rr, 2)
        
        quadr = {}
        field1 = args.fields[0]
        field2 = args.fields[1]
        field3 = args.fields[2 % num_fields]
        field4 = args.fields[-1]
        
        
        quadr[field1] = var[0]
        quadr[field2] = var[3 % num_fields if num_fields != 3 else num_fields]
        
        if x2max == np.pi:
            # Handle case of degenerate fields
            if field3 == field4:
                quadr[field3] = {}
                quadr[field4] = {}
                quadr[field3][0] = var[4 % var.shape[0]]
                quadr[field4][1] = var[-1]
            else:
                quadr[field3] =  var[4 % var.shape[0]]
                quadr[field4] = var[-1]


        kwargs = {}
        for idx, key in enumerate(quadr.keys()):
            field = key
            if idx == 0:
                # 'norm': mcolors.PowerNorm(gamma=1.0, vmin=vmin, vmax=vmax)}
                # kwargs[field] = {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)} 
                kwargs[field] =  {'vmin': vmin, 'vmax': vmax} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)} 
            else:
                if field == field3 == field4:
                    ovmin = None if len(args.cbar) == 2 else args.cbar[2]
                    ovmax = None if len(args.cbar) == 2 else args.cbar[3]
                else:
                    ovmin = None if len(args.cbar) == 2 else args.cbar[idx+1]
                    ovmax = None if len(args.cbar) == 2 else args.cbar[idx+2]
                kwargs[field] = {'norm': mcolors.LogNorm(vmin = ovmin, vmax = ovmax)} 
                kwargs[field] =  {'norm': mcolors.PowerNorm(gamma=1.0, vmin=ovmin, vmax=ovmax)} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = ovmin, vmax = ovmax)} 

        if x2max < np.pi:
            cs[0] = ax.pcolormesh(tt[:: 1], rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
            cs[1] = ax.pcolormesh(t2[::-1], rr,  var[1], cmap=args.cmap2, shading='auto', **kwargs[field2])
            
            # If simulation only goes to pi/2, if bipolar flag is set, mirror the fields accross the equator
            if args.bipolar:
                cs[2] = ax.pcolormesh(tt[:: 1] + np.pi/2, rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
                cs[3] = ax.pcolormesh(t2[::-1] + np.pi/2, rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
        else:
            if num_fields == 2:
                cs[0] = ax.pcolormesh(tt[:: 1], rr,  np.vstack((var[0],var[1])), cmap=color_map, shading='auto', **kwargs[field1])
                cs[1] = ax.pcolormesh(t2[::-1], rr,  np.vstack((var[2],var[3])), cmap=args.cmap2, shading='auto', **kwargs[field2])
            else:
                cs[0] = ax.pcolormesh(tchop[0], rchop[0],  quadr[field1], cmap=color_map, shading='auto', **kwargs[field1])
                cs[1] = ax.pcolormesh(tchop[1], rchop[0],  quadr[field2], cmap=color_map, shading='auto', **kwargs[field2])
                
                if field3 == field4:
                    cs[2] = ax.pcolormesh(trchop[1][::-1], rchop[0],  quadr[field3][0], cmap=color_map, shading='auto', **kwargs[field3])
                    cs[3] = ax.pcolormesh(trchop[0][::-1], rchop[0],  quadr[field4][1], cmap=color_map, shading='auto', **kwargs[field4])
                else:
                    cs[2] = ax.pcolormesh(trchop[1][::-1], rchop[0],  quadr[field3], cmap=color_map, shading='auto', **kwargs[field3])
                    cs[3] = ax.pcolormesh(trchop[0][::-1], rchop[0],  quadr[field4], cmap=color_map, shading='auto', **kwargs[field4])
            
    else:
        if args.log:
            kwargs = {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)}
        else:
            kwargs = {'vmin': vmin, 'vmax': vmax}
            
        cs = np.zeros(len(args.fields), dtype=object)
        
        if args.fields[0] in derived:
            var = units * util.prims2var(fields, args.fields[0])
        else:
            var = units * fields[args.fields[0]]
        
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto',
                              linewidth=0, rasterized=True, **kwargs)
        cs[0] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, 
                              linewidth=0,rasterized=True, shading='auto', **kwargs)
        
        if args.bipolar:
            cs[0] = ax.pcolormesh(tt[:: 1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
            cs[0] = ax.pcolormesh(t2[::-1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
    
    if args.pictorial: 
        ax.set_position([0.1, -0.15, 0.8, 1.30])
    
    #========================================================
    #               ORANGE DASHED CURVE
    #========================================================
    # angs    = np.linspace(x2min, x2max, 1000)
    # eps     = 0.2
    # a       = 0.50 * (1 - eps)**(-1/3)
    # b       = 0.50 * (1 - eps)**(2/3)
    # radius  = lambda theta: a*b/((a*np.cos(theta))**2 + (b*np.sin(theta))**2)**0.5
    # r_theta = radius(angs)
    # r_theta = equipotential_surfaces()
    
    # ax.plot(np.radians(np.linspace(0, 180, tt.shape[0])),  1.4 * r_theta, linewidth=1, linestyle='--', color='white')
    # ax.plot(-np.radians(np.linspace(0, 180, tt.shape[0])), 1.4 * r_theta, linewidth=1, linestyle='--', color='white')
    
    if not args.pictorial:
        if x2max < np.pi:
            ymax = int( np.floor(x2max * 180/np.pi) )
            if not args.bipolar:                                                                                                                                                                                   
                ax.set_thetamin(-ymax)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                ax.set_thetamax(ymax)
                ax.set_position( [0.05, -0.40, 0.9, 2])
                # ax.set_position( [0.1, -0.18, 0.9, 1.43])
            else:
                ax.set_position( [0.1, -0.45, 0.9, 2])
                #ax.set_position( [0.1, -0.18, 0.9, 1.50])
            if num_fields > 1:
                cbar_orientation = args.cbar_orient
                if cbar_orientation == 'vertical':
                    ycoord  = [0.1, 0.1]
                    xcoord  = [0.88, 0.04]
                    cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                else:
                    ycoord  = [0.17, 0.17]
                    xcoord  = [0.51, 0.06]
                    cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.43, 0.05]) for i in range(num_fields)]
            else:
                cbar_orientation = 'horizontal'
                if cbar_orientation == 'horizontal':
                    cbaxes  = fig.add_axes([0.15, 0.15, 0.70, 0.05]) 
        else:  
            if not args.no_cbar:         
                cbar_orientation = args.cbar_orient
                # ax.set_position([0.1, -0.18, 0.7, 1.3])
                if num_fields > 1:
                    if num_fields == 2:
                        if cbar_orientation == 'vertical':
                            ycoord  = [0.1, 0.08] if x2max < np.pi else [0.15, 0.15]
                            xcoord  = [0.1, 0.85] if x2max < np.pi else [0.93, 0.08]
                            cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.65]) for i in range(num_fields)]
                        else:
                            if is_wedge:
                                ycoord  = [0.2, 0.20] if x2max < np.pi else [0.15, 0.15]
                                xcoord  = [0.1, 0.50] if x2max < np.pi else [0.52, 0.04]
                            else:
                                ycoord  = [0.2, 0.20] if x2max < np.pi else [0.10, 0.10]
                                xcoord  = [0.1, 0.50] if x2max < np.pi else [0.51, 0.20]
                            cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.45, 0.04]) for i in range(num_fields)]
                    if num_fields == 3:
                        ycoord  = [0.1, 0.5, 0.1]
                        xcoord  = [0.07, 0.85, 0.85]
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.4]) for i in range(1, num_fields)]
                        cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                    if num_fields == 4:
                        ycoord  = [0.5, 0.1, 0.5, 0.1]
                        xcoord  = [0.85, 0.85, 0.07, 0.07]
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
                else:
                    if not is_wedge:
                        pass
                        #plt.tight_layout()
                        
                    if cbar_orientation == 'vertical':
                        cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.85])
                    else:
                        cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.90])
        if args.log:
            if not args.no_cbar:
                if num_fields > 1:
                    fmt  = [None if field in lin_fields else tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True) for field in args.fields]
                    cbar = [fig.colorbar(cs[i], orientation=cbar_orientation, cax=cbaxes[i], format=fmt[i]) for i in range(num_fields)]
                    for cb in cbar:
                        cb.outline.set_visible(False)                                 
                else:
                    logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                    cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes, format=logfmt)
        else:
            if not args.no_cbar:
                if num_fields > 1:
                    cbar = [fig.colorbar(cs[i], orientation=cbar_orientation, cax=cbaxes[i]) for i in range(num_fields)]
                else:
                    cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes)
        ax.yaxis.grid(True, alpha=0.05)
        ax.xaxis.grid(True, alpha=0.05)
    
    if is_wedge:
        wedge_min = args.wedge_lims[0]
        wedge_max = args.wedge_lims[1]
        ang_min   = args.wedge_lims[2]
        ang_max   = args.wedge_lims[3]
        
        # Draw the wedge cutout on the main plot
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=1, color='orange')
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='orange')
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='orange')
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=1, color='orange')
   
            
        if args.nwedge == 2:
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=1, color='orange')
            ax.plot(np.radians(-np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='orange')
            ax.plot(np.radians(-np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='orange')
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=1, color='orange')
            
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='both', labelsize=15)
    rlabels = ax.get_ymajorticklabels()
    if not args.pictorial:
        for label in rlabels:
            label.set_color('white')
    else:
        ax.axes.yaxis.set_ticklabels([])
        
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_rmax(x1max) if args.xmax == 0.0 else ax.set_rmax(args.xmax)
    ax.set_rmin(x1min)
    
    field_str = util.get_field_str(args)
    if is_wedge:
        if num_fields == 1:
            wedge.set_position([0.5, -0.5, 0.3, 2])
            ax.set_position([0.05, -0.5, 0.46, 2])
        else:
            if args.nwedge == 1:
                ax.set_position([0.15, -0.5, 0.46, 2])
                wedge.set_position([0.58, -0.5, 0.3, 2])
            elif args.nwedge == 2:
                ax.set_position([0.28, -0.5, 0.45, 2.0])
                wedge.set_position([0.70, -0.5, 0.3, 2])
                axes[2].set_position([0.01, -0.5, 0.3, 2])
            
        if len(args.fields) > 1:
            if len(args.cbar2) == 4:
                vmin2, vmax2, vmin3, vmax3 = args.cbar2
            else:
                vmin2, vmax2 = args.cbar2
                vmin3, vmax3 = None, None
            kwargs = {}
            for idx, key in enumerate(quadr.keys()):
                field = args.fields[idx % num_fields]
                if idx == 0:
                    kwargs[field] =  {'vmin': vmin2, 'vmax': vmax2} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin2, vmax = vmax2)} 
                elif idx == 1:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                    kwargs[field] =  {'vmin': vmin3, 'vmax': vmax3} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin3, vmax = vmax3)} 
                else:
                    continue

            wedge.pcolormesh(tt[:: 1], rr,  np.vstack((var[0],var[1])), cmap=color_map, shading='auto', **kwargs[field1])
            if args.nwedge == 2:
                axes[2].pcolormesh(t2[::-1], rr,  np.vstack((var[2],var[3])), cmap=args.cmap2, shading='auto', **kwargs[field2])
            
        else:
            vmin2, vmax2 = args.cbar2
            if args.log:
                kwargs = {'norm': mcolors.LogNorm(vmin = vmin2, vmax = vmax2)}
            else:
                kwargs = {'vmin': vmin2, 'vmax': vmax2}
            w1 = wedge.pcolormesh(tt, rr, var, cmap=color_map, shading='nearest', **kwargs)
        
        wedge.set_theta_zero_location('N')
        wedge.set_theta_direction(-1)
        wedge.yaxis.grid(False)
        wedge.xaxis.grid(False)
        wedge.tick_params(axis='both', labelsize=17)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('white')
            
        wedge.set_ylim([wedge_min, wedge_max])
        wedge.set_rorigin(-wedge_min/4)
        wedge.set_thetamin(ang_min)
        wedge.set_thetamax(ang_max)
        wedge.yaxis.set_minor_locator(plt.MaxNLocator(1))
        wedge.yaxis.set_major_locator(plt.MaxNLocator(2))
        wedge.set_aspect('equal')
        wedge.axes.xaxis.set_ticklabels([])
        wedge.axes.yaxis.set_ticklabels([])
        
        if args.nwedge > 1:
            # force the rlabels to be outside plot area
            axes[2].tick_params(axis="y",direction="out", pad=-25)
            axes[2].set_theta_zero_location('N')
            axes[2].set_theta_direction(-1)
            axes[2].yaxis.grid(False)
            axes[2].xaxis.grid(False)
            axes[2].tick_params(axis='both', labelsize=17)               
            axes[2].axes.xaxis.set_ticklabels([])
            axes[2].set_ylim([wedge_min, wedge_max])
            axes[2].set_rorigin(-wedge_min/4)
            axes[2].set_thetamin(-ang_min)
            axes[2].set_thetamax(-ang_max)
            axes[2].yaxis.set_minor_locator(plt.MaxNLocator(1))
            axes[2].yaxis.set_major_locator(plt.MaxNLocator(2))
            axes[2].set_aspect('equal')
            axes[2].axes.yaxis.set_ticklabels([])
            
    if not args.pictorial:
        if not args.no_cbar:
            set_label = ax.set_ylabel if args.cbar_orient == 'vertical' else ax.set_xlabel
            fsize = 30 if not args.print else DEFAULT_SIZE
            if args.log:
                if x2max == np.pi:
                    if num_fields > 1:
                        for i in range(num_fields):
                            if args.fields[i] in lin_fields:
                                # labelpad = -35 if you want the labels to be on other side
                                cbar[i].set_label(r'{}'.format(field_str[i]), labelpad = -35 )
                                # xticks = [0.10, 0.20, 0.35, 0.50]
                                # cbar[i].set_ticks(xticks)
                                # cbar[i].set_ticklabels(['%.2f' % x for x in xticks])
                                # loc = tkr.MultipleLocator(base=0.12) # this locator puts ticks at regular intervals
                                # cbaxes[i].xaxis.set_major_locator(loc)
                                cbaxes[i].yaxis.set_ticks_position('left')
                            else:
                                cbar[i].set_label(r'$\log$ {}'.format(field_str[i]))
                                # cbaxes[i].xaxis.set_major_locator(plt.MaxNLocator(4))
                    else:
                        cbar.set_label(r'$\log$ {}'.format(field_str))
                else:
                    if num_fields > 1:
                        for i in range(num_fields):
                            if args.fields[i] in lin_fields:
                                cbar[i].set_label(r'{}'.format(field_str[i]))
                            else:
                                cbar[i].set_label(r'$\log$ {}'.format(field_str[i]))
                    else:
                        cbar.set_label(r'$\log$ {}'.format(field_str))
            else:
                if x2max >= np.pi:
                    if num_fields > 1:
                        for i in range(num_fields):
                            cbar[i].set_label(r'{}'.format(field_str[i]))
                    else:
                        cbar.set_label(f'{field_str}')
                else:
                    cbar.set_label(r'{}'.format(field_str))
        fsize = 25 if not args.print else DEFAULT_SIZE
        if args.setup:
            fig.suptitle('{} at t = {:.2f}'.format(args.setup, tend), y=1.03)
        else:
            fig.suptitle('(c) t = {:d} s'.format(int(tend.value if args.units else tend)), y=0.95)

def plot_cartesian_plot(
    fields: dict, 
    args: argparse.ArgumentParser, 
    mesh: dict, 
    dset: dict,
    mirror: bool = False) -> None:
    
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    x1, x2 = mesh['x1'], mesh['x2']
    vmin,vmax = args.cbar

    if args.log:
        kwargs = {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'norm': mcolors.PowerNorm(gamma=args.power, vmin=vmin, vmax=vmax)}
        # kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
    
    if args.fields[0] in derived:
        var = util.prims2var(fields, args.fields[0])
    else:
        var = fields[args.fields[0]]
    tend = dset['time']
    if args.oned_slice is None:
        c = ax.pcolormesh(x1, x2, var, cmap=color_map, shading='auto', **kwargs)
        if mirror:
            ax.pcolormesh(-x1, x2, var,  cmap=color_map, shading='auto', **kwargs)
        divider = make_axes_locatable(ax)
        cbaxes  = divider.append_axes('right', size='5%', pad=0.05)

        if not args.no_cbar:
            if args.log:
                logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                cbar = fig.colorbar(c, orientation='vertical', cax=cbaxes, format=logfmt)
            else:
                cbar = fig.colorbar(c, orientation='vertical', cax=cbaxes)

        ax.tick_params(axis='both', labelsize=10)
        # Change the format of the field
        field_str = util.get_field_str(args)
        if args.log:
            cbar.ax.set_ylabel(r'$\log$[{}]'.format(field_str))
        else:
            cbar.ax.set_ylabel(r'{}'.format(field_str))
        
        ax.set_aspect('equal')
    else:
        field_labels = util.get_field_str(args)
        labels = field_labels if isinstance(field_labels, list) else [None]
        scale_cycle = cycle(args.scale_downs)
        one_axis_var = x1[args.oned_slice] if args.oned_proj == 2 else x2[:,args.oned_slice]
        for idx, field in enumerate(args.fields):
            if field in derived:
                var = util.prims2var(fields, field)
            else:
                var = fields[field]
                
            scale = next(scale_cycle)
            var /= scale
            if scale != 1:
                if scale.is_integer():
                    scale = int(scale)
                labels[idx] = rf'{labels[idx]} / {scale}'
            
            proj_var = var[args.oned_slice] if args.oned_proj == 2 else var[:, args.oned_slice]
            ax.plot(one_axis_var, proj_var, label=labels[idx])
        
        if isinstance(field_labels, list):
            ax.legend()
        else:
            ax.set_ylabel(field_labels)
        ax.set_xlim(one_axis_var[0], one_axis_var[-1])
        
    if args.setup:
        ax.set_title('{} at t = {:.2f}'.format(args.setup, tend))
    
def plot_1d_curve(
    fields: dict, 
    args:       argparse.ArgumentParser, 
    mesh:       dict, 
    dset:       dict,
    overplot:   bool=False,
    ax:         bool=None, 
    case:       int =0) -> None:
    
    num_fields = len(args.fields)
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.files)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['x1'], mesh['x2']
    theta    = theta
    tidx,_   = util.find_nearest(theta, np.deg2rad(args.viewing))
    x1max        = dset['x1max']
    x1min        = dset['x1min']
    
    vmin,vmax = args.cbar[:2]
    var = [field for field in args.fields] if num_fields > 1 else args.fields[0]
    
    # Change the format of the field
    field_labels = util.get_field_str(args)
    tend = dset['time'] * util.time_scale
    if args.mass:
        dV          = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
        mass        = dV * fields['W'] * fields['rho']
        # linestyle = '-.'
        if args.labels is None:
            ax.loglog(r, mass[tidx]/ np.max(mass[tidx]), label = 'mass', linestyle='-.', color=colors[case])
            ax.loglog(r, fields['p'][tidx] / np.max(fields['p'][tidx]), label = 'pressure', color=colors[case])
        else:
            ax.loglog(r, mass[tidx]/ np.max(mass[tidx]), label = f'{args.labels[case]} mass', linestyle='-.', color=colors[case])
            ax.loglog(r, fields['p'][tidx] / np.max(fields['p'][tidx]), label = f'{args.labels[case]} pressure', color=colors[case])
        ax.legend()
        ax.axvline(0.65, linestyle='--', color='red')
        ax.axvline(1.00, linestyle='--', color='blue')
    else:
        for idx, field in enumerate(args.fields):
            if field in derived:
                var = util.prims2var(fields, field)
            else:
                var = fields[field]
            
            if args.units:
                if field == 'p' or field == 'energy':
                    var *= util.e_scale.value
                elif field == 'D' or field == 'rho':
                    var *= util.rho_scale.value
                    
            if args.labels:
                label = r'$\rm {}, t={:.1f}$'.format(args.labels[case], tend)
            else:
                label = r'$t-{:.1f}$'.format(tend)
                
            if len(args.fields) > 1:
                label = field_labels[idx] + ', ' + label
                
            
            ax.plot(r[0], var[tidx], label=label)
            if args.log:
                ax.set_xscale('log')
                if field not in lin_fields:
                    ax.set_yscale('log')
            if args.extra_files is not None:
                for one_file in args.extra_files:
                    oned_var = util.read_file(args, one_file, ndim=1)[0]
                    ax.loglog(r, oned_var[field], label='Spherical')
    
    
    ax.set_xlim(x1min, x1max)
    ax.set_xlabel(r'$r/R_0$')
    ax.tick_params(axis='both', labelsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if args.legend:
        ax.legend()
    if len(args.fields) == 1:
        ax.set_ylabel(r'{}'.format(field_labels))
        
    
    if args.setup:
        ax.set_title(r'$\theta = {:.2f}^{{\circ}}$ time: {:.3f}'.format(mesh['x2'][tidx] * 180 / np.pi, tend))
    if not overplot:
        return fig
    # fig.suptitle(r'{} at $\theta = {:.2f}$ deg, t = {:.2f} s'.format(args.setup,theta[args.tidx], tend), y=0.95)
    
def plot_per_theta(
    fields:    dict, 
    args:      argparse.ArgumentParser, 
    mesh:      dict , 
    dset:      dict, 
    overplot:  bool=False, 
    ax:        bool=None, 
    case:      int =0) -> None:
    print('plotting vs theta...')
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.90, len(args.files)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    theta = mesh['x2']
    

    for field in args.fields:
        fields = fields if args.extra_files is None else util.read_1d_file(args.extra_files[0])[0]
        if field in derived:
            var = util.prims2var(fields, field)
        else:
            var = fields[field].copy()
        if args.units:
            if field == 'p' or field == 'energy':
                var *= util.e_scale.value
            elif field == 'D' or field == 'rho':
                var *= util.rho_scale.value

    theta = theta * 180 / np.pi
    
    if var.ndim > 1:
        if not args.tau_s:
            pts = np.max(var, axis=1)
        else:
            beta  = calc_beta(fields)
            pts   = 1.0 / np.max(beta, axis=1)
            pts   = running_mean(pts, 50)
            theta = running_mean(theta, 50)
    else:
        if not tau_s:
            pts   = np.max(var)
            pts   = running_mean(pts, 50)
            theta = running_mean(theta, 50)
        else:
            beta = calc_beta(fields)
            pts = 1.0 / np.max(beta)
    
    label = args.labels[case] if args.labels is not None else None
    
    if args.cmap != 'grayscale':
        ax.plot(theta, pts,label=label, color=colors[case])
    else:
        ax.plot(theta, pts,label=label)
    
    if args.log:
        ax.set_yscale('log')
    
    inds   = np.argmax(fields['gamma_beta'], axis=1)
    vw     = 1e8 * u.cm/u.s
    mdot   = (1e-6 * u.M_sun / u.yr).to(u.g/u.s)
    a_star = mdot / (4.0 * np.pi * vw)
    x      = 0.01
    kappa  = 0.2 * (1.0 + x) * u.cm**2 / u.g
    
    if args.tau_s:
        tau = np.zeros_like(mesh['x2'])
        for tidx, t in enumerate(mesh['x2']):
            ridx      = np.argmax(fields['gamma_beta'][tidx])
            tau[tidx] = kappa * a_star / (mesh['x1'][tidx,ridx] * R_0)
            
        mean_tau = running_mean(tau, 50)
        ax.plot(theta, 32*mean_tau, linestyle='--', label=label+"(x32)")
    if not args.tau_s:
        ylabel = util.get_field_str(args)
        if args.units:
            for idx, char in enumerate(ylabel):
                if char == "[":
                    unit_idx = idx
            
            ylabel = ylabel[:unit_idx-1]+r"$_{\rm max}$" + ylabel[unit_idx:]
            
        else:
            ylabel = ylabel + r"$_{\rm max}$"
    else:
        ylabel = r'$\tau_s$'

        
    # Aesthetic 
    if case == 0:
        if args.anot_loc is not None:
            dV = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
            etot = np.sum(util.prims2var(fields, "energy") * dV * util.e_scale.value)
            place_annotation(args, fields, ax, etot)
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$\theta [\rm deg]$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(theta[0], theta[-1])
    
def plot_dec_rad(
    fields:    dict, 
    args:      argparse.ArgumentParser, 
    mesh:      dict , 
    dset:      dict, 
    overplot:  bool=False, 
    ax:        bool=None, 
    case:      int =0) -> None:
    print('plotting deceleration radius...')
    
    file_num = len(args.files)
    
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    theta = mesh['x2']
    mdots = np.logspace(np.log10(1e-6), np.log10(24e-4),128)

    colors = plt.cm.viridis(np.linspace(0.1, 0.90, file_num if file_num > 1 else len(mdots)))
    
    tvert   = util.calc_theta_verticies(mesh['x2'])
    tcent   = 0.5 * (tvert[1:,0] + tvert[:-1,0])
    dtheta  = tvert[1:,0] - tvert[:-1,0]
    domega  = 2.0 * np.pi * np.sin(tcent) * dtheta 
    vw      = 1e8 * u.cm / u.s
    mdots   = (mdots * u.M_sun / u.yr).to(u.g/u.s)
    factor  = np.array([0.75 * 4.0 * np.pi * vw.value / mdot.value for mdot in mdots])
    gb      = fields['gamma_beta']
    W       = util.calc_lorentz_gamma(fields)
    dV      = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
    mass    = W * dV * fields['rho'] * util.m.value

    theta = theta * 180 / np.pi
    
    if file_num > 1:
        pts   = np.zeros(shape=(1, theta.size))
    else:
        pts   = np.zeros(shape=(len(mdots), theta.size))
    
    label = args.labels[case] if args.labels is not None else None
    
    window     = 100
    mean_theta = running_mean(theta, window)
    
    cutoffs = np.linspace(args.cutoffs[0], args.cutoffs[1], 128)
    for cidx, cutoff in enumerate(cutoffs):
        if cidx == 0:
            label = r"$\Gamma \beta = {}$".format(cutoff)
        elif cidx + 1 == len(cutoffs):
            label = r"$\Gamma \beta = {}$".format(cutoff)
        else:
            label = None
        mass[gb < cutoff] = 0
        W = (1 + cutoff**2)**0.5
        for midx, val in enumerate(mdots):
            if midx > 0:
                break
            
            for tidx, angle in enumerate(theta):
                ridx_max         = np.argmax(gb[tidx])
                r                = np.sum(mass[tidx]) / domega[tidx] / W * factor[midx]
                pts[midx][tidx]  = r
            
            mean_r     = running_mean(pts[midx], window)
            if args.cmap != 'grayscale':
                ax.plot(mean_theta, mean_r, label=label, color = colors[case if file_num > 1 else cidx])
            else:
                ax.plot(mean_theta, mean_r, label=label)
    
    #=============
    # Aesthetic 
    #=============
    if args.log:
        ax.set_yscale('log')
    
    ylabel = r'$r_{\rm dec} [\rm cm]$'
    
    axins = inset_axes(ax, width='20%', height='4%', loc='upper left', borderpad=2.5)

    # setup the colorbar
    scalarmappaple = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    scalarmappaple.set_array(cutoffs)
    
    cbar = fig.colorbar(scalarmappaple, cax=axins, orientation='horizontal', ticks=[cutoffs[0], cutoffs[-1]])
    cbar.ax.set_xticklabels([r'$\Gamma\beta = {}$'.format(cutoffs[0]), r'$\Gamma\beta = {}$'.format(cutoffs[-1])])
    axins.xaxis.set_ticks_position('top')
    if case == 0:
        if args.anot_loc is not None:
            dV = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
            etot = np.sum(util.prims2var(fields, "energy") * dV * util.e_scale.value)
            place_annotation(args, fields, ax, etot)
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$\theta [\rm deg]$')
    ax.set_ylabel(ylabel)
    
    if not args.xlims:
        ax.set_xlim(mean_theta[0], mean_theta[-1])
    else:
        ax.set_xlim(*args.xlims)
        
    if args.ylims is not None:
        ax.set_ylim(*args.ylims)
    
def plot_hist(
    fields:      dict, 
    args:        argparse.ArgumentParser, 
    mesh:        dict, 
    dset:        dict, 
    overplot:    bool=False, 
    ax:          int =None, 
    ax_num:      int =0, 
    case:        int =0, 
    ax_col:      int =0) -> None:
    print('Computing histogram...')
    
    # Check if subplots are split amonst the file inputs. If so, roll the colors
    # to reset when on a different axes object
    color_len = args.sub_split[ax_num] if args.sub_split is not None else len(args.files)
    if args.cmap == 'grayscale':
        colors = plt.cm.gray(np.linspace(0.05, 0.75, color_len+1))
    else:
        colors = plt.cm.viridis(np.linspace(0.10, 0.75, color_len+1))

    lw = 1.0 if case < 2 else 2.5
    def calc_1d_hist(fields: dict, mesh: dict):
        dV_1d    = util.calc_cell_volume1D(mesh['x1'])
        
        if args.kinetic:
            W        = util.calc_lorentz_gamma(fields)
            mass     = dV_1d * fields['rho'] * W
            var      = (W - 1.0) * mass * util.e_scale.value # Kinetic Energy in [erg]
        elif args.mass:
            W        = util.calc_lorentz_gamma(fields)
            mass     = dV_1d * fields['rho'] * W
            var      = mass * util.m.value            # Mass in [g]
        elif args.enthalpy:
            h   = util.calc_enthalpy(fields)
            var = (h - 1.0) * util.e_scale.value      # Specific Enthalpy in [erg]
        elif args.dm_du:
            u   = fields['gamma_beta']
            var = u* fields['rho'] * dV_1d   / (1 + u**2)**0.5 
        else:
            edens_1d  = util.prims2var(fields, 'energy')
            var       = edens_1d * dV_1d * util.e_scale.value          # Total Energy in [erg]
            
        u1d       = fields['gamma_beta']
        gbs_1d    = np.logspace(np.log10(1.e-3), np.log10(u1d.max()), 128)
        var       = np.asanyarray([var[np.where(u1d > gb)].sum() for gb in gbs_1d])
        
        label = r'$\varepsilon = 0$'
        if args.labels is not None:
            if len(args.labels) == len(args.files) and not args.sub_split:
                etot         = np.sum(util.prims2var(fields, "energy") * dV_1d * util.e_scale.value)
                order_of_mag = np.floor(np.log10(etot))
                scale        = int(etot / 1e51)
                front_factor = int(etot / 10**order_of_mag)
                if front_factor != 1 or scale != 1:
                    if scale > 1e3:
                        scale = 1000
                    label = r"${}E_{{51}}$".format(scale) + f"({label})"     
                else:
                    label = r"$E_{51}$" + f"({label})" 
                
        if args.norm:
            var /= var.max()
            util.fill_below_intersec(gbs_1d, var, 1e-6, colors[0])
            
        ax.hist(gbs_1d, bins=gbs_1d, weights=var, alpha=0.8, label= label,
                color=colors[0], histtype='step', linewidth=lw)
        
        
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)
    
    tend        = dset['time'] * util.time_scale
    theta       = mesh['x2']
    r           = mesh['x1']
    dV          = util.calc_cell_volume2D(r, theta)
    
    if args.kinetic:
        W    = util.calc_lorentz_gamma(fields)
        mass = dV * fields['rho'] * W
        var  = (W - 1.0) * mass * util.e_scale.value
    elif args.enthalpy:
        h   = calc_enthalpy(fields)
        var = (h - 1.0) *  dV * util.e_scale.value
    elif args.mass:
        W   = util.calc_lorentz_gamma(fields)
        var = dV * fields['rho'] * W * util.m.value
    elif args.dm_du:
        u   = fields['gamma_beta']
        var = u * fields['rho'] * dV / (1 + u**2)**0.5 * util.m.value
    else:
        var = util.prims2var(fields, "energy") * dV * util.e_scale.value

    # Create 4-Velocity bins as well as the Y-value bins directly
    u         = fields['gamma_beta']
    gbs       = np.logspace(np.log10(1.e-3), np.log10(u.max()), 128)
    var       = np.asanyarray([var[u > gb].sum() for gb in gbs]) 
    
    # if case == 0:
    #     oned_field, setups1d, mesh1d   = util.read_1d_file(args.extra_files[0])
    #     calc_1d_hist(oned_field, mesh1d)
    # if case == 2:
    #     oned_field, setups1d, mesh1d   = util.read_1d_file(args.extra_files[1])
    #     calc_1d_hist(oned_field, mesh1d)
        
    if ax_col == 0:     
        if args.anot_loc is not None:
            etot = np.sum(util.prims2var(fields, "energy") * dV * util.e_scale.value)
            place_annotation(args, fields, ax, etot)
        
        #1D Comparison 
        if args.extra_files is not None:
            if args.sub_split is None:
                for file in args.extra_files:
                    oned_field, oned_setup, oned_mesh = util.read_1d_file(file)
                    calc_1d_hist(oned_field, oned_mesh)
            else:
                oned_field, one_setup, one_mesh = util.read_1d_file(args.extra_files[ax_num])
                calc_1d_hist(oned_field)

    if args.norm:
        var /= var.max()

    if args.labels is not None:
        label = '%s'%(args.labels[case])
            
        if len(args.labels) == len(args.files) and not args.sub_split:
            etot         = np.sum(util.prims2var(fields, "energy") * dV * util.e_scale.value)
            order_of_mag = np.floor(np.log10(etot))
            scale        = int(etot / 1e51)
            front_factor = int(etot / 10**order_of_mag)
            if front_factor != 1 or scale != 1:
                label = r"${}E_{{51}}$".format(scale) + "(%s)"%(label)     
            else:
                label = r"$E_{51}$" + "(%s)"%(label)  
    else:
        label = None
    
    if case < 2:
        lw = 1.0
    elif case < 4:
        lw = 2.0
    else:
        lw = 3.5
    c = colors[case + 1]
    if case % 2 == 0:
        c = colors[1]
    else:
        c = colors[-1]
    ax.hist(gbs, bins=gbs, weights=var, label=label, histtype='step', 
            rwidth=1.0, linewidth=lw, color=c, alpha=0.9)
    
    if args.fill_scale is not None:
        util.fill_below_intersec(gbs, var, args.fill_scale*var.max(), colors[case])
                    
            
    ax.set_xscale('log')
    ax.set_yscale('log')

    if args.xlims is None:
        ax.set_xlim(1e-3, 1e2)
    else:
        ax.set_xlim(args.xlims[0], args.xlims[1])
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    
    if args.ylims is None:
        if args.mass:
            ax.set_ylim(1e-3*var.max(), 10.0*var.max())
        else:
            ax.set_ylim(1e-9*var.max(), 10.0*var.max())
    else:
        ax.set_ylim(args.ylims[0],args.ylims[1])
        
    if args.sub_split is None:
        ax.set_xlabel(r'$\Gamma\beta $')
        if args.kinetic:
            ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$')
        elif args.enthalpy:
            ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$')
        elif args.dm_du:
            ax.set_ylabel(r'$dM/d\Gamma\beta ( > \Gamma \beta) \ [\rm{g}]$')
        elif args.mass:
            ax.set_ylabel(r'$M(> \Gamma \beta) \ [\rm{g}]$')
        else:
            ax.set_ylabel(r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$')
            
        ax.tick_params('both', labelsize=15)
    else:        
        ax.tick_params('x', labelsize=15)
        ax.tick_params('y', labelsize=10)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if args.setup:
        ax.set_title(r'{}, t ={:.2f} s'.format(args.setup, tend))
    
    if overplot and args.sub_split is None:
        if args.labels is not None:
            fsize = 15 if not args.print else DEFAULT_SIZE
            ax.legend(loc=args.legend_loc, fancybox=True, framealpha=0.1)
        

def plot_dx_domega(
    fields:        dict, 
    args:          argparse.ArgumentParser, 
    mesh:          dict, 
    dset:          dict, 
    overplot:      bool=False, 
    subplot:       bool=False, 
    ax:            Union[None,plt.Axes]=None, 
    case:          int=0, 
    ax_col:        int=0,
    ax_num:        int=0) -> None:
    
    energy_and_mass = False
    if not overplot:
        if 0 in args.cutoffs:
            if args.dm_domega and args.de_domega:
                energy_and_mass = True
                fig, (ax0, ax1, ax2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 3, 3]}, 
                                        figsize=(5,12), sharex=True)
            else:
                fig, (ax0, ax) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, 
                                        figsize=(10,9), sharex=True)
                
        else:
            fig = plt.figure(figsize=[10, 9])
            ax  = fig.add_subplot(1, 1, 1)
        
    def calc_dec_rad(gb: float, M_ej: float):
        vw   = 1e8 * u.cm / u.s 
        mdot = (1e-6 * u.M_sun / u.yr).to(u.g/u.s)
        a    = vw / mdot 
        r    = M_ej * a.value
        return r 
        
    def calc_1d_dx_domega(ofield: dict) -> None:
        edens_1d = util.prims2var(ofield, 'energy')
        dV_1d    = util.calc_cell_volume1D(ofield['x1'])
        mass     = dV_1d * ofield['rho'] * ofield['W']
        e_k      = (ofield['W'] - 1.0) * mass * util.e_scale.value
        etotal_1d = edens_1d * dV_1d * util.e_scale.value
        
        if args.kinetic:
            var = e_k
        elif args.dm_domega:
            var = mass * util.m.value
        else:
            var = etotal_1d
        
        for cutoff in args.cutoffs:
            total_var = sum(var[ofield['gamma_beta'] > cutoff])
            print(f"1D var sum with GB > {cutoff}: {total_var:.2e}")
            ax.axhline(total_var, linestyle='--', color='black', label='$\varepsilon = 0$')
                
    def de_domega(var, gamma_beta, gamma_beta_cut, tz, domega, bin_edges):
        var = var.copy()
        var[gamma_beta < gamma_beta_cut] = 0.0
        de = np.hist(tz, weights=energy, bins=theta_bin_edges)
        dw = np.hist(tz, weights=domega, bins=theta_bin_edges)
        return de / dwplot_dx_d
    
    col       = case % len(args.sub_split) if args.sub_split is not None else case
    color_len = len(args.sub_split) if args.sub_split is not None else len(args.files)
    colors    = plt.cm.viridis(np.linspace(0.1, 0.80, color_len if color_len > 1 else len(args.cutoffs)))
    coloriter = cycle(colors)
    
    tend        = dset['time'] * util.time_scale
    theta       = mesh['x2']
    tv          = util.calc_theta_verticies(theta)
    r           = mesh['x1']
    dV          = util.calc_cell_volume2D(r, theta)
    
    if ax_col == 0:
        if args.anot_loc is not None:
            etot = np.sum(util.prims2var(fields, "energy") * dV * util.e_scale.value)
            if not energy_and_mass:
                place_annotation(args, fields, ax, etot)
            else:
                place_annotation(args, fields, ax1, etot)
                place_annotation(args, fields, ax2, etot)
            
        #1D Comparison 
        if args.extra_files is not None:
            if args.sub_split is None:
                for file in args.extra_files:
                    oned_field, one_setup, one_mesh  = util.read_1d_file(file)
                    calc_1d_dx_domega(oned_field)
            else:
                oned_field, one_setup, one_mesh = util.read_1d_file(args.extra_files[ax_num%len(args.extra_files)])
                calc_1d_dx_domega(oned_field)  
    
    if energy_and_mass:
        if args.kinetic:
            W    = util.calc_lorentz_gamma(fields)
            mass = dV * fields['rho'] * W
            ek   = (W - 1.0) * mass * util.e_scale.value
    elif args.de_domega:
        if args.kinetic:
            W    = util.calc_lorentz_gamma(fields)
            mass = dV * fields['rho'] * W
            var  = (W - 1.0) * mass * util.e_scale.value
        elif args.enthalpy:
            h   = calc_enthalpy(fields)
            var = (h - 1.0) *  dV * util.e_scale.value
        elif 'temperature' in args.fields:
            var = util.prims2var(fields, 'temperature')
        else:
            edens_total = util.prims2var(fields, 'energy')
            var = edens_total * dV * util.e_scale.value
    elif args.dm_domega:
        W   = util.calc_lorentz_gamma(fields)
        var = dV * fields['rho'] * W * util.m.value
    
        
    gb      = fields['gamma_beta']
    tcenter = 0.5 * (tv[1:] + tv[:-1])
    dtheta  = (theta[-1,0] - theta[0,0])/theta.shape[0] * (180 / np.pi)
    domega  = 2.0 * np.pi * np.sin(tcenter) *(tv[1:] - tv[:-1])

    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    if args.inset:
        axins = inset_axes(ax, width="25%", height="30%",loc='upper left', borderpad=3.25)
    if args.dec_rad:
        ax_extra = ax.twinx() if not energy_and_mass else ax2.twinx()
        ax_extra.tick_params('y', labelsize=15)
        ax_extra.spines['top'].set_visible(False)
        
    lw = 2.0
    for cidx, cutoff in enumerate(args.cutoffs):
        if not energy_and_mass:
            var[gb < cutoff] = 0
            if args.labels:
                label = '%s'%(args.labels[case])
            else:
                label = None
            
            print(f'2D var sum with GB > {cutoff}: {var.sum():.2e}')
        if args.hist:
            deg_per_bin      = 3 # degrees in bin 
            num_bins         = int(deg_per_bin / dtheta) 
            theta_bin_edges  = np.linspace(theta[0,0], theta[-1,0], num_bins + 1)
            dx, _            = np.histogram(tcenter, weights=var,    bins=theta_bin_edges)
            dw, _            = np.histogram(tcenter[:,0], weights=domega[:,0], bins=theta_bin_edges)

            domega_cone = np.array([sum(domega[i:i+num_bins]) for i in range(0, len(domega), num_bins)])
            dx_cone     = np.array([sum(var[i:i+num_bins]) for i in range(0, len(var), num_bins)])
            dx_domega   = 4.0 * np.pi * np.sum(dx_cone, axis=1) / domega_cone[:,0]
            
            iso_var         = 4.0 * np.pi * dx/dw
            theta_bin_edges = np.rad2deg(theta_bin_edges)
            ax.step(theta_bin_edges[:-1], iso_var, label=label)
        else:
            if cutoff.is_integer():
                cut_fmt = int(cutoff)
            else:
                cut_fmt = cutoff
            if args.labels[0] == "":
                if args.dm_domega:
                    label=r"$M~(\Gamma \beta > {})$".format(cut_fmt)
                elif args.kinetic:
                    label=r"$E_k~(\Gamma \beta > {})$".format(cut_fmt)
                else:
                    label=r"$E_T~(\Gamma \beta > {})$".format(cut_fmt)
                if args.norm:
                    label += r" / {:.1e} ergs)".format(var.sum())
            else:
                label=label+r"$ > {}$".format(cut_fmt)
                
            if not energy_and_mass:
                quant_func     = np.sum if args.fields[0] != 'temperature' else np.mean
                iso_correction = 4.0 * np.pi
                var_per_theta  = iso_correction * quant_func(var, axis=1) / domega[:,0]
                
                axes = ax if cutoff != 0 else ax0
                linestyle = '-' if cutoff != 0 else '-'
                if cutoff == 0:
                    ax0_ylims = [var_per_theta.min(), var_per_theta.max()]
                if args.cmap == 'grayscale':
                    axes.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw, label=label, linestyle=linestyle)
                else:
                    axes.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw, label=label, color=colors[cidx], linestyle=linestyle)
            else:
                ek  [gb < cutoff] = 0
                mass[gb < cutoff] = 0
                quant_func     = np.sum if args.fields[0] != 'temperature' else np.mean
                iso_correction = 4.0 * np.pi
                ek_iso         = iso_correction * quant_func(ek, axis=1) / domega[:,0]
                m_iso          = iso_correction * quant_func(mass, axis=1) / domega[:,0] * util.m.cgs.value
                eklabel        = r"$E_k~(\Gamma \beta > %s)$"%(cutoff)
                mlabel         = r"$M~(\Gamma \beta > %s)$"%(cutoff)
                if cutoff == 0:
                    ax0_ylims = [ek_iso.min(), ek_iso.max()]
                    if args.cmap == 'grayscale':
                        ax0.plot(np.rad2deg(theta[:, 0]), ek_iso, lw=1, label=eklabel)
                    else:
                        ax0.plot(np.rad2deg(theta[:, 0]), ek_iso, lw=1, label=eklabel, color=colors[cidx])
                else:
                    if args.cmap == 'grayscale':
                        ax1.plot(np.rad2deg(theta[:, 0]), ek_iso, lw=lw, label=eklabel)
                        ax2.plot(np.rad2deg(theta[:, 0]), m_iso, lw=lw, label=mlabel)
                    else:
                        ax1.plot(np.rad2deg(theta[:, 0]), ek_iso, lw=lw, label=eklabel, color=colors[cidx])
                        ax2.plot(np.rad2deg(theta[:, 0]),  m_iso, lw=lw, label=mlabel, color=colors[cidx])
                    
                   
                

                
            if args.inset:
                axins.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw)
            lw = 1
    
    if args.dec_rad:
        ax_extra.set_ylabel(r'$r_{\rm dec} [\rm{cm}]$')  # we already handled the x-label with ax
        
    if args.xlims is None:
        ax = ax if not energy_and_mass else ax1
        ax.set_xlim(np.rad2deg(theta[0,0]), np.rad2deg(theta[-1,0]))
    else:
        ax = ax if not energy_and_mass else ax1
        ax.set_xlim(args.xlims[0], args.xlims[1])
    if args.inset:
        axins.set_xlim(80,100)
    
    if args.ylims is not None:
        ax.set_ylim(args.ylims[0], args.ylims[1])
        if args.inset:
            axins.set_ylim(args.ylims[0],args.ylims[1])
    
        # axins.set_xticklabels([])
        # axins.set_yticklabels([])
    if energy_and_mass:
        ax1.set_ylim(3e45, 4e49)
        ax2.set_ylim(3e23, 4e29)
    fsize = 15 if not args.print else DEFAULT_SIZE
    if args.sub_split is None:
        if energy_and_mass:
            ax2.set_xlabel(r'$\theta [\rm deg]$')
        else:
            ax.set_xlabel(r'$\theta [\rm deg]$')
        if 'ax0' in locals():
            ycoord = 0.5 if not energy_and_mass else 0.67
            if args.kinetic:
                fig.text(-0.06, ycoord, r'$E_{\rm K, iso}( > \Gamma \beta) \ [\rm{erg}]$', va='center', rotation='vertical')
            elif args.dm_domega:
                fig.text(0.01, ycoord, r'$M_{\rm iso}( > \Gamma \beta) \ [\rm{erg}]$', va='center', rotation='vertical')
            else:
                fig.text(0.01, ycoord, r'$E_{\rm T, iso}( > \Gamma \beta) \ [\rm{erg}]$', va='center', rotation='vertical')
                
            if energy_and_mass:
                ax2.set_ylabel(r'$M_{\rm iso} \ (>\Gamma \beta) \ [\rm g]$')
        else:
            if len(args.cutoffs) == 1:
                if args.kinetic:
                    ax.set_ylabel(r'$E_{{\rm K, iso}} \ (\Gamma \beta > {})\ [\rm{{erg}}]$'.format(args.cutoffs[0]))
                elif args.enthalpy:
                    ax.set_ylabel(r'$H_{\rm iso} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]))
                elif args.dm_domega:
                    ax.set_ylabel(r'$M_{\rm{iso}} \ (\Gamma \beta > {}) \ [\rm{{g}}]$'.format(args.cutoffs[0]))
                elif args.fields[0] == 'temperature':
                    ax.set_ylabel(r'$\bar{T}_{\rm{iso}} \ (\Gamma \beta > {}) \ [\rm{{eV}}]$'.format(args.cutoffs[0]))
                else:
                    ax.set_ylabel(r'$E_{{\rm T, iso}} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]))
            else:
                units = r'[\rm{{erg}}]' if not args.norm else ''
                if not args.dm_domega:
                    if args.kinetic:
                        ax.set_ylabel(r'$E_{{\rm K, iso}} \ (> \Gamma \beta)\ %s$'%(units))
                    elif args.enthalpy:
                        ax.set_ylabel(r'$H_{\rm iso} \ (>\Gamma \beta) \ %s$'%(units))
                    else:
                        ax.set_ylabel(r'$E_{{\rm T, iso}} \ (>\Gamma \beta) \ %s$'%(units))
                elif args.dm_domega:
                    units = r'[\rm{{g}}]' if not args.norm else ''
                    ax.set_ylabel(r'$M_{\rm iso} \ (>\Gamma \beta) \ %s$'%(units))
                elif args.fields[0] == 'temperature':
                    ax.set_ylabel(r'$\bar{T}_{\rm{iso}} \ (>\Gamma \beta) \ [\rm{{eV}}]$')
        
        ax.tick_params('both', labelsize=fsize)
    else:
        ax.tick_params('x', labelsize=fsize)
        ax.tick_params('y', labelsize=fsize)
    
    if energy_and_mass:
        ax0.tick_params('x', direction ='in')
        ax1.tick_params('x', direction ='in')
        #ax0.spines['right'].set_visible(False)
        ax0.spines['top'].set_visible(False)
        #ax1.spines['right'].set_visible(False)
        #ax1.spines['top'].set_visible(False)
        #ax2.spines['right'].set_visible(False)
        #ax2.spines['top'].set_visible(False)
    else:
        # ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    if args.inset:
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)

    if args.setup:
        ax.set_title(r'{}, t ={:.2f}'.format(args.setup, tend))
    if args.log:
        if energy_and_mass:
            ax1.set_yscale('log')
            ax2.set_yscale('log')
        else:
            ax.set_yscale('log')
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        # ax0.set_yscale('log')
        if 'ax0' in locals():
            # ax0.set_yscale('log')
            ax0.spines['top'].set_visible(False)
            # ax0.spines['right'].set_visible(False)
            # ax0.spines['bottom'].set_visible(False)
            # ax0.axes.get_xaxis().set_ticks([])
            # ax0.set_ylim(ax0_ylims)
            plt.subplots_adjust(hspace=0.0, wspace=0.0)
            ax0.yaxis.set_minor_locator(plt.MaxNLocator(2))
            tsize = 15 if not args.print else SMALL_SIZE
            ax0.tick_params('both',which='major', labelsize=tsize)
            ax.tick_params('both', which='major', labelsize=tsize)
            #ax.set_xlim(theta[0,0], theta[-1,0])
            
        if args.inset:
            axins.set_yscale('log')
        if args.dec_rad:
            ax_extra.set_yscale('log')
            if energy_and_mass:
                ticks = calc_dec_rad(0.0, np.array(ax2.axes.get_ylim())) 
            else:
                ticks = calc_dec_rad(0.0, np.array(ax.axes.get_ylim())) 
            ax_extra.set_ylim(ticks[0],ticks[-1])
            
        # ax0.yaxis.set_major_formatter(logfmt)
        # ax0.set_ylim(1e52,5e52)
        
    if not args.sub_split:
        if args.labels:
            size =  SMALL_SIZE
            if not energy_and_mass:
                ax.legend(loc=args.legend_loc, fancybox=True, framealpha=0.1, borderpad=0.3)
                if 'ax0' in locals():
                    ax0.legend(loc='best', fancybox=True, framealpha=0.1, borderpad=0.3)
            else:
                ax1.legend(loc=args.legend_loc, fancybox=True, framealpha=0.1, borderpad=0.3)
                ax2.legend(loc=args.legend_loc, fancybox=True, framealpha=0.1, borderpad=0.3)
                if 'ax0' in locals():
                    ax0.legend(loc='best', fancybox=True, framealpha=0.1, borderpad=0.3)

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(counter=0)
def plot_vs_time(
    args: argparse.ArgumentParser,
    ax: plt.Axes,
    label: str,
    color: float,
    time: np.ndarray,
    data: np.ndarray,
    ylog: bool = False,
    weights: Union[np.ndarray, float] = 1.0) -> None:
    
    ylabel = util.get_field_str(args)
    if len(ylabel) > 1:
        ylabel = ylabel[0]
    
    ax.set_title(rf'{args.setup}')
    ax.set_ylabel(r'$t$')
    ax.set_ylabel(rf"$\langle$ {ylabel} $\rangle$")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    time = np.asanyarray(time)
    data = np.asanyarray(data)
    ax.plot(time, data, label=label, color=color, alpha=1.0)

    if args.fields[0] == 'gamma_beta' or args.fields[0] == 'u1':
        if plot_vs_time.counter == 0:
            ax.plot(time, data[0] * np.exp(1 - time / time[0]), label =r'$\propto \exp(-t)$', color='grey', linestyle='-.')
            ax.plot(time, data[0] * (time / time[0]) ** (-3/2), label =r'$\propto t^{-3/2}$', color='grey', linestyle=':')
            ax.plot(time, data[0] * (time / time[0]) ** (-3), label =r'$\propto t^{-3}$', color='grey', linestyle='--')
    
    if args.log:
        ax.set_xscale('log')
        if ylog or args.fields[0] == 'gamma_beta':
            ax.set(yscale = 'log')
    
    if args.xlims:
        ax.set_xlim(*args.xlims)
    plot_vs_time.counter += 1
    
def snapshot(parser: argparse.ArgumentParser):
    plot_parser = get_subparser(parser, 1)
    plot_parser.add_argument('--cbar_sub', dest = 'cbar2', metavar='Range of Color Bar for secondary plot',nargs='+',type=float, default =[None, None], help='The colorbar range you\'d like to plot')
    plot_parser.add_argument('--no_cbar', dest ='no_cbar',action='store_true', default=False, help='colobar visible siwtch')
    plot_parser.add_argument('--cmap2', dest ='cmap2', metavar='Color Bar Colarmap 2', default = 'magma', help='The secondary colorbar cmap you\'d like to plot')
    plot_parser.add_argument('--rev_cmap', dest='rcmap', action='store_true',default=False, help='True if you want the colormap to be reversed')
    plot_parser.add_argument('--x', dest='x', nargs='+', default = None, type=float, help='List of x values to plot field max against')
    plot_parser.add_argument('--xlabel', dest='xlabel', nargs=1, default = 'X',  help='X label name')
    plot_parser.add_argument('--de_domega', dest='de_domega', action='store_true',default=False, help='Plot the dE/dOmega plot')
    plot_parser.add_argument('--dm_domega', dest='dm_domega', action='store_true',default=False, help='Plot the dM/dOmega plot')
    plot_parser.add_argument('--dec_rad', dest='dec_rad', default = False, action='store_true', help='Compute dr as function of angle')
    plot_parser.add_argument('--cutoffs', dest='cutoffs', default=[0.0], type=float, nargs='+', help='The 4-velocity cutoff value for the dE/dOmega plot')
    plot_parser.add_argument('--nwedge', dest='nwedge', default=0, type=int, help='Number of wedges')
    plot_parser.add_argument('--cbar_orient', dest='cbar_orient', default='vertical', type=str, help='Colorbar orientation', choices=['horizontal', 'vertical'])
    plot_parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 70, 110], type=float, nargs=4, help="wedge limits")
    plot_parser.add_argument('--bipolar', dest='bipolar', default = False, action='store_true')
    plot_parser.add_argument('--subplots', dest='subplots', default = None, type=int)
    plot_parser.add_argument('--sub_split', dest='sub_split', default = None, nargs='+', type=int)
    plot_parser.add_argument('--tau_s', dest='tau_s', action= 'store_true', default=False, help='The shock optical depth')
    plot_parser.add_argument('--viewing', help = 'viewing angle of simulation in [deg]', type=float, default=None, nargs='+')
    plot_parser.add_argument('--oned_slice', help='index of x1 array for one-d projection', default=None, type=int)
    plot_parser.add_argument('--oned_proj', help='axes to project 2d solution onto', default=None, type=int, choices=[1,2])
    args = parser.parse_args()
    
    vmin, vmax = args.cbar[:2]
    fields = {}
    setup = {}
    
    if args.cmap == 'grayscale':
        plt.style.use('grayscale')
    else:
        plt.style.use('seaborn-v0_8-colorblind')
    
    if args.dbg:
        plt.style.use('dark_background')
        
    
    num_subplots = len(args.sub_split) if args.sub_split is not None else 1
    flist, _     = util.get_file_list(args.files)
    if len(flist) > 1 or isinstance(flist, dict):
        if num_subplots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            lines_per_plot = len(args.files)
        else:
            fig,axs = plt.subplots(num_subplots, 1, figsize=(8,4 * num_subplots), sharex=True, tight_layout=False)
            if args.setup:
                fig.suptitle(f'{args.setup}')
            if args.de_domega or args.dm_domega:
                axs[-1].set_xlabel(r'$\theta \ \rm[deg]$')
            else:
                axs[-1].set_xlabel(r'$\Gamma \beta$')
            if args.de_domega or args.dm_domega:
                if len(args.cuotff) == 1:
                    if args.kinetic:
                        fig.text(0.030, 0.5, r'$E_{{\rm K, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), va='center', rotation='vertical')
                    elif args.dm_domega:
                        fig.text(0.030, 0.5, r'$M_{{\rm iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), va='center', rotation='vertical')
                    else:
                        fig.text(0.030, 0.5, r'$E_{{\rm T, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), va='center', rotation='vertical')
            else:
                units = r"[\rm{erg}]" if not args.norm else ""
                xpos  = 0.030 if not args.print else -0.020
                fsize = 20 if not args.print else DEFAULT_SIZE
                if args.kinetic:
                    fig.text(xpos, 0.5, r'$E_{\rm K}( > \Gamma \beta) \ %s$'%(units), va='center', rotation='vertical')
                elif args.enthalpy:
                    fig.text(xpos, 0.5, r'$H( > \Gamma \beta) \ %s$'%(units), va='center', rotation='vertical')
                else:
                    fig.text(xpos, 0.5, r'$E_{\rm T}( > \Gamma \beta) \ %s$'%(units), va='center', rotation='vertical')
            axs_iter       = iter(axs)            # iterators for the multi-plot
            subplot_iter   = iter(args.sub_split) # iterators for the subplot splitting
            
            # first elements of subplot iterator tells how many files belong on axs[0]
            lines_per_plot = next(subplot_iter)   
        
        i        = 0       
        ax_col   = 0
        ax_shift = True
        ax_num   = 0    
        if isinstance(flist, (list, np.ndarray)):
            if args.weighted_vs_time:
                weighted_vars = []
                times = []
                colors = ['red', 'black']
                label = args.labels[0] if args.labels else None
                for idx, file in enumerate(flist):
                    print(f'processing file {file}...', flush=True)
                    fields, setup, mesh = util.read_file(args, file, 2)
                    if args.fields[0] in derived:
                        var = util.prims2var(fields, args.fields[0])
                    else:
                        var = fields[args.fields[0]]
                        
                    if len(args.fields) > 1:
                        weights = util.prims2var(fields, args.fields[1])
                    else:
                        weights = 1.0 
                        
                    dV              = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
                    weighted        = np.sum(weights * var * dV) / np.sum(weights * dV)
                    weighted_vars  += [weighted]
                    times          += [setup['time']]
                plot_vs_time(args, ax, label, colors[0], times, weighted_vars, ylog = (args.fields[0] not in lin_fields))
                ax.legend()
            else:
                for idx, file in enumerate(flist):
                    print(f'processing file {file}...', flush=True)
                    fields, setup, mesh = util.read_file(args, file, ndim=2)
                    i += 1
                    if args.hist and (not args.de_domega and not args.dm_domega):
                        if args.sub_split is None:
                            plot_hist(fields, args, mesh, setup, overplot=True, ax=ax, case=idx, ax_col=idx)
                        else:
                            if ax_shift:
                                ax_col   = 0
                                ax       = next(axs_iter)   
                                ax_shift = False
                            plot_hist(fields, args, mesh, setup, overplot=True, ax=ax, ax_num=ax_num, case=i-1, ax_col=ax_col)
                    elif args.de_domega or args.dm_domega:
                        if args.sub_split is None:
                            plot_dx_domega(fields, args, mesh, setup, overplot=True, ax=ax, case=i-1, ax_col=idx)
                        else:
                            if ax_shift:
                                ax_col   = 0
                                ax       = next(axs_iter)   
                                ax_shift = False
                            plot_dx_domega(fields, args, mesh, setup, overplot=True, ax=ax, ax_num=ax_num, case=idx, ax_col=ax_col)
                    elif args.x is not None:
                        plot_per_theta(fields, args, mesh, setup, True, ax, idx)
                    elif args.dec_rad:
                        plot_dec_rad(fields, args, mesh, setup, True, ax, idx)
                    else:
                        plot_1d_curve(fields, args, mesh, setup, True, ax, idx)
                    
                    ax_col += 1
                    if i == lines_per_plot:
                        i        = 0
                        ax_num  += 1
                        ax_shift = True
                        try:
                            if num_subplots > 1:
                                lines_per_plot = next(subplot_iter)
                        except StopIteration:
                            break
        else:
            ylabel = util.get_field_str(args)
            ax.set_xlabel(r'$t [\rm s]$')
            ax.set_ylabel(ylabel)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            colors = ['red', 'black']
            for key in flist.keys():
                weighted_vars = []
                times = []
                label = args.labels[key] if args.labels else key
                for idx, file in enumerate(flist[key]):
                    print(f'processing file {file}...', flush=True)
                    fields, setup, mesh = util.read_file(args, file, 2)
                    if args.fields[0] in derived:
                        var = util.prims2var(fields, args.fields[0])
                    else:
                        var = fields[args.fields[0]]
                        
                    if len(args.fields) > 1:
                        weights = util.prims2var(fields, args.fields[1])
                    else:
                        weights = 1.0 
                        
                    dV              = util.calc_cell_volume2D(mesh['x1'], mesh['x2'])
                    weighted        = np.sum(weights * var * dV) / np.sum(weights * dV)
                    weighted_vars  += [weighted]
                    times          += [setup['time']]

                plot_vs_time(args, ax, label, colors[key], times, weighted_vars, ylog = args.fields[0] not in lin_fields)
                    
                
            ax.legend()
        if args.sub_split is not None:
            for ax in axs:
                ax.label_outer()
    else:
        fields, setup, mesh = util.read_file(args, args.files[0], ndim=2)
        if args.hist and (not args.de_domega and not args.dm_domega):
            plot_hist(fields, args, mesh, setup)
        elif args.viewing != None:
            plot_1d_curve(fields, args, mesh, setup)
        elif args.de_domega or args.dm_domega:
            plot_dx_domega(fields, args, mesh, setup)
        elif args.x is not None:
            plot_per_theta(fields, args, mesh, setup, overplot=False)
        elif args.dec_rad:
            plot_dec_rad(fields, args, mesh, setup, overplot=False)
        else:
            if setup['is_cartesian']:
                plot_cartesian_plot(fields, args, mesh, setup, mirror=setup['coord_system'] == 'axis_cylindrical')
            else:
                plot_polar_plot(fields, args, mesh, setup)
    
    if args.sub_split is not None:
        fsize = 15 if not args.print else DEFAULT_SIZE
        if args.labels is not None:
            if not args.legend_loc:
                axs[0].legend(loc='upper right', fancybox=True, framealpha=0.1)
            else:
                axs[0].legend(loc=args.legend_loc, fancybox=True, framealpha=0.1)
        plt.subplots_adjust(hspace=0.0)

    
    if not args.save:
        import matplotlib
        backend = matplotlib.get_backend()
        if backend == 'agg':
            print("Detected agg backend, so saving fig as 'backup_save' instead...", flush=True)
            ext = 'pdf' if not args.png else 'png'
            dpi = 600
            plt.savefig('backup_save.{}'.format(ext), dpi=dpi, bbox_inches='tight')
        else:
            plt.show()
    else:
        if args.print:
            fig = plt.gcf()
            all_axes = fig.get_axes()
            
            for i, ax in enumerate(all_axes):
                for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(DEFAULT_SIZE)
                
                # for item in (ax.get_xticklabels() + ax.get_yticklabels()):
                #     item.set_fontsize(SMALL_SIZE)
                try:
                    for item in ax.get_legend().get_texts():
                        item.set_fontsize(5)
                except AttributeError:
                    pass
            
            fig.set_size_inches(*args.fig_dims)
            
        # \fig.tight_layout(pad=0.00)
        # plt.subplots_adjust(wspace=0, hspace=0)
        ext = 'pdf' if not args.png else 'png'
        dpi = 600
        plt.savefig('{}.{}'.format(args.save.replace(' ', '_'), ext), dpi=dpi, bbox_inches='tight')
