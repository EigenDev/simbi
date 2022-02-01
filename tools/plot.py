#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as mcolors
import argparse 
import h5py 
import astropy.constants as const
import astropy.units as u 
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from itertools import cycle

try:
    import cmasher as cmr 
except:
    print('No Cmasher module, so defaulting to matplotlib colors')

#================================
#   constants of nature
#================================
R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * c **2
pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
time_scale = R_0 / c

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

def find_nearest(arr: list, val: float) -> Union[int, float]:
    arr = np.asarray(arr)
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
    
def fill_below_intersec(x: np.ndarray, y: np.ndarray, constraint: float, color: float) -> None:
    ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
def get_2d_file(args: argparse.ArgumentParser, filename: str) -> dict:
    setup  = {}
    fields = {}
    is_cartesian = False
    with h5py.File(filename, 'r') as hf: 
        ds          = hf.get('sim_info')
        rho         = hf.get('rho')[:]
        v1          = hf.get('v1')[:]
        v2          = hf.get('v2')[:]
        p           = hf.get('p')[:]
        t           = ds.attrs['current_time']
        xmax        = ds.attrs['xmax']
        xmin        = ds.attrs['xmin']
        ymax        = ds.attrs['ymax']
        ymin        = ds.attrs['ymin']
        
        # New checkpoint files, so check if new attributes were
        # implemented or not
        try:
            nx          = ds.attrs['nx']
            ny          = ds.attrs['ny']
        except:
            nx          = ds.attrs['NX']
            ny          = ds.attrs['NY']
        
        try:
            chi = hf.get('chi')[:]
        except:
            chi = np.zeros((ny, nx))
            
        try:
            gamma = ds.attrs['adiabatic_gamma']
        except:
            gamma = 4./3.
        
        # Check for garbage value
        if gamma < 1:
            gamma = 4./3. 
            
        try:
            coord_sysem = ds.attrs['geometry'].decode('utf-8')
        except:
            coord_sysem = 'spherical'
            
        try:
            is_linspace = ds.attrs['linspace']
        except:
            is_linspace = False
        
        setup['xmax'] = xmax 
        setup['xmin'] = xmin 
        setup['ymax'] = ymax 
        setup['ymin'] = ymin 
        setup['time'] = t * time_scale
        
        rho = rho.reshape(ny, nx)
        v1  = v1.reshape(ny, nx)
        v2  = v2.reshape(ny, nx)
        p   = p.reshape(ny, nx)
        chi = chi.reshape(ny, nx)
        
        
        if args.forder:
            rho = rho[1:-1, 1: -1]
            v1  = v1 [1:-1, 1: -1]
            v2  = v2 [1:-1, 1: -1]
            p   = p  [1:-1, 1: -1]
            chi = chi[1:-1, 1: -1]
            xactive = nx - 2
            yactive = ny - 2
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            chi = chi[2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            setup['xactive'] = xactive
            setup['yactive'] = yactive
        
        if is_linspace:
            setup['x1'] = np.linspace(xmin, xmax, xactive)
            setup['x2'] = np.linspace(ymin, ymax, yactive)
        else:
            setup['x1'] = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
            setup['x2'] = np.linspace(ymin, ymax, yactive)
        
        if coord_sysem == 'cartesian':
            is_cartesian = True
        
        W    = 1/np.sqrt(1.0 -(v1**2 + v2**2))
        beta = np.sqrt(v1**2 + v2**2)
        
        fields['rho']          = rho
        fields['v1']           = v1 
        fields['v2']           = v2 
        fields['p']            = p
        fields['chi']          = chi
        fields['gamma_beta']   = W*beta
        fields['ad_gamma']     = gamma
        setup['is_cartesian']  = is_cartesian
        # fields[idx]['gamma_beta_1'] = abs(W*v1)
        # fields[idx]['gamma_beta_2'] = abs(W*v2)
        
        
    ynpts, xnpts = rho.shape 
    mesh = {}
    if setup['is_cartesian']:
        xx, yy = np.meshgrid(setup['x1'], setup['x2'])
        mesh['xx'] = xx
        mesh['yy'] = yy
    else:      
        rr, tt = np.meshgrid(setup['x1'], setup['x2'])
        mesh['theta'] = tt 
        mesh['rr']    = rr
        mesh['r']     = setup['x1']
        mesh['th']    = setup['x2']
        
    return fields, setup, mesh 

def get_1d_equiv_file(filename: str) -> dict:
    file = filename
    ofield = {}
    with h5py.File(file, 'r') as hf:
        ds = hf.get('sim_info')
        
        rho         = hf.get('rho')[:]
        v           = hf.get('v')[:]
        p           = hf.get('p')[:]
        nx          = ds.attrs['Nx']
        t           = ds.attrs['current_time']
        xmax        = ds.attrs['xmax']
        xmin        = ds.attrs['xmin']

        rho = rho[2:-2]
        v   = v  [2:-2]
        p   = p  [2:-2]
        xactive = nx - 4
            
        W    = 1/np.sqrt(1 - v**2)
        beta = v
        

        a    = (4 * const.sigma_sb.cgs / c)
        k    = const.k_B.cgs
        T    = (3 * p * pre_scale  / a)**(1./4.)
        T_eV = (k * T).to(u.eV)
        
        h = 1.0 + 4/3 * p / (rho * (4/3 - 1))
        
        ofield['ad_gamma']    = 4./3.
        ofield['rho']         = rho
        ofield['v']           = v
        ofield['p']           = p
        ofield['W']           = W
        ofield['enthalpy']    = h
        ofield['gamma_beta']  = W*beta
        ofield['temperature'] = T_eV
        ofield['r']           = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
    return ofield
    
derived       = ['D', 'momentum', 'energy', 'energy_rst', 'enthalpy', 'temperature', 'mass', 'chi_dens',
                 'gamma_beta_1', 'gamma_beta_2']
field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'chi'] + derived
lin_fields    = ['chi', 'gamma_beta', 'gamma_beta_1', 'gamma_beta_2']

def compute_rverticies(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return rvertices 

def compute_theta_verticies(theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    return tvertices 

def calc_cell_volume1D(r: np.ndarray) -> np.ndarray:
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return 4.0 * np.pi * (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3)

def calc_cell_volume(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    dcos      = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    rvertices = np.sqrt(r[:, 1:] * r[:, :-1])
    rvertices = np.insert(rvertices,  0, r[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], r[:, -1], axis=1)
    dr        = rvertices[:, 1:] - rvertices[:, :-1]
    
    return (2.0 * np.pi *  (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos )
        
def get_field_str(args: argparse.ArgumentParser) -> str:
    field_str_list = []
    for field in args.field:
        if field == 'rho' or field == 'D':
            var = r'\rho' if field == 'rho' else 'D'
            if args.units:
                field_str_list.append( r'${}$ [g cm$^{{-3}}$]'.format(var))
            else:
                field_str_list.append( r'${}/{}_0$'.format(var,var))
            
        elif field == 'gamma_beta':
            field_str_list.append( r'$\Gamma \beta$')
        elif field == 'gamma_beta_1':
            field_str_list.append( r'$\Gamma \beta_1$')
        elif field == 'gamma_beta_2':
            field_str_list.append( r'$\Gamma \beta_2$')
        elif field == 'energy' or field == 'p':
            if args.units:
                if field == 'energy':
                    field_str_list.append( r'$\tau [\rm erg \ cm^{{-3}}]$')
                else:
                    field_str_list.append( r'$p [\rm erg \ cm^{{-3}}]$')
            else:
                if field == 'energy':
                    field_str_list.append( r'$\tau/\tau_0$')
                else:
                    field_str_list.append( r'$p/p_0$')
        elif field == 'energy_rst':
            if args.units:
                field_str_list.append( r'$\tau + D \  [\rm erg \ cm^{-3}]$')
            else:
                field_str_list.append( r'$\tau + D')
        elif field == 'chi':
            field_str_list.append( r'$\chi$')
        elif field == 'chi_dens':
            field_str_list.append( r'$D \cdot \chi$')
        elif field == 'temperature':
            field_str_list.append("T [eV]" if args.units else "T")
        else:
            field_str_list.append(field)
    
    
    return field_str_list if len(args.field) > 1 else field_str_list[0]

def calc_enthalpy(fields: dict) -> np.ndarray:
    return 1.0 + fields['p']*fields['ad_gamma'] / (fields['rho'] * (fields['ad_gamma'] - 1.0))
    
def calc_lorentz_gamma(fields: dict) -> np.ndarray:
    return (1.0 + fields['gamma_beta']**2)**0.5

def calc_beta(fields: dict) -> np.ndarray:
    W = calc_lorentz_gamma(fields)
    return (1.0 - 1.0 / W**2)**0.5

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def prims2var(fields: dict, var: str) -> np.ndarray:
    h = calc_enthalpy(fields)
    W = calc_lorentz_gamma(fields)
    if var == 'D':
        # Lab frame density
        return fields['rho'] * W
    elif var == 'S':
        # Lab frame momentum density
        return fields['rho'] * W**2 * calc_enthalpy(fields) * fields['v']
    elif var == 'energy':
        # Energy minus rest mass energy
        return fields['rho']*h*W**2 - fields['p'] - fields['rho']*W
    elif var == 'energy_rst':
        # Total Energy
        return fields['rho']*h*W**2 - fields['p']
    elif var == 'temperature':
        a    = (4.0 * const.sigma_sb.cgs / c)
        T    = (3.0 * fields['p'] * pre_scale  / a)**0.25
        T_eV = (const.k_B.cgs * T).to(u.eV)
        return T_eV.value
    elif var == 'chi_dens':
        return fields['chi'] * fields['rho'] * W
    elif var == 'gamma_beta_1':
        return W * fields['v1']
    elif var == 'gamma_beta_2':
        return W * fields['v2']
    elif var =='sp_enthalpy':
        # Specific enthalpy
        return h - 1.0  

def place_anotation(args: argparse.ArgumentParser, fields: dict, ax: plt.Axes, etot: float) -> None:
    order_of_mag = np.floor(np.log10(etot))
    front_factor = int(etot / 10**order_of_mag)
    if front_factor != 1:
        anchor_text = r"$E_{\rm exp} = %i \times 10^{%i}$ erg"%(front_factor, order_of_mag)     
    else:
        anchor_text = r"$E_{\rm exp} = 10^{%i}$ erg"%(order_of_mag)
    
    if args.anot_text is not None:
        extra_text   = args.anot_text
        anchor_text += "\n     %s"%(extra_text)
    at = AnchoredText(
    anchor_text, prop=dict(size=15), frameon=False, loc=args.anot_loc)
    at.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2")
    ax.add_artist(at)
    
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
    num_fields = len(args.field)
    is_wedge   = args.nwedge > 0
    if not subplots:
        if is_wedge != 0:
            nplots = args.nwedge + 1
            fig, axes = plt.subplots(1, nplots, subplot_kw={'projection': 'polar'},
                                figsize=(15, 12), constrained_layout=True)
            ax    = axes[0]
            wedge = axes[1]
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                                figsize=(10, 8), constrained_layout=False)
    else:
        if is_wedge:
            ax    = axs[0]
            wedge = axs[1]
        else:
            ax = axs
        
    rr, tt      = mesh['rr'], mesh['theta']
    t2          = - tt[::-1]
    xmax        = dset['xmax']
    xmin        = dset['xmin']
    ymax        = dset['ymax']
    ymin        = dset['ymin']
    
    vmin,vmax = args.cbar[:2]

    unit_scale = np.ones(num_fields)
    if (args.units):
        for idx, field in enumerate(args.field):
            if field == 'rho' or field == 'D':
                unit_scale[idx] = rho_scale.value
            elif field == 'p' or field == 'energy' or field == 'energy_rst':
                unit_scale[idx] = pre_scale.value
    
    units = unit_scale if args.units else np.ones(num_fields)
     
    if args.rcmap:
        color_map = (plt.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.get_cmap(args.cmap)
        
    tend = dset['time']
    # If plotting multiple fields on single polar projection, split the 
    # field projections into their own quadrants
    if num_fields > 1:
        cs  = np.zeros(4, dtype=object)
        var = []
        kwargs = []
        for field in args.field:
            if field in derived:
                if ymax == np.pi:
                    var += np.split(prims2var(fields, field), 2)
                else:
                    var.append(prims2var(fields, field))
            else:
                if ymax == np.pi:
                    var += np.split(fields[field], 2)
                else:
                    var.append(fields[field])
                
        if ymax == np.pi: 
            units  = np.repeat(units, 2)
            
        var    = np.asarray(var)
        var    = np.array([units[idx] * var[idx] for idx in range(var.shape[0])])
        
        tchop  = np.split(tt, 2)
        trchop = np.split(t2, 2)
        rchop  = np.split(rr, 2)
        
        quadr = {}
        field1 = args.field[0]
        field2 = args.field[1]
        field3 = args.field[2 % num_fields]
        field4 = args.field[-1]
        
        
        quadr[field1] = var[0]
        quadr[field2] = var[3 % num_fields if num_fields != 3 else num_fields]
        
        if ymax == np.pi:
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
                kwargs[field] =  {'vmin': vmin, 'vmax': vmax} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)} 
            else:
                if field == field3 == field4:
                    ovmin = None if len(args.cbar) == 2 else args.cbar[2]
                    ovmax = None if len(args.cbar) == 2 else args.cbar[3]
                else:
                    ovmin = None if len(args.cbar) == 2 else args.cbar[idx+1]
                    ovmax = None if len(args.cbar) == 2 else args.cbar[idx+2]
                kwargs[field] =  {'vmin': ovmin, 'vmax': ovmax} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = ovmin, vmax = ovmax)} 

        if ymax < np.pi:
            cs[0] = ax.pcolormesh(tt[:: 1], rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
            cs[1] = ax.pcolormesh(t2[::-1], rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
            
            # If simulation only goes to pi/2, if bipolar flag is set, mirror the fields accross the equator
            if args.bipolar:
                cs[2] = ax.pcolormesh(tt[:: 1] + np.pi/2, rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
                cs[3] = ax.pcolormesh(t2[::-1] + np.pi/2, rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
        else:
            if num_fields == 2:
                cs[0] = ax.pcolormesh(tt[:: 1], rr,  np.vstack((var[0],var[1])), cmap=args.cmap, shading='auto', **kwargs[field1])
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
            
        cs = np.zeros(len(args.field), dtype=object)
        
        if args.field[0] in derived:
            var = units * prims2var(fields, args.field[0])
        else:
            var = units * fields[args.field[0]]
            
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto',
                              linewidth=0, rasterized=True, **kwargs)
        cs[0] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, 
                              linewidth=0,rasterized=True, shading='auto', **kwargs)
        
        if args.bipolar:
            cs[0] = ax.pcolormesh(tt[:: 1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
            cs[0] = ax.pcolormesh(t2[::-1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
    
    if args.pictorial: 
        ax.set_position( [0.1, -0.15, 0.8, 1.30])
            
    if not args.pictorial:
        if ymax < np.pi:
            ymd = int( np.floor(ymax * 180/np.pi) )
            if not args.bipolar:                                                                                                                                                                                   
                ax.set_thetamin(-ymd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                ax.set_thetamax(ymd)
                ax.set_position( [0.1, -0.18, 0.8, 1.43])
            else:
                ax.set_position( [0.1, -0.18, 0.9, 1.43])
            if num_fields > 1:
                ycoord  = [0.1, 0.1 ]
                xcoord  = [0.88, 0.04]
                cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                cbar_orientation = 'vertical'
            else:
                cbar_orientation = 'horizontal'
                if cbar_orientation == 'horizontal':
                    cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
        else:  
            if not args.no_cbar:         
                cbar_orientation = 'horizontal'
                # ax.set_position([0.1, -0.18, 0.7, 1.3])
                if num_fields > 1:
                    if num_fields == 2:
                        if cbar_orientation == 'vertical':
                            ycoord  = [0.1, 0.08] if ymax < np.pi else [0.10, 0.10]
                            xcoord  = [0.1, 0.85] if ymax < np.pi else [0.86, 0.05]
                            cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                        else:
                            if not is_wedge:
                                ycoord  = [0.2, 0.20] if ymax < np.pi else [0.10, 0.10]
                                xcoord  = [0.1, 0.50] if ymax < np.pi else [0.51, 0.20]
                            else:
                                ycoord  = [0.2, 0.20] if ymax < np.pi else [0.10, 0.10]
                                xcoord  = [0.1, 0.50] if ymax < np.pi else [0.51, 0.20]
                            cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.3, 0.04]) for i in range(num_fields)]
                    if num_fields == 3:
                        ycoord  = [0.1, 0.5, 0.1]
                        xcoord  = [0.07, 0.85, 0.85]
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8 * 0.5]) for i in range(1, num_fields)]
                        cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                    if num_fields == 4:
                        ycoord  = [0.5, 0.1, 0.5, 0.1]
                        xcoord  = [0.85, 0.85, 0.07, 0.07]
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
                else:
                    if not is_wedge:
                        plt.tight_layout()
                        
                    if cbar_orientation == 'vertical':
                        cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.85])
                    else:
                        cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.85])
        if args.log:
            if not args.no_cbar:
                if num_fields > 1:
                    fmt  = [None if field in lin_fields else tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True) for field in args.field]
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
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color='white')
        
        if args.nwedge == 2:
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color='white')
            ax.plot(np.radians(-np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
            ax.plot(np.radians(-np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color='white')
            
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
    ax.set_rmax(xmax) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    ax.set_rmin(xmin)
    
    field_str = get_field_str(args)
    
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
            
        if len(args.field) > 1:
            if len(args.cbar2) == 4:
                vmin2, vmax2, vmin3, vmax3 = args.cbar2
            else:
                vmin2, vmax2 = args.cbar2
                vmin3, vmax3 = None, None
            kwargs = {}
            for idx, key in enumerate(quadr.keys()):
                field = args.field[idx % num_fields]
                if idx == 0:
                    kwargs[field] =  {'vmin': vmin2, 'vmax': vmax2} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin2, vmax = vmax2)} 
                elif idx == 1:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                    kwargs[field] =  {'vmin': vmin3, 'vmax': vmax3} if field in lin_fields else {'norm': mcolors.LogNorm(vmin = vmin3, vmax = vmax3)} 
                else:
                    continue

            wedge.pcolormesh(tt[:: 1], rr,  np.vstack((var[0],var[1])), cmap=args.cmap, shading='auto', **kwargs[field1])
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
            
        wedge.axes.xaxis.set_ticklabels([])
        wedge.set_ylim([wedge_min, wedge_max])
        wedge.set_rorigin(-wedge_min)
        wedge.set_thetamin(ang_min)
        wedge.set_thetamax(ang_max)
        wedge.yaxis.set_minor_locator(plt.MaxNLocator(1))
        wedge.yaxis.set_major_locator(plt.MaxNLocator(2))
        # wedge.set_aspect(1.)
        
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
            axes[2].set_rorigin(-wedge_min)
            axes[2].set_thetamin(-ang_min)
            axes[2].set_thetamax(-ang_max)
            axes[2].yaxis.set_minor_locator(plt.MaxNLocator(1))
            axes[2].yaxis.set_major_locator(plt.MaxNLocator(2))
            # axes[2].set_aspect(1.)
            
        
        
    if not args.pictorial:
        if not args.no_cbar:
            if args.log:
                if ymax == np.pi:
                    set_label = ax.set_ylabel if cbar_orientation == 'vertical' else ax.set_xlabel
                    if num_fields > 1:
                        for i in range(num_fields):
                            if args.field[i] in lin_fields:
                                cbar[i].set_label(r'{}'.format(field_str[i]), fontsize=30)
                            else:
                                cbar[i].set_label(r'$\log$ {}'.format(field_str[i]), fontsize=30)
                    else:
                        cbar.ax.set_ylabel(r'$\log$ {}'.format(field_str), fontsize=20)
                else:
                    if num_fields > 1:
                        for i in range(num_fields):
                            if args.field[i] in lin_fields:
                                cbar[i].ax.set_ylabel(r'{}'.format(field_str[i]), fontsize=20)
                            else:
                                cbar[i].ax.set_ylabel(r'$\log$ {}'.format(field_str[i]), fontsize=20)
                    else:
                        cbar.ax.set_xlabel(r'$\log$ {}'.format(field_str), fontsize=20)
            else:
                if ymax >= np.pi:
                    if num_fields > 1:
                        for i in range(num_fields):
                            cbar[i].ax.set_ylabel(r'{}'.format(field_str[i]), fontsize=20)
                    else:
                        cbar.ax.set_ylabel(f'{field_str}', fontsize=20)
                else:
                    cbar.ax.set_xlabel(r'{}'.format(field_str), fontsize=20)
        if args.setup != "":
            fig.suptitle('{} at t = {:.2f}'.format(args.setup, tend), fontsize=25, y=1)
        else:
            fig.suptitle('t = {:.2f}'.format(tend), fontsize=25, y=0.83)

def plot_cartesian_plot(
    fields: dict, 
    args: argparse.ArgumentParser, 
    mesh: dict, 
    dset: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10,10), constrained_layout=False)

    xx, yy = mesh['xx'], mesh['yy']
    xmax        = dset['xmax']
    xmin        = dset['xmin']
    ymax        = dset['ymax']
    ymin        = dset['ymin']
    
    vmin,vmax = args.cbar

    if args.log:
        kwargs = {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = dset['time']
    c = ax.pcolormesh(xx, yy, fields[args.field], cmap=color_map, shading='auto', **kwargs)
    
    divider = make_axes_locatable(ax)
    cbaxes  = divider.append_axes('right', size='5%', pad=0.05)
    
    if not args.no_cbar:
        if args.log:
            logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
            cbar = fig.colorbar(c, orientation='vertical', cax=cbaxes, format=logfmt)
        else:
            cbar = fig.colorbar(c, orientation='vertical', cax=cbaxes)

    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    ax.tick_params(axis='both', labelsize=10)
    
    # Change the format of the field
    field_str = get_field_str(args)
    if args.log:
        cbar.ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
    
    if args.setup != "":
        fig.suptitle('{} at t = {:.2f}'.format(args.setup, tend), fontsize=20, y=0.95)
    
def plot_1d_curve(
    fields: dict, 
    args:       argparse.ArgumentParser, 
    mesh:       dict, 
    dset:       dict, 
    overplot:   bool=False,
    a:          bool=None, 
    case:       int =0) -> None:
    
    num_fields = len(args.field)
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['r'], mesh['th']
    theta    = theta * 180 / np.pi 
    
    xmax        = dset['xmax']
    xmin        = dset['xmin']
    ymax        = dset['ymax']
    ymin        = dset['ymin']
    
    vmin,vmax = args.cbar
    var = [field for field in args.field] if num_fields > 1 else args.field[0]
    
    tend = dset['time']
    if args.mass:
        dV          = calc_cell_volume(mesh['rr'], mesh['theta'])
        mass        = dV * fields['W'] * fields['rho']
        # linestyle = '-.'
        if args.labels is None:
            ax.loglog(r, mass[args.tidx]/ np.max(mass[args.tidx]), label = 'mass', linestyle='-.', color=colors[case])
            ax.loglog(r, fields['p'][args.tidx] / np.max(fields['p'][args.tidx]), label = 'pressure', color=colors[case])
        else:
            ax.loglog(r, mass[args.tidx]/ np.max(mass[args.tidx]), label = f'{args.labels[case]} mass', linestyle='-.', color=colors[case])
            ax.loglog(r, fields['p'][args.tidx] / np.max(fields['p'][args.tidx]), label = f'{args.labels[case]} pressure', color=colors[case])
        ax.legend(fontsize=20)
        ax.axvline(0.65, linestyle='--', color='red')
        ax.axvline(1.00, linestyle='--', color='blue')
    else:
        for idx, field in enumerate(args.field):
            if field in derived:
                var = prims2var(fields, field)
            else:
                var = fields[field]
                
            label = None if args.labels is None else '{}'.format(field_str[idx] if num_fields > 1 else field_str)
            ax.loglog(r, var[args.tidx], label=label)
    

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$r/R_\odot$', fontsize=20)
    ax.tick_params(axis='both', labelsize=10)
    
    # Change the format of the field
    field_str = get_field_str(args)
    
    if args.log:
        if num_fields == 1:
            ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
        else:
            ax.legend()
    else:
        if num_fields == 1:
            ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        else:
            ax.legend()
    
    if args.setup != "":
        ax.set_title(r'$\theta = {:.2f}$ time: {:.3f}'.format(mesh['th'][args.tidx] * 180 / np.pi, tend))
    if not overplot:
        return fig
    # fig.suptitle(r'{} at $\theta = {:.2f}$ deg, t = {:.2f} s'.format(args.setup,theta[args.tidx], tend), fontsize=20, y=0.95)
    
def plot_per_theta(
    fields:    dict, 
    args:      argparse.ArgumentParser, 
    mesh:      dict , 
    dset:      dict, 
    overplot:  bool=False, 
    ax:        bool=None, 
    case:      int =0) -> None:
    print('plotting vs theta...')
    
    colors = plt.cm.viridis(np.linspace(0.1, 0.90, len(args.filename)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    theta = mesh['th']
    

    for field in args.field:
        fields = fields if args.oned_files is None else get_1d_equiv_file(args.oned_files[0])
        if field in derived:
            var = prims2var(fields, field)
        else:
            var = fields[field].copy()
        if args.units:
            if field == 'p' or field == 'energy':
                var *= pre_scale.value
            elif field == 'D' or field == 'rho':
                var *= rho_scale.value

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
    if args.tex:
        label = f"$\{label}$"
    
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
        tau = np.zeros_like(mesh['th'])
        for tidx, t in enumerate(mesh['th']):
            ridx      = np.argmax(fields['gamma_beta'][tidx])
            tau[tidx] = kappa * a_star / (mesh['rr'][tidx,ridx] * R_0)
            
        mean_tau = running_mean(tau, 50)
        ax.plot(theta, 32*mean_tau, linestyle='--', label=label+"(x32)")
    if not args.tau_s:
        ylabel = get_field_str(args)
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
            dV = calc_cell_volume(mesh['rr'], mesh['theta'])
            etot = np.sum(prims2var(fields, "energy") * dV * e_scale.value)
            place_anotation(args, fields, ax, etot)
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$\theta [\rm deg]$', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
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
    
    file_num = len(args.filename)
    
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    theta = mesh['th']
    mdots = np.logspace(np.log10(1e-6), np.log10(24e-4),128)

    colors = plt.cm.viridis(np.linspace(0.1, 0.90, file_num if file_num > 1 else len(mdots)))
    
    tvert   = compute_theta_verticies(mesh['theta'])
    tcent   = 0.5 * (tvert[1:,0] + tvert[:-1,0])
    dtheta  = tvert[1:,0] - tvert[:-1,0]
    domega  = 2.0 * np.pi * np.sin(tcent) * dtheta 
    vw      = 1e8 * u.cm / u.s
    mdots   = (mdots * u.M_sun / u.yr).to(u.g/u.s)
    factor  = np.array([0.75 * 4.0 * np.pi * vw.value / mdot.value for mdot in mdots])
    gb      = fields['gamma_beta']
    W       = calc_lorentz_gamma(fields)
    dV      = calc_cell_volume(mesh['rr'], mesh['theta'])
    mass    = W * dV * fields['rho'] * m.value

    theta = theta * 180 / np.pi
    
    if file_num > 1:
        pts   = np.zeros(shape=(1, theta.size))
    else:
        pts   = np.zeros(shape=(len(mdots), theta.size))
    
    label = args.labels[case] if args.labels is not None else None
    if args.tex:
        label = f"$\{label}$"
    
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
            dV = calc_cell_volume(mesh['rr'], mesh['theta'])
            etot = np.sum(prims2var(fields, "energy") * dV * e_scale.value)
            place_anotation(args, fields, ax, etot)
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel(r'$\theta [\rm deg]$', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    
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
    color_len = args.sub_split[ax_num] if args.sub_split is not None else len(args.filename)
    if args.cmap == 'grayscale':
        colors = plt.cm.gray(np.linspace(0.05, 0.75, color_len+1))
    else:
        colors = plt.cm.viridis(np.linspace(0.25, 0.75, color_len+1))

    lw = 3.0 + 1.5*(case // 2)
    def calc_1d_hist(fields):
        dV_1d    = calc_cell_volume1D(fields['r'])
        
        if args.kinetic:
            W        = calc_lorentz_gamma(fields)
            mass     = dV_1d * fields['rho'] * W
            var      = (W - 1.0) * mass * e_scale.value # Kinetic Energy in [erg]
        elif args.mass:
            W        = calc_lorentz_gamma(fields)
            mass     = dV_1d * fields['rho'] * W
            var      = mass * m.value            # Mass in [g]
        elif args.enthalpy:
            h   = calc_enthalpy(fields)
            var = (h - 1.0) * e_scale.value      # Specific Enthalpy in [erg]
        elif args.dm_du:
            u   = fields['gamma_beta']
            var = u* fields['rho'] * dV_1d   / (1 + u**2)**0.5 
        else:
            edens_1d  = prims2var(fields, 'energy')
            var       = edens_1d * dV_1d * e_scale.value          # Total Energy in [erg]
            
        u1d       = fields['gamma_beta']
        gbs_1d    = np.logspace(np.log10(1.e-3), np.log10(u1d.max()), 128)
        var       = np.asarray([var[np.where(u1d > gb)].sum() for gb in gbs_1d])
        
        label = r'$\epsilon = 0$'
        if args.labels is not None:
            if len(args.labels) == len(args.filename) and not args.sub_split:
                etot         = np.sum(prims2var(fields, "energy") * dV_1d * e_scale.value)
                order_of_mag = np.floor(np.log10(etot))
                scale        = int(etot / 1e51)
                front_factor = int(etot / 10**order_of_mag)
                if front_factor != 1 or scale != 1:
                    label = r"${}E_{{51}}$".format(scale) + f"({label})"     
                else:
                    label = r"$E_{51}$" + f"({label})" 
                
        if args.norm:
            var /= var.max()
            fill_below_intersec(gbs_1d, var, 1e-6, colors[0])
            
        ax.hist(gbs_1d, bins=gbs_1d, weights=var, alpha=0.8, label= label,
                color=colors[0], histtype='step', linewidth=lw)
        
        
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)
    
    tend        = dset['time']
    theta       = mesh['theta']
    r           = mesh['rr']
    dV          = calc_cell_volume(r, theta)
    
    if args.kinetic:
        W    = calc_lorentz_gamma(fields)
        mass = dV * fields['rho'] * W
        var  = (W - 1.0) * mass * e_scale.value
    elif args.enthalpy:
        h   = calc_enthalpy(fields)
        var = (h - 1.0) *  dV * e_scale.value
    elif args.mass:
        W   = calc_lorentz_gamma(fields)
        var = dV * fields['rho'] * W * m.value
    elif args.dm_du:
        u   = fields['gamma_beta']
        var = u * fields['rho'] * dV / (1 + u**2)**0.5 * m.value
    else:
        var = prims2var(fields, "energy") * dV * e_scale.value

    # Create 4-Velocity bins as well as the Y-value bins directly
    u         = fields['gamma_beta']
    gbs       = np.logspace(np.log10(1.e-3), np.log10(u.max()), 128)
    var       = np.asarray([var[u > gb].sum() for gb in gbs]) 
    
    # if case == 0:
    #     oned_field   = get_1d_equiv_file(args.oned_files[0])
    #     calc_1d_hist(oned_field)
    # if case == 2:
    #     oned_field   = get_1d_equiv_file(args.oned_files[1])
    #     calc_1d_hist(oned_field)
    if ax_col == 0:     
        if args.anot_loc is not None:
            etot = np.sum(prims2var(fields, "energy") * dV * e_scale.value)
            place_anotation(args, fields, ax, etot)
        
        #1D Comparison 
        if args.oned_files is not None:
            if args.sub_split is None:
                for file in args.oned_files:
                    oned_field   = get_1d_equiv_file(file)
                    calc_1d_hist(oned_field)
            else:
                oned_field = get_1d_equiv_file(args.oned_files[ax_num])
                calc_1d_hist(oned_field)

    if args.norm:
        var /= var.max()

    if args.labels is not None:
        if args.tex:
            label = '$\%s$'%(args.labels[case])
        else:
            label = '%s'%(args.labels[case])
            
        if len(args.labels) == len(args.filename) and not args.sub_split:
            etot         = np.sum(prims2var(fields, "energy") * dV * e_scale.value)
            order_of_mag = np.floor(np.log10(etot))
            scale        = int(etot / 1e51)
            front_factor = int(etot / 10**order_of_mag)
            if front_factor != 1 or scale != 1:
                label = r"${}E_{{51}}$".format(scale) + "(%s)"%(label)     
            else:
                label = r"$E_{51}$" + "(%s)"%(label)  
    else:
        label = None
    
    c = colors[case + 1]
    # if case % 2 == 0:
    #     c = colors[1]
    # else:
    #     c = colors[-1]
    ax.hist(gbs, bins=gbs, weights=var, label=label, histtype='step', 
            rwidth=1.0, linewidth=lw, color=c, alpha=0.9)
    
    if args.fill_scale is not None:
        fill_below_intersec(gbs, var, args.fill_scale*var.max(), colors[case])
                    
    ax.set_xscale('log')
    ax.set_yscale('log')

    if args.xlims is None:
        ax.set_xlim(1e-3, 1e2)
    else:
        ax.set_xlim(args.xlims[0], args.xlims[1])
    
    if args.ylims is None:
        if args.mass:
            ax.set_ylim(1e-3*var.max(), 10.0*var.max())
        else:
            ax.set_ylim(1e-9*var.max(), 10.0*var.max())
    else:
        ax.set_ylim(args.ylims[0],args.ylims[1])
        
    if args.sub_split is None:
        ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
        if args.kinetic:
            ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
        elif args.enthalpy:
            ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
        elif args.dm_du:
            ax.set_ylabel(r'$dM/d\Gamma\beta ( > \Gamma \beta) \ [\rm{g}]$', fontsize=20)
        elif args.mass:
            ax.set_ylabel(r'$M(> \Gamma \beta) \ [\rm{g}]$', fontsize=20)
        else:
            ax.set_ylabel(r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
            
        ax.tick_params('both', labelsize=15)
    else:        
        ax.tick_params('x', labelsize=15)
        ax.tick_params('y', labelsize=10)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if args.setup != "":
        ax.set_title(r'{}, t ={:.2f} s'.format(args.setup, tend), fontsize=20)
    
    if overplot and args.sub_split is None:
        if args.labels is not None:
                ax.legend(fontsize=15, loc=args.legend_loc)
        

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
    
    if not overplot:
        if 0 in args.cutoffs:
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
        edens_1d = prims2var(ofield, 'energy')
        dV_1d    = calc_cell_volume1D(ofield['r'])
        mass     = dV_1d * ofield['rho'] * ofield['W']
        e_k      = (ofield['W'] - 1.0) * mass * e_scale.value
        etotal_1d = edens_1d * dV_1d * e_scale.value
        
        if args.kinetic:
            var = e_k
        elif args.dm_domega:
            var = mass * m.value
        else:
            var = etotal_1d
        
        for cutoff in args.cutoffs:
            total_var = sum(var[ofield['gamma_beta'] > cutoff])
            print(f"1D var sum with GB > {cutoff}: {total_var:.2e}")
            ax.axhline(total_var, linestyle='--', color='black', label='$\epsilon = 0$')
                
    def de_domega(var, gamma_beta, gamma_beta_cut, tz, domega, bin_edges):
        var = var.copy()
        var[gamma_beta < gamma_beta_cut] = 0.0
        de = np.hist(tz, weights=energy, bins=theta_bin_edges)
        dw = np.hist(tz, weights=domega, bins=theta_bin_edges)
        return de / dwplot_dx_d
    
    col       = case % len(args.sub_split) if args.sub_split is not None else case
    color_len = len(args.sub_split) if args.sub_split is not None else len(args.filename)
    colors    = plt.cm.viridis(np.linspace(0.1, 0.80, color_len if color_len > 1 else len(args.cutoffs)))
    coloriter = cycle(colors)
    
    tend        = dset['time']
    theta       = mesh['theta']
    tv          = compute_theta_verticies(theta)
    r           = mesh['rr']
    dV          = calc_cell_volume(r, theta)
    
    if ax_col == 0:
        if args.anot_loc is not None:
            etot = np.sum(prims2var(fields, "energy") * dV * e_scale.value)
            place_anotation(args, fields, ax, etot)
            
        #1D Comparison 
        if args.oned_files is not None:
            if args.sub_split is None:
                for file in args.oned_files:
                    oned_field   = get_1d_equiv_file(file)
                    calc_1d_dx_domega(oned_field)
            else:
                oned_field   = get_1d_equiv_file(args.oned_files[ax_num%len(args.oned_files)])
                calc_1d_dx_domega(oned_field)  
                
    if args.de_domega:
        if args.kinetic:
            W = calc_lorentz_gamma(fields)
            mass   = dV * fields['rho'] * W
            var = (W - 1.0) * mass * e_scale.value
        elif args.enthalpy:
            h   = calc_enthalpy(fields)
            var = (h - 1.0) *  dV * e_scale.value
        elif 'temperature' in args.field:
            var = prims2var(fields, 'temperature')
        else:
            edens_total = prims2var(fields, 'energy')
            var = edens_total * dV * e_scale.value
    elif args.dm_domega:
        var = dV * fields['rho'] * m.value
    
    gb      = fields['gamma_beta']
    tcenter = 0.5 * (tv[1:] + tv[:-1])
    dtheta  = (theta[-1,0] - theta[0,0])/theta.shape[0] * (180 / np.pi)
    domega  = 2.0 * np.pi * np.sin(tcenter) *(tv[1:] - tv[:-1])

    # Create inset of width 1.3 inches and height 0.9 inches
    # at the default upper right location
    if args.inset:
        axins = inset_axes(ax, width="25%", height="30%",loc='upper left', borderpad=3.25)
    if args.dec_rad:
        ax2 = ax.twinx()
        ax2.tick_params('y', labelsize=15)
        ax2.spines['top'].set_visible(False)
        
    lw = 1
    for cidx, cutoff in enumerate(args.cutoffs):
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
            quant_func     = np.sum if args.field[0] != 'temperature' else np.mean
            iso_correction = 4.0 * np.pi if dm_domega else 1.0
            var_per_theta  = iso_correction * quant_func(var, axis=1) / domega[:,0]
            
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
                
            if args.norm:
                var_per_theta /= var_per_theta.max()
            
            axes = ax if cutoff != 0 else ax0
            linestyle = '-' if cutoff != 0 else '--'
            if cutoff == 0:
                ax0_ylims = [var_per_theta.min(), var_per_theta.max()]
            if args.cmap == 'grayscale':
                axes.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw, label=label, linestyle=linestyle)
            else:
                axes.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw, label=label, color=colors[cidx], linestyle=linestyle)

            # window = 50
            # mean_theta = running_mean(theta[:,0], window)
            # mean_vpt   = running_mean(var_per_theta, window)
            # ax.plot(np.rad2deg(mean_theta), mean_vpt, lw=lw, color=colors[cidx], linestyle='--')
            # if args.dec_rad:
            #     r = calc_dec_rad(cutoff, var_per_theta)
            #     ax2.plot(np.rad2deg(theta[:, 0]), r, linestyle='--', color='black')
                
            if args.inset:
                axins.plot(np.rad2deg(theta[:, 0]), var_per_theta, lw=lw)
            lw *= 1.5
    
    if args.dec_rad:
        ax2.set_ylabel(r'$r_{\rm dec} [\rm{cm}]$', fontsize=15)  # we already handled the x-label with ax
        
    if args.xlims is None:
        ax.set_xlim(np.rad2deg(theta[0,0]), np.rad2deg(theta[-1,0]))
    else:
        ax.set_xlim(args.xlims[0], args.xlims[1])
    if args.inset:
        axins.set_xlim(80,100)
    
    if args.ylims is not None:
        ax.set_ylim(args.ylims[0], args.ylims[1])
        
        if args.dec_rad:
            pass
        if args.inset:
            axins.set_ylim(args.ylims[0],args.ylims[1])
    
        # axins.set_xticklabels([])
        # axins.set_yticklabels([])
    if args.sub_split is None:
        ax.set_xlabel(r'$\theta [\rm deg]$', fontsize=20)
        if 'ax0' in locals():
            if args.kinetic:
                fig.text(0.030, 0.5, r'$E_{\rm K, \Omega}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=15, va='center', rotation='vertical')
            elif args.dm_domega:
                fig.text(0.010, 0.5, r'$M_{\theta}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=15, va='center', rotation='vertical')
            else:
                fig.text(0.010, 0.5, r'$E_{\rm T, iso}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=15, va='center', rotation='vertical')
        else:
            if len(args.cutoffs) == 1:
                if args.kinetic:
                    ax.set_ylabel(r'$E_{{\rm K, iso}} \ (\Gamma \beta > {})\ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=15)
                elif args.enthalpy:
                    ax.set_ylabel(r'$H_{\rm iso} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=15)
                elif args.dm_domega:
                    ax.set_ylabel(r'$M_{\rm{iso}} \ (\Gamma \beta > {}) \ [\rm{{g}}]$'.format(args.cutoffs[0]), fontsize=15)
                elif args.field[0] == 'temperature':
                    ax.set_ylabel(r'$\bar{T}_{\rm{iso}} \ (\Gamma \beta > {}) \ [\rm{{eV}}]$'.format(args.cutoffs[0]), fontsize=15)
                else:
                    ax.set_ylabel(r'$E_{{\rm T, iso}} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=15)
            else:
                units = r'[\rm{{erg}}]' if not args.norm else ''
                if not args.dm_domega:
                    if args.kinetic:
                        ax.set_ylabel(r'$E_{{\rm K, iso}} \ (> \Gamma \beta)\ %s$'%(units), fontsize=15)
                    elif args.enthalpy:
                        ax.set_ylabel(r'$H_{\rm iso} \ (>\Gamma \beta) \ %s$'%(units), fontsize=15)
                    else:
                        ax.set_ylabel(r'$E_{{\rm T, iso}} \ (>\Gamma \beta) \ %s$'%(units), fontsize=15)
                elif args.dm_domega:
                    units = r'[\rm{{g}}]' if not args.norm else ''
                    ax.set_ylabel(r'$M_{\Omega} \ (>\Gamma \beta) \ %s$'%(units), fontsize=15)
                elif args.field[0] == 'temperature':
                    ax.set_ylabel(r'$\bar{T}_{\rm{iso}} \ (>\Gamma \beta) \ [\rm{{eV}}]$', fontsize=15)
        
        ax.tick_params('both', labelsize=15)
    else:
        ax.tick_params('x', labelsize=15)
        ax.tick_params('y', labelsize=10)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    if args.inset:
        axins.spines['right'].set_visible(False)
        axins.spines['top'].set_visible(False)

    if args.setup != "":
        ax.set_title(r'{}, t ={:.2f}'.format(args.setup, tend), fontsize=20)
    if args.log:
        ax.set_yscale('log')
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        # ax0.set_yscale('log')
        if 'ax0' in locals():
            # ax0.set_yscale('log')
            ax0.spines['top'].set_visible(False)
            ax0.spines['right'].set_visible(False)
            ax0.spines['bottom'].set_linewidth(2)
            # ax0.axes.get_xaxis().set_ticks([])
            ax0.set_ylim(ax0_ylims)
            plt.subplots_adjust(hspace=0)
            ax0.yaxis.set_minor_locator(plt.MaxNLocator(2))
            ax0.tick_params('both',which='major', labelsize=15)
            #ax.set_xlim(theta[0,0], theta[-1,0])
            
        if args.inset:
            axins.set_yscale('log')
        if args.dec_rad:
            ax2.set_yscale('log')
            ticks = calc_dec_rad(0.0, np.array(ax.axes.get_ylim())) 
            ax2.set_ylim(ticks[0],ticks[-1])
            
        
        
        # ax0.yaxis.set_major_formatter(logfmt)
        # ax0.set_ylim(1e52,5e52)
        
    if not args.sub_split:
        if args.labels:
            ax.legend(fontsize=15, loc=args.legend_loc)
            if 'ax0' in locals():
                ax0.legend(fontsize=15, loc='best')
                
def plot_ligthcurve(
    fields:        dict, 
    args:          argparse.ArgumentParser, 
    mesh:          dict, 
    dset:          dict, 
    points:        np.ndarray,
    overplot:      bool=False, 
    subplot:       bool=False, 
    ax:            Union[None,plt.Axes]=None, 
    case:          int=0, 
    ax_col:        int=0,
    ax_num:        int=0) -> None:
        
    if not overplot:
        fig, ax = plt.subplots(1,1,figsize=(9,9))
        
    eb      = 0.1 
    ec      = 0.1
    tday    = dset['time'].to(u.day)
    dV      = calc_cell_volume(mesh['rr'], mesh['theta'])
    etot    = 1e51 * u.erg # np.sum(prims2var(fields, "energy") * dV * e_scale)
    r       = mesh['rr'] * R_0
    R3      = 0.65 * R_0
    mdot    = (1e-6 * u.M_sun/u.yr).to(u.g/u.s)
    vw      = 1e8 * u.cm/u.s
    nwind   = mdot / (4 * np.pi * vw * r**2) / const.m_p.cgs
    e52     = etot.to(u.erg) / (1e52 * u.erg)
    td      = tday / u.day 
    eb_m1   = eb / 0.1 
    ec_m1   = ec / 0.1 
    d_28    = (r / (1e28 * u.cm))
    n1      = nwind / (1 * u.cm**(-3))
    
    # print(td)
    # print(e52)
    # zzz = input('')
    
    nu_c_sphere = 2.70e12 * eb_m1**(-3/2) * e52 **(-1/2) * n1**(-1)  * td  **(-1/2) * u.Hz
    nu_m_sphere = 5.70e14 * eb_m1**( 1/2) * e52 **( 1/2) * ec_m1**2  * td  **(-3/2) * u.Hz
    fnu_max     = 1.1e5   * eb_m1**( 1/2) * e52  *         n1**(1/2) * d_28**(-2)  

    p = 2.5
    def flux(nu, tidx, r_shock):
        nu_c     = nu_c_sphere[tidx][r_shock]
        nu_m     = nu_m_sphere
        
        gtr_crit = nu > nu_c 
        between  = (nu < nu_m) & (nu > nu_c) 
        gtr_sync = nu > nu_c
        
        if nu < nu_c:
            print("1")
            zzz = input('')
            return (nu/nu_c)**( 1/3)*fnu_max[tidx][r_shock] 
        elif (nu < nu_m) & (nu > nu_c):
            return (nu/nu_c) **(-1/2)*fnu_max[tidx][r_shock]
        elif nu > nu_m:
            print("3")
            zzz = input('')
            return (nu_m/nu_c)**(-1/2)*(nu/nu_m)**(-p/2)*fnu_max[tidx][r_shock]     
        # fnu[gtr_crit] = (nu[gtr_crit]/nu_c[gtr_crit])**( 1/3)*fnu_max[tidx][gtr_crit] 
        # fnu[between]  = (nu[between]/nu_c[between]) **(-1/2)*fnu_max[tidx][between]
        # fnu[gtr_sync] = (nu_m/nu_c[gtr_sync])**(-1/2)*(nu[gtr_sync]/nu_m)**(-p/2)*fnu_max[tidx][gtr_sync]
        # return fnu

    theta1    = 0
    theta2    = int(mesh['theta'].shape[0]/2)
    r_shock   = np.argmax(fields['gamma_beta'][theta2])
    wshock    = calc_lorentz_gamma(fields)[theta2][r_shock]
    b_field   = (32 * eb * nwind[theta2][r_shock] * const.m_p.cgs)**0.5 * wshock * const.c.cgs 
    nu        = 2.80e6 * b_field.value * wshock**2 * u.Hz
    
    f       = flux(nu, theta1, r_shock)
    f2      = flux(nu, theta2, r_shock)
    points += [[nu.value, f2]] 
    
    if args.log:
        ax.set_yscale('log')
        ax.set_xscale('log')
        
    ax.set_xlabel(r'$\nu [\rm{Hz}]$', fontsize=15)
    ax.set_ylabel(r'Flux $[\mu \rm{J}]$', fontsize=15)
    # ax.legend()
    
    
def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog='This Only Supports H5 Files Right Now')
    
    parser.add_argument('filename', metavar='Filename', nargs='+',
                        help='A Data Source to Be Plotted')
    
    parser.add_argument('setup', metavar='Setup', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = 'field', metavar='Field Variable', nargs='+',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default=['rho'])
    
    parser.add_argument('--1d_files', dest='oned_files', nargs='+', help='1D files to check against', default=None)
    
    parser.add_argument('--rmax', dest = 'rmax', metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    
    parser.add_argument('--cbar_range', dest = 'cbar', metavar='Range of Color Bar', nargs='+',
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cbar_sub', dest = 'cbar2', metavar='Range of Color Bar for secondary plot',nargs='+',type=float,
                        default =[None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--no_cbar', dest ='no_cbar',action='store_true',
                        default=False, help='colobar visible siwtch')
    
    parser.add_argument('--cmap', dest ='cmap', metavar='Color Bar Colarmap',
                        default = 'magma', help='The colorbar cmap you\'d like to plot')
    parser.add_argument('--cmap2', dest ='cmap2', metavar='Color Bar Colarmap 2',
                        default = 'magma', help='The secondary colorbar cmap you\'d like to plot')
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')
    
    parser.add_argument('--rev_cmap', dest='rcmap', action='store_true',
                        default=False,
                        help='True if you want the colormap to be reversed')
    
    parser.add_argument('--x', dest='x', nargs='+', default = None, type=float,
                        help='List of x values to plot field max against')
    
    parser.add_argument('--xlabel', dest='xlabel', nargs=1, default = 'X',
                        help='X label name')
    
    parser.add_argument('--kinetic', dest='kinetic', action='store_true', default=False,
                        help='Plot the kinetic energy on the histogram')
    
    parser.add_argument('--enthalpy', dest='enthalpy', action='store_true',
                        default=False,
                        help='Plot the enthalpy on the histogram')
    
    parser.add_argument('--hist', dest='hist', action='store_true',
                        default=False,
                        help='Convert plot to histogram')
    
    parser.add_argument('--mass', dest='mass', action='store_true',
                        default=False,
                        help='Compute mass histogram')
    
    parser.add_argument('--dm_du', dest='dm_du', default = False, action='store_true',
                        help='Compute dM/dU over whole domain')
    
    parser.add_argument('--de_domega', dest='de_domega', action='store_true',
                        default=False,
                        help='Plot the dE/dOmega plot')
    
    parser.add_argument('--dm_domega', dest='dm_domega', action='store_true',
                        default=False,
                        help='Plot the dM/dOmega plot')
    
    parser.add_argument('--dec_rad', dest='dec_rad', default = False, action='store_true',
                        help='Compute dr as function of angle')
    
    parser.add_argument('--cutoffs', dest='cutoffs', default=[0.0], type=float, nargs='+',
                        help='The 4-velocity cutoff value for the dE/dOmega plot')
    
    parser.add_argument('--fill_scale', dest ='fill_scale', metavar='Filler maximum', type=float,
                        default = None, help='Set the y-scale to start plt.fill_between')
    
    parser.add_argument('--ax_anchor', dest='ax_anchor', type=str, nargs='+', default=None, 
                        help='Anchor annotation text for each plot')
    
    parser.add_argument('--norm', dest='norm', action='store_true',
                        default=False, help='True if you want the plot normalized to max value')
    
    parser.add_argument('--labels', dest='labels', nargs='+', default = None,
                        help='Optionally give a list of labels for multi-file plotting')
    
    parser.add_argument('--tidx', dest='tidx', type=int, default = None,
                        help='Set to a value if you wish to plot a 1D curve about some angle')
    
    parser.add_argument('--nwedge', dest='nwedge', default=0, type=int)
    parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 70, 110], type=float, nargs=4)
    parser.add_argument('--xlims', dest='xlims', default = None, type=float, nargs=2)
    parser.add_argument('--ylims', dest='ylims', default = None, type=float, nargs=2)
    parser.add_argument('--units', dest='units', default = False, action='store_true')
    parser.add_argument('--dbg', dest='dbg', default = False, action='store_true')
    parser.add_argument('--tex', dest='tex', default = False, action='store_true')
    parser.add_argument('--bipolar', dest='bipolar', default = False, action='store_true')
    parser.add_argument('--pictorial', dest='pictorial', default = False, action='store_true')
    parser.add_argument('--subplots', dest='subplots', default = None, type=int)
    parser.add_argument('--sub_split', dest='sub_split', default = None, nargs='+', type=int)
    parser.add_argument('--anot_loc', dest='anot_loc', default = None, type=str)
    parser.add_argument('--legend_loc', dest='legend_loc', default = None, type=str)
    parser.add_argument('--anot_text', dest='anot_text', default = None, type=str)
    parser.add_argument('--inset', dest='inset', action= 'store_true', default=False)
    parser.add_argument('--png', dest='png', action= 'store_true', default=False)
    parser.add_argument('--tau_s', dest='tau_s', action= 'store_true', default=False, 
                        help='The shock optical depth')
    parser.add_argument('--light_curve', dest='light_curve', action='store_true',default=False)
    
    parser.add_argument('--save', dest='save', type=str,
                        default=None,
                        help='Save the fig with some name')

    args = parser.parse_args()
    vmin, vmax = args.cbar[:2]
    fields = {}
    setup = {}
    
    if args.cmap == 'grayscale':
        plt.style.use('grayscale')
    else:
        plt.style.use('seaborn-colorblind')
    
    if args.dbg:
        plt.style.use('dark_background')
        
    
    num_subplots   = len(args.sub_split) if args.sub_split is not None else 1
    if len(args.filename) > 1:
        if num_subplots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            lines_per_plot = len(args.filename)
        else:
            fig,axs = plt.subplots(num_subplots, 1, figsize=(8,4 * num_subplots), sharex=True, tight_layout=False)
            if args.setup != "":
                fig.suptitle(f'{args.setup}')
            if args.de_domega or args.dm_domega:
                axs[-1].set_xlabel(r'$\theta \ \rm[deg]$', fontsize=20)
            else:
                axs[-1].set_xlabel(r'$\Gamma \beta$', fontsize=20)
            if args.de_domega or args.dm_domega:
                if len(args.cuotff) == 1:
                    if args.kinetic:
                        fig.text(0.030, 0.5, r'$E_{{\rm K, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=20, va='center', rotation='vertical')
                    elif args.dm_domega:
                        fig.text(0.030, 0.5, r'$M_{{\rm iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=20, va='center', rotation='vertical')
                    else:
                        fig.text(0.030, 0.5, r'$E_{{\rm T, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoffs[0]), fontsize=20, va='center', rotation='vertical')
            else:
                units = r"[\rm{erg}]" if not args.norm else ""
                if args.kinetic:
                    fig.text(0.030, 0.5, r'$E_{\rm K}( > \Gamma \beta) \ %s$'%(units), fontsize=20, va='center', rotation='vertical')
                elif args.enthalpy:
                    fig.text(0.030, 0.5, r'$H( > \Gamma \beta) \ %s$'%(units), fontsize=20, va='center', rotation='vertical')
                else:
                    fig.text(0.030, 0.5, r'$E_{\rm T}( > \Gamma \beta) \ %s$'%(units), fontsize=20, va='center', rotation='vertical')
            axs_iter       = iter(axs)            # iterators for the multi-plot
            subplot_iter   = iter(args.sub_split) # iterators for the subplot splitting
            
            # first elements of subplot iterator tells how many files belong on axs[0]
            lines_per_plot = next(subplot_iter)   
        
        i        = 0       
        ax_col   = 0
        ax_shift = True
        ax_num   = 0    
        
        if args.light_curve:
            points = [] 
            t = []
        for idx, file in enumerate(args.filename):
            fields, setup, mesh = get_2d_file(args, file)
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
            elif args.light_curve:
                plot_ligthcurve(fields, args, mesh, setup,points, overplot=True, ax=ax)
                t += [setup['time'].value]
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
                
        if args.light_curve:
            points = np.asarray(points)
            ax.plot(points[:,0], points[:,1])
        if args.sub_split is not None:
            for ax in axs:
                ax.label_outer()
    else:
        fields, setup, mesh = get_2d_file(args, args.filename[0])
        if args.hist and (not args.de_domega and not args.dm_domega):
            plot_hist(fields, args, mesh, setup)
        elif args.tidx != None:
            plot_1d_curve(fields, args, mesh, setup)
        elif args.de_domega or args.dm_domega:
            plot_dx_domega(fields, args, mesh, setup)
        elif args.x is not None:
            plot_per_theta(fields, args, mesh, setup, overplot=False)
        elif args.dec_rad:
            plot_dec_rad(fields, args, mesh, setup, overplot=False)
        elif args.light_curve:
            plot_ligthcurve(fields, args, mesh, setup)
        else:
            if setup['is_cartesian']:
                plot_cartesian_plot(fields, args, mesh, setup)
            else:
                plot_polar_plot(fields, args, mesh, setup)
    
    if args.sub_split is not None:
        if args.labels is not None:
                if not args.legend_loc:
                    axs[0].legend(fontsize=15, loc='upper right')
                else:
                    axs[0].legend(fontsize=15, loc=args.legend_loc)
        plt.subplots_adjust(hspace=0.1)

            
    if not args.save:
        plt.show()
    else:
        try:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                    "font.serif": "Times New Roman",
                }
            )
        except RuntimeError:
            pass 
        
        ext = 'pdf' if not args.png else 'png'
        plt.savefig('{}.{}'.format(args.save.replace(' ', '_'), ext), dpi=500, bbox_inches='tight')
    
if __name__ == '__main__':
    main()
