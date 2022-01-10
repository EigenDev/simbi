#! /usr/bin/env python

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const
import astropy.units as u 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Union

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

def find_nearest(arr: list, val: float) -> Union[int, float]:
    arr = np.asarray(arr)
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
    
def fill_below_intersec(x: np.ndarray, y: np.ndarray, constraint: float, color: float) -> None:
    ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
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
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        k = const.k_B.cgs.value
        m = const.m_p.cgs.value
        me = const.m_e.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        
        h = 1.0 + 4/3 * p / (rho * (4/3 - 1))
        
        ofield['rho']         = rho
        ofield['v']           = v
        ofield['p']           = p
        ofield['W']           = W
        ofield['enthalpy']    = h
        ofield['gamma_beta']  = W*beta
        ofield['temperature'] = T
        ofield['r']           = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
    return ofield
    
cons          = ['D', 'momentum', 'energy', 'energy_rst', 'enthalpy']
field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature', 'gamma_beta_1', 'gamma_beta_2', 'energy', 'mass', 'chi', 'chi_dens'] + cons 
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
    return (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3)

def calc_cell_volume(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    dcos = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    
    rvertices = np.sqrt(r[:, 1:] * r[:, :-1])
    rvertices = np.insert(rvertices,  0, r[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], r[:, -1], axis=1)
    dr = rvertices[:, 1:] - rvertices[:, :-1]
    
    theta_mean  = 0.5 * (tvertices[1:] + tvertices[:-1])
    dtheta      = tvertices[1:] - tvertices[:-1]
    return ( (1./3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos )
        
def get_field_str(args: argparse.ArgumentParser) -> str:
    field_str_list = []
    for field in args.field:
        if field == 'rho' or field == 'D':
            var = r'\rho' if field == 'rho' else 'D'
            if args.units:
                field_str_list.append( r'${}$ [g cm$^{{-3}}$]'.format(var))
            else:
                field_str_list.append( r'${}$'.format(var))
            
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
                    field_str_list.append( r'$\tau$')
                else:
                    field_str_list.append( r'$p$')
        elif field == 'energy_rst':
            if args.units:
                field_str_list.append( r'$\tau + D \  [\rm erg \ cm^{-3}]$')
            else:
                field_str_list.append( r'$\tau + D')
        elif field == 'chi':
            field_str_list.append( r'$\chi$')
        elif field == 'chi_dens':
            field_str_list.append( r'$D \cdot \chi$')
        else:
            field_str_list.append( field)
    
    
    return field_str_list if len(args.field) > 1 else field_str_list[0]
    
    
def prims2cons(fields: dict, cons: str) -> np.ndarray:
    if cons == 'D':
        # Lab frame density
        return fields['rho'] * fields['W']
    elif cons == 'S':
        # Lab frame momentum density
        return fields['rho'] * fields['W']**2 * fields['enthalpy'] * fields['v']
    elif cons == 'energy':
        # Energy minus rest mass energy
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']
    elif cons == 'energy_rst':
        # Total Energy
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p']
    elif cons =='enthalpy':
        # Specific enthalpy
        return fields['enthalpy'] - 1.0


def plot_polar_plot(
    field_dict: dict,                            # Field dict
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
    if not subplots:
        if args.wedge:
            fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                                figsize=(15, 10), constrained_layout=True)
            ax    = axes[0]
            wedge = axes[1]
        else:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                                figsize=(10, 8), constrained_layout=False)
    else:
        if args.wedge:
            ax    = axs[0]
            wedge = axs[1]
        else:
            ax = axs
        
    rr, tt      = mesh['rr'], mesh['theta']
    t2          = mesh['t2']
    xmax        = dset[0]['xmax']
    xmin        = dset[0]['xmin']
    ymax        = dset[0]['ymax']
    ymin        = dset[0]['ymin']
    
    vmin,vmax = args.cbar

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
        
    tend = dset[0]['time']
    # If plotting multiple fields on single polar projection, split the 
    # field projections into their own quadrants
    if num_fields > 1:
        cs  = np.zeros(4, dtype=object)
        var = []
        kwargs = []
        for field in args.field:
            if field in cons:
                if ymax == np.pi:
                    var += np.split(prims2cons(field_dict, field), 2)
                else:
                    var.append(prims2cons(field_dict, field))
            else:
                if ymax == np.pi:
                    var += np.split(field_dict[field], 2)
                else:
                    var.append(field_dict[field])
                
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
        quadr[field2] = var[3% num_fields]
        
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
                kwargs[field] =  {'vmin': vmin, 'vmax': vmax} if field in lin_fields else {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)} 
            else:
                if field == field3 == field4:
                    ovmin = quadr[field][0].min()
                    ovmax = quadr[field][0].max()
                else:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                kwargs[field] =  {'vmin': ovmin, 'vmax': ovmax} if field in lin_fields else {'norm': colors.LogNorm(vmin = ovmin, vmax = ovmax)} 

        if ymax < np.pi:
            cs[0] = ax.pcolormesh(tt[:: 1], rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
            cs[1] = ax.pcolormesh(t2[::-1], rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
            
            # If simulation only goes to pi/2, if bipolar flag is set, mirror the fields accross the equator
            if args.bipolar:
                cs[2] = ax.pcolormesh(tt[:: 1] + np.pi/2, rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
                cs[3] = ax.pcolormesh(t2[::-1] + np.pi/2, rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
        else:
            if num_fields == 2:
                cs[0] = ax.pcolormesh(tt[:: 1], rr,  np.vstack((var[0],var[1])), cmap=color_map, shading='auto', **kwargs[field1])
                cs[1] = ax.pcolormesh(t2[::-1], rr,  np.vstack((var[2],var[3])), cmap=color_map, shading='auto', **kwargs[field2])
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
            kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
        else:
            kwargs = {'vmin': vmin, 'vmax': vmax}
            
        cs = np.zeros(len(args.field), dtype=object)
        
        if args.field[0] in cons:
            var = units * prims2cons(field_dict, args.field[0])
        else:
            var = units * field_dict[args.field[0]]
            
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto', **kwargs)
        cs[0] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, shading='auto', **kwargs)
        
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
                cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
                cbar_orientation = 'horizontal'
                
            
        else:
            if num_fields > 1:
                if num_fields == 2:
                    ycoord  = [0.1, 0.08] if ymax < np.pi else [0.1, 0.1]
                    xcoord  = [0.1, 0.85] if ymax < np.pi else [0.87, 0.05]
                    cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                    
                if num_fields == 3:
                    ycoord  = [0.1, 0.5, 0.1]
                    xcoord  = [0.07, 0.85, 0.85]
                    cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8 * 0.5]) for i in range(1, num_fields)]
                    cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                if num_fields == 4:
                    ycoord  = [0.5, 0.1, 0.5, 0.1]
                    xcoord  = [0.85, 0.85, 0.07, 0.07]
                    cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
                    
                cbar_orientation = 'vertical'
            else:
                cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
                cbar_orientation = 'vertical'
        
        if args.log:
            if num_fields > 1:
                fmt  = [None if field in lin_fields else tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True) for field in args.field]
                cbar = [fig.colorbar(cs[i], orientation=cbar_orientation, cax=cbaxes[i], format=fmt[i]) for i in range(num_fields)]
                for cb in cbar:
                    cb.outline.set_visible(False)                                 
            else:
                logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
                cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes, format=logfmt)
        else:
            if num_fields > 1:
                cbar = [fig.colorbar(cs[i], orientation=cbar_orientation, cax=cbaxes[i]) for i in range(num_fields)]
            else:
                cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes)
        ax.yaxis.grid(True, alpha=0.05)
        ax.xaxis.grid(True, alpha=0.05)
    
    if args.wedge:
        wedge_min = args.wedge_lims[0]
        wedge_max = args.wedge_lims[1]
        ang_min   = args.wedge_lims[2]
        ang_max   = args.wedge_lims[3]
        
        # Draw the wedge cutout on the main plot
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color='white')
    
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='both', labelsize=10)
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
    
    if args.wedge:
        if num_fields == 1:
            wedge.set_position( [0.5, -0.5, 0.3, 2])
            ax.set_position( [0.05, -0.5, 0.46, 2])
        else:
            ax.set_position( [0.15, -0.5, 0.46, 2])
            wedge.set_position( [0.58, -0.5, 0.3, 2])
            
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
                    kwargs[field] =  {'vmin': vmin2, 'vmax': vmax2} if field in lin_fields else {'norm': colors.LogNorm(vmin = vmin2, vmax = vmax2)} 
                elif idx == 1:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                    kwargs[field] =  {'vmin': vmin3, 'vmax': vmax3} if field in lin_fields else {'norm': colors.LogNorm(vmin = vmin3, vmax = vmax3)} 
                else:
                    continue
                
            wedge.pcolormesh(tchop[0], rchop[0], quadr[field1], cmap=color_map, shading='nearest', **kwargs[field1])
            wedge.pcolormesh(tchop[1], rchop[1], quadr[field2], cmap=color_map, shading='nearest', **kwargs[field2])
        else:
            vmin2, vmax2 = args.cbar2
            if args.log:
                kwargs = {'norm': colors.LogNorm(vmin = vmin2, vmax = vmax2)}
            else:
                kwargs = {'vmin': vmin2, 'vmax': vmax2}
            wedge.pcolormesh(tt, rr, var, cmap=color_map, shading='nearest', **kwargs)
            
        wedge.set_theta_zero_location('N')
        wedge.set_theta_direction(-1)
        wedge.yaxis.grid(False)
        wedge.xaxis.grid(False)
        wedge.tick_params(axis='both', labelsize=6)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('white')
            
        wedge.axes.xaxis.set_ticklabels([])
        wedge.set_ylim([wedge_min, wedge_max])
        wedge.set_rorigin(-wedge_min)
        wedge.set_thetamin(ang_min)
        wedge.set_thetamax(ang_max)
        wedge.set_aspect(1.)
        
        
    if not args.pictorial:
        if args.log:
            if ymax == np.pi:
                if num_fields > 1:
                    for i in range(num_fields):
                        if args.field[i] in lin_fields:
                            cbar[i].ax.set_ylabel(r'{}'.format(field_str[i]), fontsize=20)
                        else:
                            cbar[i].ax.set_ylabel(r'$\log$ {}'.format(field_str[i]), fontsize=20)
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
                cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
            else:
                cbar.ax.set_xlabel(r'{}'.format(field_str), fontsize=20)
        if args.setup != "":
            fig.suptitle('{} at t = {:.2f}'.format(args.setup, tend), fontsize=20, y=1)

def plot_cartesian_plot(
    field_dict: dict, 
    args: argparse.ArgumentParser, 
    mesh: dict, 
    dset: dict) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10,10), constrained_layout=False)

    xx, yy = mesh['xx'], mesh['yy']
    xmax        = dset[0]['xmax']
    xmin        = dset[0]['xmin']
    ymax        = dset[0]['ymax']
    ymin        = dset[0]['ymin']
    
    vmin,vmax = args.cbar

    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = dset[0]['time']
    c = ax.pcolormesh(xx, yy, field_dict[args.field], cmap=color_map, shading='auto', **kwargs)
    
    divider = make_axes_locatable(ax)
    cbaxes  = divider.append_axes('right', size='5%', pad=0.05)
    
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
    field_dict: dict, 
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
    
    xmax        = dset[0]['xmax']
    xmin        = dset[0]['xmin']
    ymax        = dset[0]['ymax']
    ymin        = dset[0]['ymin']
    
    vmin,vmax = args.cbar
    var = [field for field in args.field] if num_fields > 1 else args.field[0]
    
    tend = dset[0]['time']
    if args.mass:
        dV          = calc_cell_volume(mesh['rr'], mesh['theta'])
        mass        = 2.0 * np.pi * dV * field_dict['W'] * field_dict['rho']
        # linestyle = '-.'
        if args.labels is None:
            ax.loglog(r, mass[args.tidx]/ np.max(mass[args.tidx]), label = 'mass', linestyle='-.', color=colors[case])
            ax.loglog(r, field_dict['p'][args.tidx] / np.max(field_dict['p'][args.tidx]), label = 'pressure', color=colors[case])
        else:
            ax.loglog(r, mass[args.tidx]/ np.max(mass[args.tidx]), label = f'{args.labels[case]} mass', linestyle='-.', color=colors[case])
            ax.loglog(r, field_dict['p'][args.tidx] / np.max(field_dict['p'][args.tidx]), label = f'{args.labels[case]} pressure', color=colors[case])
        ax.legend(fontsize=20)
        ax.axvline(0.65, linestyle='--', color='red')
        ax.axvline(1.00, linestyle='--', color='blue')
    else:
        for idx, field in enumerate(args.field):
            if field in cons:
                var = prims2cons(field_dict, field)
            else:
                var = field_dict[field]
                
            label = None if args.labels is None else '{}'.format(field_str[idx] if num_fields > 1 else field_str)
            ax.loglog(r, var[args.tidx], label=label)
    
    # ax.set_position( [0.1, -0.18, 0.8, 1.43])
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
    
def plot_max(
    fields:    dict, 
    args:      argparse.ArgumentParser, 
    mesh:      dict , 
    dset:      dict, 
    overplot:  bool=False, 
    ax:        bool=None, 
    case:      int =0) -> None:
    
    print('plotting max values along x...')
    
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['r'], mesh['th']
    theta    = theta * 180 / np.pi 
    
    xmax        = dset[0]['xmax']
    xmin        = dset[0]['xmin']
    ymax        = dset[0]['ymax']
    ymin        = dset[0]['ymin']
    
    vmin,vmax = args.cbar
    ofield = get_1d_equiv_file(16384)
    #1D test 
    tend = dset[0]['time']
    ax.set_title(r'{} at t={:.3f}'.format(args.setup, tend), fontsize=20)
    
    if args.field == 'gamma_beta':
        edens_total = prims2cons(fields, 'energy')
        theta       = mesh['theta']
        r           = mesh['rr']
        dV          = calc_cell_volume(r, theta)
        etotal      = edens_total * 2.0 * np.pi * dV * e_scale.value
        
        u = fields['gamma_beta']
        w = 0.001 #np.diff(u).max()*1e-1
        n = int(np.ceil( (u.max() - u.min() ) / w ) )
        gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), n)
        ets = np.asarray([etotal[np.where(u > gb)].sum() for gb in gbs])
        ets /= ets.max()
        expl_ind = find_nearest(ets, 1e-6)[0]
        ax.scatter(args.x[case], gbs[expl_ind])
    else:
        ax.scatter(args.x[case], np.max(fields[args.field]))
        
    if case == 0:
        #1D Check
        if args.oned_files is not None:
            for file in args.oned_files:
                ofield = get_1d_equiv_file(file)
                edens_1d = prims2cons(ofield, 'energy')
                dV_1d    = 4.0 * np.pi * calc_cell_volume1D(ofield['r'])
                etotal_1d = edens_1d * dV_1d * e_scale.value
                u1d       = ofield['gamma_beta']
                w = 0.001
                n = int(np.ceil( (u1d.max() - u1d.min() ) / w ) )
                gbs_1d = np.logspace(np.log10(1.e-4), np.log10(u1d.max()), n)
                ets_1d = np.asarray([etotal_1d[np.where(u1d > gb)].sum() for gb in gbs_1d])
                ets_1d /= ets_1d.max()
                expl_ind = find_nearest(ets_1d, 1e-6)[0]
                
                ax.scatter(0.0, gbs_1d[expl_ind])
    
        
    
    ax.set_xlabel(f'{args.xlabel[0]}', fontsize=20)
    ax.tick_params(axis='both', labelsize=8)
    
    # Change the format of the field
    field_str = get_field_str(args)
    
    if args.log:
        ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
    if not overplot:
        return fig
    
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
    
    def calc_1d_hist(fields):
        dV_1d    = 4.0 * np.pi * calc_cell_volume1D(fields['r'])
        
        if args.kinetic:
            mass     = dV_1d * fields['rho'] * fields['W']
            var      = (fields['W'] - 1.0) * mass * e_scale.value # Kinetic Energy in [erg]
        elif args.mass:
            var = mass * m.value                                  # Mass in [g]
        elif args.enthalpy:
            var = (fields['enthalpy'] - 1.0) * e_scale.value      # Specific Enthalpy in [erg]
        else:
            edens_1d  = prims2cons(fields, 'energy')
            var       = edens_1d * dV_1d * e_scale.value          # Total Energy in [erg]
            
        u1d       = fields['gamma_beta']
        gbs_1d    = np.logspace(np.log10(1.e-3), np.log10(u1d.max()), 128)
        var       = np.asarray([var[np.where(u1d > gb)].sum() for gb in gbs_1d])
        if args.norm:
            var /= var.max()
            fill_below_intersec(gbs_1d, var, 1e-6, colors[0])
            
        ax.hist(gbs_1d, bins=gbs_1d, weights=var, alpha=0.8, label= r'Sphere', histtype='step', linewidth=3.0)
        
        
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)
        
    tend        = dset[case]['time']
    edens_total = prims2cons(fields, 'energy')
    theta       = mesh['theta']
    r           = mesh['rr']
    dV          = calc_cell_volume(r, theta)
    
    
    if args.kinetic:
        mass = 2.0 * np.pi * dV * fields['rho'] * fields['W']
        var  = (fields['W'] - 1.0) * mass * e_scale.value
    elif args.enthalpy:
        var = (fields['enthalpy'] - 1.0) *  2.0 * np.pi * dV * e_scale.value
    elif args.mass:
        var = 2.0 * np.pi * dV * fields['rho'] * fields['W'] * m.value
    else:
        var = edens_total * 2.0 * np.pi * dV * e_scale.value

    # Check if subplots are split amonst the file inputs. If so, roll the colors
    # to reset when on a different axes object
    col       = case % len(args.sub_split) if args.sub_split is not None else case
    color_len = len(args.sub_split) if args.sub_split is not None else len(args.filename)
    colors    = plt.cm.twilight_shifted(np.linspace(0.1, 0.8, color_len))
    
    # Create 4-Velocity bins as well as the Y-value bins directly
    u         = fields['gamma_beta']
    gbs       = np.logspace(np.log10(1.e-3), np.log10(u.max()), 128)
    var       = np.asarray([var[u > gb].sum() for gb in gbs]) 
    
    
    if args.norm:
        var /= var.max()

    label = None if args.labels is None else r'${}$'.format(args.labels[case])
    ax.hist(gbs, bins=gbs, weights=var, label=label, histtype='step', rwidth=1.0, linewidth=3.0, color=colors[col], alpha=0.7)
    
    if args.fill_scale is not None:
        fill_below_intersec(gbs, var, args.fill_scale*var.max(), colors[case])

    if ax_col == 0:
        #1D Comparison 
        if args.oned_files is not None:
            if args.sub_split is None:
                for file in args.oned_files:
                    oned_field   = get_1d_equiv_file(file)
                    calc_1d_hist(oned_field)
            else:
                oned_field   = get_1d_equiv_file(args.oned_files[ax_num])
                calc_1d_hist(oned_field)
                    
    
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(1e-3, 1e2)
    if args.mass:
        ax.set_ylim(1e-3*var.max(), 10.0*var.max())
    else:
        ax.set_ylim(1e-9*var.max(), 10.0*var.max())
        
    if args.sub_split is None:
        ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
        if args.kinetic:
            ax.set_ylabel(r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
        elif args.enthalpy:
            ax.set_ylabel(r'$H ( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
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
        
    if not overplot:
        return fig

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
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)
    
    def calc_1d_dx_domega(ofield: dict) -> None:
        edens_1d = prims2cons(ofield, 'energy')
        dV_1d    = 4.0 * np.pi * calc_cell_volume1D(ofield['r'])
        mass     = dV_1d * ofield['rho'] * ofield['W']
        e_k      = (ofield['W'] - 1.0) * mass * e_scale.value
        etotal_1d = edens_1d * dV_1d * e_scale.value
        
        if args.kinetic:
            var = e_k
        elif args.dm_domega:
            var = mass * m.value
        else:
            var = etotal_1d
        
        total_var = sum(var[ofield['gamma_beta'] > args.cutoff])
        print(f"1D var sum with GB > {args.cutoff}: {total_var}")
        ax.axhline(total_var, linestyle='--', color='black')
                
    def de_domega(var, gamma_beta, gamma_beta_cut, tz, domega, bin_edges):
        var = var.copy()
        var[gamma_beta < gamma_beta_cut] = 0.0
        de = np.hist(tz, weights=energy, bins=theta_bin_edges)
        dw = np.hist(tz, weights=domega, bins=theta_bin_edges)
        return de / dw
    
    color_len = len(args.sub_split) if args.sub_split is not None else len(args.filename)
    colors    = plt.cm.plasma(np.linspace(0.1, 0.8, color_len))
    
    tend        = dset[case]['time']
    theta       = mesh['theta']
    tv          = compute_theta_verticies(theta)
    r           = mesh['rr']
    dV          = calc_cell_volume(r, theta)
    
    if args.de_domega:
        if args.kinetic:
            mass   = 2.0 * np.pi * dV * fields['rho'] * fields['W']
            var = (fields['W'] - 1.0) * mass * e_scale.value
        elif args.enthalpy:
            var = (fields['enthalpy'] - 1.0) *  2.0 * np.pi * dV * e_scale.value
        else:
            edens_total = prims2cons(fields, 'energy')
            var = edens_total * 2.0 * np.pi * dV * e_scale.value
    elif args.dm_domega:
        var = 2.0 * np.pi * dV * fields['rho'] * m.value

    col       = case % len(args.sub_split) if args.sub_split is not None else case
    color_len = len(args.sub_split) if args.sub_split is not None else len(args.filename)
    colors    = plt.cm.twilight_shifted(np.linspace(0.1, 0.8, color_len))
    
    u                          = fields['gamma_beta']
    tcenter                    = 0.5 * (tv[1:] + tv[:-1])
    dtheta                     = (theta[-1,0] - theta[0,0])/theta.shape[0] * (180 / np.pi)
    domega                     = 2 * np.pi * np.sin(tcenter) *(tv[1:] - tv[:-1])
    var[u < args.cutoff]       = 0

    label = f'{args.labels[case]}' if args.labels is not None else None
    
    print(f'2D var sum with GB > {args.cutoff}: {var.sum()}')
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
        var_per_theta = 4.0 * np.pi * np.sum(var, axis=1) / domega[:,0]
        ax.plot(np.rad2deg(theta[:, 0]), var_per_theta, color=colors[case], label=label)
    
    if ax_col == 0:
        #1D Comparison 
        if args.oned_files is not None:
            if args.sub_split is None:
                for file in args.oned_files:
                    oned_field   = get_1d_equiv_file(file%len(args.oned_files))
                    calc_1d_dx_domega(oned_field)
            else:
                oned_field   = get_1d_equiv_file(args.oned_files[ax_num%len(args.oned_files)])
                calc_1d_dx_domega(oned_field)      
                
    ax.set_xlim(np.rad2deg(theta[0,0]), np.rad2deg(theta[-1,0]))
    if args.sub_split is None:
        ax.set_xlabel(r'$\theta [\rm deg]$', fontsize=20)
        if args.kinetic:
            ax.set_ylabel(r'$E_{{\rm K, iso}} \ (\Gamma \beta > {})\ [\rm{{erg}}]$'.format(args.cutoff), fontsize=15)
        elif args.enthalpy:
            ax.set_ylabel(r'$H_{\rm iso} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoff), fontsize=15)
        elif args.dm_domega:
            ax.set_ylabel(r'$M_{\rm{iso}} \ (\Gamma \beta > {}) \ [\rm{{g}}]$'.format(args.cutoff), fontsize=15)
        else:
            ax.set_ylabel(r'$E_{{\rm T, iso}} \ (\Gamma \beta > {}) \ [\rm{{erg}}]$'.format(args.cutoff), fontsize=15)
        
    
        ax.tick_params('both', labelsize=15)
    else:
        ax.tick_params('x', labelsize=15)
        ax.tick_params('y', labelsize=10)
        
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if args.sub_split is None:
        ax.set_title(r'{}, t ={:.2f}'.format(args.setup, tend), fontsize=20)
        
    if not overplot:
        ax.set_title(r'{}, t ={:.2f}'.format(args.setup, tend), fontsize=20)
        return fig
    
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
    
    parser.add_argument('--cbar_range', dest = 'cbar', metavar='Range of Color Bar', nargs=2,
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cbar_sub', dest = 'cbar2', metavar='Range of Color Bar for secondary plot',nargs='+',type=float,
                        default =[None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cmap', dest = 'cmap', metavar='Color Bar Colarmap',
                        default = 'magma', help='The colorbar cmap you\'d like to plot')
    
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
    
    parser.add_argument('--de_domega', dest='de_domega', action='store_true',
                        default=False,
                        help='Plot the dE/dOmega plot')
    
    parser.add_argument('--dm_domega', dest='dm_domega', action='store_true',
                        default=False,
                        help='Plot the dM/dOmega plot')
    
    parser.add_argument('--cutoff', dest='cutoff', default=0.0, type=float,
                        help='The 4-velocity cutoff value for the dE/dOmega plot')
    
    parser.add_argument('--fill_scale', dest = 'fill_scale', metavar='Filler maximum', type=float,
                        default = None, help='Set the y-scale to start plt.fill_between')
    
    parser.add_argument('--norm', dest='norm', action='store_true',
                        default=False, help='True if you want the plot normalized to max value')
    
    parser.add_argument('--labels', dest='labels', nargs='+', default = None,
                        help='Optionally give a list of labels for multi-file plotting')
    
    parser.add_argument('--tidx', dest='tidx', type=int, default = None,
                        help='Set to a value if you wish to plot a 1D curve about some angle')
    
    parser.add_argument('--wedge', dest='wedge', default=False, action='store_true')
    parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 80, 110], type=float, nargs=4)

    parser.add_argument('--units', dest='units', default = False, action='store_true')
    parser.add_argument('--dbg', dest='dbg', default = False, action='store_true')
    parser.add_argument('--bipolar', dest='bipolar', default = False, action='store_true')
    parser.add_argument('--pictorial', dest='pictorial', default = False, action='store_true')
    parser.add_argument('--subplots', dest='subplots', default = None, type=int)
    parser.add_argument('--sub_split', dest='sub_split', default = None, nargs='+', type=int)
    
    parser.add_argument('--save', dest='save', type=str,
                        default=None,
                        help='Save the fig with some name')

    is_cartesian = False
    args = parser.parse_args()
    vmin, vmax = args.cbar
    field_dict = {}
    setup_dict = {}
    
    if args.dbg:
        plt.style.use('dark_background')
        
    for idx, file in enumerate(args.filename):
        field_dict[idx] = {}
        setup_dict[idx] = {}
        with h5py.File(file, 'r') as hf:
            
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
            
            setup_dict[idx]['xmax'] = xmax 
            setup_dict[idx]['xmin'] = xmin 
            setup_dict[idx]['ymax'] = ymax 
            setup_dict[idx]['ymin'] = ymin 
            setup_dict[idx]['time'] = t * time_scale
            
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
                setup_dict[idx]['xactive'] = xactive
                setup_dict[idx]['yactive'] = yactive
            else:
                rho = rho[2:-2, 2: -2]
                v1  = v1 [2:-2, 2: -2]
                v2  = v2 [2:-2, 2: -2]
                p   = p  [2:-2, 2: -2]
                chi = chi[2:-2, 2: -2]
                xactive = nx - 4
                yactive = ny - 4
                setup_dict[idx]['xactive'] = xactive
                setup_dict[idx]['yactive'] = yactive
            
            if is_linspace:
                setup_dict[idx]['x1'] = np.linspace(xmin, xmax, xactive)
                setup_dict[idx]['x2'] = np.linspace(ymin, ymax, yactive)
            else:
                setup_dict[idx]['x1'] = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
                setup_dict[idx]['x2'] = np.linspace(ymin, ymax, yactive)
            
            if coord_sysem == 'cartesian':
                is_cartesian = True
            
            W    = 1/np.sqrt(1.0 -(v1**2 + v2**2))
            beta = np.sqrt(v1**2 + v2**2)
            
            
            e = 3*p/rho 
            c = const.c.cgs.value
            a = (4 * const.sigma_sb.cgs.value / c)
            k = const.k_B.cgs
            m = const.m_p.cgs.value
            T = (3 * p * c ** 2  / a)**(1./4.)
            h = 1.0 + gamma*p/(rho*(gamma - 1.0))
            field_dict[idx]['rho']          = rho
            field_dict[idx]['v1']           = v1 
            field_dict[idx]['v2']           = v2 
            field_dict[idx]['p']            = p
            field_dict[idx]['chi']          = chi
            field_dict[idx]['chi_dens']     = chi * rho * W
            field_dict[idx]['gamma_beta']   = W*beta
            field_dict[idx]['gamma_beta_1'] = abs(W*v1)
            field_dict[idx]['gamma_beta_2'] = abs(W*v2)
            field_dict[idx]['temperature']  = T
            field_dict[idx]['enthalpy']     = h
            field_dict[idx]['W']            = W
        
        
    ynpts, xnpts = rho.shape 
    cdict = {}
    
    mesh = {}
    if is_cartesian:
        xx, yy = np.meshgrid(setup_dict[0]['x1'], setup_dict[0]['x2'])
        mesh['xx'] = xx
        mesh['yy'] = yy
    else:      
        rr, tt = np.meshgrid(setup_dict[0]['x1'], setup_dict[0]['x2'])
        rr, t2 = np.meshgrid(setup_dict[0]['x1'], -setup_dict[0]['x2'][::-1])
        mesh['theta'] = tt 
        mesh['rr']    = rr
        mesh['r']     = setup_dict[0]['x1']
        mesh['th']    = setup_dict[0]['x2']
    
    num_subplots   = len(args.sub_split) if args.sub_split is not None else 1
    if len(args.filename) > 1:
        if num_subplots == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8,8))
            lines_per_plot = len(args.filename)
        else:
            fig,axs = plt.subplots(num_subplots, 1, figsize=(8,2 * num_subplots), sharex=True, tight_layout=False)
            if args.setup != "":
                fig.suptitle(f'{args.setup}')
            if args.de_domega or args.dm_domega:
                axs[-1].set_xlabel(r'$\theta \ \rm[deg]$', fontsize=20)
            else:
                axs[-1].set_xlabel(r'$\log \Gamma \beta$', fontsize=20)
            if args.de_domega or args.dm_domega:
                if args.kinetic:
                    fig.text(0.030, 0.5, r'$E_{{\rm K, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoff), fontsize=20, va='center', rotation='vertical')
                elif args.dm_domega:
                    fig.text(0.030, 0.5, r'$M_{{\rm iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoff), fontsize=20, va='center', rotation='vertical')
                else:
                    fig.text(0.030, 0.5, r'$E_{{\rm T, iso}}( > {}) \ [\rm{{erg}}]$'.format(args.cutoff), fontsize=20, va='center', rotation='vertical')
            else:
                if args.kinetic:
                    fig.text(0.030, 0.5, r'$E_{\rm K}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20, va='center', rotation='vertical')
                elif args.enthalpy:
                    fig.text(0.030, 0.5, r'$H( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20, va='center', rotation='vertical')
                else:
                    fig.text(0.030, 0.5, r'$E_{\rm T}( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20, va='center', rotation='vertical')
            axs_iter       = iter(axs)            # iterators for the multi-plot
            subplot_iter   = iter(args.sub_split) # iterators for the subplot splitting
            
            # first elements of subplot iterator tells how many files belong on axs[0]
            lines_per_plot = next(subplot_iter)   
        
        i        = 0       
        ax_col   = 0
        ax_shift = True
        ax_num   = 0     
        for idx, file in enumerate(args.filename):
            i += 1
            if args.hist and (not args.de_domega and not args.dm_domega):
                if args.sub_split is None:
                    plot_hist(field_dict[idx], args, mesh, setup_dict, overplot=True, ax=ax, case=idx, ax_col=idx)
                else:
                    if ax_shift:
                        ax_col   = 0
                        ax       = next(axs_iter)   
                        ax_shift = False
                    plot_hist(field_dict[idx], args, mesh, setup_dict, overplot=True, ax=ax, ax_num=ax_num, case=idx, ax_col=ax_col)
            elif args.de_domega or args.dm_domega:
                if args.sub_split is None:
                    plot_dx_domega(field_dict[idx], args, mesh, setup_dict, overplot=True, ax=ax, case=idx, ax_col=idx)
                else:
                    if ax_shift:
                        ax_col   = 0
                        ax       = next(axs_iter)   
                        ax_shift = False
                    plot_dx_domega(field_dict[idx], args, mesh, setup_dict, overplot=True, ax=ax, ax_num=ax_num, case=idx, ax_col=ax_col)
            elif args.x is not None:
                plot_max(field_dict[idx], args, mesh, setup_dict, True, ax, idx)
            else:
                plot_1d_curve(field_dict[idx], args, mesh, setup_dict, True, ax, idx)
            
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
                
        if args.sub_split is not None:
            for ax in axs:
                ax.label_outer()
    else:
        if args.hist and (not args.de_domega and not args.dm_domega):
            plot_hist(field_dict[0], args, mesh, setup_dict)
        elif args.tidx != None:
            plot_1d_curve(field_dict[0], args, mesh, setup_dict)
        elif args.de_domega or args.dm_domega:
            plot_dx_domega(field_dict[0], args, mesh, setup_dict)
        else:
            if is_cartesian:
                plot_cartesian_plot(field_dict[0], args, mesh, setup_dict)
            else:
                mesh['t2'] = t2
                plot_polar_plot(field_dict[0], args, mesh, setup_dict)
                
    if args.labels is not None:
        if args.sub_split is not None:
            for ax in axs:
                ax.legend(fontsize=7, loc='upper right')
        else:
            plt.legend()
            
    if not args.save:
        plt.show()
    else:
         plt.savefig('{}.png'.format(args.save.replace(' ', '_')), dpi=500)
    
if __name__ == '__main__':
    main()
