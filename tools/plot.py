#! /usr/bin/env python

# Read in a File and Plot it

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import time
import scipy.special as spc
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const
import astropy.units as u 
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import os


import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist import SubplotHost
from mpl_toolkits.axisartist import GridHelperCurveLinear
from matplotlib.projections import get_projection_class
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def curvelinear_test(fig, *data):
    """
    polar projection, but in a rectangular box.
    """
    global ax1
    # see demo_curvelinear_grid.py for details
    tr = Affine2D().scale(np.pi / 180., 1.) + PolarAxes.PolarTransform()

    extreme_finder = angle_helper.ExtremeFinderCycle(20,
                                                     20,
                                                     lon_cycle=360,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0,
                                                                 np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)

    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    ax1 = SubplotHost(fig, 1, 1, 1, grid_helper=grid_helper)

    fig.add_subplot(ax1)

    # Now creates floating axis

    # floating axis whose first coordinate (theta) is fixed at 60
    ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    axis.label.set_text(r"$\theta = 60^{\circ}$")
    axis.label.set_visible(True)
    
    # A parasite axes with given transform
    ax2 = ax1.get_aux_axes(tr)
    # note that ax2.transData == tr + ax1.transData
    # Anything you draw in ax2 will match the ticks and grids of ax1.
    ax1.parasites.append(ax2)
    
    ax2.pcolormesh(*data, shading="auto")

    # floating axis whose second coordinate (r) is fixed at 6
    ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    axis.label.set_text(r"$r = 6$")

    ax1.set_aspect(1.)
    ax1.set_xlim(-5, 12)
    ax1.set_ylim(-5, 10)

    ax1.grid(True)
    

def curvelinear_test2(ax1, args, *data):
    """
    polar projection, but in a rectangular box.
    """
    # see demo_curvelinear_grid.py for details
    tr = Affine2D().scale(np.pi / 180., 1.) + PolarAxes.PolarTransform()

    extreme_finder = angle_helper.ExtremeFinderCycle(20,
                                                     20,
                                                     lon_cycle=180,
                                                     lat_cycle=None,
                                                     lon_minmax=None,
                                                     lat_minmax=(0,
                                                                 np.inf),
                                                     )

    grid_locator1 = angle_helper.LocatorDMS(12)

    tick_formatter1 = angle_helper.FormatterDMS()

    grid_helper = GridHelperCurveLinear(tr,
                                        extreme_finder=extreme_finder,
                                        grid_locator1=grid_locator1,
                                        tick_formatter1=tick_formatter1
                                        )

    # Now creates floating axis

    # floating axis whose first coordinate (theta) is fixed at 60
    # ax1.axis["lat"] = axis = ax1.new_floating_axis(0, 60)
    vmin, vmax = args.cbar
    ax1.pcolormesh(*data, shading='auto', cmap=args.cmap, norm = colors.LogNorm(vmin = vmin, vmax = vmax))
    #axis.label.set_text(r"$\theta = 60^{\circ}$")
    #axis.label.set_visible(True)

    # floating axis whose second coordinate (r) is fixed at 6
    # ax1.axis["lon"] = axis = ax1.new_floating_axis(1, 6)
    # axis.label.set_text(r"$r = 6$")

    ax1.set_aspect(1.)
    ax1.set_xlim(0.1, 5.0)
    ax1.set_ylim(0.1, 3.0)

    ax1.grid(True)

def find_nearest(arr, val):
    arr = np.asarray(arr)
    idx = np.argmin(np.abs(arr - val))
    return idx, arr[idx]
    
def fill_below_intersec(x, y, constraint, color):
    # colors = plt.cm.plasma(np.linspace(0.25, 0.75, len(x)))
    ind = find_nearest(y, constraint)[0]
    plt.fill_between(x[ind:],y[ind:], color=color, alpha=0.1, interpolate=True)
    
def get_1d_equiv_file(rzones: int):
    file = 'data/srhd/e53/m0/{}.chkpt.010_000.h5'.format(rzones)
    ofield = {}
    with h5py.File(file, 'r+') as hf:
            
            ds = hf.get("sim_info")
            
            rho         = hf.get("rho")[:]
            v           = hf.get("v")[:]
            p           = hf.get("p")[:]
            nx          = ds.attrs["Nx"]
            t           = ds.attrs["current_time"]
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]

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
            
            ofield["rho"]         = rho
            ofield["v"]           = v
            ofield["p"]           = p
            ofield["W"]           = W
            ofield["enthalpy"]    = h
            ofield["gamma_beta"]  = W*beta
            ofield["temperature"] = T
            ofield["r"]           = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
    return ofield

cons = ['D', 'momentum', 'energy', 'energy_rst']
field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature', 'line_profile', 'energy'] + cons 
col = plt.cm.jet([0.25,0.75])  

R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * const.c.cgs.value**2
pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
vel_scale  = c 
time_scale = R_0 / c

def compute_rvertcies(r):
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return rvertices 

def compute_theta_vertcies(theta):
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    return tvertices 

def calc_cell_volume1D(r):
    rvertices = np.sqrt(r[1:] * r[:-1])
    rvertices = np.insert(rvertices,  0, r[0])
    rvertices = np.insert(rvertices, r.shape, r[-1])
    return (1./3.) * (rvertices[1:]**3 - rvertices[:-1]**3)

def calc_cell_volume(r, theta):
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
        
def get_field_str(args):
    if args.field == "rho":
        if args.units:
            return r'$\rho$ [g cm$^{-3}$]'
        else:
            return r'$\rho$'
    elif args.field == "gamma_beta":
        return r"$\Gamma \ \beta$"
    elif args.field == "energy":
        if args.units:
            return r"$\tau [\rm erg \ cm^{-3}]$"
        else:
            return r"$\tau $"
    elif args.field == "energy_rst":
        if args.units:
            return r"$\tau - D \  [\rm erg \ cm^{-3}]$"
        else:
            return r"$\tau - D"
    else:
        return args.field
    
def prims2cons(fields, cons):
    if cons == "D":
        return fields['rho'] * fields['W']
    elif cons == "S":
        return fields['rho'] * fields['W']**2 * fields['v']
    elif cons == "energy":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']
    elif cons == "energy_rst":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p']


def plot_polar_plot(field_dict, args, mesh, ds):
    # fig = plt.figure(figsize=(10,8), constrained_layout=False)
    # ax  = fig.add_subplot(1, 1, 1, polar=True)
    if args.wedge:
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                            figsize=(15, 10), constrained_layout=True)
        
        ax    = axes[0]
        wedge = axes[1]
    else:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'},
                            figsize=(8, 8), constrained_layout=False)
    # fig, ax = plt.subplots(1, 1, figsize=(10,8), subplot_kw=dict(projection='polar'), constrained_layout=False)

    rr, tt = mesh['rr'], mesh['theta']
    t2 = mesh['t2']
    xmax        = ds[0]["xmax"]
    xmin        = ds[0]["xmin"]
    ymax        = ds[0]["ymax"]
    ymin        = ds[0]["ymin"]
    
    vmin,vmax = args.cbar

    unit_scale = 1.0
    if (args.units):
        if args.field == "rho" or args.field == "D":
            unit_scale = rho_scale
        elif args.field == "p" or args.field == "energy" or args.field == "energy_rst":
            unit_scale = pre_scale
    
    units = unit_scale.value if args.units else 1.0
    if args.field in cons:
        var = units * prims2cons(field_dict, args.field)
    else:
        var = units * field_dict[args.field]
        
    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds[0]["time"]
    c1 = ax.pcolormesh(tt[::-1], rr, var, cmap=color_map, shading='auto', **kwargs)
    c2 = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, shading='auto', **kwargs)
    
    # ax.set_thetamin(0)
    # ax.set_thetamax(180)
    if ymax < np.pi:
        ax.set_position( [0.1, -0.18, 0.8, 1.43])
        cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
        cbar_orientation = "horizontal"
        ymd = int( np.floor(ymax * 180/np.pi) )                                                                                                                                                                                     
        ax.set_thetamin(-ymd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        ax.set_thetamax(ymd)
    else:
        cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        cbar_orientation = "vertical"
        
    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c1, orientation=cbar_orientation, cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c1, orientation=cbar_orientation, cax=cbaxes)
    
    if args.wedge:
        wedge_min = args.wedge_lims[0]
        wedge_max = args.wedge_lims[1]
        ang_min   = args.wedge_lims[2]
        ang_max   = args.wedge_lims[3]
        
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color="white")
        
    # ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.2)
    ax.xaxis.grid(True, alpha=0.2)
    ax.tick_params(axis='both', labelsize=10)
    rlabels = ax.get_ymajorticklabels()
    for label in rlabels:
        label.set_color('white')
        
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(xmax) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    ax.set_rmin(xmin)
    
    if args.wedge:
        ax.set_position( [0.05, -0.5, 0.46, 2])
    
    field_str = get_field_str(args)
    
    if args.wedge:
        vmin2, vmax2 = args.cbar2
        if args.log:
            kwargs = {'norm': colors.LogNorm(vmin = vmin2, vmax = vmax2)}
        else:
            kwargs = {'vmin': vmin2, 'vmax': vmax2}
            
        wedge.pcolormesh(tt, rr, var, cmap=color_map, shading='nearest', **kwargs)
        wedge.set_theta_zero_location("N")
        wedge.set_theta_direction(-1)
        wedge.yaxis.grid(True, alpha=0.2)
        wedge.xaxis.grid(True, alpha=0.2)
        wedge.tick_params(axis='both', labelsize=10)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('white')
            
        #wedge.axes.xaxis.set_ticklabels([])
        wedge.set_ylim([wedge_min, wedge_max])
        wedge.set_rorigin(-wedge_min)
        wedge.set_thetamin(ang_min)
        wedge.set_thetamax(ang_max)
        wedge.set_aspect(1.)
        wedge.set_position( [0.5, -0.5, 0.3, 2])
        
    # fig.set_tight_layout(True)
    if args.log:
        if ymax >= np.pi:
            cbar.ax.set_ylabel(r'$\log$ {}'.format(field_str), fontsize=20)
        else:
            cbar.ax.set_xlabel(r'$\log$ {}'.format(field_str), fontsize=20)
    else:
        if ymax >= np.pi:
            cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        else:
            cbar.ax.set_xlabel(r'{}'.format(field_str), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f}'.format(args.setup[0], tend), fontsize=20, y=0.95)

def plot_cartesian_plot(field_dict, args, mesh, ds):
    fig, ax= plt.subplots(1, 1, figsize=(10,10), constrained_layout=False)

    xx, yy = mesh['xx'], mesh['yy']
    xmax        = ds[0]["xmax"]
    xmin        = ds[0]["xmin"]
    ymax        = ds[0]["ymax"]
    ymin        = ds[0]["ymin"]
    
    vmin,vmax = eval(args.cbar)

    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds[0]["time"]
    c = ax.pcolormesh(xx, yy, field_dict[args.field], cmap=color_map, shading='auto', **kwargs)
    
    divider = make_axes_locatable(ax)
    cbaxes = divider.append_axes('right', size='5%', pad=0.05)
    
    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c, orientation="vertical", cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c, orientation="vertical", cax=cbaxes)

    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    ax.tick_params(axis='both', labelsize=10)
    
    # Change the format of the field
    if args.field == "rho":
        field_str = r'$\rho$'
    elif args.field == "gamma_beta":
        field_str = r"$\Gamma \ \beta$"
    elif args.field == "temperature":
        field_str = r"T [K]"
    else:
        field_str = args.field
    
    if args.log:
        cbar.ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f}'.format(args.setup[0], tend), fontsize=20, y=0.95)
    
def plot_1d_curve(field_dict, args, mesh, ds, overplot=False, ax=None, case=0):
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['r'], mesh['th']
    theta    = theta * 180 / np.pi 
    
    xmax        = ds[0]["xmax"]
    xmin        = ds[0]["xmin"]
    ymax        = ds[0]["ymax"]
    ymin        = ds[0]["ymin"]
    
    vmin,vmax = eval(args.cbar)
    ofield = get_1d_equiv_file(16384)
    #1D test 
    tend = ds[0]["time"]
    # for idx in range(len(theta)):
    #     ax.loglog(r, field_dict[args.field][idx])
    ax.loglog(r, field_dict[args.field][args.tidx])
    ax.loglog(ofield["r"], ofield[args.field], 'ro')
        
    # ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r'$r/R_\odot$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    # Change the format of the field
    if args.field == "rho":
        field_str = r'$\rho$'
    elif args.field == "gamma_beta":
        field_str = r"$\Gamma \ \beta$"
    elif args.field == "temperature":
        field_str = r"T [K]"
    else:
        field_str = args.field
    
    if args.log:
        ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        ax.set_ylabel(r'$[{}]$'.format(args.field), fontsize=20)
    
    if not overplot:
        return fig
    # fig.suptitle(r'{} at $\theta = {:.2f}$ deg, t = {:.2f} s'.format(args.setup[0],theta[args.tidx], tend), fontsize=20, y=0.95)
def plot_max(fields, args, mesh, ds, overplot=False, ax=None, case=0):
    print("plotting max values along x...")
    
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['r'], mesh['th']
    theta    = theta * 180 / np.pi 
    
    xmax        = ds[0]["xmax"]
    xmin        = ds[0]["xmin"]
    ymax        = ds[0]["ymax"]
    ymin        = ds[0]["ymin"]
    
    vmin,vmax = eval(args.cbar)
    ofield = get_1d_equiv_file(16384)
    #1D test 
    tend = ds[0]["time"]
    ax.set_title(r'{} at t={:.3f}'.format(args.setup[0], tend), fontsize=20)
    
    if args.field == "gamma_beta":
        edens_total = prims2cons(fields, "energy")
        theta       = mesh['theta']
        r           = mesh["rr"]
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
        ofield = get_1d_equiv_file(16384)
        edens_1d = prims2cons(ofield, "energy")
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
    if args.field == "rho":
        field_str = r'$\rho$'
    elif args.field == "gamma_beta":
        field_str = r"$\Gamma \ \beta > 10^{-6} E_{\rm inj}$"
    elif args.field == "temperature":
        field_str = r"T [K]"
    else:
        field_str = args.field
    
    if args.log:
        ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
    if not overplot:
        return fig
    
def plot_hist(fields, args, mesh, ds, overplot=False, ax=None, case=0):
    print("Computing histogram...")
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(args.filename)))
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = ds[case]["time"]
    edens_total = prims2cons(fields, "energy")
    theta       = mesh['theta']
    r           = mesh["rr"]
    dV          = calc_cell_volume(r, theta)
    
    etotal = edens_total * 2.0 * np.pi * dV * e_scale.value
    mass   = 2.0 * np.pi * dV * fields["W"] * fields["rho"]
    e_k    = (fields['W'] - 1.0) * mass * e_scale.value
    
    
    
    u = fields['gamma_beta']
    w = 0.1 #np.diff(u).max()*1e-1
    n = int(np.ceil( (u.max() - u.min() ) / w ) )
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), 100)
    eks = np.asarray([e_k[np.where(u > gb)].sum() for gb in gbs])
    ets = np.asarray([etotal[np.where(u > gb)].sum() for gb in gbs])
    
    if args.norm:
        ets /= ets.max()

    if args.labels is None:
        ax.hist(gbs, bins=gbs, weights=ets, label= r'$E_T$', histtype='step', rwidth=1.0, linewidth=3.0, color=colors[case])
    else:
        ax.hist(gbs, bins=gbs, weights=ets, label=r'$\{}$'.format(args.labels[case]), histtype='step', rwidth=1.0, linewidth=3.0, color=colors[case])
    fill_below_intersec(gbs, ets, 1e-6, colors[case])
    if case == 0:
        #1D Check 
        ofield = get_1d_equiv_file(16384)
        edens_1d = prims2cons(ofield, "energy")
        dV_1d    = 4.0 * np.pi * calc_cell_volume1D(ofield['r'])
        etotal_1d = edens_1d * dV_1d * e_scale.value
        u1d       = ofield['gamma_beta']
        w = 0.001
        n = int(np.ceil( (u1d.max() - u1d.min() ) / w ) )
        gbs_1d = np.logspace(np.log10(1.e-4), np.log10(u1d.max()), n)
        ets_1d = np.asarray([etotal_1d[np.where(u1d > gb)].sum() for gb in gbs_1d])
        if args.norm:
            ets_1d /= ets_1d.max()
        fill_below_intersec(gbs_1d, ets_1d, 1e-6, colors[0])
        ax.hist(gbs_1d, bins=gbs_1d, weights=ets_1d, alpha=0.8, label= r'Sphere', histtype='step', linewidth=3.0)
    
    sorted_energy = np.sort(ets)
    plt.xscale('log')
    plt.yscale('log')
    #ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
    ax.set_ylabel(r'$E( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    ax.tick_params('both', labelsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(r'{}'.format(args.setup[0]), fontsize=20)
    # ax.legend(fontsize=15)
    if not overplot:
        ax.set_title(r'{}, t ={:.2f} s'.format(args.setup[0], tend), fontsize=20)
        return fig
    
def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('filename', metavar='Filename', nargs='+',
                        help='A Data Source to Be Plotted')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = "field", metavar='Field Variable', nargs='?',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default="rho")
    parser.add_argument('--rmax', dest = "rmax", metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar', nargs=2,
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cbar_sub', dest = "cbar2", metavar='Range of Color Bar for secondary plot',nargs=2,type=float,
                        default =[None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cmap', dest = "cmap", metavar='Color Bar Colarmap',
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
    
    parser.add_argument('--x', dest='x', nargs="+", default = None, type=float,
                        help='List of x values to plot field max against')
    
    parser.add_argument('--xlabel', dest='xlabel', nargs=1, default = 'X',
                        help='X label name')
    
    parser.add_argument('--ehist', dest='ehist', action='store_true',
                        default=False, help='True if you want the plot the energy histogram')
    
    parser.add_argument('--norm', dest='norm', action='store_true',
                        default=False, help='True if you want the plot normalized to max value')
    
    parser.add_argument('--labels', dest='labels', nargs="+", default = None,
                        help='Optionally give a list of labels for multi-file plotting')
    
    parser.add_argument('--tidx', dest='tidx', type=int, default = None,
                        help='Set to a value if you wish to plot a 1D curve about some angle')
    
    parser.add_argument('--wedge', dest='wedge', default=False, action='store_true')
    parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 80, 110], type=float, nargs=4)

    parser.add_argument('--units', dest='units', default = False, action='store_true')
    
    parser.add_argument('--save', dest='save', type=str,
                        default=None,
                        help='Save the fig with some name')

    is_cartesian = False
    args = parser.parse_args()
    vmin, vmax = args.cbar
    field_dict = {}
    setup_dict = {}
    for idx, file in enumerate(args.filename):
        field_dict[idx] = {}
        setup_dict[idx] = {}
        with h5py.File(file, 'r+') as hf:
            
            ds = hf.get("sim_info")
            
            rho         = hf.get("rho")[:]
            v1          = hf.get("v1")[:]
            v2          = hf.get("v2")[:]
            p           = hf.get("p")[:]
            nx          = ds.attrs["NX"]
            ny          = ds.attrs["NY"]
            t           = ds.attrs["current_time"]
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]
            ymax        = ds.attrs["ymax"]
            ymin        = ds.attrs["ymin"]
            
            # New checkpoint files, so check if new attributes were
            # implemented or not
            try:
                gamma = ds.attrs["adiabatic_gamma"]
                
                # Check for garbage value
                if gamma < 1:
                    gamma = 4.0/3.0
            except:
                gamma = 4./3.
                
            try:
                coord_sysem = ds.attrs["geometry"].decode('utf-8')
            except:
                coord_sysem = "spherical"
                
            try:
                is_linspace = ds.attrs["linspace"]
            except:
                is_linspace = False
            
            setup_dict[idx]["xmax"] = xmax 
            setup_dict[idx]["xmin"] = xmin 
            setup_dict[idx]["ymax"] = ymax 
            setup_dict[idx]["ymin"] = ymin 
            setup_dict[idx]["time"] = t * time_scale
            
            rho = rho.reshape(ny, nx)
            v1  = v1.reshape(ny, nx)
            v2  = v2.reshape(ny, nx)
            p   = p.reshape(ny, nx)
            
            if args.forder:
                rho = rho[1:-1, 1: -1]
                v1  = v1 [1:-1, 1: -1]
                v2  = v2 [1:-1, 1: -1]
                p   = p  [1:-1, 1: -1]
                xactive = nx - 2
                yactive = ny - 2
                setup_dict[idx]["xactive"] = xactive
                setup_dict[idx]["yactive"] = yactive
            else:
                rho = rho[2:-2, 2: -2]
                v1  = v1 [2:-2, 2: -2]
                v2  = v2 [2:-2, 2: -2]
                p   = p  [2:-2, 2: -2]
                xactive = nx - 4
                yactive = ny - 4
                setup_dict[idx]["xactive"] = xactive
                setup_dict[idx]["yactive"] = yactive
            
            if is_linspace:
                setup_dict[idx]["x1"] = np.linspace(xmin, xmax, xactive)
                setup_dict[idx]["x2"] = np.linspace(ymin, ymax, yactive)
            else:
                setup_dict[idx]["x1"] = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
                setup_dict[idx]["x2"] = np.linspace(ymin, ymax, yactive)
            
            if coord_sysem == "cartesian":
                is_cartesian = True
            
            W    = 1/np.sqrt(1 -(v1**2 + v2**2))
            beta = np.sqrt(v1**2 + v2**2)
            
            
            e = 3*p/rho 
            c = const.c.cgs.value
            a = (4 * const.sigma_sb.cgs.value / c)
            k = const.k_B.cgs
            m = const.m_p.cgs.value
            T = (3 * p * c ** 2  / a)**(1./4.)
            h = 1.0 + gamma*p/(rho*(gamma - 1.0))
            tau = (rho * h * W**2 - p - rho * W )
            field_dict[idx]["rho"]         = rho
            field_dict[idx]["v1"]          = v1 
            field_dict[idx]["v2"]          = v2 
            field_dict[idx]["p"]           = p
            field_dict[idx]["gamma_beta"]  = W*beta
            field_dict[idx]["temperature"] = T
            field_dict[idx]["enthalpy"]    = h
            field_dict[idx]["W"]           = W
            field_dict[idx]["energy"]      = rho * h * W * W  - p - rho * W
        
        
    ynpts, xnpts = rho.shape 
    cdict = {}
    
    mesh = {}
    if is_cartesian:
        xx, yy = np.meshgrid(setup_dict[0]["x1"], setup_dict[0]["x2"])
        mesh["xx"] = xx
        mesh["yy"] = yy
    else:      
        rr, tt = np.meshgrid(setup_dict[0]["x1"], setup_dict[0]["x2"])
        rr, t2 = np.meshgrid(setup_dict[0]["x1"], -setup_dict[0]["x2"][::-1])
        mesh["theta"] = tt 
        mesh["rr"]    = rr
        mesh["r"]     = setup_dict[0]["x1"]
        mesh["th"]    = setup_dict[0]["x2"]
    
    
    if len(args.filename) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        # if args.x is not None: plt.xlim(, max(args.x))
        case = 0
        for idx, file in enumerate(args.filename):
            if args.ehist:
                plot_hist(field_dict[idx], args, mesh, setup_dict, True, ax, case)
            elif args.x is not None:
                plot_max(field_dict[idx], args, mesh, setup_dict, True, ax, case)
            else:
                plot_1d_curve(field_dict[idx], args, mesh, setup_dict, True, ax, case)
            case += 1
    else:
        if args.ehist:
            plot_hist(field_dict[0], args, mesh, setup_dict)
        elif args.tidx != None:
            plot_1d_curve(field_dict[0], args, mesh, setup_dict)
        else:
            if is_cartesian:
                plot_cartesian_plot(field_dict[0], args, mesh, setup_dict)
            else:
                mesh["t2"] = t2
                plot_polar_plot(field_dict[0], args, mesh, setup_dict)
                
    if args.labels is not None:
        plt.legend()
    if not args.save:
        plt.show()
    else:
         plt.savefig("{}.png".format(args.save.replace(" ", "_")), dpi=500)
    
if __name__ == "__main__":
    main()
