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

from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
import os

field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature', 'line_profile', 'energy']

ofield = {}
with h5py.File('data/srhd/702.chkpt.0100000.h5', 'r+') as hf:
        
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
        
        h = 1.0 + 5/3 * p / (rho * (5/3 - 1))
        
        ofield["rho"]         = rho
        ofield["v"]           = v
        ofield["p"]           = p
        ofield["W"]           = W
        ofield["enthalpy"]    = h
        ofield["gamma_beta"]  = W*beta
        ofield["temperature"] = T
        ofield["r"]           = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
        
R_0 = 7e10 * u.cm 
def prims2cons(fields, cons):
    if cons == "D":
        return fields['rho'] * fields['W']
    elif cons == "energy":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']


def plot_polar_plot(field_dict, args, mesh, ds):
    fig, ax= plt.subplots(1, 1, figsize=(15,8), subplot_kw=dict(projection='polar'), constrained_layout=False)

    rr, tt = mesh['rr'], mesh['theta']
    t2 = mesh['t2']
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
    c1 = ax.pcolormesh(tt, rr, field_dict[args.field], cmap=color_map, shading='auto', **kwargs)
    c2 = ax.pcolormesh(t2[::-1], rr, field_dict[args.field],  cmap=color_map, shading='auto', **kwargs)
    
    
        
    if ymax < np.pi:
        ax.set_position( [0.1, -0.18, 0.8, 1.43])
        cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
        cbar_orientation = "horizontal"
    else:
        cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        cbar_orientation = "vertical"
        
    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c2, orientation=cbar_orientation, cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c2, orientation=cbar_orientation, cax=cbaxes)
        
    
        
    # ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    ax.tick_params(axis='both', labelsize=10)
    # cbaxes.tick_params(axis='x', labelsize=10)
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(xmax) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    
    ymd = int( np.ceil(ymax * 180/np.pi) )
    ax.set_thetamin(-ymd)
    ax.set_thetamax(ymd)

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
        cbar.ax.set_ylabel(r'$[{}]$'.format(args.field), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f} s'.format(args.setup[0], tend), fontsize=20, y=0.95)

def plot_1d_curve(field_dict, args, mesh, ds):
    fig, ax= plt.subplots(1, 1, figsize=(10,10),constrained_layout=False)

    r, theta = mesh['r'], mesh['th']
    theta    = theta * 180 / np.pi 
    
    xmax        = ds[0]["xmax"]
    xmin        = ds[0]["xmin"]
    ymax        = ds[0]["ymax"]
    ymin        = ds[0]["ymin"]
    
    vmin,vmax = eval(args.cbar)
    
    #1D test 
    tend = ds[0]["time"]
    for idx in range(len(theta)):
        ax.loglog(r, field_dict[args.field][idx])
    # ax.loglog(r, field_dict[args.field][args.tidx])
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
        
    # fig.suptitle(r'{} at $\theta = {:.2f}$ deg, t = {:.2f} s'.format(args.setup[0],theta[args.tidx], tend), fontsize=20, y=0.95)
    
def plot_hist(fields, args, mesh, ds, overplot=False, ax=None, case=0):
    if not overplot:
        fig = plt.figure(figsize=[9, 9], constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)

    tend = ds[0]["time"]
    e_scale = 2e33 * const.c.cgs.value**2
    edens_total = prims2cons(fields, "energy")
    theta       = mesh['theta']
    r           = mesh["rr"]
    
    tvertices = 0.5 * (theta[1:] + theta[:-1])
    tvertices = np.insert(tvertices, 0, theta[0], axis=0)
    tvertices = np.insert(tvertices, tvertices.shape[0], theta[-1], axis=0)
    
    rvertices = np.sqrt(r[:, 1:] * r[:, :-1])
    rvertices = np.insert(rvertices,  0, r[:, 0], axis=1)
    rvertices = np.insert(rvertices, rvertices.shape[1], r[:, -1], axis=1)
    dr = rvertices[:, 1:] - rvertices[:, :-1]
        
    theta_mean  = 0.5 * (tvertices[1:] + tvertices[:-1])
    dtheta      = tvertices[1:] - tvertices[:-1]
    dcos        = np.cos(tvertices[:-1]) - np.cos(tvertices[1:])
    dV          =  ( (2.0*np.pi/3.) * (rvertices[:, 1:]**3 - rvertices[:, :-1]**3) *  dcos )
    
    etotal = edens_total * dV * e_scale
    mass   = dV * fields["W"] * fields["rho"]
    e_k    = (fields['W'] - 1.0) * mass * e_scale
    
    #1D Check 
    # edens_1d = prims2cons(ofield, "energy")
    # dV_1d    = (4 * np.pi/3.) * (rvertices[0, 1:]**3 - rvertices[0, :-1]**3)
    # etotal_1d = edens_1d * dV_1d * e_scale
    # u1d       = ofield['gamma_beta']
    # w = np.diff(u1d).max()*1e-1
    # n = int(np.ceil( (u1d.max() - u1d.min() ) / w ) )
    # gbs_1d = np.logspace(np.log10(1.e-4), np.log10(u1d.max()), n)
    # ets_1d = np.asarray([etotal_1d[np.where(u1d > gb)].sum() for gb in gbs_1d])
    
    u = fields['gamma_beta']
    w = np.diff(u).max()*1e-1
    n = int(np.ceil( (u.max() - u.min() ) / w ) )
    gbs = np.logspace(np.log10(1.e-4), np.log10(u.max()), n)
    eks = np.asarray([e_k[np.where(u > gb)].sum() for gb in gbs])
    ets = np.asarray([etotal[np.where(u > gb)].sum() for gb in gbs])
    
    bins    = np.arange(min(gbs), max(gbs) + w, w)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]), len(bins))

    if args.labels is None:
        ax.hist(gbs, bins=gbs, weights=ets, label= r'$E_T$', histtype='step', rwidth=1.0, linewidth=3.0)
    else:
        ax.hist(gbs, bins=gbs, weights=ets, label=r'$\{}$'.format(args.labels[case]), histtype='step', rwidth=1.0, linewidth=3.0)
    
    # if case == 0:
    #     ax.hist(gbs_1d, bins=gbs_1d, weights=ets_1d, alpha=0.8, label= r'1D Sphere', histtype='step', linewidth=3.0)
    
    sorted_energy = np.sort(ets)
    plt.xscale('log')
    plt.yscale('log')
    #ax.set_ylim(sorted_energy[1], 1.5*ets.max())
    ax.set_xlabel(r'$\Gamma\beta $', fontsize=20)
    ax.set_ylabel(r'$E( > \Gamma \beta) \ [\rm{erg}]$', fontsize=20)
    ax.tick_params('both', labelsize=20)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title(r'Roche lobe overflow, t ={:.2f} s'.format(tend), fontsize=20)
    ax.legend(fontsize=15)
    if not overplot:
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
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar',
                        default ='None, None', help='The colorbar range you\'d like to plot')
    
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
    
    parser.add_argument('--ehist', dest='ehist', action='store_true',
                        default=False,
                        help='True if you want the plot the energy histogram')
    parser.add_argument('--labels', dest='labels', nargs="+", default = None,
                        help='Optionally give a list of labels for multi-file plotting')
    
    parser.add_argument('--tidx', dest='tidx', type=int, default = None,
                        help='Set to a value if you wish to plot a 1D curve about some angle')

    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='True if you want save the fig')

   
    args = parser.parse_args()
    vmin, vmax = eval(args.cbar)
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
            try:
                gamma = ds.attrs["adiabatic_gamma"]
            except:
                gamma = 4./3.
            
            
            setup_dict[idx]["xmax"] = xmax 
            setup_dict[idx]["xmin"] = xmin 
            setup_dict[idx]["ymax"] = ymax 
            setup_dict[idx]["ymin"] = ymin 
            setup_dict[idx]["time"] = t
            
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
                
            W    = 1/np.sqrt(1 -(v1**2 + v2**2))
            beta = np.sqrt(v1**2 + v2**2)
            
            
            e = 3*p/rho 
            c = const.c.cgs.value
            a = (4 * const.sigma_sb.cgs.value / c)
            k = const.k_B.cgs
            m = const.m_p.cgs.value
            T = (3 * p * c ** 2  / a)**(1./4.)
            h = 1. + gamma*p/(rho*(gamma - 1.))
            
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

    
    if (args.log):
        r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
        norm = colors.LogNorm(vmin=rho.min(), vmax=3.*rho.min())
    else:
        r = np.linspace(xmin, xmax, xactive)
        # norm = colors.LinearNorm(vmin=None, vmax=None)
        
    # r = np.logspace(np.log10(0.01), np.log10(0.5), xnpts)
    theta = np.linspace(ymin, ymax, yactive)
    theta_mirror = - theta[::-1]
    # theta_mirror[-1] *= -1.
    
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)
    
    mesh = {}
    mesh["theta"] = tt 
    mesh["rr"]    = rr
    mesh["r"]     = r 
    mesh["th"]     = theta
    
    if len(args.filename) > 1:
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        case = 0
        for idx, file in enumerate(args.filename):
            if (args.log):
                r = np.logspace(np.log10(setup_dict[idx]["xmin"]), np.log10(setup_dict[idx]["xmax"]), setup_dict[idx]["xactive"])
                norm = colors.LogNorm(vmin=rho.min(), vmax=3.*rho.min())
            else:
                r = np.linspace(xmin, xmax, xactive)
                # norm = colors.LinearNorm(vmin=None, vmax=None)
                
            # r = np.logspace(np.log10(0.01), np.log10(0.5), xnpts)
            theta = np.linspace(setup_dict[idx]["ymin"], setup_dict[idx]["ymax"], setup_dict[idx]["yactive"])
            theta_mirror = - theta[::-1]
            # theta_mirror[-1] *= -1.
            
            rr, tt = np.meshgrid(r, theta)
            rr, t2 = np.meshgrid(r, theta_mirror)
            
            mesh = {}
            mesh["theta"] = tt 
            mesh["rr"]    = rr
            if args.ehist:
                plot_hist(field_dict[idx], args, mesh, setup_dict, True, ax, case)
            case += 1
    else:
        if args.ehist:
            plot_hist(field_dict[0], args, mesh, setup_dict)
        elif args.tidx != None:
            plot_1d_curve(field_dict[0], args, mesh, setup_dict)
        else:
            mesh["t2"] = t2
            plot_polar_plot(field_dict[0], args, mesh, setup_dict)
        
        

    
    # divider = make_axes_locatable(ax)
    # cbaxes  = divider.append_axes('right', size='5%', pad=0.1)
    # cbar    = fig.colorbar(c2, orientation='vertical')
    # cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
    
    
    if args.save:
        plt.savefig("plots/2D/SR/{}.png".format(args.setup[0].replace(" ", "_")), dpi=500)
    else:
        plt.show()
    
    # if args.save:
    #     
    
if __name__ == "__main__":
    main()
