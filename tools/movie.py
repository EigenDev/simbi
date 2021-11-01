#! /usr/bin/env python

# Tool for making movies from h5 files in directory

import numpy as np 
import matplotlib.pyplot as plt #! /usr/bin/env python
import matplotlib.ticker as tkr
import time
import matplotlib.colors as colors
import argparse 
import h5py 
import astropy.constants as const

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import os, os.path


cons = ['D', 'momentum', 'energy', 'energy_rst']
field_choices = ['rho', 'v1', 'v2', 'pre', 'gamma_beta', 'temperature', 'gamma_beta_1', 'gamma_beta_2', 'energy', 'mass', 'chi', 'chi_dens'] + cons 
lin_fields = ['chi', 'gamma_beta', 'gamma_beta_1', 'gamma_beta_2']
col = plt.cm.jet([0.25,0.75])  

R_0 = const.R_sun.cgs 
c   = const.c.cgs
m   = const.M_sun.cgs
 
rho_scale  = m / (4./3. * np.pi * R_0 ** 3) 
e_scale    = m * const.c.cgs.value**2
pre_scale  = e_scale / (4./3. * np.pi * R_0**3)
vel_scale  = c 
time_scale = R_0 / c

def prims2cons(fields, cons):
    if cons == "D":
        return fields['rho'] * fields['W']
    elif cons == "S":
        return fields['rho'] * fields['W']**2 * fields['v']
    elif cons == "energy":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p'] - fields['rho']*fields['W']
    elif cons == "energy_rst":
        return fields['rho']*fields['enthalpy']*fields['W']**2 - fields['p']
    
def get_field_str(args):
    if args.field[0] == "rho":
        if args.units:
            return r'$\rho$ [g cm$^{-3}$]'
        else:
            return r'$\rho$'
    elif args.field[0] == "gamma_beta":
        return r"$\Gamma \ \beta$"
    elif args.field[0] == "energy":
        if args.units:
            return r"$\tau [\rm erg \ cm^{-3}]$"
        else:
            return r"$\tau $"
    elif args.field[0] == "energy_rst":
        if args.units:
            return r"$\tau - D \  [\rm erg \ cm^{-3}]$"
        else:
            return r"$\tau - D"
        
    elif args.field[0] == "pre":
        if args.units:
            return r"$p [\rm erg \ cm^{-3}]$"
        else:
            return r"$p$"
    else:
        return args.field[0]


def get_frames(dir, max_file_num):
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames.sort(key=len, reverse=False) # sorts by ascending length
    frames = frames[:max_file_num]
    total_frames = len(frames)
    return total_frames, frames

def read_file(filepath, filename, args):
    is_cartesian = False
    field_dict = {}
    setup_dict = {}
    with h5py.File(filepath + filename, 'r+') as hf:
        ds = hf.get("sim_info")
        rho         = hf.get("rho")[:]
        v1          = hf.get("v1")[:]
        v2          = hf.get("v2")[:]
        p           = hf.get("p")[:]
        t           = ds.attrs["current_time"] * time_scale
        xmax        = ds.attrs["xmax"]
        xmin        = ds.attrs["xmin"]
        ymax        = ds.attrs["ymax"]
        ymin        = ds.attrs["ymin"]
        
        # New checkpoint files, so check if new attributes were
        # implemented or not
        try:
            nx          = ds.attrs["nx"]
            ny          = ds.attrs["ny"]
        except:
            nx          = ds.attrs["NX"]
            ny          = ds.attrs["NY"]
        try:
            gamma = ds.attrs["adiabatic_gamma"]
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
            
        try:
            chi = hf.get("chi")[:]
        except:
            chi = np.zeros((ny, nx))
        
        setup_dict["xmax"] = xmax 
        setup_dict["xmin"] = xmin 
        setup_dict["ymax"] = ymax 
        setup_dict["ymin"] = ymin 
        setup_dict["time"] = t
        
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
            setup_dict["xactive"] = xactive
            setup_dict["yactive"] = yactive
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            chi = chi[2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            setup_dict["xactive"] = xactive
            setup_dict["yactive"] = yactive
        
        if is_linspace:
            setup_dict["x1"] = np.linspace(xmin, xmax, xactive)
            setup_dict["x2"] = np.linspace(ymin, ymax, yactive)
        else:
            setup_dict["x1"] = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
            setup_dict["x2"] = np.linspace(ymin, ymax, yactive)
        
        if coord_sysem == "cartesian":
            is_cartesian = True
        
        setup_dict["is_cartesian"] = is_cartesian
        setup_dict["dataset"] = ds
        W    = 1/np.sqrt(1 -(v1**2 + v2**2))
        beta = np.sqrt(v1**2 + v2**2)
        
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        k = const.k_B.cgs
        m = const.m_p.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        h = 1. + gamma*p/(rho*(gamma - 1.))
        
        field_dict["rho"]          = rho
        field_dict["v1"]           = v1 
        field_dict["v2"]           = v2 
        field_dict["pre"]          = p
        field_dict["chi"]          = chi
        field_dict["chi_dens"]     = chi * rho * W
        field_dict["gamma_beta"]   = W*beta
        field_dict["gamma_beta_1"] = abs(W*v1)
        field_dict["gamma_beta_2"] = abs(W*v2)
        field_dict["temperature"]  = T
        field_dict["enthalpy"]     = h
        field_dict["W"]            = W
        field_dict["energy"]       = rho * h * W * W  - p - rho * W
        
    return field_dict, setup_dict

def plot_polar_plot(fig, axs, cbaxes, field_dict, args, mesh, ds):
    num_fields = len(args.field)
    if args.wedge:
        ax    = axs[0]
        wedge = axs[1]
    else:
        ax = axs
        
    rr, tt = mesh['rr'], mesh['theta']
    t2 = mesh['t2']
    xmax        = ds["xmax"]
    xmin        = ds["xmin"]
    ymax        = ds["ymax"]
    ymin        = ds["ymin"]
    
    vmin,vmax = args.cbar

    unit_scale = np.ones(num_fields)
    if (args.units):
        for idx, field in enumerate(args.field):
            if field == "rho" or field == "D":
                unit_scale[idx] = rho_scale.value
            elif field == "pre" or field == "energy" or field == "energy_rst":
                unit_scale[idx] = pre_scale.value
    
    units = unit_scale if args.units else np.ones(num_fields)
     
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds["time"]
    if num_fields > 1:
        cs  = np.zeros(4, dtype=object)
        var = []
        kwargs = []
        for field in args.field:
            if field in cons:
                var += np.split(prims2cons(field_dict, field), 2)
            else:
                var += np.split(field_dict[field], 2)
        units  = np.repeat(units, 2)
        print(units)
        zzz = input('')
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
        quadr[field2] = var[3]
        
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
                if field3 == field4 and field == field3:
                    ovmin = quadr[field][0].min()
                    ovmax = quadr[field][0].max()
                else:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                kwargs[field] =  {'vmin': ovmin, 'vmax': ovmax} if field in lin_fields else {'norm': colors.LogNorm(vmin = ovmin, vmax = ovmax)} 

        cs[0] = ax.pcolormesh(tchop[0],        rchop[0],  quadr[field1], cmap=color_map, shading='auto', **kwargs[field1])
        cs[1] = ax.pcolormesh(tchop[1],        rchop[0],  quadr[field2], cmap=color_map, shading='auto', **kwargs[field2])
        
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
        
        if args.field in cons:
            var = units * prims2cons(field_dict, args.field[0])
        else:
            var = units * field_dict[args.field[0]]
        
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto', **kwargs)
        cs[0] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, shading='auto', **kwargs)
    
    if ymax < np.pi:
        ax.set_position( [0.1, -0.18, 0.8, 1.43])
        # cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
        cbar_orientation = "horizontal"
        ymd = int( np.floor(ymax * 180/np.pi) )                                                                                                                                                                                     
        ax.set_thetamin(-ymd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        ax.set_thetamax(ymd)
    else:
        # cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        cbar_orientation = "vertical"
        
    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(cs[0], orientation=cbar_orientation, cax=cbaxes)
    
    if args.wedge:
        wedge_min = args.wedge_lims[0]
        wedge_max = args.wedge_lims[1]
        ang_min   = args.wedge_lims[2]
        ang_max   = args.wedge_lims[3]
        
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color="white")
    
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
        # ax.set_position( [0.1, -0.18, 0.8, 1.43])
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
        wedge.set_theta_zero_location("N")
        wedge.set_theta_direction(-1)
        wedge.yaxis.grid(True, alpha=0.2)
        wedge.xaxis.grid(True, alpha=0.2)
        wedge.tick_params(axis='both', labelsize=10)
        rlabels = ax.get_ymajorticklabels()
        for label in rlabels:
            label.set_color('white')
            
        wedge.axes.xaxis.set_ticklabels([])
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
        
    fig.suptitle('{} at t = {:.2f}'.format(args.setup[0], tend), fontsize=20, y=1)
    
def plot_cartesian_plot(fig, ax, cbaxes, field_dict, args, mesh, ds):
    xx, yy = mesh['xx'], mesh['yy']
    xmax        = ds["xmax"]
    xmin        = ds["xmin"]
    ymax        = ds["ymax"]
    ymin        = ds["ymin"]
    
    vmin,vmax = args.cbar

    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds["time"]
    c = ax.pcolormesh(xx, yy, field_dict[args.field], cmap=color_map, shading='auto', **kwargs)

        
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
        cbar.ax.set_ylabel(r'$\log$ {}'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f} s'.format(args.setup[0], tend), fontsize=20, y=0.95)
    
def create_mesh(fig, ax, filepath, filename, cbaxes, args):
    fields, setups = read_file(filepath, filename, args)
    
    ynpts, xnpts = fields["rho"].shape 

    mesh = {}
    if setups["is_cartesian"]:
        xx, yy = np.meshgrid(setups["x1"], setups["x2"])
        mesh["xx"] = xx
        mesh["yy"] = yy
        plot_cartesian_plot(fig, ax, cbaxes, fields, args, mesh, setups)
    else:      
        rr, tt = np.meshgrid(setups["x1"],  setups["x2"])
        rr, t2 = np.meshgrid(setups["x1"], -setups["x2"][::-1])
        mesh["theta"] = tt 
        mesh["rr"]    = rr
        mesh["t2"]    = t2
        mesh["r"]     = setups["x1"]
        mesh["th"]    = setups["x2"]
        plot_polar_plot(fig, ax, cbaxes, fields, args, mesh, setups)
    
    
        
    return ax

def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('data_dir', metavar='dir', nargs='+',
                        help='A data directory to retrieve the h5 files')
    
    parser.add_argument('setup', metavar='Setup', nargs='+', type=str,
                        help='The name of the setup you are plotting (e.g., Blandford McKee)')
    
    parser.add_argument('--field', dest = "field", metavar='Field Variable', nargs='+',
                        help='The name of the field variable you\'d like to plot',
                        choices=field_choices, default=["rho"])
    parser.add_argument('--rmax', dest = "rmax", metavar='Radial Domain Max',
                        default = 0.0, help='The domain range')
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar', nargs=2,
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cbar_sub', dest = "cbar2", metavar='Range of Color Bar for secondary plot',nargs='+',type=float,
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
    
    parser.add_argument('--file_max', dest='file_max', default = None, type=int)
    
    parser.add_argument('--save', dest='save', type=str,
                        default=None,
                        help='Save the fig with some name')
    
    parser.add_argument('--half', dest='half', action='store_true',
                        default=False,
                        help='True if you want half a polar plot')

   
    args = parser.parse_args()
    frame_count, flist = get_frames(args.data_dir[0], args.file_max)
    
    # read the first file and infer the system configuration from it
    init_setup = read_file(args.data_dir[0], flist[0], args)[1]
    if init_setup["is_cartesian"]:
        fig, ax = plt.subplots(1, 1, figsize=(10,10), constrained_layout=False)
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes('right', size='5%', pad=0.05)
        cbar_orientation = "vertical"
    else:
        if args.wedge:
            fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                         figsize=(15, 10), constrained_layout=True)
        else:
            fig = plt.figure(figsize=(15,8), constrained_layout=False)
            ax  = fig.add_subplot(111, projection='polar')
            
        if init_setup["ymax"] < np.pi:
            ax.set_position( [0.1, -0.18, 0.8, 1.43])
            cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
            cbar_orientation = "horizontal"
        else:
            cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
            cbar_orientation = "vertical"
    
    def init_mesh(filename):
        p = create_mesh(fig, ax, args.data_dir[0], filename, cbaxes, args)
        
        return p
        
    def update(frame, args):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """
        cbaxes.cla()
        if isinstance(ax, (list, np.ndarray)):
            for axs in ax:
                axs.cla()
        else:
            ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        pcolor_mesh = create_mesh(fig, ax, args.data_dir[0], flist[frame], cbaxes, args)

        return pcolor_mesh

    # Initialize plot
    inital_im = init_mesh(flist[0])

    animation = FuncAnimation(
        # Your Matplotlib Figure object
        fig,
        # The function that does the updating of the Figure
        update,
        # Frame information (here just frame number)
        np.arange(frame_count),
        #blit = True,
        # Extra arguments to the animate function
        fargs=[args],
        # repeat=False,
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval= 1000 / 5
    )

    if not args.save:
        plt.show()
    else:
        animation.save("{}.mp4".format(args.save.replace(" ", "_")))
    
    
    
if __name__ == "__main__":
    main()