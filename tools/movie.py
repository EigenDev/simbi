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

def get_frames(dir):
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    total_frames = len(frames)
    frames.sort(key=len, reverse=False) # sorts by ascending length
    
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
        
        setup_dict["xmax"] = xmax 
        setup_dict["xmin"] = xmin 
        setup_dict["ymax"] = ymax 
        setup_dict["ymin"] = ymin 
        setup_dict["time"] = t
        
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
            setup_dict["xactive"] = xactive
            setup_dict["yactive"] = yactive
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
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
        
        field_dict["rho"]         = rho
        field_dict["v1"]          = v1 
        field_dict["v2"]          = v2 
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        field_dict["enthalpy"]    = h
        field_dict["W"]           = W
        field_dict["energy"]      = rho * h * W * W  - p - rho * W
        
    return field_dict, setup_dict

def plot_polar_plot(fig, ax, cbaxes, field_dict, args, mesh, ds):
    rr, tt = mesh['rr'], mesh['theta']
    t2 = mesh['t2']
    xmax        = ds["xmax"]
    xmin        = ds["xmin"]
    ymax        = ds["ymax"]
    ymin        = ds["ymin"]
    
    vmin,vmax = eval(args.cbar)

    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds["time"]
    c1 = ax.pcolormesh(tt, rr, field_dict[args.field], cmap=color_map, shading='auto', **kwargs)
    c2 = ax.pcolormesh(t2[::-1], rr, field_dict[args.field],  cmap=color_map, shading='auto', **kwargs)
    
    
        
    if ymax < np.pi:
        cbar_orientation = "horizontal"
        ymd = int( np.floor(ymax * 180/np.pi) )
        ax.set_thetamin(-ymd)
        ax.set_thetamax(ymd)
    else:
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
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(xmax) if args.rmax == 0.0 else ax.set_rmax(args.rmax)

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
        if ymax >= np.pi:
            cbar.ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
        else:
            cbar.ax.set_xlabel(r'$\log$[{}]'.format(field_str), fontsize=20)
    else:
        if ymax >= np.pi:
            cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        else:
            cbar.ax.set_xlabel(r'{}'.format(field_str), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f} s'.format(args.setup[0], tend), fontsize=20, y=0.95)
    
def plot_cartesian_plot(fig, ax, cbaxes, field_dict, args, mesh, ds):
    xx, yy = mesh['xx'], mesh['yy']
    xmax        = ds["xmax"]
    xmin        = ds["xmin"]
    ymax        = ds["ymax"]
    ymin        = ds["ymin"]
    
    vmin,vmax = eval(args.cbar)

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
        cbar.ax.set_ylabel(r'$\log$[{}]'.format(field_str), fontsize=20)
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
    

field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature']

def main():
    parser = argparse.ArgumentParser(
        description='Plot a 2D Figure From a File (H5).',
        epilog="This Only Supports H5 Files Right Now")
    
    parser.add_argument('data_dir', metavar='dir', nargs='+',
                        help='A data directory to retrieve the h5 files')
    
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

    parser.add_argument('--save', dest='save', action='store_true',
                        default=False,
                        help='True if you want save the fig')
    
    parser.add_argument('--half', dest='half', action='store_true',
                        default=False,
                        help='True if you want half a polar plot')

   
    args = parser.parse_args()
    frame_count, flist = get_frames(args.data_dir[0])
    
    # read the first file and infer the system configuration from it
    init_setup = read_file(args.data_dir[0], flist[0], args)[1]
    if init_setup["is_cartesian"]:
        fig, ax = plt.subplots(1, 1, figsize=(10,10), constrained_layout=False)
        divider = make_axes_locatable(ax)
        cbaxes = divider.append_axes('right', size='5%', pad=0.05)
        cbar_orientation = "vertical"
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
        interval= 1000 / 20
    )

    if args.save:
        animation.save("{}.mp4".format(args.setup[0]).replace(" ", "_"))
    else:
        plt.show()
    
    
    
if __name__ == "__main__":
    main()

import h5py 