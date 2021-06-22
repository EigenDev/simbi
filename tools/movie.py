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

from matplotlib.animation import FuncAnimation
import os, os.path

def get_frames(dir):
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    total_frames = len(frames)
    frames.sort(key=len, reverse=False) # sorts by ascending length
    
    return total_frames, frames

def create_mesh(fig, ax, filepath, filename, field, setup, cbaxes, vmin = None, vmax = None, 
                log=False, forder=False, rcmap = False, cmap='magma',
                rmax= 0.0,
                ):
    
    field_dict = {}
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
        
        
        rho = rho.reshape(ny, nx)
        v1  = v1.reshape(ny, nx)
        v2  = v2.reshape(ny, nx)
        p   = p.reshape(ny, nx)
        
        if forder:
            rho = rho[1:-1, 1: -1]
            v1  = v1 [1:-1, 1: -1]
            v2  = v2 [1:-1, 1: -1]
            p   = p  [1:-1, 1: -1]
            xactive = nx - 2
            yactive = ny - 2
        else:
            rho = rho[2:-2, 2: -2]
            v1  = v1 [2:-2, 2: -2]
            v2  = v2 [2:-2, 2: -2]
            p   = p  [2:-2, 2: -2]
            xactive = nx - 4
            yactive = ny - 4
            
        W    = 1/np.sqrt(1 - v1**2 + v2**2)
        beta = np.sqrt(v1**2 + v2**2)
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        m = const.m_p.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        
        
        field_dict["rho"]         = rho
        field_dict["v1"]          = v1 
        field_dict["v2"]          = v2 
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        
        
    ynpts, xnpts = rho.shape 

    if (log):
        r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
    else:
        r = np.linspace(xmin, xmax, xactive)
        
    # r = np.logspace(np.log10(0.01), np.log10(0.5), xnpts)
    theta = np.linspace(ymin, ymax, yactive)
    theta_mirror = - theta[::-1]
    theta_mirror[-1] *= -1.
    
    rr, tt = np.meshgrid(r, theta)
    rr, t2 = np.meshgrid(r, theta_mirror)
    
    norm  = colors.LogNorm(vmin = vmin, vmax = vmax)
    
    if rcmap:
        color_map = (plt.cm.get_cmap(cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(cmap)


    c1 = ax.pcolormesh(tt, rr, field_dict[field], cmap=color_map, shading='auto', norm = norm)
    c2 = ax.pcolormesh(t2[::-1], rr, field_dict[field],  cmap=color_map, shading='auto', norm = norm)


    if log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c2, orientation='vertical', cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
        
    fig.suptitle('SIMBI: {} at t = {:.2f} s'.format(setup, t), fontsize=20, y=0.95)
    
    # ax.set_position( [0.1, -0.18, 0.8, 1.43])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    ax.tick_params(axis='both', labelsize=10)
    cbaxes.tick_params(axis='x', labelsize=10)
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(xmax) if rmax == 0.0 else ax.set_rmax(rmax)
    
    ymd = int( np.ceil(ymax * 180/np.pi) )
    ax.set_thetamin(-ymd)
    ax.set_thetamax(ymd)

    # Change the format of the field
    if   field == "rho":
        field_str = r'$\rho$'
    elif field == "gamma_beta":
        field_str = r"$\Gamma \ \beta$"
    else:
        field_str = field
    
    if log:
        cbar.ax.set_xlabel('Log [{}]'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_xlabel('[{}]'.format(field), fontsize=20)
        
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
    vmin, vmax = eval(args.cbar)
    
    fig = plt.figure(figsize=(15,8), constrained_layout=False)
    ax  = fig.add_subplot(111, projection='polar')
    if args.half:
        ax.set_position( [0.1, -0.18, 0.8, 1.43])
        cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
        cbar_orientation = "horizontal"
    else:
        cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
        cbar_orientation = "vertical"
    # cbaxes = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
    
    frame_count, flist = get_frames(args.data_dir[0])
    
    def init_mesh(filename):
        p = create_mesh(fig, ax, args.data_dir[0], filename, args.field,
                        args.setup[0], cbaxes,  vmin,
                        vmax, args.log, args.forder,
                        args.rcmap, args.cmap, args.rmax)
        
        return p
        
    def update(frame, *fargs):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """
        cbaxes.cla()
        ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        pcolor_mesh = create_mesh(fig, ax, args.data_dir[0], flist[frame], *fargs)

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
        fargs=[args.field,
                args.setup[0], cbaxes, vmin,
                vmax, args.log, args.forder,
                args.rcmap, args.cmap, args.rmax],
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