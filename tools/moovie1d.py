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
    # Get number of files in dir
    total_frames = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])

    return total_frames, frames

def create_grid(fig, ax, filepath, filename, field, setup, 
                log=False, forder=False, rmax= 0.0,
                ):
    
    field_dict = {}
    with h5py.File(filepath + filename, 'r+') as hf:
        
        ds = hf.get("sim_info")
        
        rho         = hf.get("rho")[:]
        v           = hf.get("v")[:]
        p           = hf.get("p")[:]
        nx          = ds.attrs["Nx"]
        t           = ds.attrs["current_time"]
        xmax        = ds.attrs["xmax"]
        xmin        = ds.attrs["xmin"]
        
        
        if forder:
            rho = rho[1:-1]
            v   = v[1:-1]
            p   = p  [1:-1]
            xactive = nx - 2
        else:
            rho = rho[2:-2]
            v   = v [2:-2]
            p   = p  [2:-2]
            xactive = nx - 4
            
        W    = 1/np.sqrt(1 - v**2)
        beta = v
        
        e = 3*p/rho 
        c = const.c.cgs.value
        a = (4 * const.sigma_sb.cgs.value / c)
        m = const.m_p.cgs.value
        T = (3 * p * c ** 2  / a)**(1./4.)
        
        
        field_dict["rho"]         = rho
        field_dict["v"]           = v
        field_dict["p"]           = p
        field_dict["gamma_beta"]  = W*beta
        field_dict["temperature"] = T
        
        
    xnpts = xactive

    if (log):
        r = np.logspace(np.log10(xmin), np.log10(xmax), xactive)
        if field != "v" and field != "gamma_beta":
            ax.loglog(r, field_dict[field])
        else:
            ax.semilogx(r, field_dict[field])
    else:
        r = np.linspace(xmin, xmax, xactive)
        ax.plot(r, field_dict[field])
        
    ax.set_xlabel('r/R0', fontsize=20)


    # if log:
    #     logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
    #     cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes, format=logfmt)
    # else:
    #     cbar = fig.colorbar(c2, orientation='horizontal', cax=cbaxes)
        
    ax.set_title('{} at t = {:.2f} s'.format(setup, t), fontsize=20)
    
    ax.tick_params(axis='both', labelsize=10)
    ax.set_xlim(xmin, xmax) if rmax == 0.0 else ax.set_xlim(xmin, rmax)

    # Change the format of the field
    if log:
        if   field == "rho":
            field_str = r'Log $\rho$'
        elif field == "gamma_beta":
            field_str = r"$\Gamma \ \beta$"
        elif field == "temperature":
            field_str = "Log T [K]"
        else:
            field_str = field
    else:
        if  field == "rho":
            field_str = r' $\rho$'
        elif field == "gamma_beta":
            field_str = r" $\Gamma \ \beta$"
        elif field == "temperature":
            field_str = " T [K]"
        else:
            field_str = field
    
    
    ax.set_ylabel('{}'.format(field_str), fontsize=20)
    
    return ax
    

field_choices = ['rho', 'v', 'p', 'gamma_beta', 'temperature']

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
    
    parser.add_argument('--log', dest='log', action='store_true',
                        default=False,
                        help='Logarithmic Radial Grid Option')
    
    parser.add_argument('--first_order', dest='forder', action='store_true',
                        default=False,
                        help='True if this is a grid using RK1')

   
    args = parser.parse_args()
    
    fig = plt.figure(figsize=(15,8))
    ax  = fig.add_subplot(111)
    
    frame_count, flist = get_frames(args.data_dir[0])
    
    def init_mesh(filename):
        p = create_grid(fig, ax, args.data_dir[0], filename, args.field,
                        args.setup[0], args.log, args.forder,
                        args.rmax)
        
        return p
        
    def update(frame, *fargs):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """

        ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        pcolor_mesh = create_grid(fig, ax, args.data_dir[0], flist[frame], *fargs)

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
                args.setup[0], args.log, args.forder,
                args.rmax],
        # repeat=False,
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval= 1000 / 25
    )

    # plt.show()
    # Try to set the DPI to the actual number of pixels you're plotting
    animation.save("science/{}_{}.mp4".format(args.setup[0], args.field))
    
    
if __name__ == "__main__":
    main()

import h5py 