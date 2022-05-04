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
import utility as util 

from utility import DEFAULT_SIZE, SMALL_SIZE, BIGGER_SIZE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import os, os.path

try:
    import cmasher as cmr 
except:
    print("No cmasher, so defaulting to matplotlib colormaps")

cons = ['D', 'momentum', 'energy', 'energy_rst']
field_choices = ['rho', 'v1', 'v2', 'p', 'gamma_beta', 'temperature', 'gamma_beta_1', 'gamma_beta_2', 'energy', 'mass', 'chi', 'chi_dens'] + cons 
lin_fields = ['chi', 'gamma_beta', 'gamma_beta_1', 'gamma_beta_2']


def get_frames(dir, max_file_num):
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames.sort(key=len, reverse=False) # sorts by ascending length
    frames = frames[:max_file_num]
    total_frames = len(frames)
    return total_frames, frames

def plot_polar_plot(fig, axs, cbaxes, field_dict, args, mesh, ds):
    num_fields = len(args.field)
    if args.wedge:
        ax    = axs[0]
        wedge = axs[1]
    else:
        ax = axs
        
    rr, tt = mesh['rr'], mesh['theta']
    t2     = -tt
    x1max        = ds["x1max"]
    x1min        = ds["x1min"]
    x2max        = ds["x2max"]
    x2min        = ds["x2min"]
    
    vmin,vmax = args.cbar[:2]

    unit_scale = np.ones(num_fields)
    if (args.units):
        for idx, field in enumerate(args.field):
            if field == "rho" or field == "D":
                unit_scale[idx] = rho_scale.value
            elif field == "p" or field == "energy" or field == "energy_rst":
                unit_scale[idx] = pre_scale.value
    
    units = unit_scale if args.units else np.ones(num_fields)
     
    if args.rcmap:
        color_map = (plt.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.get_cmap(args.cmap)
        
    tend = ds["time"]
    if num_fields > 1:
        cs  = np.zeros(4, dtype=object)
        var = []
        kwargs = []
        for field in args.field:
            if field in cons:
                if x2max == np.pi:
                    var += np.split(util.prims2var(field_dict, field), 2)
                else:
                    var.append(util.prims2var(field_dict, field))
            else:
                if x2max == np.pi:
                    var += np.split(field_dict[field], 2)
                else:
                    var.append(field_dict[field])
                
        if x2max == np.pi: 
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
                kwargs[field] =  {'vmin': vmin, 'vmax': vmax} if field in lin_fields else {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)} 
            else:
                if field3 == field4 and field == field3:
                    ovmin = quadr[field][0].min()
                    ovmax = quadr[field][0].max()
                else:
                    ovmin = quadr[field].min()
                    ovmax = quadr[field].max()
                kwargs[field] =  {'vmin': ovmin, 'vmax': ovmax} if field in lin_fields else {'norm': colors.LogNorm(vmin = ovmin, vmax = ovmax)} 

        if x2max < np.pi:
            cs[0] = ax.pcolormesh(tt[:: 1], rr,  var[0], cmap=color_map, shading='auto', **kwargs[field1])
            cs[1] = ax.pcolormesh(t2[::-1], rr,  var[1], cmap=color_map, shading='auto', **kwargs[field2])
            
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
            kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
        else:
            kwargs = {'vmin': vmin, 'vmax': vmax}
            
        cs = np.zeros(len(args.field), dtype=object)
        
        if args.field[0] in cons:
            var = units * util.prims2var(field_dict, args.field[0])
        else:
            var = units * field_dict[args.field[0]]
            
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto', **kwargs)
        cs[0] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, shading='auto', **kwargs)
        
        if args.bipolar:
            cs[0] = ax.pcolormesh(tt[:: 1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
            cs[0] = ax.pcolormesh(t2[::-1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
    
    if args.pictorial: 
        ax.set_position( [0.1, -0.15, 0.8, 1.3])
            
    if not args.pictorial:
        if x2max < np.pi:
            ymd = int( np.floor(x2max * 180/np.pi) )
            if not args.bipolar:                                                                                                                                                                                   
                ax.set_thetamin(-ymd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                ax.set_thetamax(ymd)
                ax.set_position( [0.1, -0.18, 0.8, 1.43])
            else:
                ax.set_position( [0.1, -0.18, 0.9, 1.43])
            if num_fields > 1:
                ycoord  = [0.1, 0.1 ]
                xcoord  = [0.88, 0.04]
                # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                cbar_orientation = "vertical"
            else:
                # cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
                cbar_orientation = "horizontal"
                
            
        else:
            if num_fields > 1:
                if num_fields == 2:
                    ycoord  = [0.1, 0.1 ]
                    xcoord  = [0.1, 0.85]
                    # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                    
                if num_fields == 3:
                    ycoord  = [0.1, 0.5, 0.1]
                    xcoord  = [0.07, 0.85, 0.85]
                    # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8 * 0.5]) for i in range(1, num_fields)]
                    # cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                if num_fields == 4:
                    ycoord  = [0.5, 0.1, 0.5, 0.1]
                    xcoord  = [0.85, 0.85, 0.07, 0.07]
                    # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
                    
                cbar_orientation = "vertical"
            else:
                # cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
                cbar_orientation = "vertical"
        
        if args.log:
            if num_fields > 1:
                fmt  = [None if field in lin_fields else tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True) for field in args.field]
                cbar = [fig.colorbar(cs[i], orientation=cbar_orientation, cax=cbaxes[i],       format=fmt[i]) for i in range(num_fields)]
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
        
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=2, color="white")
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=2, color="white")
    
    
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    ax.tick_params(axis='both', labelsize=10)
    rlabels = ax.get_ymajorticklabels()
    if not args.pictorial:
        for label in rlabels:
            label.set_color('white')
    else:
        ax.axes.yaxis.set_ticklabels([])
        
    ax.axes.xaxis.set_ticklabels([])
    ax.set_rmax(x1max) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    ax.set_rmin(x1min)
    
    field_str = util.get_field_str(args)
    
    if args.wedge:
        if num_fields == 1:
            wedge.set_position( [0.5, -0.5, 0.3, 2])
            ax.set_position( [0.05, -0.5, 0.46, 2])
        else:
            ax.set_position( [0.16, -0.5, 0.46, 2])
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
            wedge.pcolormesh(tchop[1], rchop[1], quadr[field2], cmap=args.cmap2, shading='nearest', **kwargs[field2])
        else:
            vmin2, vmax2 = args.cbar2
            if args.log:
                kwargs = {'norm': colors.LogNorm(vmin = vmin2, vmax = vmax2)}
            else:
                kwargs = {'vmin': vmin2, 'vmax': vmax2}
            wedge.pcolormesh(tt, rr, var, cmap=color_map, shading='nearest', **kwargs)
            
        wedge.set_theta_zero_location("N")
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
            if x2max == np.pi:
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
            if x2max >= np.pi:
                cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
            else:
                cbar.ax.set_xlabel(r'{}'.format(field_str), fontsize=20)
        
        fig.suptitle('{} at t = {:.2f}'.format(args.setup[0], tend), fontsize=20, y=1)
    
def plot_cartesian_plot(fig, ax, cbaxes, field_dict, args, mesh, ds):
    xx, yy = mesh['xx'], mesh['yy']
    x1max        = ds["x1max"]
    x1min        = ds["x1min"]
    x2max        = ds["x2max"]
    x2min        = ds["x2min"]
    
    vmin,vmax = args.cbar

    if args.log:
        kwargs = {'norm': colors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'norm': colors.PowerNorm(2.0, vmin=vmin, vmax=vmax)}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds["time"]
    ax.yaxis.grid(True, alpha=0.1)
    ax.xaxis.grid(True, alpha=0.1)
    c = ax.pcolormesh(xx, yy, field_dict[args.field[0]], cmap=color_map, shading='auto', **kwargs)

    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(c, orientation="vertical", cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(c, orientation="vertical", cax=cbaxes)

    ax.tick_params(axis='both', labelsize=10)
    
    # Change the format of the field
    field_str = util.get_field_str(args)
    
    if args.log:
        cbar.ax.set_ylabel(r'$\log$ {}'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        
    fig.suptitle('{} at t = {:.2f} s'.format(args.setup[0], tend), fontsize=20, y=0.95)
    
def create_mesh(fig, ax, filepath, filename, cbaxes, args):
    fields, setups, mesh = util.read_2d_file(args, filepath+filename)
    if setups["is_cartesian"]:
        plot_cartesian_plot(fig, ax, cbaxes, fields, args, mesh, setups)
    else:      
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
    
    parser.add_argument('--cbar_range', dest = "cbar", metavar='Range of Color Bar(s)', nargs='+',
                        default = [None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cbar_sub', dest = "cbar2", metavar='Range of Color Bar for secondary plot',nargs='+',type=float,
                        default =[None, None], help='The colorbar range you\'d like to plot')
    
    parser.add_argument('--cmap', dest = "cmap", metavar='Color Bar Colarmap',
                        default = 'magma', help='The colorbar cmap you\'d like to plot')
    parser.add_argument('--cmap2', dest = "cmap2", metavar='Color Bar #2 Colarmap',
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
    
    parser.add_argument('--tex', dest='tex', action='store_true',
                        default=False, help='True if you want the latex formatting')
    
    parser.add_argument('--labels', dest='labels', nargs="+", default = None,
                        help='Optionally give a list of labels for multi-file plotting')
    
    parser.add_argument('--tidx', dest='tidx', type=int, default = None,
                        help='Set to a value if you wish to plot a 1D curve about some angle')
    
    parser.add_argument('--wedge', dest='wedge', default=False, action='store_true')
    parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 80, 110], type=float, nargs=4)
    parser.add_argument('--units', dest='units', default = False, action='store_true')
    parser.add_argument('--file_max', dest='file_max', default = None, type=int)
    parser.add_argument('--frame_range', dest='frame_range', default = [None, None], nargs=2, type=int)
    parser.add_argument('--dbg', dest='dbg', default = False, action='store_true')
    parser.add_argument('--bipolar', dest='bipolar', default = False, action='store_true')
    parser.add_argument('--pictorial', dest='pictorial', default = False, action='store_true')
    
    parser.add_argument('--save', dest='save', type=str,
                        default=None,
                        help='Save the fig with some name')
    
    parser.add_argument('--half', dest='half', action='store_true',
                        default=False,
                        help='True if you want half a polar plot')
        
    args = parser.parse_args()
    
    if args.tex:
        plt.rc('font',   size=DEFAULT_SIZE)          # controls default text sizes
        plt.rc('axes',   titlesize=DEFAULT_SIZE)     # fontsize of the axes title
        plt.rc('axes',   labelsize=DEFAULT_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('ytick',  labelsize=DEFAULT_SIZE)     # fontsize of the tick labels
        plt.rc('legend', fontsize=DEFAULT_SIZE)      # legend fontsize
        plt.rc('figure', titlesize=DEFAULT_SIZE)    # fontsize of the figure title
        
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": "Times New Roman",
                "font.size": DEFAULT_SIZE
            }
        )
            
    if args.dbg:
        plt.style.use('dark_background')
        
    frame_count, flist = get_frames(args.data_dir[0], args.file_max)
    
    flist      = flist[args.frame_range[0]: args.frame_range[1]]
    frame_count = len(flist)
    
    num_fields = len(args.field)
    if num_fields > 1:
        if len(args.cbar) == 2*num_fields:
            pass
        else:
            args.cbar += (num_fields - 1) * [None, None]
            
    # read the first file and infer the system configuration from it
    init_setup = util.read_2d_file(args, args.data_dir[0] + flist[0])[1]
    if init_setup["is_cartesian"]:
        fig, ax = plt.subplots(1, 1, figsize=(11,10), constrained_layout=False)
        divider = make_axes_locatable(ax)
        if not args.pictorial:
            cbaxes = divider.append_axes('right', size='5%', pad=0.05)
        cbar_orientation = "vertical"
    else:
        if args.wedge:
            fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'},
                         figsize=(15, 10), constrained_layout=True)
        else:
            fig = plt.figure(figsize=(15,8), constrained_layout=False)
            ax  = fig.add_subplot(111, projection='polar')
            ax.grid(False)
            
        if init_setup["x2max"] < np.pi:
            ax.set_position( [0.1, -0.18, 0.8, 1.43])
            if not args.pictorial:
                cbaxes  = fig.add_axes([0.2, 0.1, 0.6, 0.04]) 
            cbar_orientation = "horizontal"
        else:
            if num_fields > 1:
                if num_fields == 2:
                    ycoord  = [0.1, 0.1 ]
                    xcoord  = [0.1, 0.85]
                    if not args.pictorial:
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                    
                if num_fields == 3:
                    ycoord  = [0.1, 0.5, 0.1]
                    xcoord  = [0.07, 0.85, 0.85]
                    if args.pictorial:
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8 * 0.5]) for i in range(1, num_fields)]
                        cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                if num_fields == 4:
                    ycoord  = [0.5, 0.1, 0.5, 0.1]
                    xcoord  = [0.85, 0.85, 0.07, 0.07]
                    if args.pictorial:
                        cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
            else:
                if args.pictorial:
                    pass
                    # cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
            cbar_orientation = "vertical"
    
    def init_mesh(filename):
        if not args.pictorial:
            p = create_mesh(fig, ax, args.data_dir[0], filename, cbaxes, args)
        else:
            p = create_mesh(fig, ax, args.data_dir[0], filename, None, args)
        
        return p
        
    def update(frame, args):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """
        if not args.pictorial:
            try:
                for cbax in cbaxes:
                    cbax.cla()
            except:
                cbaxes.cla()
            
        if isinstance(ax, (list, np.ndarray)):
            for axs in ax:
                axs.cla()
        else:
            ax.cla()
        # Not strictly neccessary, just so we know we are stealing these from
        # the global scope
        if not args.pictorial:
            pcolor_mesh = create_mesh(fig, ax, args.data_dir[0], flist[frame], cbaxes, args)
        else:
            pcolor_mesh = create_mesh(fig, ax, args.data_dir[0], flist[frame], None, args)

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
        interval= 1000 / 10
    )

    if not args.save:
        plt.show()
    else:
        dpi = 600
        animation.save("{}.mp4".format(args.save.replace(" ", "_")), codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
    
    
    
if __name__ == "__main__":
    main()