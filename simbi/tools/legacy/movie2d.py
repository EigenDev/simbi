
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as tkr
import time
import matplotlib.colors as mcolors
import argparse 
import h5py 
import astropy.constants as const
import utility as util 
from visual import derived, lin_fields
from ..detail import get_subparser
from utility import DEFAULT_SIZE, SMALL_SIZE, BIGGER_SIZE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
import os, os.path

try:
    import cmasher as cmr 
except:
    print("No cmasher, so defaulting to matplotlib colormaps")
    
def get_frames(dir, max_file_num):
    frames       = sorted([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    frames.sort(key=len, reverse=False) # sorts by ascending length
    frames = frames[:max_file_num]
    total_frames = len(frames)
    return total_frames, frames

def plot_polar_plot(fig, axs, cbaxes, fields, args, mesh, dset, subplots=False):
    num_fields = len(args.fields)
    is_wedge   = args.nwedge > 0
    rr, tt = mesh['x1'], mesh['x2']
    t2     = - tt[::-1]
    x1max  = dset['x1max']
    x1min  = dset['x1min']
    x2max  = dset['x2max']
    x2min  = dset['x2min']
    if not subplots:
        if is_wedge:
            nplots = args.nwedge + 1
            ax    = axs[0]
            wedge = axs[1]
        else:
            if x2max < np.pi:
                figsize = (8, 5)
            else:
                figsize = (10, 8)
            ax = axs
    else:
        if is_wedge:
            ax    = axs[0]
            wedge = axs[1]
        else:
            ax = axs
    
    vmin,vmax = args.cbar[:2]

    unit_scale = np.ones(num_fields)
    if (args.units):
        for idx, field in enumerate(args.fields):
            if field == 'rho' or field == 'D':
                unit_scale[idx] = util.rho_scale.value
            elif field == 'p' or field == 'energy' or field == 'energy_rst':
                unit_scale[idx] = util.pre_scale.value
    
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
                    var += np.split(util.prims2var(fields, field), 2)
                else:
                    var.append(util.prims2var(fields, field))
            else:
                if x2max == np.pi:
                    var += np.split(fields[field], 2)
                else:
                    var.append(fields[field])
                
        if x2max == np.pi: 
            units  = np.repeat(units, 2)
            
        var    = np.asanyarray(var)
        var    = np.array([units[idx] * var[idx] for idx in range(var.shape[0])])
        
        tchop  = np.split(tt, 2)
        trchop = np.split(t2, 2)
        rchop  = np.split(rr, 2)
        
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
        ax.grid(False)
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
            
        cs = np.empty(2 if not args.bipolar else 4, dtype=object)
        
        if args.fields[0] in derived:
            var = units * util.prims2var(fields, args.fields[0])
        else:
            var = units * fields[args.fields[0]]
        
        cs[0] = ax.pcolormesh(tt, rr, var, cmap=color_map, shading='auto',
                              linewidth=0, rasterized=True, **kwargs)
        cs[1] = ax.pcolormesh(t2[::-1], rr, var,  cmap=color_map, 
                              linewidth=0,rasterized=True, shading='auto', **kwargs)
        
        if args.bipolar:
            cs[2] = ax.pcolormesh(tt[:: 1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
            cs[3] = ax.pcolormesh(t2[::-1] + np.pi, rr,  var, cmap=color_map, shading='auto', **kwargs)
    
    if args.pictorial: 
        ax.set_position([0.1, -0.15, 0.8, 1.30])
    
    # =================================================
    #                   DRAW DASHED LINE
    # =================================================
    # angs    = np.linspace(x2min, x2max, 1000)
    # eps     = 0.2
    # a       = 0.47 * (1 - eps)**(-1/3)
    # b       = 0.47 * (1 - eps)**(2/3)
    # radius  = lambda theta: a*b/((a*np.cos(theta))**2 + (b*np.sin(theta))**2)**0.5
    # r_theta = radius(angs)

    # ax.plot(np.radians(np.linspace(0, 180, 1000)), r_theta, linewidth=1, linestyle='--', color='orange')
    # ax.plot(-np.radians(np.linspace(0, 180, 1000)), r_theta, linewidth=1, linestyle='--', color='orange')
    
    if not args.pictorial:
        if x2max < np.pi:
            ymd = int( np.floor(x2max * 180/np.pi) )
            if not args.bipolar:                                                                                                                                                                                   
                ax.set_thetamin(-ymd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                ax.set_thetamax(ymd)
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
                    # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8]) for i in range(num_fields)]
                else:
                    ycoord  = [0.15, 0.15]
                    xcoord  = [0.51, 0.06]
                    # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.43, 0.05]) for i in range(num_fields)]
            else:
                cbar_orientation = 'horizontal'
                if cbar_orientation == 'horizontal':
                    pass
                    # cbaxes  = fig.add_axes([0.15, 0.15, 0.70, 0.05]) 
        else:  
            if not args.no_cbar:         
                cbar_orientation = args.cbar_orient
                # ax.set_position([0.1, -0.18, 0.7, 1.3])
                if num_fields > 1:
                    if num_fields == 2:
                        if cbar_orientation == 'vertical':
                            ycoord  = [0.1, 0.08] if x2max < np.pi else [0.15, 0.15]
                            xcoord  = [0.1, 0.85] if x2max < np.pi else [0.93, 0.08]
                            # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.65]) for i in range(num_fields)]
                        else:
                            if is_wedge:
                                ycoord  = [0.2, 0.20] if x2max < np.pi else [0.15, 0.15]
                                xcoord  = [0.1, 0.50] if x2max < np.pi else [0.52, 0.04]
                            else:
                                ycoord  = [0.2, 0.20] if x2max < np.pi else [0.10, 0.10]
                                xcoord  = [0.1, 0.50] if x2max < np.pi else [0.51, 0.20]
                            # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.45, 0.04]) for i in range(num_fields)]
                    if num_fields == 3:
                        ycoord  = [0.1, 0.5, 0.1]
                        xcoord  = [0.07, 0.85, 0.85]
                        # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.4]) for i in range(1, num_fields)]
                        # cbaxes.append(fig.add_axes([xcoord[0], ycoord[0] ,0.03, 0.8]))
                    if num_fields == 4:
                        ycoord  = [0.5, 0.1, 0.5, 0.1]
                        xcoord  = [0.85, 0.85, 0.07, 0.07]
                        # cbaxes  = [fig.add_axes([xcoord[i], ycoord[i] ,0.03, 0.8/(0.5 * num_fields)]) for i in range(num_fields)]
                else:
                    if not is_wedge:
                        pass
                        #plt.tight_layout()
                        
                    # if cbar_orientation == 'vertical':
                    #     # cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.85])
                    # else:
                    #     # cbaxes  = fig.add_axes([0.86, 0.07, 0.03, 0.90])
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
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=1, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='white')
        ax.plot(np.radians(np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='white')
        ax.plot(np.radians(np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=1, color='white')
   
            
        if args.nwedge == 2:
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_max, wedge_max, 1000), linewidth=1, color='white')
            ax.plot(np.radians(-np.linspace(ang_min, ang_min, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='white')
            ax.plot(np.radians(-np.linspace(ang_max, ang_max, 1000)), np.linspace(wedge_min, wedge_max, 1000), linewidth=1, color='white')
            ax.plot(np.radians(-np.linspace(ang_min, ang_max, 1000)), np.linspace(wedge_min, wedge_min, 1000), linewidth=1, color='white')
            
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.tick_params(axis='both', labelsize=15)
    rlabels = ax.get_ymajorticklabels()
    # if not args.pictorial:
    #     for label in rlabels:
    #         label.set_color('white')
    # else:
    #     ax.axes.yaxis.set_ticklabels([])
        
    # ax.axes.xaxis.set_ticklabels([])
    # ax.axes.yaxis.set_ticklabels([])
    # ax.set_rmax(x1max) if args.rmax == 0.0 else ax.set_rmax(args.rmax)
    # ax.set_rmin(x1min)
    
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
                                cbar[i].set_label(r'{}'.format(field_str[i]), fontsize=fsize, labelpad = -40 )
                                # xticks = [0.10, 0.20, 0.35, 0.50]
                                # cbar[i].set_ticks(xticks)
                                # cbar[i].set_ticklabels(['%.2f' % x for x in xticks])
                                # loc = tkr.MultipleLocator(base=0.12) # this locator puts ticks at regular intervals
                                # cbaxes[i].xaxis.set_major_locator(loc)
                                cbaxes[i].yaxis.set_ticks_position('left')
                            else:
                                cbar[i].set_label(r'$\log$ {}'.format(field_str[i]), fontsize=fsize)
                                # cbaxes[i].xaxis.set_major_locator(plt.MaxNLocator(4))
                    else:
                        cbar.set_label(r'$\log$ {}'.format(field_str), fontsize=fsize)
                else:
                    if num_fields > 1:
                        for i in range(num_fields):
                            if args.fields[i] in lin_fields:
                                cbar[i].set_label(r'{}'.format(field_str[i]), fontsize=fsize)
                            else:
                                cbar[i].set_label(r'$\log$ {}'.format(field_str[i]), fontsize=fsize)
                    else:
                        cbar.set_label(r'$\log$ {}'.format(field_str), fontsize=fsize)
            else:
                if x2max >= np.pi:
                    if num_fields > 1:
                        for i in range(num_fields):
                            cbar[i].set_label(r'{}'.format(field_str[i]), fontsize=fsize)
                    else:
                        cbar.set_label(f'{field_str}', fontsize=fsize)
                else:
                    cbar.set_label(r'{}'.format(field_str), fontsize=fsize)
        
        if args.setup != "":
            fig.suptitle('{} at t = {:.2f}'.format(args.setup, tend), fontsize=25, y=1)
        else:
            fsize = 25 if not args.print else DEFAULT_SIZE
            fig.suptitle('t = {:d}'.format(int(tend.value)), fontsize=fsize, y=0.95)
    
    return cs
    
def plot_cartesian_plot(fig, ax, cbaxes, fields, args, mesh, ds):
    plots = []
    x1, x2 = mesh['x1'], mesh['x2']
    
    vmin,vmax = args.cbar
    if args.log:
        kwargs = {'norm': mcolors.LogNorm(vmin = vmin, vmax = vmax)}
    else:
        kwargs = {'vmin': vmin, 'vmax': vmax}
        # kwargs = {'norm': mcolors.PowerNorm(2.0, vmin=vmin, vmax=vmax)}
        
    if args.rcmap:
        color_map = (plt.cm.get_cmap(args.cmap)).reversed()
    else:
        color_map = plt.cm.get_cmap(args.cmap)
        
    tend = ds["time"] * (util.time_scale if args.units else 1.0)
    ax.grid(False)
    if args.fields[0] in derived:
        var = util.prims2var(fields, args.fields[0])
    else:
        var = fields[args.fields[0]]
        
    plots += [ax.pcolormesh(x1, x2, var, cmap=color_map, shading='auto', **kwargs)]
    if ds['coord_system'] == 'axis_cylindrical':
        plots += [ax.pcolormesh(-x1, x2, var, cmap=color_map, shading='auto', **kwargs)]

    if args.log:
        logfmt = tkr.LogFormatterExponent(base=10.0, labelOnlyBase=True)
        cbar = fig.colorbar(plots[0], orientation="vertical", cax=cbaxes, format=logfmt)
    else:
        cbar = fig.colorbar(plots[0], orientation="vertical", cax=cbaxes)

    ax.tick_params(axis='both', labelsize=10)
    
    # Change the format of the field
    field_str = util.get_field_str(args)
    if args.log:
        cbar.ax.set_ylabel(r'$\log$ {}'.format(field_str), fontsize=20)
    else:
        cbar.ax.set_ylabel(r'{}'.format(field_str), fontsize=20)
        
    ax.set_title('{} at t = {:.2f}'.format(args.setup, tend), fontsize=20)
    
    return plots
    
def create_mesh(fig, ax, filename, cbaxes, args):
    fields, setups, mesh = util.read_file(args, filename, ndim=2)
    if setups["is_cartesian"]:
        c = plot_cartesian_plot(fig, ax, cbaxes, fields, args, mesh, setups)
    else:      
        c = plot_polar_plot(fig, ax, cbaxes, fields, args, mesh, setups)        
    return c

def get_data(filename, args):
    fields, setups, _ = util.read_file(args, filename, ndim=2)
    if args.fields[0] in derived:
        var = util.prims2var(fields, args.fields[0])
    else:
        var = fields[args.fields[0]]
        
    return setups, var
    
def movie(parser: argparse.ArgumentParser) -> None:
    plot_parser = get_subparser(parser, 1)
    plot_parser.add_argument('--cbar_sub', dest = "cbar2", metavar='Range of Color Bar for secondary plot',nargs='+',type=float,default =[None, None], help='The colorbar range you\'d like to plot')
    plot_parser.add_argument('--cmap2', dest = "cmap2", metavar='Color Bar #2 Colarmap', default = 'magma', help='The colorbar cmap you\'d like to plot')
    plot_parser.add_argument('--rev_cmap', dest='rcmap', action='store_true', default=False, help='True if you want the colormap to be reversed')
    plot_parser.add_argument('--x', dest='x', nargs="+", default = None, type=float,help='List of x values to plot field max against')
    plot_parser.add_argument('--xlabel', dest='xlabel', nargs=1, default = 'X', help='X label name')
    plot_parser.add_argument('--nwedge', dest='nwedge', default=0, type=int, help='Number of wedges')
    plot_parser.add_argument('--wedge_lims', dest='wedge_lims', default = [0.4, 1.4, 80, 110], type=float, nargs=4)
    plot_parser.add_argument('--file_max', dest='file_max', default = None, type=int)
    plot_parser.add_argument('--frame_range', dest='frame_range', default = [None, None], nargs=2, type=int)
    plot_parser.add_argument('--bipolar', dest='bipolar', default = False, action='store_true')
    plot_parser.add_argument('--half', dest='half', action='store_true', default=False, help='True if you want half a polar plot')
    plot_parser.add_argument('--no_cbar', dest='no_cbar', help="Set if ignore cbar", action='store_true', default=False)
    plot_parser.add_argument('--cbar_orient', dest='cbar_orient', default='vertical', type=str, help='Colorbar orientation', choices=['horizontal', 'vertical'])
    plot_parser.add_argument('--sub_split', dest='sub_split', default = None, nargs='+', type=int)
    plot_parser.add_argument('--tau_s', dest='tau_s', action= 'store_true', default=False, help='The shock optical depth')
    plot_parser.add_argument('--transparent', help='transparent bg flag', default=False, action='store_true')
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
    
    flist, frame_count = util.get_file_list(args.files)
    flist              = flist[args.frame_range[0]: args.frame_range[1]]
    frame_count        = len(flist)
    cbar               = args.cbar 
    num_fields         = len(args.fields)
    if num_fields > 1:
        cbar += (num_fields - 1) * [None, None]
    
    # read the first file and infer the system configuration from it
    init_setup = util.read_file(args, flist[0], ndim=2)[1]
    cartesian = False
    if init_setup["is_cartesian"]:
        fig, ax = plt.subplots(1, 1, figsize=(11,10))
        ax.grid(False)
        ax.set_aspect('equal')
        divider = make_axes_locatable(ax)
        if not args.pictorial:
            cbaxes = divider.append_axes('right', size='5%', pad=0.05)
        cartesian = True
    else:
        if args.nwedge > 0:
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
                if not args.pictorial:
                    cbaxes  = fig.add_axes([0.8, 0.1, 0.03, 0.8]) 
    
    
    def init_mesh():
        if not args.pictorial:
            p = create_mesh(fig, ax, flist[0], cbaxes, args)
        else:
            p = create_mesh(fig, ax, flist[0], None, args)
        return p
    
    drawings  = init_mesh()
    ticks_loc = ax.get_yticks().tolist()
    def update(frame, args):
        """
        Animation function. Takes the current frame number (to select the potion of
        data to plot) and a line object to update.
        """
        if init_setup['mesh_motion']:
            if not args.pictorial:
                try:
                    for cbax in cbaxes:
                        cbax.cla()
                except TypeError:
                    cbaxes.cla()
                
            try:
                for axs in ax:
                    axs.cla()
            except TypeError:
                ax.cla()
                
            ax.grid(False)
            if not args.pictorial:
                p = create_mesh(fig, ax, flist[frame], cbaxes, args)
            else:
                p = create_mesh(fig, ax, flist[frame], None, args)
            
            return p
        else:
            setups, data = get_data(flist[frame], args)
            time = setups['time'] * (util.time_scale if args.units else 1.0)
            if cartesian:
                ax.set_title('{} at t = {:.2f}'.format(args.setup, time), fontsize=20)
            else:
                fig.suptitle('{} at t = {:.2f}'.format(args.setup, setups['time']), fontsize=20, y=1.0)
            
            for drawing in drawings:
                drawing.set_array(data.ravel())
            return drawings,
        

    animation = FuncAnimation(
        # Your Matplotlib Figure object
        fig,
        # The function that does the updating of the Figure
        update,
        # Frame information (here just frame number)
        np.arange(frame_count),
        # blit = True,
        # init_func=init_mesh,
        # Extra arguments to the animate function
        fargs=[args],
        # repeat=False,
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval= 1000 / 10,
        repeat=True,
    )

    if not args.save:
        plt.show()
    else:
        animation.save("{}.mp4".format(args.save.replace(" ", "_")),
                       progress_callback = lambda i, n: print(f'Saving frame {i} of {n}', end='\r', flush=True))
                    #    savefig_kwargs={"transparent": args.transparent, "facecolor": "none"},)
    