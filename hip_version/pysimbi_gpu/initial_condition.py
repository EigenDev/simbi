# Module to config the initial condition for the SIMBI
# hydro setup. From here on, I will fragment the code 
# to try and reduce the confusion between functions
import numpy as np 
import h5py 

def load_checkpoint(model, filename, dim):
    
    with h5py.File(filename, 'r+') as hf:
        t = 0
        ds = hf.get("sim_info")
        
        if dim == 1:
            rho         = hf.get("rho")[:]
            v           = hf.get("v")[:]
            p           = hf.get("p")[:]
            nx          = ds.attrs["Nx"]
            model.t     = ds.attrs["current_time"]
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]
            try:
                ad_gamma = ds.attrs["ad_gamma"]
            except:
                ad_gamma = 4./3.
            
            
            h = 1. + ad_gamma*p/(rho*(ad_gamma - 1.0))
            
            W   = 1./np.sqrt(1. - v ** 2)
            model.D   = rho * W 
            model.S   = W**2 * rho*h*v
            model.tau = W**2 * rho*h - p - rho*W
            
            model.u = np.array([model.D, model.S, model.tau])
            
        else:
            rho         = hf.get("rho")[:]
            v1          = hf.get("v1")[:]
            v2          = hf.get("v2")[:]
            p           = hf.get("p")[:]
            nx          = ds.attrs["NX"]
            ny          = ds.attrs["NY"]
            model.t     = ds.attrs["current_time"]
            xmax        = ds.attrs["xmax"]
            xmin        = ds.attrs["xmin"]
            ymax        = ds.attrs["ymax"]
            ymin        = ds.attrs["ymin"]
            try:
                ad_gamma = ds.attrs["ad_gamma"]
            except:
                ad_gamma = 4./3.
            
            
            rho = rho.reshape(ny, nx)
            v1  = v1.reshape(ny, nx)
            v2  = v2.reshape(ny, nx)
            p   = p.reshape(ny, nx)
            
            h = 1. + ad_gamma*p/(rho*(ad_gamma - 1.0))
            
            W   = 1./np.sqrt(1. - (v1*v1 + v2*v2))
            model.D   = rho * W 
            model.S1  = W*W*rho*h*v1 
            model.S2  = W*W*rho*h*v2 
            model.tau = W*W*rho*h - p - rho*W
            
            model.u = np.array([model.D, model.S1, model.S2, model.tau])
            

def initializeModel(model, first_order = False, periodic = False):
    
    # Check if u-array is empty. If it is, generate an array.
    if model.dimensions == 1:
        if not model.u.any():
            if periodic:
                if model.regime == "classical":
                    model.u = np.empty(shape = (model.n_vars, model.Npts), dtype = float)
                    
                    model.u[:, :] = np.array([model.init_rho, model.init_rho*model.init_v, 
                                            model.init_energy])
                else:
                    model.u = np.empty(shape = (model.n_vars, model.Npts), dtype = float)
                    
                    model.u[:, :] = np.array([model.initD, model.initS, 
                                            model.init_tau])
                    
            else:
                if first_order:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.Npts), dtype=float)
                        model.u[:, :] = np.array([model.init_rho, model.init_rho*model.init_v, 
                                            model.init_energy])
                        
                        # Add boundary ghosts
                        right_ghost = model.u[:, -1]
                        left_ghost = model.u[:, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=1)
                        model.u = np.insert(model.u, 0, left_ghost , axis=1)
                        
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.Npts), dtype=float)
                        model.u[:, :] = np.array([model.initD, model.initS, 
                                            model.init_tau])
                        
                        # Add boundary ghosts
                        right_ghost = model.u[:, -1]
                        left_ghost = model.u[:, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=1)
                        model.u = np.insert(model.u, 0, left_ghost , axis=1)
                        
                        
                    
                else:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.Npts), dtype=float)
                        model.u[:, :] = np.array([model.init_rho, model.init_rho*model.init_v, 
                                            model.init_energy])
                        
                        # Add boundary ghosts
                        right_ghost = model.u[:, -1]
                        left_ghost = model.u[:, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[-1], 
                                        (right_ghost, right_ghost) , axis=1)
                        
                        model.u = np.insert(model.u, 0,
                                        (left_ghost, left_ghost) , axis=1)
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.Npts), dtype=float)
                        model.u[:, :] = np.array([model.initD, model.initS, 
                                            model.init_tau])
                        
                        # Add boundary ghosts
                        right_ghost = model.u[:, -1]
                        left_ghost = model.u[:, 0]
                        
                        
                        model.u = np.insert(model.u, model.u.shape[-1], 
                                        (right_ghost, right_ghost) , axis=1)
                        
                        model.u = np.insert(model.u, 0,
                                        (left_ghost, left_ghost) , axis=1)
                        
                
        else:
            if not first_order:
                # Add the extra ghost cells for i-2, i+2
                right_ghost = model.u[:, -1]
                left_ghost = model.u[:, 0]
                model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=1)
                model.u = np.insert(model.u, 0, left_ghost , axis=1)
                
                    
    elif model.dimensions == 2:
        if not model.u.any():
            if periodic:
                model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype = float)
                
                model.u[:, :, :] = np.array([model.init_rho, model.init_rho*model.init_vx,
                                            model.init_rho*model.init_vy, 
                                            model.init_energy])
            else:
                if first_order:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.init_rho, model.init_rho*model.init_vx,
                                                    model.init_rho*model.init_vy, model.init_energy])
                        
                        # Add boundary ghosts
                        right_ghost = model.u[:, :, -1]
                        left_ghost = model.u[:, :, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=2)
                        model.u = np.insert(model.u, 0, left_ghost , axis=2)
                        
                        upper_ghost = model.u[:, 0]
                        bottom_ghost = model.u[:, -1]
                        
                        model.u = np.insert(model.u, model.u.shape[1], bottom_ghost , axis=1)
                        model.u = np.insert(model.u, 0, upper_ghost , axis=1)
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.initD, model.initS1,
                                                    model.initS2, model.init_tau])
                        
                        # Add boundary ghosts
                        bottom_ghost = model.u[:, -1]
                        upper_ghost = model.u[:, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[1], 
                                        bottom_ghost , axis=1)
                        
                        model.u = np.insert(model.u, 0,
                                        upper_ghost , axis=1)
                        
                        
                        left_ghost = model.u[:, :, 0]
                        right_ghost = model.u[:, :, -1]
                        
                        
                        
                        model.u = np.insert(model.u, 0, 
                                        left_ghost , axis=2)
                        
                        model.u = np.insert(model.u, model.u.shape[2],
                                        right_ghost , axis=2)
                        
                    
                else:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.init_rho, model.init_rho*model.init_vx,
                                                    model.init_rho*model.init_vy, model.init_energy])
                        
                        # Add boundary ghosts
                        bottom_ghost = model.u[:, -1]
                        upper_ghost = model.u[:, 0]
                        
                        
                        model.u = np.insert(model.u, model.u.shape[1], 
                                        (bottom_ghost, bottom_ghost) , axis=1)
                        
                        model.u = np.insert(model.u, 0,
                                        (upper_ghost, upper_ghost) , axis=1)
                        
                        left_ghost = model.u[:, :, 0]
                        right_ghost = model.u[:, :, -1]
                        
                        model.u = np.insert(model.u, 0, 
                                        (left_ghost, left_ghost) , axis=2)
                        
                        model.u = np.insert(model.u, model.u.shape[2],
                                        (right_ghost, right_ghost) , axis=2)
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.initD, model.initS1,
                                                    model.initS2, model.init_tau])
                        
                        # Add boundary ghosts
                        bottom_ghost = model.u[:, -1]
                        upper_ghost = model.u[:, 0]
                        
                        model.u = np.insert(model.u, model.u.shape[1], 
                                        (bottom_ghost, bottom_ghost) , axis=1)
                        
                        model.u = np.insert(model.u, 0,
                                        (upper_ghost, upper_ghost) , axis=1)
                        
                        left_ghost = model.u[:, :, 0]
                        right_ghost = model.u[:, :, -1]
                        
                        model.u = np.insert(model.u, 0, 
                                        (left_ghost, left_ghost) , axis=2)
                        
                        model.u = np.insert(model.u, model.u.shape[2],
                                        (right_ghost, right_ghost) , axis=2)
                        
                
        else:
            if not first_order:
                # Add the extra ghost cells for i-2, i+2
                right_ghost = model.u[:, :, -1]
                left_ghost = model.u[:, :, 0]
                
                model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=2)
                model.u = np.insert(model.u, 0, left_ghost , axis=2)
                
    else:
        if not model.u.any():
            if periodic:
                model.u = np.empty(shape = (model.n_vars, model.zNpts, model.yNpts, model.xNpts), dtype = float)
                
                model.u[:, :, :] = np.array([model.init_rho, model.init_rho*model.init_vx,
                                            model.init_rho*model.init_vy, model.init_rho*model.init_vz,
                                            model.init_energy])
            else:
                if first_order:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.zNpts, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :, :] = np.array([model.init_rho, 
                                                     model.init_rho*model.init_vx,
                                                     model.init_rho*model.init_vy,
                                                     model.init_rho*model.init_vz,
                                                     model.init_energy])
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.zNpts, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.initD, model.initS1,
                                                    model.initS2, model.initS3, model.init_tau])
                        
                    # Add boundary ghosts
                    zupper_ghost  = model.u[:, 0]
                    zlower_ghost  = model.u[:,-1]
                    
                    model.u = np.insert(model.u, model.u.shape[1], zupper_ghost , axis=1)
                    model.u = np.insert(model.u, 0, zlower_ghost , axis=1)
                    
                    yupper_ghost = model.u[:, :,  0]
                    ylower_ghost = model.u[:, :, -1]
                    
                    model.u = np.insert(model.u, model.u.shape[2], ylower_ghost , axis=2)
                    model.u = np.insert(model.u, 0, yupper_ghost , axis=2)
                    
                    xleft_ghost  = model.u[:, :, :, 0]
                    xright_ghost = model.u[:, :, :, -1]
                    
                    model.u = np.insert(model.u, model.u.shape[3], xright_ghost , axis=3)
                    model.u = np.insert(model.u, 0, xleft_ghost , axis=3)
                        
                    
                else:
                    if model.regime == "classical":
                        model.u = np.empty(shape = (model.n_vars, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.init_rho, model.init_rho*model.init_vx,
                                                    model.init_rho*model.init_vy, model.init_energy])
                        
                    else:
                        model.u = np.empty(shape = (model.n_vars, model.zNpts, model.yNpts, model.xNpts), dtype=float)
                        model.u[:, :, :] = np.array([model.initD, model.initS1,
                                                    model.initS2, model.initS3, model.init_tau])
                        
                    # Add boundary ghosts
                    zupper_ghost  = model.u[:, 0]
                    zlower_ghost  = model.u[:,-1]
                    
                    model.u = np.insert(model.u, model.u.shape[1], (zupper_ghost, zupper_ghost) , axis=1)
                    model.u = np.insert(model.u, 0, (zlower_ghost, zlower_ghost) , axis=1)
                    
                    yupper_ghost = model.u[:, :,  0]
                    ylower_ghost = model.u[:, :, -1]
                    
                    model.u = np.insert(model.u, model.u.shape[2], (ylower_ghost, ylower_ghost) , axis=2)
                    model.u = np.insert(model.u, 0, (yupper_ghost, yupper_ghost) , axis=2)
                    
                    xleft_ghost  = model.u[:, :, :, 0]
                    xright_ghost = model.u[:, :, :, -1]
                    
                    model.u = np.insert(model.u, model.u.shape[3], (xright_ghost, xright_ghost) , axis=3)
                    model.u = np.insert(model.u, 0, (xleft_ghost, xleft_ghost) , axis=3)
                    
        else:
            if not first_order:
                # Add the extra ghost cells for i-2, i+2
                right_ghost = model.u[:, :, -1]
                left_ghost = model.u[:, :, 0]
                
                model.u = np.insert(model.u, model.u.shape[-1], right_ghost , axis=2)
                model.u = np.insert(model.u, 0, left_ghost , axis=2)
                