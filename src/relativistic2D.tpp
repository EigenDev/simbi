

template <class T>
T UstateSR2D::cons2prim2D(T &u_state2D, T &lorentz_gamma){
    /**
     * Return a 2D matrix containing the primitive
     * variables density (rho), pressure, and
     * velocity (v)
     */
    double rho, S1,S2, S, D, tau, pmin, tol;
    double pressure, W;
    double v1, v2, vtot;
     
    int n_vars     = u_state2D.size();
    int ny_gridpts = u_state2D[0].size();
    int nx_gridpts = u_state2D[0][0].size();

    vector<vector<vector<double> > > prims(n_vars, vector<vector<double> > 
                                            (ny_gridpts, vector<double>(nx_gridpts)));
   
    double epsilon, D_0, S1_0, S2_0, tau_0;
    double S0, Sn, D_n, S1_n, S2_n, tau_n, pn, p0;
    
    for (int jj=0; jj < ny_gridpts; jj++){
        for(int ii=0; ii< nx_gridpts; ii++){
            D   =  u_state2D[0][jj][ii];      // Relativistic Density
            S1  =  u_state2D[1][jj][ii];      // X1-Momentum Denity
            S2  =  u_state2D[2][jj][ii];      // x2-Momentum Density
            tau =  u_state2D[3][jj][ii];      // Energy Density
            W   =  lorentz_gamma[jj][ii]; 

            S = sqrt(S1*S1 + S2*S2);

            pmin = abs(S - tau - D);

            tol = 1.e-6; //D*1.e-12;

            pressure = newton_raphson(pmin, pressure_func, dfdp, tol, D, tau, W, gamma, S);

            v1 = S1/(tau + D + pressure);

            v2 = S2/(tau + D + pressure);

            vtot = sqrt( v1*v1 + v2*v2 );

            W = 1./sqrt(1 - vtot*vtot);

            rho = D/W;

            
            prims[0][jj][ii] = rho;
            prims[1][jj][ii] = v1;
            prims[2][jj][ii] = v2;
            prims[3][jj][ii] = pressure;
            

        }
    }
    

    return prims;
};