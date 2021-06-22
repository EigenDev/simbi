/**
 * Implementation for the 2D coordinate lattice 
 * header file
*/
#include "clattice.h"
#include <cmath>
#include <iostream>

simbi::CLattice::CLattice () {}

simbi::CLattice::CLattice(std::vector<double> &x1, std::vector<double> &x2, simbi::Geometry geom)
{
    this->x1ccenters = x1;
    this->x2ccenters = x2;
    this->_geom      = geom;
}

simbi::CLattice::~CLattice () {}

void simbi::CLattice::set_nx1_zones()
{
    nx1zones = x1ccenters.size();
}

void simbi::CLattice::set_nx2_zones()
{
    nx2zones = x2ccenters.size();
}

void simbi::CLattice::compute_x1_vertices(simbi::Cellspacing spacing)
{
    x1vertices.resize(nx1zones + 1);
    x1vertices[0]        = x1ccenters[0];
    x1vertices[nx1zones] = x1ccenters[nx1zones - 1];
    switch (spacing)
    {
        case simbi::Cellspacing::LOGSPACE:
            for (size_t i = 1; i < x1ccenters.size(); i++)
            {
                x1vertices[i] = std::sqrt(x1ccenters[i] * x1ccenters[i-1]);
            }
            break;
        
        case simbi::Cellspacing::LINSPACE:
            for (size_t i = 1; i < x1ccenters.size(); i++)
            {
                x1vertices[i] = 0.5 * (x1ccenters[i] + x1ccenters[i-1]);
            }
            
            break;
    }
    
}

void simbi::CLattice::compute_x2_vertices(simbi::Cellspacing spacing)
{
    x2vertices.resize(nx2zones + 1);
    x2vertices[0]        = x2ccenters[0];
    x2vertices[nx2zones] = x2ccenters[nx2zones - 1];
    switch (spacing)
    {
        case simbi::Cellspacing::LOGSPACE:
            for (size_t i = 1; i < x2ccenters.size(); i++)
            {
                x2vertices[i] = std::sqrt(x2ccenters[i] * x2ccenters[i-1]);
            }
            break;
        
        case simbi::Cellspacing::LINSPACE:
            for (size_t i = 1; i < x2ccenters.size(); i++)
            {
                x2vertices[i] = 0.5 * (x2ccenters[i] + x2ccenters[i-1]);
            }
            
            break;
    }
    
}

void simbi::CLattice::compute_x1face_areas()
{
    x1_face_areas.reserve((nx1zones + 1));
    double tl, tr, dcos;
    for(auto &xvertex: x1vertices)
    {
        x1_face_areas.push_back(xvertex * xvertex);
    }
}

void simbi::CLattice::compute_x2face_areas()
{
    x2_face_areas.reserve((nx2zones + 1));
    double rl, rr, rdiff;
    for(auto &yvertex: x2vertices)
    {
        x2_face_areas.push_back(std::sin(yvertex));
    }
}

void simbi::CLattice::compute_s1face_areas()
{
    s1_face_areas.reserve((nx1zones + 1)*nx2zones);
    double tl, tr, dcos;
    for (int jj = 1; jj < nx2zones + 1; jj++)
    {
        tl   = x2vertices[jj - 1];
        tr   = x2vertices[jj];
        dcos = std::cos(tl) - std::cos(tr);
        for(auto &xvertex: x1vertices)
        {
            s1_face_areas.push_back(xvertex * xvertex * dcos);
        }
    }
}

void simbi::CLattice::compute_s2face_areas()
{
    s2_face_areas.reserve((nx2zones + 1)*nx1zones);
    double rl, rr, rdiff;
    for(auto &yvertex: x2vertices)
    {
        for (int ii = 1; ii < nx1zones + 1; ii++)
        {
            rl    = x1vertices[ii - 1];
            rr    = x1vertices[ii];
            rdiff = 0.5*(rr*rr - rl*rl);
            s2_face_areas.push_back(std::sin(yvertex) * rdiff);
        }
    }
}

void simbi::CLattice::compute_dx1()
{
    dx1.reserve(nx1zones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx1.push_back(x1vertices[ii] - x1vertices[ii - 1]);
    }
    
}

void simbi::CLattice::compute_dx2()
{
    dx2.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx2.push_back(x2vertices[ii] - x2vertices[ii - 1]);
    }
    
}

void simbi::CLattice::compute_dV1()
{
    double rr, rl, rmean, dr, dV;
    dV1.reserve(nx1zones);
    int nvx = nx1zones + 1;
    for (int ii = 1; ii < nvx; ii++)
    {
        rr = x1vertices[ii];
        rl = x1vertices[ii - 1];

        dr = rr - rl;
        rmean = 0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
        dV = rmean * rmean * dr;
        
        dV1.push_back(dV);
    }
    
}

void simbi::CLattice::compute_dV2()
{
    double x2mean, x2r, x2l, dx2_bar;
    dV2.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t jj = 1; jj < size; jj++)
    {
        x2r = x2vertices[jj];
        x2l = x2vertices[jj - 1];
        x2mean = 0.5*(x2r + x2l);
        // x2mean = 
        //     (std::sin(x2r) - x2r*std::cos(x2r) - std::sin(x2l) + x2l*std::cos(x2l))/(std::cos(x2l) - std::cos(x2r));
        dx2_bar = std::sin(x2mean)*(x2r - x2l);
        dV2.push_back(dx2_bar);
    }
    
}

void simbi::CLattice::compute_dV()
{
    double rr, rl, rmean, tl, tr, dV;
    dVc.reserve(nx1zones*nx2zones);
    int nvx = nx1zones + 1;
    int nvy = nx2zones + 1;
    for (int jj = 1; jj < nvy; jj++){
        tl = x2vertices[jj - 1];
        tr = x2vertices[jj];
        for (int ii = 1; ii < nvx; ii++)
        {
            rr = x1vertices[ii];
            rl = x1vertices[ii - 1];
            dV = (1./3.) * (rr*rr*rr - rl*rl*rl)*(std::cos(tl) - std::cos(tr));
            
            dVc.push_back(dV);
        }
    }
    
}

void simbi::CLattice::compute_cot()
{
    double x2mean, x2r, x2l;
    cot.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t jj = 1; jj < size; jj++)
    {
        x2r = x2vertices[jj];
        x2l = x2vertices[jj - 1];
        x2mean = 0.5 * (x2l + x2r);
        // x2mean = 
        //     (std::sin(x2r) - x2r*std::cos(x2r) - std::sin(x2l) + x2l*std::cos(x2l))/(std::cos(x2l) - std::cos(x2r));
        cot.push_back(std::cos(x2mean)/std::sin(x2mean));
    }
}

void simbi::CLattice::compute_x1mean()
{
    double xr, xl;
    x1mean.reserve(nx1zones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        xr = x1vertices[ii];
        xl = x1vertices[ii - 1];
        x1mean.push_back(0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl));
    }
}

void simbi::CLattice::config_lattice(simbi::Cellspacing xcellspacing, simbi::Cellspacing ycellspacing)
{
    set_nx1_zones();

    set_nx2_zones();

    compute_x1_vertices(xcellspacing);

    compute_x2_vertices(ycellspacing);

    compute_x1face_areas();

    compute_x2face_areas();

    compute_s1face_areas();
    compute_s2face_areas();

    compute_dx1();

    compute_dx2();

    compute_dV1();

    compute_dV2();

    compute_dV();

    compute_x1mean();

    compute_cot();
}

 