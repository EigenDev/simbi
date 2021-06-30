/**
 * Implementation for the 2D coordinate lattice 
 * header file
*/
#include "clattice_1d.hpp"
#include <cmath>
#include <iostream>

simbi::CLattice1D::CLattice1D () {}

simbi::CLattice1D::CLattice1D(std::vector<double> &x1, simbi::Geometry geom)
{
    this->x1ccenters = x1;
    this->_geom      = geom;
}

simbi::CLattice1D::~CLattice1D () {}

void simbi::CLattice1D::set_nzones()
{
    nzones = x1ccenters.size();
}

void simbi::CLattice1D::compute_x1_vertices(simbi::Cellspacing spacing)
{
    x1vertices.resize(nzones + 1);
    x1vertices[0]        = x1ccenters[0];
    x1vertices[nzones] = x1ccenters[nzones - 1];
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

void simbi::CLattice1D::compute_face_areas()
{
    face_areas.reserve((nzones + 1));
    for(auto &xvertex: x1vertices)
    {
        face_areas.push_back(xvertex * xvertex);
    }
}

void simbi::CLattice1D::compute_dx1()
{
    dx1.reserve(nzones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx1.push_back(x1vertices[ii] - x1vertices[ii - 1]);
    }
    
}

void simbi::CLattice1D::compute_dV()
{
    double rr, rl, rmean, dr;
    dV.reserve(nzones);
    int nvx = nzones + 1;
    for (int ii = 1; ii < nvx; ii++)
    {
        rr = x1vertices[ii];
        rl = x1vertices[ii - 1];

        dr = rr - rl;
        rmean = 0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
        
        dV.push_back(rmean * rmean * dr);
    }
    
}

void simbi::CLattice1D::compute_x1mean()
{
    double xr, xl;
    x1mean.reserve(nzones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        xr = x1vertices[ii];
        xl = x1vertices[ii - 1];
        x1mean.push_back(0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl));
    }
}

void simbi::CLattice1D::config_lattice(simbi::Cellspacing cellspacing)
{
    set_nzones();

    compute_x1_vertices(cellspacing);

    compute_face_areas();

    compute_dx1();

    compute_dV();

    compute_x1mean();
}

 