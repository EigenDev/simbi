/**
 * Implementation for the 2D coordinate lattice 
 * header file
*/
#include "clattice.h"
#include <cmath>
#include <iostream>

simbi::CLattice::CLattice () {}

simbi::CLattice::CLattice(std::vector<double> &x1, std::vector<double> &x2, simbi::GEOMETRY geom)
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

void simbi::CLattice::compute_x1_vertices(simbi::CELLSPACING spacing)
{
    x1vertices.resize(nx1zones + 1);
    x1vertices[0]        = x1ccenters[0];
    x1vertices[nx1zones] = x1ccenters[nx1zones - 1];
    switch (spacing)
    {
        case simbi::CELLSPACING::LOGSPACE:
            for (size_t i = 1; i < x1ccenters.size(); i++)
            {
                x1vertices[i] = std::sqrt(x1ccenters[i] * x1ccenters[i-1]);
            }
            break;
        
        case simbi::CELLSPACING::LINSPACE:
            for (size_t i = 1; i < x1ccenters.size(); i++)
            {
                x1vertices[i] = 0.5 * (x1ccenters[i] + x1ccenters[i-1]);
            }
            
            break;
    }
    
}

void simbi::CLattice::compute_x2_vertices(simbi::CELLSPACING spacing)
{
    x2vertices.resize(nx2zones + 1);
    x2vertices[0]        = x2ccenters[0];
    x2vertices[nx2zones] = x2ccenters[nx2zones - 1];
    switch (spacing)
    {
        case simbi::CELLSPACING::LOGSPACE:
            for (size_t i = 1; i < x2ccenters.size(); i++)
            {
                x2vertices[i] = std::sqrt(x2ccenters[i] * x2ccenters[i-1]);
            }
            break;
        
        case simbi::CELLSPACING::LINSPACE:
            for (size_t i = 1; i < x2ccenters.size(); i++)
            {
                x2vertices[i] = 0.5 * (x2ccenters[i] + x2ccenters[i-1]);
            }
            
            break;
    }
    
}

void simbi::CLattice::compute_x1face_areas()
{
    x1_face_areas.reserve(nx1zones + 1);
    for(auto &vertex: x1vertices)
    {
        x1_face_areas.push_back(vertex * vertex);
    }
}

void simbi::CLattice::compute_x2face_areas()
{
    x2_face_areas.reserve(nx2zones + 1);
    for(auto &vertex: x2vertices)
    {
        x2_face_areas.push_back(std::sin(vertex));
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
    double rr, rl, rmean, dr;
    dV1.reserve(nx1zones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        rr = x1vertices[ii];
        rl = x1vertices[ii - 1];
        dr = rr - rl;

        rmean = 0.75 * (rr * rr * rr * rr - rl * rl * rl * rl) / (rr * rr * rr - rl * rl * rl);
        dV1.push_back(rmean * rmean * dr);
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
        x2mean = 0.5 *(x2r + x2l);
        dx2_bar = std::sin(x2mean)*(x2r - x2l);
        dV2.push_back(dx2_bar);
    }
    
}

void simbi::CLattice::compute_cot()
{
    double x2mean, x2r, x2l, dx2_bar;
    cot.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t jj = 1; jj < size; jj++)
    {
        x2r = x2vertices[jj];
        x2l = x2vertices[jj - 1];
        x2mean = 0.5 *(x2r + x2l);
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

void simbi::CLattice::config_lattice(simbi::CELLSPACING xcellspacing, simbi::CELLSPACING ycellspacing)
{
    set_nx1_zones();

    set_nx2_zones();

    compute_x1_vertices(xcellspacing);

    compute_x2_vertices(ycellspacing);

    compute_x1face_areas();

    compute_x2face_areas();

    compute_dx1();

    compute_dx2();

    compute_dV1();

    compute_dV2();

    compute_x1mean();

    compute_cot();
}

 