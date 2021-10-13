/**
 * Implementation for the 2D coordinate lattice 
 * header file
*/
#include "clattice3D.hpp"
#include <cmath>
#include <iostream>

simbi::CLattice3D::CLattice3D () {}

simbi::CLattice3D::CLattice3D(
    std::vector<real> &x1, 
    std::vector<real> &x2,
    std::vector<real> &x3,
    simbi::Geometry geom)
:
    x1ccenters(x1),
    x2ccenters(x2),
    x3ccenters(x3),
    _geom(geom)
{

}

simbi::CLattice3D::~CLattice3D () {}

void simbi::CLattice3D::set_nx1_zones()
{
    nx1zones = x1ccenters.size();
}

void simbi::CLattice3D::set_nx2_zones()
{
    nx2zones = x2ccenters.size();
}

void simbi::CLattice3D::set_nx3_zones()
{
    nx3zones = x3ccenters.size();
}

void simbi::CLattice3D::compute_x1_vertices(simbi::Cellspacing spacing)
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

void simbi::CLattice3D::compute_x2_vertices(simbi::Cellspacing spacing)
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

void simbi::CLattice3D::compute_x3_vertices(simbi::Cellspacing spacing)
{
    x3vertices.resize(nx3zones + 1);
    x3vertices[0]        = x3ccenters[0];
    x3vertices[nx3zones] = x3ccenters[nx3zones - 1];
    switch (spacing)
    {
        case simbi::Cellspacing::LOGSPACE:
            for (size_t i = 1; i < x3ccenters.size(); i++)
            {
                x3vertices[i] = std::sqrt(x3ccenters[i] * x3ccenters[i-1]);
            }
            break;
        
        case simbi::Cellspacing::LINSPACE:
            for (size_t i = 1; i < x3ccenters.size(); i++)
            {
                x3vertices[i] = 0.5 * (x3ccenters[i] + x3ccenters[i-1]);
            }
            
            break;
    }
    
}

void simbi::CLattice3D::compute_x1face_areas()
{
    x1_face_areas.reserve((nx1zones + 1));
    for(auto &xvertex: x1vertices)
    {
        x1_face_areas.push_back(xvertex * xvertex);
    }
}

void simbi::CLattice3D::compute_x2face_areas()
{
    x2_face_areas.reserve((nx2zones + 1));
    for(auto &yvertex: x2vertices)
    {
        x2_face_areas.push_back(std::sin(yvertex));
    }
}

void simbi::CLattice3D::compute_x3face_areas()
{
    x3_face_areas.reserve((nx3zones + 1));
    for(auto &zvertex: x3vertices)
    {
        x3_face_areas.push_back(1.0);
    }
}

void simbi::CLattice3D::compute_s1face_areas()
{
    s1_face_areas.reserve((nx1zones + 1)*nx2zones);
    real tl, tr, dcos;
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

void simbi::CLattice3D::compute_s2face_areas()
{
    s2_face_areas.reserve((nx2zones + 1)*nx1zones);
    real rl, rr;
    for(auto &yvertex: x2vertices)
    {
        for (int ii = 1; ii < nx1zones + 1; ii++)
        {
            rl    = x1vertices[ii - 1];
            rr    = x1vertices[ii];
            s2_face_areas.push_back(std::sin(yvertex) * 0.5*(rr*rr - rl*rl));
        }
    }
}

void simbi::CLattice3D::compute_dx1()
{
    dx1.reserve(nx1zones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx1.push_back(x1vertices[ii] - x1vertices[ii - 1]);
    }
    
}

void simbi::CLattice3D::compute_dx2()
{
    dx2.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx2.push_back(x2vertices[ii] - x2vertices[ii - 1]);
    }
    
}

void simbi::CLattice3D::compute_dx3()
{
    dx3.reserve(nx3zones);
    size_t size = x3vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        dx3.push_back(x3vertices[ii] - x3vertices[ii - 1]);
    }
    
}

void simbi::CLattice3D::compute_dV1()
{
    real rr, rl, rmean, dr, dV;
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

void simbi::CLattice3D::compute_dV2()
{
    real x2mean, x2r, x2l, dx2_bar;
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

void simbi::CLattice3D::compute_dV3()
{
    real x3r, x3l;
    dV3.reserve(nx3zones);
    size_t size = x3vertices.size();
    for (size_t kk = 1; kk < size; kk++)
    {
        x3r = x3vertices[kk];
        x3l = x3vertices[kk - 1];
        dV2.push_back((x3r - x3l));
    }
    
}

void simbi::CLattice3D::compute_dV()
{
    real rr, rl, tl, tr, dV;
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

void simbi::CLattice3D::compute_cot()
{
    real x2mean, x2r, x2l;
    cot.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t jj = 1; jj < size; jj++)
    {
        x2r = x2vertices[jj];
        x2l = x2vertices[jj - 1];
        x2mean = 0.5 * (x2l + x2r);
        cot.push_back(std::cos(x2mean)/std::sin(x2mean));
    }
}

void simbi::CLattice3D::compute_sin()
{
    real x2mean, x2r, x2l;
    sin.reserve(nx2zones);
    size_t size = x2vertices.size();
    for (size_t jj = 1; jj < size; jj++)
    {
        x2r = x2vertices[jj];
        x2l = x2vertices[jj - 1];
        x2mean = 0.5 * (x2l + x2r);
        sin.push_back(std::sin(x2mean));
    }
}

void simbi::CLattice3D::compute_x1mean()
{
    real xr, xl;
    x1mean.reserve(nx1zones);
    size_t size = x1vertices.size();
    for (size_t ii = 1; ii < size; ii++)
    {
        xr = x1vertices[ii];
        xl = x1vertices[ii - 1];
        x1mean.push_back(0.75 * (xr * xr * xr * xr - xl * xl * xl * xl) / (xr * xr * xr - xl * xl * xl));
    }
}

void simbi::CLattice3D::config_lattice(
    simbi::Cellspacing xcellspacing, 
    simbi::Cellspacing ycellspacing,
    simbi::Cellspacing zcellspacing)
{
    set_nx1_zones();
    set_nx2_zones();
    set_nx3_zones();

    compute_x1_vertices(xcellspacing);
    compute_x2_vertices(ycellspacing);
    compute_x3_vertices(zcellspacing);

    compute_x1face_areas();
    compute_x2face_areas();
    compute_x3face_areas();

    compute_s1face_areas();
    compute_s2face_areas();

    compute_dx1();
    compute_dx2();
    compute_dx3();

    compute_dV1();
    compute_dV2();
    // compute_dV3();

    compute_x1mean();
    compute_cot();
    compute_sin();
}

 