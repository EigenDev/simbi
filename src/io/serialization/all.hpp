// =============================================================================
// all.hpp
//
// convenience header for all hdf5 serialization specializations.
// this file simply includes all other headers in the `io/serialization`
// directory, providing a single point of inclusion for all types that have
// implemented the `h5_serializable` trait.
//
// usage:
//   #include "io/serialization/all.hpp"
// =============================================================================
#pragma once

// convenience header that includes all serialization specializations
#include "body_serial.hpp"
#include "domain_serial.hpp"
#include "field_serial.hpp"
#include "hydro_serial.hpp"
#include "mesh_serial.hpp"
#include "metadata_serial.hpp"
