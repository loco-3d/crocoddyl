///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, ???
//
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <crocoddyl/config.hpp>

#ifdef CROCODDYL_EXPLICIT_INSTANTIATION_INSTANTIATE
#define CROCODDYL_EXPLICIT_INSTANTIATION_EXTERN
#else
#define CROCODDYL_EXPLICIT_INSTANTIATION_EXTERN extern
#endif

#define CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR double
#include CROCODDYL_EXPLICIT_INSTANTIATION_HEADER
#undef CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR
#define CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR float
#include CROCODDYL_EXPLICIT_INSTANTIATION_HEADER

#undef CROCODDYL_EXPLICIT_INSTANTIATION_SCALAR
#undef CROCODDYL_EXPLICIT_INSTANTIATION_EXTERN
#undef CROCODDYL_EXPLICIT_INSTANTIATION_HEADER
#undef CROCODDYL_EXPLICIT_INSTANTIATION_INSTANTIATE
