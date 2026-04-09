// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#ifndef COLMPC_PYTHON_HPP_
#define COLMPC_PYTHON_HPP_

#include "colmpc/fwd.hpp"
// include fwd first
#include <eigenpy/eigenpy.hpp>

namespace colmpc {
namespace python {

void exposeStateMultibody();
void exposeDynamicFreeFwd();
void exposeActivationModelQuadExp();
void exposeActivationModelDistanceQuad();
void exposeResidualDistanceCollision();
void exposeResidualDistanceCollision2();
void exposeResidualVelocityAvoidance();

}  // namespace python
}  // namespace colmpc

#endif  // COLMPC_PYTHON_HPP_
