///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CORE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CORE_HPP_

#include "python/crocoddyl/fwd.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollector();
void exposeStateAbstract();
void exposeActuationAbstract();
void exposeActionAbstract();
void exposeDifferentialActionAbstract();
void exposeActivationAbstract();
void exposeSquashingAbstract();
void exposeSquashingSmoothSat();
void exposeActuationSquashing();
void exposeDataCollectorActuation();
void exposeIntegratedActionEuler();
void exposeIntegratedActionRK4();
void exposeActionNumDiff();
void exposeDifferentialActionNumDiff();
void exposeActivationNumDiff();
void exposeShootingProblem();
void exposeSolverAbstract();
void exposeStateEuclidean();
void exposeActionUnicycle();
void exposeActionLQR();
void exposeDifferentialActionLQR();
void exposeActivationQuad();
void exposeActivationWeightedQuad();
void exposeActivationQuadraticBarrier();
void exposeActivationWeightedQuadraticBarrier();
void exposeActivationSmoothAbs();
void exposeActivationNorm2Barrier();
void exposeSolverDDP();
void exposeSolverKKT();
void exposeSolverFDDP();
void exposeSolverBoxQP();
void exposeSolverBoxDDP();
void exposeSolverBoxFDDP();
void exposeCallbacks();

void exposeCore();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CORE_HPP_
