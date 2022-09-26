///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Trento
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
void exposeControlParametrizationAbstract();
void exposeActuationAbstract();
void exposeActionAbstract();
void exposeDifferentialActionAbstract();
void exposeIntegratedActionAbstract();
void exposeResidualAbstract();
void exposeActivationAbstract();
void exposeSquashingAbstract();
void exposeSquashingSmoothSat();
void exposeActuationSquashing();
void exposeDataCollectorActuation();
void exposeIntegratedActionEuler();
void exposeIntegratedActionRK();
void exposeIntegratedActionRK4();
void exposeCostAbstract();
void exposeResidualControl();
void exposeCostSum();
void exposeCostResidual();
void exposeCostControl();
void exposeConstraintAbstract();
void exposeActionNumDiff();
void exposeDifferentialActionNumDiff();
void exposeActivationNumDiff();
void exposeStateNumDiff();
void exposeShootingProblem();
void exposeSolverAbstract();
void exposeStateEuclidean();
void exposeControlParametrizationPolyZero();
void exposeControlParametrizationPolyOne();
void exposeControlParametrizationPolyTwoRK();
void exposeActionUnicycle();
void exposeActionLQR();
void exposeDifferentialActionLQR();
void exposeActivationQuad();
void exposeActivationQuadFlatExp();
void exposeActivationQuadFlatLog();
void exposeActivationWeightedQuad();
void exposeActivationQuadraticBarrier();
void exposeActivationWeightedQuadraticBarrier();
void exposeActivationSmooth1Norm();
void exposeActivationSmooth2Norm();
void exposeActivation2NormBarrier();
void exposeSolverDDP();
void exposeSolverKKT();
void exposeSolverFDDP();
void exposeSolverBoxQP();
void exposeSolverBoxDDP();
void exposeSolverBoxFDDP();
#ifdef CROCODDYL_WITH_IPOPT
void exposeSolverIpopt();
#endif
void exposeCallbacks();
void exposeStopWatch();

void exposeCore();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CORE_HPP_
