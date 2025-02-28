///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University, University of Trento
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
#ifdef CROCODDYL_WITH_CODEGEN
void exposeActionCodeGen();
#endif
void exposeDifferentialActionAbstract();
void exposeIntegratedActionAbstract();
void exposeResidualAbstract();
void exposeActivationAbstract();
void exposeSquashingAbstract();
void exposeSquashingSmoothSat();
void exposeActuationSquashing();
void exposeDataCollectorActuation();
void exposeDataCollectorJoint();
void exposeIntegratedActionEuler();
void exposeIntegratedActionRK();
void exposeCostAbstract();
void exposeResidualControl();
void exposeResidualJointEffort();
void exposeResidualJointAcceleration();
void exposeCostSum();
void exposeCostResidual();
void exposeConstraintAbstract();
void exposeConstraintManager();
void exposeConstraintResidual();
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
void exposeSolverIntro();
#ifdef CROCODDYL_WITH_IPOPT
void exposeSolverIpopt();
#endif
void exposeCallbacks();
void exposeException();
void exposeStopWatch();

void exposeCore();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CORE_HPP_
