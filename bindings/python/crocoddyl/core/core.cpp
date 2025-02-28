///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeCore() {
  exposeDataCollector();
  exposeStateAbstract();
  exposeControlParametrizationAbstract();
  exposeActuationAbstract();
  exposeActionAbstract();
#ifdef CROCODDYL_WITH_CODEGEN
  exposeActionCodeGen();
#endif
  exposeIntegratedActionAbstract();
  exposeDifferentialActionAbstract();
  exposeResidualAbstract();
  exposeActivationAbstract();
  exposeSquashingAbstract();
  exposeSquashingSmoothSat();
  exposeActuationSquashing();
  exposeDataCollectorActuation();
  exposeDataCollectorJoint();
  exposeIntegratedActionEuler();
  exposeIntegratedActionRK();
  exposeCostAbstract();
  exposeResidualControl();
  exposeResidualJointEffort();
  exposeResidualJointAcceleration();
  exposeCostSum();
  exposeCostResidual();
  exposeConstraintAbstract();
  exposeConstraintManager();
  exposeConstraintResidual();
  exposeActionNumDiff();
  exposeDifferentialActionNumDiff();
  exposeActivationNumDiff();
  exposeStateNumDiff();
  exposeShootingProblem();
  exposeSolverAbstract();
  exposeStateEuclidean();
  exposeControlParametrizationPolyZero();
  exposeControlParametrizationPolyOne();
  exposeControlParametrizationPolyTwoRK();
  exposeActionUnicycle();
  exposeActionLQR();
  exposeDifferentialActionLQR();
  exposeActivationQuad();
  exposeActivationQuadFlatExp();
  exposeActivationQuadFlatLog();
  exposeActivationWeightedQuad();
  exposeActivationQuadraticBarrier();
  exposeActivationWeightedQuadraticBarrier();
  exposeActivationSmooth1Norm();
  exposeActivationSmooth2Norm();
  exposeActivation2NormBarrier();
  exposeSolverKKT();
  exposeSolverDDP();
  exposeSolverFDDP();
  exposeSolverBoxQP();
  exposeSolverBoxDDP();
  exposeSolverBoxFDDP();
  exposeSolverIntro();
#ifdef CROCODDYL_WITH_IPOPT
  exposeSolverIpopt();
#endif
  exposeCallbacks();
  exposeException();
  exposeStopWatch();
}

}  // namespace python
}  // namespace crocoddyl
