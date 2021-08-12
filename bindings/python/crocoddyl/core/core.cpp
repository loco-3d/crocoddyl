///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, LAAS-CNRS, University of Trento
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
  exposeIntegratedActionAbstract();
  exposeDifferentialActionAbstract();
  exposeResidualAbstract();
  exposeActivationAbstract();
  exposeSquashingAbstract();
  exposeSquashingSmoothSat();
  exposeActuationSquashing();
  exposeDataCollectorActuation();
  exposeIntegratedActionEuler();
  exposeIntegratedActionRK4();
  exposeCostAbstract();
  exposeResidualControl();
  exposeCostSum();
  exposeCostResidual();
  exposeCostControl();
  exposeActionNumDiff();
  exposeDifferentialActionNumDiff();
  exposeActivationNumDiff();
  exposeShootingProblem();
  exposeSolverAbstract();
  exposeStateEuclidean();
  exposeControlParametrizationPolyZero();
  exposeControlParametrizationPolyOne();
  exposeControlParametrizationPolyTwoRK4();
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
  exposeCallbacks();
  exposeStopWatch();
}

}  // namespace python
}  // namespace crocoddyl
