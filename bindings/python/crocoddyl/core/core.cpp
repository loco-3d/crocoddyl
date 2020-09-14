///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeCore() {
  exposeDataCollector();
  exposeStateAbstract();
  exposeActuationAbstract();
  exposeActionAbstract();
  exposeDifferentialActionAbstract();
  exposeActivationAbstract();
  exposeSquashingAbstract();
  exposeSquashingSmoothSat();
  exposeActuationSquashing();
  exposeDataCollectorActuation();
  exposeIntegratedActionEuler();
  exposeIntegratedActionRK4();
  exposeCostAbstract();
  exposeCostSum();
  exposeCostControl();
  exposeActionNumDiff();
  exposeDifferentialActionNumDiff();
  exposeActivationNumDiff();
  exposeShootingProblem();
  exposeSolverAbstract();
  exposeStateEuclidean();
  exposeActionUnicycle();
  exposeActionLQR();
  exposeDifferentialActionLQR();
  exposeActivationQuad();
  exposeActivationWeightedQuad();
  exposeActivationQuadraticBarrier();
  exposeActivationWeightedQuadraticBarrier();
  exposeActivationSmoothAbs();
  exposeActivationCollision();
  exposeSolverKKT();
  exposeSolverDDP();
  exposeSolverFDDP();
  exposeSolverBoxQP();
  exposeSolverBoxDDP();
  exposeSolverBoxFDDP();
  exposeCallbacks();
}

}  // namespace python
}  // namespace crocoddyl
