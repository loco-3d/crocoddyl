///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, LAAS-CNRS,
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
  exposeObserverAbstract();
  exposeDynamicsAbstract();
#ifdef CROCODDYL_WITH_CODEGEN
  exposeActionCodeGen();
#endif
  exposeIntegratorTime();
  exposeIntegratedActionAbstract();
  exposeDifferentialActionAbstract();
  exposeResidualAbstract();
  exposeTaskAbstract();
  exposeGuidanceAbstract();
  exposeGuidanceLinear();
  exposeGuidanceSmoothSaturation();
  exposeGuidanceComponentwiseSaturation();
  exposeActivationAbstract();
  exposeSquashingAbstract();
  exposeSquashingSmoothSat();
  exposeActuationSquashing();
  exposeParamsAbstract();
  exposeParamsData();
  exposeParameterManager();
  exposeIntegratorTimeoptParams();
  exposeDataCollectorActuation();
  exposeDataCollectorJoint();
  exposeIntegratedActionEuler();
  exposeIntegratedActionRK();
  exposeDiscretizedAction();
  exposeCostAbstract();
  exposeResidualControl();
  exposeResidualJointEffort();
  exposeResidualJointAcceleration();
  exposeResidualTaskFirstOrder();
  exposeResidualTaskSecondOrder();
  exposeResidualParameters();
  exposeCostSum();
  exposeCostResidual();
  exposeConstraintAbstract();
  exposeConstraintManager();
  exposeParameterPhase();
  exposeConstraintResidual();
  exposeIntegratedObserverAbstract();
  exposeIntegratedObserverEuler();
  exposeIntegratedObserverRK();
  exposeDiscretizedObserver();
  exposeResidualNumDiff();
  exposeActuationNumDiff();
  exposeActionNumDiff();
  exposeDynamicsNumDiff();
  exposeObserverNumDiff();
  exposeDifferentialActionNumDiff();
  exposeActivationNumDiff();
  exposeStateNumDiff();
  exposeProblemAbstract();
  exposeShootingProblem();
  exposeObservationProblem();
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
  exposeActivationWeightedSmooth1Norm();
  exposeActivationSmooth2Norm();
  exposeActivation2NormBarrier();
  exposeSolverKKT();
  exposeSolverDDP();
  exposeSolverFDDP();
#ifdef CROCODDYL_WITH_ODYN
  exposeSolverOdynSQP();
#endif
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
