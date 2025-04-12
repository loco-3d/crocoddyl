///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_FWD_HPP_
#define CROCODDYL_CORE_FWD_HPP_

#include "crocoddyl/core/pch.hpp"

namespace crocoddyl {

inline bool& enableMultithreading() {
  static bool enable = true;
  return enable;
}

enum AssignmentOp { setto, addto, rmfrom };

inline bool is_a_AssignmentOp(AssignmentOp op) {
  return (op == setto || op == addto || op == rmfrom);
}

// action
template <typename Scalar>
class ActionModelAbstractTpl;

template <typename Scalar>
struct ActionDataAbstractTpl;

template <typename Scalar>
class ActionModelUnicycleTpl;
template <typename Scalar>
struct ActionDataUnicycleTpl;

template <typename Scalar>
class ActionModelLQRTpl;
template <typename Scalar>
struct ActionDataLQRTpl;

// differential action
template <typename Scalar>
class DifferentialActionModelAbstractTpl;
template <typename Scalar>
struct DifferentialActionDataAbstractTpl;

template <typename Scalar>
class DifferentialActionModelLQRTpl;
template <typename Scalar>
struct DifferentialActionDataLQRTpl;

// integrated action
template <typename Scalar>
class IntegratedActionModelAbstractTpl;
template <typename Scalar>
struct IntegratedActionDataAbstractTpl;

template <typename Scalar>
class IntegratedActionModelEulerTpl;
template <typename Scalar>
struct IntegratedActionDataEulerTpl;

template <typename Scalar>
class IntegratedActionModelRKTpl;
template <typename Scalar>
struct IntegratedActionDataRKTpl;

// residual
template <typename Scalar>
class ResidualModelAbstractTpl;
template <typename Scalar>
struct ResidualDataAbstractTpl;

// activation
template <typename Scalar>
struct ActivationBoundsTpl;
template <typename Scalar>
class ActivationModelQuadraticBarrierTpl;
template <typename Scalar>
struct ActivationDataQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelWeightedQuadraticBarrierTpl;

template <typename Scalar>
class ActivationModelQuadTpl;

template <typename Scalar>
class ActivationModelQuadFlatExpTpl;
template <typename Scalar>
struct ActivationDataQuadFlatExpTpl;

template <typename Scalar>
class ActivationModelQuadFlatLogTpl;
template <typename Scalar>
struct ActivationDataQuadFlatLogTpl;

template <typename Scalar>
class ActivationModelWeightedQuadTpl;
template <typename Scalar>
struct ActivationDataWeightedQuadTpl;

template <typename Scalar>
class ActivationModelSmooth1NormTpl;
template <typename Scalar>
struct ActivationDataSmooth1NormTpl;

template <typename Scalar>
class ActivationModelSmooth2NormTpl;
template <typename Scalar>
struct ActivationDataSmooth2NormTpl;

template <typename Scalar>
class ActivationModel2NormBarrierTpl;
template <typename Scalar>
struct ActivationData2NormBarrierTpl;

template <typename Scalar>
class ActivationModelAbstractTpl;
template <typename Scalar>
struct ActivationDataAbstractTpl;

// state
template <typename Scalar>
class StateAbstractTpl;

template <typename Scalar>
class StateVectorTpl;

// control
template <typename Scalar>
class ControlParametrizationModelAbstractTpl;
template <typename Scalar>
struct ControlParametrizationDataAbstractTpl;

template <typename Scalar>
class ControlParametrizationModelPolyZeroTpl;

template <typename Scalar>
class ControlParametrizationModelPolyOneTpl;
template <typename Scalar>
struct ControlParametrizationDataPolyOneTpl;

template <typename Scalar>
class ControlParametrizationModelPolyTwoRKTpl;
template <typename Scalar>
struct ControlParametrizationDataPolyTwoRKTpl;

// actuation
template <typename Scalar>
class ActuationModelAbstractTpl;
template <typename Scalar>
struct ActuationDataAbstractTpl;

template <typename Scalar>
class ActuationSquashingModelTpl;
template <typename Scalar>
struct ActuationSquashingDataTpl;

// squashing
template <typename Scalar>
class SquashingModelAbstractTpl;
template <typename Scalar>
struct SquashingDataAbstractTpl;

template <typename Scalar>
class SquashingModelSmoothSatTpl;

// joint
template <typename Scalar>
struct JointDataAbstractTpl;

// data collector
template <typename Scalar>
struct DataCollectorAbstractTpl;

template <typename Scalar>
struct DataCollectorActuationTpl;

template <typename Scalar>
struct DataCollectorJointTpl;

template <typename Scalar>
struct DataCollectorJointActuationTpl;

// residual
template <typename Scalar>
class ResidualModelControlTpl;
template <typename Scalar>
struct ResidualDataControlTpl;

template <typename Scalar>
class ResidualModelJointEffortTpl;
template <typename Scalar>
struct ResidualDataJointEffortTpl;

template <typename Scalar>
class ResidualModelJointAccelerationTpl;
template <typename Scalar>
struct ResidualDataJointAccelerationTpl;

// cost
template <typename Scalar>
class CostModelAbstractTpl;
template <typename Scalar>
struct CostDataAbstractTpl;

template <typename Scalar>
struct CostItemTpl;
template <typename Scalar>
class CostModelSumTpl;
template <typename Scalar>
struct CostDataSumTpl;

template <typename Scalar>
class CostModelResidualTpl;
template <typename Scalar>
struct CostDataResidualTpl;

// constraint
template <typename Scalar>
class ConstraintModelAbstractTpl;
template <typename Scalar>
struct ConstraintDataAbstractTpl;

template <typename Scalar>
struct ConstraintItemTpl;
template <typename Scalar>
class ConstraintModelManagerTpl;
template <typename Scalar>
struct ConstraintDataManagerTpl;

template <typename Scalar>
class ConstraintModelResidualTpl;
template <typename Scalar>
struct ConstraintDataResidualTpl;

// shooting
template <typename Scalar>
class ShootingProblemTpl;

// Numdiff
template <typename Scalar>
class ActionModelNumDiffTpl;
template <typename Scalar>
struct ActionDataNumDiffTpl;

template <typename Scalar>
class DifferentialActionModelNumDiffTpl;
template <typename Scalar>
struct DifferentialActionDataNumDiffTpl;

template <typename Scalar>
class ActuationModelNumDiffTpl;
template <typename Scalar>
struct ActuationDataNumDiffTpl;

template <typename Scalar>
class ActivationModelNumDiffTpl;
template <typename Scalar>
struct ActivationDataNumDiffTpl;

template <typename Scalar>
class ResidualModelNumDiffTpl;
template <typename Scalar>
struct ResidualDataNumDiffTpl;

template <typename Scalar>
class ConstraintModelNumDiffTpl;
template <typename Scalar>
struct ConstraintDataNumDiffTpl;

template <typename Scalar>
class StateNumDiffTpl;

template <typename Scalar>
class ControlParametrizationModelNumDiffTpl;
template <typename Scalar>
struct ControlParametrizationDataNumDiffTpl;

template <typename Scalar>
class ActuationModelNumDiffTpl;
template <typename Scalar>
struct ActuationDataNumDiffTpl;

template <typename Scalar>
class ActionModelCodeGenTpl;

template <typename Scalar>
struct ActionDataCodeGenTpl;

/********************Template Instantiation*************/
typedef ActionModelAbstractTpl<double> ActionModelAbstract;
typedef ActionDataAbstractTpl<double> ActionDataAbstract;
typedef ActionModelUnicycleTpl<double> ActionModelUnicycle;
typedef ActionDataUnicycleTpl<double> ActionDataUnicycle;
typedef ActionModelLQRTpl<double> ActionModelLQR;
typedef ActionDataLQRTpl<double> ActionDataLQR;

typedef DifferentialActionModelAbstractTpl<double>
    DifferentialActionModelAbstract;
typedef DifferentialActionDataAbstractTpl<double>
    DifferentialActionDataAbstract;
typedef DifferentialActionModelLQRTpl<double> DifferentialActionModelLQR;
typedef DifferentialActionDataLQRTpl<double> DifferentialActionDataLQR;

typedef IntegratedActionModelAbstractTpl<double> IntegratedActionModelAbstract;
typedef IntegratedActionDataAbstractTpl<double> IntegratedActionDataAbstract;
typedef IntegratedActionModelEulerTpl<double> IntegratedActionModelEuler;
typedef IntegratedActionDataEulerTpl<double> IntegratedActionDataEuler;
typedef IntegratedActionModelRKTpl<double> IntegratedActionModelRK;
typedef IntegratedActionDataRKTpl<double> IntegratedActionDataRK;

typedef ResidualModelAbstractTpl<double> ResidualModelAbstract;
typedef ResidualDataAbstractTpl<double> ResidualDataAbstract;
typedef ResidualModelControlTpl<double> ResidualModelControl;
typedef ResidualDataControlTpl<double> ResidualDataControl;
typedef ResidualModelJointEffortTpl<double> ResidualModelJointEffort;
typedef ResidualDataJointEffortTpl<double> ResidualDataJointEffort;
typedef ResidualModelJointAccelerationTpl<double>
    ResidualModelJointAcceleration;
typedef ResidualDataJointAccelerationTpl<double> ResidualDataJointAcceleration;

typedef ActivationDataQuadraticBarrierTpl<double>
    ActivationDataQuadraticBarrier;
typedef ActivationModelQuadraticBarrierTpl<double>
    ActivationModelQuadraticBarrier;
typedef ActivationBoundsTpl<double> ActivationBounds;
typedef ActivationModelWeightedQuadraticBarrierTpl<double>
    ActivationModelWeightedQuadraticBarrier;
typedef ActivationModelQuadTpl<double> ActivationModelQuad;
typedef ActivationModelQuadFlatExpTpl<double> ActivationModelQuadFlatExp;
typedef ActivationDataQuadFlatExpTpl<double> ActivationDataQuadFlatExp;
typedef ActivationModelQuadFlatLogTpl<double> ActivationModelQuadFlatLog;
typedef ActivationDataQuadFlatLogTpl<double> ActivationDataQuadFlatLog;
typedef ActivationModelWeightedQuadTpl<double> ActivationModelWeightedQuad;
typedef ActivationDataWeightedQuadTpl<double> ActivationDataWeightedQuad;
DEPRECATED(
    "Use ActivationModelSmooth1Norm",
    typedef ActivationModelSmooth1NormTpl<double> ActivationModelSmoothAbs;)
DEPRECATED(
    "Use ActivationDataSmooth1Norm",
    typedef ActivationDataSmooth1NormTpl<double> ActivationDataSmoothAbs;)
typedef ActivationModelSmooth1NormTpl<double> ActivationModelSmooth1Norm;
typedef ActivationDataSmooth1NormTpl<double> ActivationDataSmooth1Norm;
typedef ActivationModelSmooth2NormTpl<double> ActivationModelSmooth2Norm;
typedef ActivationDataSmooth2NormTpl<double> ActivationDataSmooth2Norm;
typedef ActivationModel2NormBarrierTpl<double> ActivationModel2NormBarrier;
typedef ActivationData2NormBarrierTpl<double> ActivationData2NormBarrier;
typedef ActivationModelAbstractTpl<double> ActivationModelAbstract;
typedef ActivationDataAbstractTpl<double> ActivationDataAbstract;

typedef StateAbstractTpl<double> StateAbstract;
typedef StateVectorTpl<double> StateVector;

typedef ControlParametrizationModelAbstractTpl<double>
    ControlParametrizationModelAbstract;
typedef ControlParametrizationDataAbstractTpl<double>
    ControlParametrizationDataAbstract;
typedef ControlParametrizationModelPolyZeroTpl<double>
    ControlParametrizationModelPolyZero;
typedef ControlParametrizationModelPolyOneTpl<double>
    ControlParametrizationModelPolyOne;
typedef ControlParametrizationDataPolyOneTpl<double>
    ControlParametrizationDataPolyOne;
typedef ControlParametrizationModelPolyTwoRKTpl<double>
    ControlParametrizationModelPolyTwoRK;
typedef ControlParametrizationDataPolyTwoRKTpl<double>
    ControlParametrizationDataPolyTwoRK;

typedef ActuationDataAbstractTpl<double> ActuationDataAbstract;
typedef ActuationModelAbstractTpl<double> ActuationModelAbstract;
typedef ActuationSquashingDataTpl<double> ActuationSquashingData;
typedef ActuationSquashingModelTpl<double> ActuationSquashingModel;

typedef SquashingDataAbstractTpl<double> SquashingDataAbstract;
typedef SquashingModelAbstractTpl<double> SquashingModelAbstract;
typedef SquashingModelSmoothSatTpl<double> SquashingModelSmoothSat;

typedef JointDataAbstractTpl<double> JointDataAbstract;

typedef DataCollectorAbstractTpl<double> DataCollectorAbstract;
typedef DataCollectorActuationTpl<double> DataCollectorActuation;
typedef DataCollectorJointTpl<double> DataCollectorJoint;
typedef DataCollectorJointActuationTpl<double> DataCollectorJointActuation;

typedef CostModelAbstractTpl<double> CostModelAbstract;
typedef CostDataAbstractTpl<double> CostDataAbstract;
typedef CostItemTpl<double> CostItem;
typedef CostModelSumTpl<double> CostModelSum;
typedef CostDataSumTpl<double> CostDataSum;
typedef CostModelResidualTpl<double> CostModelResidual;
typedef CostDataResidualTpl<double> CostDataResidual;

typedef ConstraintModelAbstractTpl<double> ConstraintModelAbstract;
typedef ConstraintDataAbstractTpl<double> ConstraintDataAbstract;
typedef ConstraintItemTpl<double> ConstraintItem;
typedef ConstraintModelManagerTpl<double> ConstraintModelManager;
typedef ConstraintDataManagerTpl<double> ConstraintDataManager;
typedef ConstraintModelResidualTpl<double> ConstraintModelResidual;
typedef ConstraintDataResidualTpl<double> ConstraintDataResidual;

typedef ShootingProblemTpl<double> ShootingProblem;

typedef ActionModelNumDiffTpl<double> ActionModelNumDiff;
typedef ActionDataNumDiffTpl<double> ActionDataNumDiff;
typedef ControlParametrizationModelNumDiffTpl<double>
    ControlParametrizationModelNumDiff;
typedef ControlParametrizationDataNumDiffTpl<double>
    ControlParametrizationDataNumDiff;
typedef DifferentialActionModelNumDiffTpl<double>
    DifferentialActionModelNumDiff;
typedef ActuationModelNumDiffTpl<double> ActuationModelNumDiff;
typedef ActuationDataNumDiffTpl<double> ActuationDataNumDiff;
typedef DifferentialActionDataNumDiffTpl<double> DifferentialActionDataNumDiff;
typedef ActivationModelNumDiffTpl<double> ActivationModelNumDiff;
typedef ActivationDataNumDiffTpl<double> ActivationDataNumDiff;
typedef ResidualModelNumDiffTpl<double> ResidualModelNumDiff;
typedef ResidualDataNumDiffTpl<double> ResidualDataNumDiff;
typedef ConstraintModelNumDiffTpl<double> ConstraintModelNumDiff;
typedef ConstraintDataNumDiffTpl<double> ConstraintDataNumDiff;
typedef StateNumDiffTpl<double> StateNumDiff;
typedef ActuationModelNumDiffTpl<double> ActuationModelNumDiff;
typedef ActuationDataNumDiffTpl<double> ActuationDataNumDiff;

typedef ActionModelCodeGenTpl<double> ActionModelCodeGen;
typedef ActionDataCodeGenTpl<double> ActionDataCodeGen;

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_FWD_HPP_
